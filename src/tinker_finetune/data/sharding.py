"""Deterministic sharding, resumable streaming and mixture sampling.

Three things break large fine-tuning runs long before the maths does:

1. **Resharding churn.** Rows are usually split across workers with
   ``index % world_size``. Add one worker and *every* row changes owner, which
   invalidates every per-worker cache and every resume cursor. Here rows are
   hashed into a fixed number of virtual buckets and buckets are placed on
   workers by rendezvous (HRW) hashing, so growing the fleet from ``W`` to
   ``W+1`` moves only about ``1/(W+1)`` of the corpus and nothing else.
2. **Resume that silently skips or repeats data.** :class:`Cursor` records
   epoch, position *and* the shard topology it was produced under, so a resume
   against a different ``world_size`` is reported rather than quietly
   re-reading someone else's rows.
3. **Mixtures that drift.** Multi-corpus training is specified as weights but
   realized as counts; :class:`MixtureSampler` applies temperature, handles
   exhausted sources explicitly, and reports the proportions it *actually*
   emitted so the gap is visible.

Everything is pure Python and deterministic given a seed - the same plan is
reproduced on every worker without a coordination round-trip.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

__all__ = [
    "DEFAULT_BUCKETS",
    "stable_hash",
    "ShardSpec",
    "Cursor",
    "ReshardError",
    "ShardedStream",
    "ExhaustionPolicy",
    "MixtureSource",
    "MixtureSampler",
]

T = TypeVar("T")

DEFAULT_BUCKETS = 1024
_MASK64 = (1 << 64) - 1


def _to_bytes(key: Any) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8", "surrogatepass")
    if isinstance(key, (int, float, bool)) or key is None:
        return repr(key).encode()
    try:
        return json.dumps(key, sort_keys=True, default=repr).encode()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(key).encode()


def stable_hash(key: Any, *, seed: int = 0) -> int:
    """A 64-bit hash that is stable across processes, hosts and interpreter runs.

    ``hash()`` is randomized per process (PYTHONHASHSEED) and must never be used
    for shard placement; this is blake2b with the seed folded into the salt.
    """
    salt = (seed & _MASK64).to_bytes(8, "little")
    digest = hashlib.blake2b(_to_bytes(key), digest_size=8, salt=salt, person=b"tf-shard").digest()
    return int.from_bytes(digest, "little")


def _mix64(x: int) -> int:
    """splitmix64 finalizer - cheap, well-distributed, used for HRW scoring."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


@dataclass(frozen=True)
class ShardSpec:
    """Which slice of a corpus this worker owns.

    Placement is ``row -> bucket -> rank``: the row hash picks one of
    ``num_buckets`` virtual buckets, and each bucket is placed on the rank with
    the highest rendezvous score. Adding or removing a worker therefore only
    relocates the buckets that worker wins or loses.
    """

    rank: int
    world_size: int
    num_buckets: int = DEFAULT_BUCKETS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank {self.rank} outside [0, {self.world_size})")
        if self.num_buckets < self.world_size:
            raise ValueError(
                f"num_buckets ({self.num_buckets}) must be >= world_size ({self.world_size}); "
                "otherwise some workers get no data"
            )

    def bucket_of(self, key: Any) -> int:
        return stable_hash(key, seed=self.seed) % self.num_buckets

    def owner_of_bucket(self, bucket: int) -> int:
        best_rank, best_score = 0, -1
        for rank in range(self.world_size):
            score = _mix64(bucket * 0x100000001B3 ^ _mix64(rank ^ (self.seed & _MASK64)))
            if score > best_score:
                best_rank, best_score = rank, score
        return best_rank

    def owner_of(self, key: Any) -> int:
        return self.owner_of_bucket(self.bucket_of(key))

    def owns(self, key: Any) -> bool:
        return self.owner_of(key) == self.rank

    def buckets(self) -> tuple[int, ...]:
        """Buckets owned by this rank (ascending)."""
        return tuple(b for b in range(self.num_buckets) if self.owner_of_bucket(b) == self.rank)

    def sibling(self, rank: int) -> ShardSpec:
        """The same topology as seen by another rank - handy in tests and planners."""
        return ShardSpec(rank=rank, world_size=self.world_size, num_buckets=self.num_buckets, seed=self.seed)


@dataclass(frozen=True)
class Cursor:
    """Resume position: the *next* item to emit, plus the topology it assumes."""

    epoch: int = 0
    index: int = 0
    world_size: int = 1
    rank: int = 0
    num_buckets: int = DEFAULT_BUCKETS

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.index < 0:
            raise ValueError("cursor epoch/index must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "index": self.index,
            "world_size": self.world_size,
            "rank": self.rank,
            "num_buckets": self.num_buckets,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Cursor:
        known = {k: int(payload[k]) for k in cls.__dataclass_fields__ if k in payload}
        return cls(**known)


class ReshardError(RuntimeError):
    """Raised when a cursor is resumed under a different shard topology."""


class ShardedStream(Generic[T]):
    """A deterministic, resumable view of one shard of a corpus.

    ``shuffle`` reorders *within* the shard using a seed derived from
    ``(seed, epoch)``, so every epoch differs but any worker can reconstruct any
    epoch's order from the cursor alone - no shuffle state to checkpoint.
    """

    def __init__(
        self,
        rows: Sequence[T],
        spec: ShardSpec,
        *,
        key_fn: Callable[[T], Any] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        max_epochs: int | None = 1,
    ) -> None:
        if max_epochs is not None and max_epochs < 0:
            raise ValueError("max_epochs must be non-negative or None")
        self.rows = rows
        self.spec = spec
        self.shuffle = shuffle
        self.seed = seed
        self.max_epochs = max_epochs
        self._key_fn = key_fn or (lambda row: row)
        self._owned: list[int] | None = None

    @property
    def owned_indices(self) -> list[int]:
        """Indices of ``rows`` this rank owns, in corpus order (computed once)."""
        if self._owned is None:
            self._owned = [i for i, row in enumerate(self.rows) if self.spec.owns(self._key_fn(row))]
        return self._owned

    def __len__(self) -> int:
        return len(self.owned_indices)

    def epoch_order(self, epoch: int) -> list[int]:
        order = list(self.owned_indices)
        if self.shuffle and order:
            random.Random(stable_hash(f"{self.seed}:{epoch}", seed=self.seed)).shuffle(order)
        return order

    def iter_from(self, cursor: Cursor | None = None, *, allow_reshard: bool = False) -> Iterator[tuple[T, Cursor]]:
        """Yield ``(row, cursor_after_row)`` starting at ``cursor``.

        The cursor is emitted *after* each row so that persisting it means
        "this row is done" - the semantics that make at-least-once resume safe.
        """
        cursor = cursor or self.start_cursor()
        cursor = self._reconcile(cursor, allow_reshard=allow_reshard)
        epoch = cursor.epoch
        index = cursor.index
        while self.max_epochs is None or epoch < self.max_epochs:
            order = self.epoch_order(epoch)
            if not order:  # empty shard: do not spin forever on an infinite stream
                return
            while index < len(order):
                row = self.rows[order[index]]
                index += 1
                yield row, self._cursor(epoch, index)
            epoch, index = epoch + 1, 0

    def __iter__(self) -> Iterator[T]:
        return (row for row, _ in self.iter_from())

    def start_cursor(self) -> Cursor:
        return self._cursor(0, 0)

    def _cursor(self, epoch: int, index: int) -> Cursor:
        return Cursor(
            epoch=epoch,
            index=index,
            world_size=self.spec.world_size,
            rank=self.spec.rank,
            num_buckets=self.spec.num_buckets,
        )

    def _reconcile(self, cursor: Cursor, *, allow_reshard: bool) -> Cursor:
        same = (
            cursor.world_size == self.spec.world_size
            and cursor.rank == self.spec.rank
            and cursor.num_buckets == self.spec.num_buckets
        )
        if same:
            return cursor
        if not allow_reshard:
            raise ReshardError(
                f"cursor was written for rank {cursor.rank}/{cursor.world_size} "
                f"(buckets={cursor.num_buckets}) but this stream is rank {self.spec.rank}/"
                f"{self.spec.world_size} (buckets={self.spec.num_buckets}); pass "
                "allow_reshard=True to restart the current epoch under the new topology"
            )
        return self._cursor(cursor.epoch, 0)


class ExhaustionPolicy(str, Enum):
    """What to do when a source runs out of rows."""

    cycle = "cycle"  # restart the source (classic upsampling of a small corpus)
    drain = "drain"  # drop it and renormalize the remaining weights
    stop = "stop"  # end the mixture entirely


@dataclass
class MixtureSource(Generic[T]):
    name: str
    rows: Sequence[T]
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mixture source needs a name")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"source {self.name!r}: weight must be finite and >= 0")


@dataclass
class MixtureSampler(Generic[T]):
    """Weighted interleave of several corpora, deterministic given ``seed``.

    ``temperature`` reshapes the weights as ``w ** (1 / T)``: ``T > 1`` flattens
    the mixture (the usual fix for a corpus that would otherwise be 95% English),
    ``T < 1`` sharpens it, and ``T == 0`` degenerates to the single heaviest
    source. Sources whose rows run out are handled by :class:`ExhaustionPolicy`.
    """

    sources: Sequence[MixtureSource[T]]
    temperature: float = 1.0
    seed: int = 0
    policy: ExhaustionPolicy = ExhaustionPolicy.cycle
    _emitted: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("mixture needs at least one source")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("mixture source names must be unique")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if sum(s.weight for s in self.sources) <= 0:
            raise ValueError("mixture weights sum to zero")
        self.policy = ExhaustionPolicy(self.policy)
        self._emitted = {name: 0 for name in names}

    def probabilities(self) -> dict[str, float]:
        """Sampling probability per source after temperature, ignoring exhaustion."""
        return self._probabilities([s for s in self.sources if s.weight > 0])

    def _probabilities(self, live: Sequence[MixtureSource[T]]) -> dict[str, float]:
        live = [s for s in live if s.weight > 0]
        if not live:
            return {}
        if self.temperature == 0:
            best = max(live, key=lambda s: (s.weight, s.name))
            return {best.name: 1.0}
        scaled = {s.name: s.weight ** (1.0 / self.temperature) for s in live}
        total = sum(scaled.values())
        if total == 0 or not math.isfinite(total):  # underflow/overflow at extreme temperatures
            return {s.name: 1.0 / len(live) for s in live}
        return {name: value / total for name, value in scaled.items()}

    def take(self, n: int) -> list[tuple[str, T]]:
        """Draw ``n`` ``(source_name, row)`` pairs; may return fewer if sources run dry."""
        if n < 0:
            raise ValueError("n must be non-negative")
        rng = random.Random(stable_hash(f"mixture:{self.seed}", seed=self.seed))
        live = [s for s in self.sources if s.weight > 0 and s.rows]
        positions = {s.name: 0 for s in live}
        out: list[tuple[str, T]] = []
        while len(out) < n and live:
            probs = self._probabilities(live)
            if not probs:
                break
            names = list(probs)
            pick = rng.choices(names, weights=[probs[name] for name in names], k=1)[0]
            source = next(s for s in live if s.name == pick)
            pos = positions[pick]
            if pos >= len(source.rows):
                if self.policy is ExhaustionPolicy.stop:
                    break
                if self.policy is ExhaustionPolicy.drain:
                    live = [s for s in live if s.name != pick]
                    continue
                positions[pick] = pos = 0  # cycle
            out.append((pick, source.rows[pos]))
            positions[pick] = pos + 1
            self._emitted[pick] += 1
        return out

    def realized_proportions(self) -> dict[str, float]:
        """Fraction of everything emitted so far, per source - compare with :meth:`probabilities`."""
        total = sum(self._emitted.values())
        if total == 0:
            return {name: 0.0 for name in self._emitted}
        return {name: count / total for name, count in self._emitted.items()}
