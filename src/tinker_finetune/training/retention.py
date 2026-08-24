"""Checkpoint retention: keep what a rollback would actually need, delete the rest.

A 70B LoRA run that checkpoints every 200 steps fills a disk long before it
finishes, and the usual reflex - "keep the last N" - throws away the only good
checkpoint the moment a run diverges at the end. The policy here answers the
three questions an on-call engineer asks after a bad deploy:

- *What was the most recent state?* ``keep_last``
- *What was the best state?* ``keep_best`` on a recorded metric, ties broken
  toward the later step.
- *Can we bisect the run?* ``keep_every`` retains a coarse ladder of steps.

Deletion is planned first (:func:`plan_retention`) and executed second
(:func:`apply_retention`), so the plan can be logged, diffed or dry-run. The
manifest is rewritten atomically, and entries whose file has vanished are
pruned rather than crashing the run - retention must never be the thing that
kills a training job.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tinker_finetune.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "RetentionPolicy",
    "CheckpointEntry",
    "RetentionPlan",
    "plan_retention",
    "apply_retention",
    "CorruptManifestError",
]


class CorruptManifestError(RuntimeError):
    """Raised when a checkpoint manifest cannot be parsed as a list of entries."""


@dataclass(frozen=True)
class CheckpointEntry:
    """One manifest row, normalized."""

    step: int
    path: str
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> CheckpointEntry:
        if not isinstance(payload, dict) or "step" not in payload or "path" not in payload:
            raise CorruptManifestError(f"manifest entry missing step/path: {payload!r}")
        try:
            step = int(payload["step"])
        except (TypeError, ValueError) as exc:
            raise CorruptManifestError(f"non-integer step: {payload['step']!r}") from exc
        raw = payload.get("metrics") or {}
        metrics: dict[str, float] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if math.isfinite(float(value)):  # NaN/inf must never win a "best" comparison
                    metrics[str(key)] = float(value)
        return cls(step=step, path=str(payload["path"]), metrics=metrics)

    def to_dict(self) -> dict:
        return {"step": self.step, "path": self.path, "metrics": dict(self.metrics)}


@dataclass(frozen=True)
class RetentionPolicy:
    """How many checkpoints to keep, and which ones.

    A checkpoint survives if *any* rule keeps it; the union is deliberate, so
    adding a rule can never delete something the previous policy protected.
    """

    keep_last: int = 3
    keep_best: int = 1
    keep_every: int = 0
    metric: str = "eval_loss"
    mode: str = "min"
    min_age_steps: int = 0
    protect_final: bool = True

    def __post_init__(self) -> None:
        for name in ("keep_last", "keep_best", "keep_every", "min_age_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

    @property
    def keeps_nothing(self) -> bool:
        return self.keep_last == 0 and self.keep_best == 0 and self.keep_every == 0


@dataclass
class RetentionPlan:
    """The decision for every entry, with a reason per kept checkpoint."""

    keep: list[CheckpointEntry] = field(default_factory=list)
    delete: list[CheckpointEntry] = field(default_factory=list)
    reasons: dict[int, tuple[str, ...]] = field(default_factory=dict)
    missing: list[CheckpointEntry] = field(default_factory=list)

    @property
    def bytes_freed_hint(self) -> int:
        total = 0
        for entry in self.delete:
            try:
                total += Path(entry.path).stat().st_size
            except OSError:
                continue
        return total

    def summary(self) -> str:
        kept = ", ".join(
            f"{e.step}({'+'.join(self.reasons.get(e.step, ()))})" for e in sorted(self.keep, key=lambda e: e.step)
        )
        return f"keep [{kept or '-'}] delete {[e.step for e in self.delete]} missing {[e.step for e in self.missing]}"


def _normalize(entries: Iterable) -> list[CheckpointEntry]:
    out: list[CheckpointEntry] = []
    for entry in entries:
        out.append(entry if isinstance(entry, CheckpointEntry) else CheckpointEntry.from_dict(entry))
    # A restarted run can re-emit a step; the last write wins, as on disk.
    deduped: dict[int, CheckpointEntry] = {}
    for entry in out:
        deduped[entry.step] = entry
    return sorted(deduped.values(), key=lambda e: e.step)


def plan_retention(
    entries: Iterable,
    policy: RetentionPolicy,
    *,
    check_exists: bool = False,
) -> RetentionPlan:
    """Decide which checkpoints to keep. Pure - touches the filesystem only if asked."""
    ordered = _normalize(entries)
    plan = RetentionPlan()
    if not ordered:
        return plan

    if check_exists:
        present, missing = [], []
        for entry in ordered:
            (present if Path(entry.path).exists() else missing).append(entry)
        plan.missing = missing
        ordered = present
        if not ordered:
            return plan

    reasons: dict[int, list[str]] = {}
    latest_step = ordered[-1].step

    def mark(entry: CheckpointEntry, reason: str) -> None:
        reasons.setdefault(entry.step, []).append(reason)

    if policy.protect_final:
        mark(ordered[-1], "final")
    for entry in ordered[len(ordered) - policy.keep_last :] if policy.keep_last else []:
        mark(entry, "last")
    if policy.keep_best:
        scored = [e for e in ordered if policy.metric in e.metrics]
        sign = -1.0 if policy.mode == "max" else 1.0
        # Ties break toward the later step: it saw more data for the same score.
        scored.sort(key=lambda e: (sign * e.metrics[policy.metric], -e.step))
        for entry in scored[: policy.keep_best]:
            mark(entry, "best")
    if policy.keep_every:
        for entry in ordered:
            if entry.step % policy.keep_every == 0:
                mark(entry, "every")
    if policy.min_age_steps:
        for entry in ordered:
            if latest_step - entry.step < policy.min_age_steps:
                mark(entry, "young")

    for entry in ordered:
        if entry.step in reasons:
            plan.keep.append(entry)
            plan.reasons[entry.step] = tuple(dict.fromkeys(reasons[entry.step]))
        else:
            plan.delete.append(entry)
    return plan


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)  # rename is atomic: readers see old or new, never half
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_manifest(manifest_path: str | Path) -> list[CheckpointEntry]:
    path = Path(manifest_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CorruptManifestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise CorruptManifestError(f"{path} must contain a JSON list, got {type(payload).__name__}")
    return _normalize(payload)


def apply_retention(
    manifest_path: str | Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
    entries: Sequence | None = None,
) -> RetentionPlan:
    """Apply ``policy`` to a checkpoint manifest, deleting files that lost.

    Returns the plan that was applied. With ``dry_run`` nothing is touched, so
    the same call can be used to preview a policy change against a live run.
    """
    path = Path(manifest_path)
    loaded = _normalize(entries) if entries is not None else load_manifest(path)
    if not loaded:
        return RetentionPlan()
    if policy.keeps_nothing and not policy.protect_final:
        raise ValueError("policy would delete every checkpoint; set keep_last/keep_best or protect_final")

    plan = plan_retention(loaded, policy, check_exists=True)
    if dry_run:
        log.info("retention dry-run: %s", plan.summary())
        return plan

    for entry in plan.delete:
        try:
            target = Path(entry.path)
            if target.is_dir():
                for child in sorted(target.rglob("*"), reverse=True):
                    child.rmdir() if child.is_dir() else child.unlink(missing_ok=True)
                target.rmdir()
            else:
                target.unlink(missing_ok=True)
        except OSError as exc:  # a locked or already-gone file must not fail the run
            log.warning("retention could not delete %s: %s", entry.path, exc)
    _atomic_write_json(path, [e.to_dict() for e in plan.keep])
    log.info("retention applied: %s", plan.summary())
    return plan
