"""Token accounting and hard budget ceilings for a fine-tuning run.

The failure that hurts a team is not a job that crashes; it is a job that
quietly runs for eleven hours because someone typed ``epochs=100`` against a
975B-parameter MoE. This module makes spend a first-class, *enforced* quantity:

- :class:`CostModel` prices train and sample tokens per model, defaulting to a
  size-derived estimate so unknown models still get a number rather than a
  shrug.
- :class:`BudgetTracker` accumulates usage thread-safely, fires a one-shot
  warning at ``warn_at`` of the ceiling, and raises :class:`BudgetExceededError`
  the moment a hard limit is crossed.
- :meth:`BudgetTracker.preflight` answers "can this whole job fit?" *before*
  the first step, which is where a budget check is actually worth having.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.registry import get_model, is_supported

log = get_logger(__name__)

__all__ = [
    "CostModel",
    "Usage",
    "BudgetTracker",
    "BudgetExceededError",
    "estimate_sft_tokens",
]


class BudgetExceededError(RuntimeError):
    """Raised when recorded usage crosses a hard ceiling."""

    def __init__(self, limit_name: str, used: float, limit: float) -> None:
        self.limit_name = limit_name
        self.used = used
        self.limit = limit
        super().__init__(f"Budget exceeded: {limit_name} used={used:.4g} limit={limit:.4g}.")


@dataclass(frozen=True)
class Usage:
    train_tokens: int = 0
    sample_tokens: int = 0
    steps: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.train_tokens + other.train_tokens,
            self.sample_tokens + other.sample_tokens,
            self.steps + other.steps,
            self.cost_usd + other.cost_usd,
        )

    @property
    def total_tokens(self) -> int:
        return self.train_tokens + self.sample_tokens

    def to_dict(self) -> dict:
        return {
            "train_tokens": self.train_tokens,
            "sample_tokens": self.sample_tokens,
            "total_tokens": self.total_tokens,
            "steps": self.steps,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CostModel:
    """USD per million tokens, split by train (fwd+bwd) and sample (fwd only).

    Explicit per-model prices win; otherwise the price scales with *active*
    parameters per token, which is what an MoE actually costs to serve - a
    975B/41B-active model prices like a ~41B dense model, not a 975B one.
    """

    train_per_mtok: dict[str, float] = field(default_factory=dict)
    sample_per_mtok: dict[str, float] = field(default_factory=dict)
    usd_per_bparam_mtok: float = 0.02  # train price per billion active params
    sample_discount: float = 0.25      # forward-only ~ a quarter of fwd+bwd
    fallback_active_params_b: float = 8.0

    def _active_params(self, model: str) -> float:
        if is_supported(model):
            return get_model(model).active_params_b
        return self.fallback_active_params_b

    def train_rate(self, model: str) -> float:
        if model in self.train_per_mtok:
            return self.train_per_mtok[model]
        return self.usd_per_bparam_mtok * self._active_params(model)

    def sample_rate(self, model: str) -> float:
        if model in self.sample_per_mtok:
            return self.sample_per_mtok[model]
        return self.train_rate(model) * self.sample_discount

    def cost(self, model: str, *, train_tokens: int = 0, sample_tokens: int = 0) -> float:
        if train_tokens < 0 or sample_tokens < 0:
            raise ValueError("token counts must be non-negative.")
        return (
            train_tokens / 1e6 * self.train_rate(model)
            + sample_tokens / 1e6 * self.sample_rate(model)
        )


def estimate_sft_tokens(*, num_examples: int, avg_tokens: float, epochs: int) -> int:
    """Tokens an SFT run will push through the backend (ignoring packing gains)."""
    if min(num_examples, epochs) < 0 or avg_tokens < 0:
        raise ValueError("estimate inputs must be non-negative.")
    return int(num_examples * avg_tokens * epochs)


class BudgetTracker:
    """Thread-safe usage accumulator with soft warnings and hard ceilings.

    ``None`` for a limit means unlimited. Limits are inclusive: usage exactly
    equal to the ceiling is allowed; the next token is not.
    """

    def __init__(
        self,
        *,
        model: str,
        max_usd: float | None = None,
        max_tokens: int | None = None,
        max_steps: int | None = None,
        warn_at: float = 0.8,
        cost_model: CostModel | None = None,
        on_warn: Callable[[str, float, float], None] | None = None,
    ) -> None:
        for name, value in (("max_usd", max_usd), ("max_tokens", max_tokens), ("max_steps", max_steps)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when set.")
        if not 0 < warn_at <= 1:
            raise ValueError("warn_at must be in (0, 1].")
        self.model = model
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.warn_at = warn_at
        self.costs = cost_model or CostModel()
        self._on_warn = on_warn or (
            lambda name, used, limit: log.warning(
                "Budget warning: %s at %.0f%% (%.4g / %.4g)", name, 100 * used / limit, used, limit
            )
        )
        self._lock = threading.Lock()
        self._usage = Usage()
        self._warned: set[str] = set()

    # -- inspection ------------------------------------------------------
    @property
    def usage(self) -> Usage:
        with self._lock:
            return self._usage

    def _limits(self, usage: Usage) -> list[tuple[str, float, float | None]]:
        return [
            ("cost_usd", usage.cost_usd, self.max_usd),
            ("tokens", float(usage.total_tokens), float(self.max_tokens) if self.max_tokens else None),
            ("steps", float(usage.steps), float(self.max_steps) if self.max_steps else None),
        ]

    def remaining(self) -> dict[str, float | None]:
        usage = self.usage
        return {name: (None if limit is None else max(0.0, limit - used))
                for name, used, limit in self._limits(usage)}

    def fraction_used(self) -> float:
        """Highest fraction of any configured ceiling; 0.0 when unlimited."""
        usage = self.usage
        fractions = [used / limit for _, used, limit in self._limits(usage) if limit]
        return max(fractions) if fractions else 0.0

    def snapshot(self) -> dict:
        return {
            "model": self.model,
            "usage": self.usage.to_dict(),
            "limits": {"max_usd": self.max_usd, "max_tokens": self.max_tokens, "max_steps": self.max_steps},
            "remaining": self.remaining(),
            "fraction_used": round(self.fraction_used(), 4),
        }

    # -- enforcement -----------------------------------------------------
    def would_exceed(self, *, train_tokens: int = 0, sample_tokens: int = 0, steps: int = 0) -> str | None:
        """Name of the ceiling this delta would cross, or ``None``."""
        delta = Usage(train_tokens, sample_tokens, steps,
                      self.costs.cost(self.model, train_tokens=train_tokens, sample_tokens=sample_tokens))
        projected = self.usage + delta
        for name, used, limit in self._limits(projected):
            if limit is not None and used > limit:
                return name
        return None

    def preflight(self, *, train_tokens: int = 0, sample_tokens: int = 0, steps: int = 0) -> None:
        """Raise *before* a job starts if its projected total cannot fit."""
        breach = self.would_exceed(train_tokens=train_tokens, sample_tokens=sample_tokens, steps=steps)
        if breach:
            projected = self.usage + Usage(
                train_tokens, sample_tokens, steps,
                self.costs.cost(self.model, train_tokens=train_tokens, sample_tokens=sample_tokens),
            )
            used = {name: value for name, value, _ in self._limits(projected)}[breach]
            limit = {name: lim for name, _, lim in self._limits(projected)}[breach]
            raise BudgetExceededError(f"projected {breach}", used, float(limit))

    def record(self, *, train_tokens: int = 0, sample_tokens: int = 0, steps: int = 0) -> Usage:
        """Accumulate usage; raises once a hard ceiling is crossed."""
        if min(train_tokens, sample_tokens, steps) < 0:
            raise ValueError("usage deltas must be non-negative.")
        cost = self.costs.cost(self.model, train_tokens=train_tokens, sample_tokens=sample_tokens)
        with self._lock:
            self._usage = self._usage + Usage(train_tokens, sample_tokens, steps, cost)
            usage = self._usage
            breaches = [(n, u, lim) for n, u, lim in self._limits(usage) if lim is not None and u > lim]
            warnings = [
                (n, u, lim) for n, u, lim in self._limits(usage)
                if lim is not None and u >= self.warn_at * lim and n not in self._warned
            ]
            for name, _, _ in warnings:
                self._warned.add(name)
        for name, used, limit in warnings:
            self._on_warn(name, used, float(limit))
        if breaches:
            name, used, limit = breaches[0]
            raise BudgetExceededError(name, used, float(limit))
        return usage
