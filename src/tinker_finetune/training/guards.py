"""Run-time guards that stop a training job before it wastes the budget.

Three classes of failure show up in real LoRA runs and none of them announce
themselves in the final metrics:

1. **Numerical blow-ups** - a NaN loss (bad batch, fp16 overflow, a bad LR)
   poisons the optimizer state; every subsequent step is noise, but the loop
   keeps happily reporting ``nan`` for another two hours.
2. **Divergence** - loss trending *up* over a window, usually an LR an order
   of magnitude too high for the LoRA rank in use.
3. **Silent no-ops** - a loss pinned at exactly zero, which almost always
   means the loss mask is all-zero (no supervised tokens) rather than a
   perfectly fit model.

Plus the ordinary one: held-out loss stopped improving, so keep the best
checkpoint and stop paying for more steps.

Guards are pure observers - they return a :class:`GuardDecision` and never
touch the backend - so trainers stay testable and guards stay reusable across
SFT, DPO and RL loops.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from statistics import median

from tinker_finetune.logging_utils import get_logger

log = get_logger(__name__)

__all__ = [
    "GuardDecision",
    "NumericalGuard",
    "EarlyStopping",
    "TrainingGuards",
    "GuardTripped",
]


@dataclass(frozen=True)
class GuardDecision:
    """Truthy exactly when training should stop."""

    stop: bool = False
    code: str | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.stop

    @classmethod
    def go(cls) -> GuardDecision:
        return cls()

    @classmethod
    def halt(cls, code: str, reason: str) -> GuardDecision:
        return cls(True, code, reason)


class GuardTripped(RuntimeError):
    """Raised by callers that prefer a hard failure to a quiet early stop."""

    def __init__(self, decision: GuardDecision) -> None:
        self.decision = decision
        super().__init__(f"[{decision.code}] {decision.reason}")


@dataclass
class NumericalGuard:
    """Watches per-step loss and gradient norm for pathological behaviour."""

    window: int = 20
    spike_factor: float = 5.0
    spike_patience: int = 3          # consecutive spiky steps before stopping
    max_grad_norm: float | None = 1e3
    divergence_factor: float = 1.5   # recent median vs. early median
    min_steps_before_divergence: int = 20
    zero_loss_patience: int = 5
    _losses: deque[float] = field(default_factory=lambda: deque(maxlen=512), repr=False)
    _spikes: int = 0
    _zeros: int = 0

    def observe(self, loss: float | None, grad_norm: float | None = None) -> GuardDecision:
        if loss is not None and not math.isfinite(loss):
            return GuardDecision.halt(
                "non_finite_loss",
                f"Loss became {loss}; optimizer state is unrecoverable from here.",
            )
        if grad_norm is not None and not math.isfinite(grad_norm):
            return GuardDecision.halt(
                "non_finite_grad", f"Gradient norm became {grad_norm}."
            )
        if grad_norm is not None and self.max_grad_norm is not None and grad_norm > self.max_grad_norm:
            return GuardDecision.halt(
                "grad_explosion",
                f"Gradient norm {grad_norm:.4g} exceeds max_grad_norm={self.max_grad_norm:.4g}.",
            )
        if loss is None:
            return GuardDecision.go()

        # Exactly-zero loss is a masking bug far more often than a miracle.
        if loss == 0.0:
            self._zeros += 1
            if self._zeros >= self.zero_loss_patience:
                return GuardDecision.halt(
                    "zero_loss",
                    f"Loss was exactly 0.0 for {self._zeros} consecutive steps; the loss mask is "
                    "probably empty (no supervised tokens).",
                )
        else:
            self._zeros = 0

        recent = list(self._losses)[-self.window :]
        if len(recent) >= max(3, self.window // 4):
            base = median(recent)
            if base > 0 and loss > self.spike_factor * base:
                self._spikes += 1
                if self._spikes >= self.spike_patience:
                    return GuardDecision.halt(
                        "loss_spike",
                        f"Loss spiked to {loss:.4g} vs. running median {base:.4g} for "
                        f"{self._spikes} consecutive steps.",
                    )
            else:
                self._spikes = 0

        self._losses.append(loss)

        if len(self._losses) >= max(self.min_steps_before_divergence, 2 * self.window):
            history = list(self._losses)
            early = median(history[: self.window])
            late = median(history[-self.window :])
            if early > 0 and late > self.divergence_factor * early:
                return GuardDecision.halt(
                    "divergence",
                    f"Loss is trending up (median {early:.4g} -> {late:.4g} over "
                    f"{len(history)} steps); the learning rate is likely too high.",
                )
        return GuardDecision.go()


@dataclass
class EarlyStopping:
    """Patience-based stopping on a held-out metric.

    ``min_delta`` is what makes this useful rather than superstitious: an
    improvement of 1e-9 is noise, not progress, and must not reset patience.
    """

    patience: int = 3
    min_delta: float = 0.0
    mode: str = "min"
    best: float | None = None
    best_step: int | None = None
    _bad: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")
        if self.patience < 0 or self.min_delta < 0:
            raise ValueError("patience and min_delta must be non-negative.")

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def observe(self, value: float, *, step: int | None = None) -> GuardDecision:
        if not math.isfinite(value):
            return GuardDecision.halt("non_finite_eval", f"Eval metric became {value}.")
        if self._improved(value):
            self.best = value
            self.best_step = step
            self._bad = 0
            return GuardDecision.go()
        self._bad += 1
        if self._bad > self.patience:
            return GuardDecision.halt(
                "early_stop",
                f"No improvement over {self.best:.6g} for {self._bad} evaluations "
                f"(patience={self.patience}); best was at step {self.best_step}.",
            )
        return GuardDecision.go()


@dataclass
class TrainingGuards:
    """Composite façade a trainer can hold without knowing the guard zoo."""

    numerical: NumericalGuard | None = field(default_factory=NumericalGuard)
    early_stopping: EarlyStopping | None = None
    tripped: GuardDecision | None = None

    def observe_train(self, *, loss: float | None, grad_norm: float | None = None) -> GuardDecision:
        if self.numerical is None:
            return GuardDecision.go()
        decision = self.numerical.observe(loss, grad_norm)
        return self._record(decision)

    def observe_eval(self, value: float, *, step: int | None = None) -> GuardDecision:
        if self.early_stopping is None:
            return GuardDecision.go()
        return self._record(self.early_stopping.observe(value, step=step))

    def _record(self, decision: GuardDecision) -> GuardDecision:
        if decision.stop:
            self.tripped = decision
            log.warning("Training guard tripped [%s]: %s", decision.code, decision.reason)
        return decision

    def raise_if_tripped(self) -> None:
        if self.tripped is not None:
            raise GuardTripped(self.tripped)
