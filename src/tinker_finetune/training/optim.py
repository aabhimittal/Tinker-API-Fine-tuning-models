"""Learning-rate schedules with warmup.

Kept pure and dependency-free so it is trivially unit-testable. ``lr_at_step``
returns the LR to pass into the backend's ``optim_step`` at a given global step.
"""

from __future__ import annotations

import math

from tinker_finetune.models.schemas import OptimConfig


def lr_at_step(cfg: OptimConfig, step: int, total_steps: int) -> float:
    """Return the learning rate for ``step`` (0-indexed) of ``total_steps``.

    Linear warmup for the first ``warmup_ratio`` fraction of steps, then the
    configured decay (cosine / linear / constant) down to ``min_lr_ratio * lr``.
    """
    if total_steps <= 0:
        return cfg.learning_rate

    peak = cfg.learning_rate
    floor = peak * cfg.min_lr_ratio
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    if cfg.lr_schedule == "constant":
        return peak
    if cfg.lr_schedule == "linear":
        return peak + (floor - peak) * progress
    # cosine
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return floor + (peak - floor) * cosine
