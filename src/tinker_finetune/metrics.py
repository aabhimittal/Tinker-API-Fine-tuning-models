"""Metrics logging with an optional Weights & Biases backend.

Trainers emit :class:`TrainMetrics` through an ``on_metrics`` callback. A
:class:`MetricsLogger` turns those into a durable record. Implementations:

- :class:`NoOpLogger`   discard (default).
- :class:`JsonlLogger`  append one JSON line per step under the run dir.
- :class:`WandbLogger`  stream to Weights & Biases (optional dependency).

Use :func:`build_logger` to pick one from settings, and :func:`as_callback` to
compose a logger (and any extra callbacks) into a single ``on_metrics`` handler.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from tinker_finetune.config import Settings
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import TrainMetrics

log = get_logger(__name__)


@runtime_checkable
class MetricsLogger(Protocol):
    def log(self, metrics: TrainMetrics) -> None: ...

    def finish(self) -> None: ...


class NoOpLogger:
    def log(self, metrics: TrainMetrics) -> None:  # noqa: D102
        return None

    def finish(self) -> None:  # noqa: D102
        return None


class JsonlLogger:
    """Append metrics as JSONL to ``<run_dir>/metrics.jsonl``."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, metrics: TrainMetrics) -> None:
        self._fh.write(json.dumps(metrics.model_dump(exclude_none=True)) + "\n")
        self._fh.flush()

    def finish(self) -> None:
        self._fh.close()


class WandbLogger:
    """Stream metrics to Weights & Biases. Requires ``pip install wandb``."""

    def __init__(
        self,
        project: str,
        *,
        entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
        mode: str = "online",
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "wandb is not installed. Run `pip install wandb` or unset TF_WANDB_PROJECT."
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=project, entity=entity, name=run_name, config=config or {}, mode=mode
        )

    def log(self, metrics: TrainMetrics) -> None:
        payload = metrics.model_dump(exclude_none=True)
        step = payload.pop("step", None)
        self._run.log(payload, step=step)

    def finish(self) -> None:
        self._run.finish()


def build_logger(
    settings: Settings,
    *,
    run_dir: str | Path | None = None,
    run_name: str | None = None,
    config: dict | None = None,
) -> MetricsLogger:
    """Pick a logger: W&B if configured, else JSONL if a run dir is given, else no-op."""
    if settings.wandb_project and settings.wandb_mode != "disabled":
        try:
            return WandbLogger(
                settings.wandb_project,
                entity=settings.wandb_entity,
                run_name=run_name,
                config=config,
                mode=settings.wandb_mode,
            )
        except RuntimeError as exc:  # pragma: no cover - missing optional dep
            log.warning("W&B disabled: %s", exc)
    if run_dir is not None:
        return JsonlLogger(run_dir)
    return NoOpLogger()


def as_callback(
    logger: MetricsLogger,
    *extra: Callable[[TrainMetrics], None],
) -> Callable[[TrainMetrics], None]:
    """Compose a logger and any extra callbacks into one ``on_metrics`` handler."""

    def _cb(m: TrainMetrics) -> None:
        logger.log(m)
        for fn in extra:
            fn(m)

    return _cb
