"""Job runner: builds the backend + trainer for a job and drives it to completion.

Executed inside the manager's thread pool. Translates a persisted job into a
concrete SFT or RL run, wiring metric callbacks and the cooperative stop event
back into the manager.
"""

from __future__ import annotations

from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import (
    JobStatus,
    JobType,
    LoRAConfig,
    RLConfig,
    SFTConfig,
)
from tinker_finetune.rewards import get_reward_fn
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.rl_trainer import RLTrainer
from tinker_finetune.training.sft_trainer import SFTTrainer

log = get_logger(__name__)


def run_job(
    *,
    manager,
    job_id: str,
    config: SFTConfig | RLConfig,
    prompts: list[str] | None = None,
    train_path: str | None = None,
    eval_path: str | None = None,
    reward: str | None = None,
) -> None:
    settings = get_settings()
    stop = manager.stop_event(job_id)
    if stop.is_set():
        manager._mark(job_id, JobStatus.cancelled, error="Cancelled before start.")
        return

    manager._mark(job_id, JobStatus.running)
    try:
        lora: LoRAConfig = config.lora
        backend = build_backend(
            config.base_model,
            lora,
            dry_run=settings.dry_run,
            api_key=settings.tinker_api_key,
        )
        tokenizer = build_tokenizer(config.base_model, prefer_hf=not settings.dry_run)
        checkpoints = CheckpointManager(settings.artifacts_dir / "checkpoints", job_id)

        if isinstance(config, SFTConfig):
            _run_sft(manager, job_id, config, backend, tokenizer, checkpoints,
                     train_path, eval_path, stop)
        elif isinstance(config, RLConfig):
            _run_rl(manager, job_id, config, backend, tokenizer, checkpoints,
                    prompts, reward, stop)
        else:  # pragma: no cover - guarded by types
            raise TypeError(f"Unsupported config type: {type(config)!r}")

        latest = checkpoints.latest()
        manager._set_outputs(
            job_id,
            checkpoint=latest["path"] if latest else None,
            sampling_model=f"{config.base_model.split('/')[-1]}-{job_id}",
        )
        final_status = JobStatus.cancelled if stop.is_set() else JobStatus.succeeded
        manager._mark(job_id, final_status)
        log.info("Job %s finished with status %s", job_id, final_status.value)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the record
        log.exception("Job %s failed", job_id)
        manager._mark(job_id, JobStatus.failed, error=f"{type(exc).__name__}: {exc}")


def _run_sft(manager, job_id, config, backend, tokenizer, checkpoints,
             train_path, eval_path, stop) -> None:
    if not train_path:
        raise ValueError("SFT jobs require a train dataset path.")
    train = load_chat_dataset(train_path)
    eval_examples = load_chat_dataset(eval_path) if eval_path else None

    trainer = SFTTrainer(
        backend, tokenizer, config,
        checkpoints=checkpoints,
        should_stop=stop.is_set,
    )
    total_steps = trainer.steps_per_epoch(train) * config.epochs
    trainer.on_metrics = lambda m: manager._on_metrics(job_id, m, total_steps=total_steps)
    trainer.train(train, eval_examples)


def _run_rl(manager, job_id, config, backend, tokenizer, checkpoints, prompts, reward, stop) -> None:
    if not prompts:
        raise ValueError("RL jobs require a non-empty list of prompts.")
    reward_fn = get_reward_fn(reward)
    trainer = RLTrainer(
        backend, tokenizer, config, reward_fn,
        checkpoints=checkpoints,
        should_stop=stop.is_set,
    )
    trainer.on_metrics = lambda m: manager._on_metrics(job_id, m, total_steps=config.iterations)
    trainer.train(prompts)


# JobType re-export for symmetry with the manager API.
__all__ = ["run_job", "JobType"]
