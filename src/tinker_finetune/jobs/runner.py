"""Job runner: builds the backend + trainer for a job and drives it to completion.

Executed inside the manager's thread pool. Translates a persisted job into a
concrete SFT / RL / DPO run, wiring metric callbacks (persistence + optional
JSONL / Weights & Biases logging) and the cooperative stop event back into the
manager.
"""

from __future__ import annotations

from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.preferences import load_preference_dataset
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.metrics import as_callback, build_logger
from tinker_finetune.models.schemas import (
    DPOConfig,
    JobStatus,
    JobType,
    LoRAConfig,
    RLConfig,
    SFTConfig,
)
from tinker_finetune.reward_models import resolve_reward
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.dpo_trainer import DPOTrainer
from tinker_finetune.training.rl_trainer import RLTrainer
from tinker_finetune.training.sft_trainer import SFTTrainer

log = get_logger(__name__)


def run_job(
    *,
    manager,
    job_id: str,
    config: SFTConfig | RLConfig | DPOConfig,
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
    run_dir = settings.artifacts_dir / "jobs" / job_id
    logger = build_logger(
        settings, run_dir=run_dir, run_name=job_id, config=config.model_dump(mode="json")
    )
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
                     train_path, eval_path, stop, logger)
        elif isinstance(config, RLConfig):
            _run_rl(manager, job_id, config, backend, tokenizer, checkpoints,
                    prompts, reward, stop, logger)
        elif isinstance(config, DPOConfig):
            _run_dpo(manager, job_id, config, backend, tokenizer, checkpoints,
                     train_path, stop, logger)
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
    finally:
        logger.finish()


def _run_sft(manager, job_id, config, backend, tokenizer, checkpoints,
             train_path, eval_path, stop, logger) -> None:
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
    trainer.on_metrics = as_callback(
        logger, lambda m: manager._on_metrics(job_id, m, total_steps=total_steps)
    )
    trainer.train(train, eval_examples)


def _run_rl(manager, job_id, config, backend, tokenizer, checkpoints,
            prompts, reward, stop, logger) -> None:
    if not prompts:
        raise ValueError("RL jobs require a non-empty list of prompts.")
    settings = get_settings()
    reward_fn = resolve_reward(reward, dry_run=settings.dry_run, api_key=settings.tinker_api_key)
    trainer = RLTrainer(
        backend, tokenizer, config, reward_fn,
        checkpoints=checkpoints,
        should_stop=stop.is_set,
    )
    trainer.on_metrics = as_callback(
        logger, lambda m: manager._on_metrics(job_id, m, total_steps=config.iterations)
    )
    trainer.train(prompts)


def _run_dpo(manager, job_id, config, backend, tokenizer, checkpoints,
             train_path, stop, logger) -> None:
    if not train_path:
        raise ValueError("DPO jobs require a preference dataset path.")
    settings = get_settings()
    prefs = load_preference_dataset(train_path)
    # A frozen reference policy: a fresh, un-optimized backend on the same base.
    reference = None
    if not config.reference_free:
        reference = build_backend(
            config.base_model, config.lora,
            dry_run=settings.dry_run, api_key=settings.tinker_api_key,
        )
    trainer = DPOTrainer(
        backend, reference, tokenizer, config,
        checkpoints=checkpoints,
        should_stop=stop.is_set,
    )
    import math

    total_steps = math.ceil(len(prefs) / config.batch_size) * config.epochs
    trainer.on_metrics = as_callback(
        logger, lambda m: manager._on_metrics(job_id, m, total_steps=total_steps)
    )
    trainer.train(prefs)


# JobType re-export for symmetry with the manager API.
__all__ = ["run_job", "JobType"]
