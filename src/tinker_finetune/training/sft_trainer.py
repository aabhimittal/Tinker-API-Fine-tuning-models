"""Supervised fine-tuning (SFT) trainer.

Implements the canonical Tinker training loop:

    for each batch:
        result = backend.forward_backward(batch, loss_fn="cross_entropy")
        lr = lr_at_step(...)
        backend.optim_step(optim, lr)

with LR warmup/decay, gradient-norm tracking, periodic checkpointing, optional
periodic evaluation, and a metrics callback so callers (jobs/CLI) can stream
progress without the trainer knowing about them.
"""

from __future__ import annotations

from collections.abc import Callable

from tinker_finetune.data.datasets import iter_batches
from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import ChatExample, SFTConfig, TrainMetrics
from tinker_finetune.tinker_client.client import TinkerBackend
from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.optim import lr_at_step

log = get_logger(__name__)

MetricsCallback = Callable[[TrainMetrics], None]


class SFTTrainer:
    def __init__(
        self,
        backend: TinkerBackend,
        tokenizer: Tokenizer,
        config: SFTConfig,
        *,
        checkpoints: CheckpointManager | None = None,
        on_metrics: MetricsCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.checkpoints = checkpoints
        self.on_metrics = on_metrics
        self._should_stop = should_stop or (lambda: False)

    def _emit(self, m: TrainMetrics) -> None:
        if self.on_metrics:
            self.on_metrics(m)

    def _prepare_batches(self, examples: list[ChatExample]) -> list[list]:
        return list(
            iter_batches(
                examples,
                self.tokenizer,
                batch_size=self.config.batch_size,
                max_seq_len=self.config.max_seq_len,
                pack=self.config.pack_sequences,
            )
        )

    def steps_per_epoch(self, examples: list[ChatExample]) -> int:
        return len(self._prepare_batches(examples))

    def train(
        self,
        train_examples: list[ChatExample],
        eval_examples: list[ChatExample] | None = None,
    ) -> list[TrainMetrics]:
        cfg = self.config
        batches = self._prepare_batches(train_examples)
        if not batches:
            raise ValueError("No training batches were produced from the dataset.")
        total_steps = len(batches) * cfg.epochs
        history: list[TrainMetrics] = []
        step = 0

        log.info(
            "Starting SFT: model=%s epochs=%d batches/epoch=%d total_steps=%d",
            cfg.base_model, cfg.epochs, len(batches), total_steps,
        )

        for _epoch in range(cfg.epochs):
            for batch in batches:
                if self._should_stop():
                    log.info("Stop requested; halting at step %d", step)
                    return history

                fb = self.backend.forward_backward(batch, loss_fn="cross_entropy")
                lr = lr_at_step(cfg.optim, step, total_steps)
                opt = self.backend.optim_step(cfg.optim, lr)
                step += 1

                metrics = TrainMetrics(
                    step=step,
                    epoch=round(step / len(batches), 4),
                    loss=round(fb.loss, 6),
                    learning_rate=lr,
                    grad_norm=round(opt.grad_norm, 6),
                    tokens=fb.num_tokens,
                )
                history.append(metrics)
                self._emit(metrics)

                if cfg.eval_every_steps and eval_examples and step % cfg.eval_every_steps == 0:
                    self._run_eval(eval_examples, step, history)

                if (
                    self.checkpoints
                    and cfg.save_every_steps
                    and step % cfg.save_every_steps == 0
                ):
                    self.checkpoints.save(self.backend, step, metrics=metrics.model_dump())

        # Final checkpoint + sampler snapshot.
        if self.checkpoints:
            self.checkpoints.save(self.backend, step, metrics=history[-1].model_dump())
        self.backend.save_weights_for_sampler(f"sft-{cfg.base_model.split('/')[-1]}")
        log.info("SFT complete: %d steps, final loss=%.4f", step, history[-1].loss or float("nan"))
        return history

    def _run_eval(
        self, eval_examples: list[ChatExample], step: int, history: list[TrainMetrics]
    ) -> None:
        # Lightweight held-out loss via forward_backward's reported loss (no
        # optimizer step). Kept import-local to avoid a cycle.
        batches = self._prepare_batches(eval_examples)
        if not batches:
            return
        losses = [self.backend.forward_backward(b).loss for b in batches]
        eval_loss = sum(losses) / len(losses)
        m = TrainMetrics(step=step, epoch=history[-1].epoch, loss=round(eval_loss, 6))
        log.info("[eval] step=%d held-out loss=%.4f", step, eval_loss)
        self._emit(m)
