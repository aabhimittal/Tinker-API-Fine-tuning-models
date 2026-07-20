"""Direct Preference Optimization (DPO) trainer.

DPO fine-tunes a policy directly on preference pairs, with no reward model and
no sampling loop. For each pair it compares the policy's log-probability of the
*chosen* vs *rejected* response against a frozen *reference* policy:

    logits = beta * ( (logp_pol(chosen) - logp_ref(chosen))
                    - (logp_pol(rejected) - logp_ref(rejected)) )
    loss   = -log_sigmoid(logits)                       # "sigmoid" (standard DPO)

Variants supported: label smoothing (cDPO), IPO loss, and a reference-free mode.

Design note: the actual gradient update is delegated to the backend's
``forward_backward(loss_fn="dpo")`` (real autodiff lives server-side, exactly as
with SFT/RL). The trainer orchestrates, computes the reference log-probs once,
and derives the reported DPO metrics (implicit reward margin and accuracy) from
forward-only ``logprobs`` calls — which is also what makes the whole loop
observable and testable offline against the fake backend.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from tinker_finetune.data.preferences import build_preference_datums
from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import DPOConfig, PreferenceExample, TrainMetrics
from tinker_finetune.tinker_client.client import Datum, TinkerBackend
from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.optim import lr_at_step

log = get_logger(__name__)

MetricsCallback = Callable[[TrainMetrics], None]


def _log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    return min(0.0, x) - math.log1p(math.exp(-abs(x)))


class DPOTrainer:
    def __init__(
        self,
        backend: TinkerBackend,
        reference_backend: TinkerBackend | None,
        tokenizer: Tokenizer,
        config: DPOConfig,
        *,
        checkpoints: CheckpointManager | None = None,
        on_metrics: MetricsCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.backend = backend
        # A frozen snapshot of the initial policy. May be None for reference-free.
        self.reference_backend = reference_backend
        self.tokenizer = tokenizer
        self.config = config
        self.checkpoints = checkpoints
        self.on_metrics = on_metrics
        self._should_stop = should_stop or (lambda: False)

    def _emit(self, m: TrainMetrics) -> None:
        if self.on_metrics:
            self.on_metrics(m)

    def _pair_datums(self, prefs: list[PreferenceExample]) -> list[tuple[Datum, Datum]]:
        return [
            build_preference_datums(p, self.tokenizer, max_seq_len=self.config.max_seq_len)
            for p in prefs
        ]

    def _reference_logprobs(
        self, pairs: list[tuple[Datum, Datum]]
    ) -> tuple[list[float], list[float]]:
        if self.config.reference_free or self.reference_backend is None:
            zeros = [0.0] * len(pairs)
            return zeros, list(zeros)
        chosen = self.reference_backend.logprobs([c for c, _ in pairs])
        rejected = self.reference_backend.logprobs([r for _, r in pairs])
        return chosen, rejected

    def _dpo_loss(
        self, pol_c: float, pol_r: float, ref_c: float, ref_r: float
    ) -> tuple[float, float, bool]:
        """Return (loss, implicit_reward_margin, chosen_preferred)."""
        cfg = self.config
        pol_ratio = pol_c - pol_r
        ref_ratio = ref_c - ref_r
        margin = cfg.beta * (pol_ratio - ref_ratio)
        if cfg.loss_type == "ipo":
            # IPO: squared error toward a 1/(2*beta) target margin.
            diff = (pol_ratio - ref_ratio) - 1.0 / (2.0 * cfg.beta)
            loss = diff * diff
        else:
            eps = cfg.label_smoothing
            loss = -_log_sigmoid(margin) * (1 - eps) - _log_sigmoid(-margin) * eps
        return loss, margin, margin > 0

    def train(self, prefs: list[PreferenceExample]) -> list[TrainMetrics]:
        cfg = self.config
        pairs = self._pair_datums(prefs)
        if not pairs:
            raise ValueError("DPO training needs at least one preference pair.")
        ref_c_all, ref_r_all = self._reference_logprobs(pairs)

        bs = cfg.batch_size
        batches = [
            (pairs[i : i + bs], ref_c_all[i : i + bs], ref_r_all[i : i + bs])
            for i in range(0, len(pairs), bs)
        ]
        total_steps = len(batches) * cfg.epochs
        history: list[TrainMetrics] = []
        step = 0

        log.info(
            "Starting DPO: model=%s epochs=%d pairs=%d beta=%.3f loss=%s ref_free=%s",
            cfg.base_model, cfg.epochs, len(pairs), cfg.beta, cfg.loss_type,
            cfg.reference_free,
        )

        for _epoch in range(cfg.epochs):
            for batch_pairs, ref_c, ref_r in batches:
                if self._should_stop():
                    log.info("Stop requested; halting DPO at step %d", step)
                    return history

                # Forward-only policy log-probs for metric computation.
                pol_c = self.backend.logprobs([c for c, _ in batch_pairs])
                pol_r = self.backend.logprobs([r for _, r in batch_pairs])

                losses, margins, correct = [], [], 0
                for i in range(len(batch_pairs)):
                    loss, margin, pref = self._dpo_loss(pol_c[i], pol_r[i], ref_c[i], ref_r[i])
                    losses.append(loss)
                    margins.append(margin)
                    correct += int(pref)

                # Gradient update: preference-signed datums via the DPO loss_fn.
                update_batch: list[Datum] = []
                for c, r in batch_pairs:
                    update_batch.append(_signed(c, +1.0))
                    update_batch.append(_signed(r, -1.0))
                self.backend.forward_backward(update_batch, loss_fn="dpo")
                lr = lr_at_step(cfg.optim, step, total_steps)
                opt = self.backend.optim_step(cfg.optim, lr)
                step += 1

                n = len(batch_pairs)
                metrics = TrainMetrics(
                    step=step,
                    epoch=round(step / len(batches), 4),
                    loss=round(sum(losses) / n, 6),
                    learning_rate=lr,
                    grad_norm=round(opt.grad_norm, 6),
                    reward_margin=round(sum(margins) / n, 6),
                    reward_accuracy=round(correct / n, 6),
                )
                history.append(metrics)
                self._emit(metrics)

                if (
                    self.checkpoints
                    and cfg.save_every_steps
                    and step % cfg.save_every_steps == 0
                ):
                    self.checkpoints.save(self.backend, step, metrics=metrics.model_dump())

        if self.checkpoints:
            self.checkpoints.save(self.backend, step, metrics=history[-1].model_dump())
        self.backend.save_weights_for_sampler(f"dpo-{cfg.base_model.split('/')[-1]}")
        log.info(
            "DPO complete: %d steps, final margin=%.4f acc=%.3f",
            step, history[-1].reward_margin or 0.0, history[-1].reward_accuracy or 0.0,
        )
        return history


def _signed(datum: Datum, sign: float) -> Datum:
    """Copy a datum with a preference sign in ``advantages`` for the DPO loss_fn."""
    advantages = [sign if w > 0 else 0.0 for w in datum.weights]
    return Datum(
        input_tokens=datum.input_tokens,
        target_tokens=datum.target_tokens,
        weights=datum.weights,
        advantages=advantages,
        metadata={**datum.metadata, "dpo_sign": sign},
    )
