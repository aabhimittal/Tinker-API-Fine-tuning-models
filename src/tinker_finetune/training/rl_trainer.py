"""Reinforcement-learning trainer (GRPO / REINFORCE style).

The loop mirrors the Tinker RL recipe:

    for each iteration:
        for each prompt:
            sample a *group* of completions            (backend.sample)
            score each with a reward function          (reward_fn)
        compute group-relative advantages              (GRPO) or plain returns
        build advantage-weighted datums                (loss on completion tokens)
        backend.forward_backward(..., loss_fn="importance_sampling")
        backend.optim_step(...)

GRPO advantages are ``(r - mean_group) / (std_group + eps)`` which removes the
need for a learned value function. Rewards come from a user-supplied callable,
so this supports RLHF (reward model), RLVR (verifiable rewards), or heuristics.
"""

from __future__ import annotations

from collections.abc import Callable

from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import RLConfig, TrainMetrics
from tinker_finetune.tinker_client.client import Datum, TinkerBackend
from tinker_finetune.training.checkpoint import CheckpointManager
from tinker_finetune.training.optim import lr_at_step

log = get_logger(__name__)

# reward_fn(prompt, completion_text) -> scalar reward
RewardFn = Callable[[str, str], float]
MetricsCallback = Callable[[TrainMetrics], None]


class RolloutSampler:
    """Samples groups of completions for a prompt and returns tokens + text."""

    def __init__(self, backend: TinkerBackend, tokenizer: Tokenizer, cfg: RLConfig) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.cfg = cfg

    def sample_group(self, prompt: str, *, seed_base: int) -> list[tuple[list[int], str]]:
        prompt_ids = self.tokenizer.encode(prompt, add_special=True)
        out: list[tuple[list[int], str]] = []
        for k in range(self.cfg.group_size):
            res = self.backend.sample(
                prompt_ids,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                seed=seed_base + k,
            )
            text = res.text or self.tokenizer.decode(res.tokens)
            out.append((prompt_ids + res.tokens, text))
        return out


def _group_advantages(rewards: list[float], mode: str, eps: float = 1e-6) -> list[float]:
    """Compute per-sample advantages for one prompt group."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    if mode == "reinforce":
        return [r - mean for r in rewards]  # mean baseline
    # grpo: standardize within the group
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    return [(r - mean) / (std + eps) for r in rewards]


class RLTrainer:
    def __init__(
        self,
        backend: TinkerBackend,
        tokenizer: Tokenizer,
        config: RLConfig,
        reward_fn: RewardFn,
        *,
        checkpoints: CheckpointManager | None = None,
        on_metrics: MetricsCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.checkpoints = checkpoints
        self.on_metrics = on_metrics
        self._should_stop = should_stop or (lambda: False)
        self._sampler = RolloutSampler(backend, tokenizer, config)

    def _emit(self, m: TrainMetrics) -> None:
        if self.on_metrics:
            self.on_metrics(m)

    def _build_datum(
        self, full_tokens: list[int], prompt_len: int, advantage: float
    ) -> Datum:
        """Advantage-weighted datum: loss only on the completion tokens."""
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        # Weight/advantage 0 on prompt positions, `advantage` on completion.
        weights = [0.0 if i + 1 < prompt_len else 1.0 for i in range(len(input_tokens))]
        advantages = [0.0 if i + 1 < prompt_len else advantage for i in range(len(input_tokens))]
        return Datum(input_tokens, target_tokens, weights, advantages=advantages)

    def train(self, prompts: list[str]) -> list[TrainMetrics]:
        cfg = self.config
        if not prompts:
            raise ValueError("RL training needs a non-empty list of prompts.")
        history: list[TrainMetrics] = []
        total_steps = cfg.iterations
        step = 0

        log.info(
            "Starting RL: model=%s iters=%d group_size=%d prompts/batch=%d adv=%s",
            cfg.base_model, cfg.iterations, cfg.group_size, cfg.prompts_per_batch, cfg.advantage,
        )

        for it in range(cfg.iterations):
            if self._should_stop():
                log.info("Stop requested; halting at iteration %d", it)
                break

            batch_prompts = [prompts[(it * cfg.prompts_per_batch + i) % len(prompts)]
                             for i in range(cfg.prompts_per_batch)]
            datums: list[Datum] = []
            all_rewards: list[float] = []

            for p_idx, prompt in enumerate(batch_prompts):
                seed_base = cfg.seed + it * 100_000 + p_idx * 1000
                group = self._sampler.sample_group(prompt, seed_base=seed_base)
                prompt_len = len(self.tokenizer.encode(prompt, add_special=True))
                rewards = [self.reward_fn(prompt, text) for _, text in group]
                all_rewards.extend(rewards)
                advantages = _group_advantages(rewards, cfg.advantage)
                for (full_tokens, _text), adv in zip(group, advantages, strict=True):
                    datums.append(self._build_datum(full_tokens, prompt_len, adv))

            fb = self.backend.forward_backward(datums, loss_fn="importance_sampling")
            lr = lr_at_step(cfg.optim, step, total_steps)
            opt = self.backend.optim_step(cfg.optim, lr)
            step += 1

            reward_mean = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            metrics = TrainMetrics(
                step=step,
                epoch=float(it),
                loss=round(fb.loss, 6),
                learning_rate=lr,
                grad_norm=round(opt.grad_norm, 6),
                tokens=fb.num_tokens,
                reward_mean=round(reward_mean, 6),
                kl=cfg.kl_coef,
            )
            history.append(metrics)
            self._emit(metrics)
            log.info("[rl] iter=%d reward_mean=%.4f loss=%.4f", it, reward_mean, fb.loss)

            if self.checkpoints and cfg.iterations and step % max(1, cfg.iterations // 5) == 0:
                self.checkpoints.save(self.backend, step, metrics=metrics.model_dump())

        if self.checkpoints and history:
            self.checkpoints.save(self.backend, step, metrics=history[-1].model_dump())
        self.backend.save_weights_for_sampler(f"rl-{cfg.base_model.split('/')[-1]}")
        log.info("RL complete: %d iterations", step)
        return history
