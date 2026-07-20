"""Reward-model scoring for RLHF.

RLHF trains a policy against a *reward model* (RM) — a model that maps a
(prompt, completion) to a scalar quality score. This module provides:

- a :class:`RewardModel` protocol (``score(prompt, completion) -> float``),
- :class:`TinkerRewardModel`, which scores with a Tinker-served model,
- :class:`FakeRewardModel`, a deterministic offline scorer,
- :func:`resolve_reward`, which turns a reward *spec* into an RL ``RewardFn``.

Reward specs (used by the CLI/API ``reward`` field):

- ``"length_target"`` / ``"numeric_match"`` / ...  → a built-in heuristic reward
  from :mod:`tinker_finetune.rewards`.
- ``"rm:<model_name>"``                            → score with a reward model,
  e.g. ``"rm:Qwen/Qwen3-8B"`` or ``"rm:thinkingmachines/Inkling-Small"``.

This closes the RLHF loop: sample from the policy (RL trainer) → score with the
reward model → optimize the policy toward higher reward.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from tinker_finetune.data.tokenization import Tokenizer, build_tokenizer
from tinker_finetune.models.schemas import LoRAConfig, Message, Role
from tinker_finetune.rewards import RewardFn, get_reward_fn
from tinker_finetune.tinker_client.client import Datum, TinkerBackend, build_backend


@runtime_checkable
class RewardModel(Protocol):
    def score(self, prompt: str, completion: str) -> float: ...


class FakeRewardModel:
    """Deterministic offline reward model.

    Rewards completions that are non-empty, reasonably long, and lexically
    diverse — a coarse "helpfulness" proxy — with seeded jitter. Bounded to
    roughly [0, 1] so RL advantages stay well-scaled.
    """

    def __init__(self, model_name: str = "fake-rm") -> None:
        self.model_name = model_name

    def score(self, prompt: str, completion: str) -> float:
        text = completion.strip()
        if not text:
            return 0.0
        length = min(1.0, len(text) / 240.0)
        diversity = len(set(text.split())) / (len(text.split()) + 1)
        seed = (abs(hash((prompt, completion))) % 1000) / 1000.0
        score = 0.5 * length + 0.4 * diversity + 0.1 * seed
        return max(0.0, min(1.0, score))


class TinkerRewardModel:
    """Reward model backed by a Tinker-served model.

    Produces a scalar by reading the model's mean per-token log-probability of
    the completion given the prompt and squashing it to (0, 1). A production RM
    would instead load a trained scalar reward head; this logprob proxy keeps the
    same interface and makes the RLHF wiring runnable end-to-end (and offline via
    the fake backend).
    """

    def __init__(self, backend: TinkerBackend, tokenizer: Tokenizer) -> None:
        self.backend = backend
        self.tokenizer = tokenizer

    def _datum(self, prompt: str, completion: str) -> Datum:
        prompt_ids = self.tokenizer.encode(
            self._render(prompt), add_special=True
        )
        completion_ids = self.tokenizer.encode(completion, add_special=False)
        completion_ids.append(self.tokenizer.eos_id)
        tokens = prompt_ids + completion_ids
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        # Supervise only completion positions (mask the prompt).
        weights = [
            1.0 if i + 1 >= len(prompt_ids) else 0.0 for i in range(len(input_tokens))
        ]
        return Datum(input_tokens, target_tokens, weights)

    @staticmethod
    def _render(prompt: str) -> str:
        return f"{Message(role=Role.user, content=prompt).content}"

    def score(self, prompt: str, completion: str) -> float:
        datum = self._datum(prompt, completion)
        total_logprob = self.backend.logprobs([datum])[0]
        n = max(1, datum.num_supervised_tokens)
        mean_lp = total_logprob / n
        # Squash mean log-prob (<= 0) into (0, 1); higher (less negative) => higher.
        return 1.0 / (1.0 + math.exp(-mean_lp))


def build_reward_model(
    model_name: str, *, dry_run: bool, api_key: str | None = None
) -> RewardModel:
    """Construct a reward model for ``model_name`` (fake in dry-run)."""
    if dry_run:
        return FakeRewardModel(model_name)
    backend = build_backend(model_name, LoRAConfig(rank=8), dry_run=False, api_key=api_key)
    tokenizer = build_tokenizer(model_name, prefer_hf=True)
    return TinkerRewardModel(backend, tokenizer)


def resolve_reward(
    spec: str | None, *, dry_run: bool, api_key: str | None = None
) -> RewardFn:
    """Resolve a reward spec into an RL ``RewardFn``.

    ``"rm:<model>"`` builds a reward model; anything else is a built-in heuristic.
    """
    if spec and spec.startswith("rm:"):
        model_name = spec[len("rm:"):]
        rm = build_reward_model(model_name, dry_run=dry_run, api_key=api_key)
        return lambda prompt, completion: rm.score(prompt, completion)
    return get_reward_fn(spec)
