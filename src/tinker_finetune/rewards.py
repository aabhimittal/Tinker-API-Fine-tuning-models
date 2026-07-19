"""Reward function registry for RL fine-tuning.

Reward functions map ``(prompt, completion_text) -> float``. This registry ships
a few reference rewards (length shaping, a verifiable numeric-answer checker,
and a keyword reward) and lets callers register their own — e.g. a reward-model
scorer for RLHF or a task verifier for RLVR.

Register a custom reward at import time::

    from tinker_finetune.rewards import register

    @register("my_reward")
    def my_reward(prompt: str, completion: str) -> float:
        ...
"""

from __future__ import annotations

import re
from collections.abc import Callable

RewardFn = Callable[[str, str], float]

_REGISTRY: dict[str, RewardFn] = {}


def register(name: str) -> Callable[[RewardFn], RewardFn]:
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn

    return deco


def get_reward_fn(name: str | None) -> RewardFn:
    """Return a registered reward by name, defaulting to ``length_target``."""
    if not name:
        return _REGISTRY["length_target"]
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown reward {name!r}. Registered: {', '.join(sorted(_REGISTRY))}"
        ) from None


def list_rewards() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Reference rewards
# ---------------------------------------------------------------------------
@register("length_target")
def length_target(prompt: str, completion: str, target_chars: int = 200) -> float:
    """Reward completions whose length is near ``target_chars`` (peak 1.0)."""
    diff = abs(len(completion) - target_chars)
    return max(0.0, 1.0 - diff / target_chars)


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@register("numeric_match")
def numeric_match(prompt: str, completion: str) -> float:
    """Verifiable reward: 1.0 if the last number in the completion equals the
    last number in the prompt (a stand-in for checking a computed answer)."""
    p_nums = _NUM_RE.findall(prompt)
    c_nums = _NUM_RE.findall(completion)
    if not p_nums or not c_nums:
        return 0.0
    return 1.0 if p_nums[-1] == c_nums[-1] else 0.0


@register("nonempty")
def nonempty(prompt: str, completion: str) -> float:
    """Minimal reward: 1.0 for any non-whitespace completion, else 0.0."""
    return 1.0 if completion.strip() else 0.0
