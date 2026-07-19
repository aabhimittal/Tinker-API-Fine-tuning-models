"""Backend protocol and factory over Tinker's core training primitives.

Tinker exposes a deliberately small set of primitives, and this module mirrors
them so the training loops read the same whether they run live or against the
fake:

- ``forward_backward``  run a forward + backward pass over a batch of
  :class:`Datum` and accumulate gradients; returns per-datum loss/logprobs.
- ``optim_step``        apply an Adam update with the given hyper-parameters.
- ``sample``            generate completions from the current weights.
- ``save_weights_for_sampler``  snapshot weights and return a sampler handle.
- ``save_state`` / ``load_state``  full training-state checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from tinker_finetune.models.schemas import LoRAConfig, OptimConfig


@dataclass
class Datum:
    """One training example expressed as token ids + a loss target.

    Mirrors ``tinker.Datum``: ``input_tokens`` are fed to the model, ``target_tokens``
    are the next-token labels, and ``weights`` masks the loss (0 = ignore, e.g.
    for prompt tokens; 1 = supervise, e.g. for assistant tokens). ``advantages``
    is used by RL loss functions.
    """

    input_tokens: list[int]
    target_tokens: list[int]
    weights: list[float]
    advantages: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.input_tokens)
        if not (len(self.target_tokens) == len(self.weights) == n):
            raise ValueError(
                "input_tokens, target_tokens and weights must have equal length "
                f"(got {n}, {len(self.target_tokens)}, {len(self.weights)})."
            )
        if self.advantages is not None and len(self.advantages) != n:
            raise ValueError("advantages must match sequence length when provided.")

    @property
    def num_supervised_tokens(self) -> int:
        return int(sum(1 for w in self.weights if w > 0))


@dataclass
class ForwardBackwardResult:
    loss: float
    per_datum_loss: list[float]
    num_tokens: int
    logprobs: list[list[float]] | None = None


@dataclass
class OptimStepResult:
    learning_rate: float
    grad_norm: float


@dataclass
class SampleResult:
    tokens: list[int]
    text: str
    logprobs: list[float] | None = None
    stop_reason: str = "length"


@runtime_checkable
class TinkerBackend(Protocol):
    """The surface every trainer depends on. Both live and fake implement it."""

    base_model: str

    def forward_backward(
        self, batch: list[Datum], loss_fn: str = "cross_entropy"
    ) -> ForwardBackwardResult: ...

    def optim_step(self, optim: OptimConfig, lr: float) -> OptimStepResult: ...

    def sample(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> SampleResult: ...

    def save_weights_for_sampler(self, name: str) -> str: ...

    def save_state(self, path: str) -> str: ...

    def load_state(self, path: str) -> None: ...


def build_backend(
    base_model: str,
    lora: LoRAConfig,
    *,
    dry_run: bool,
    api_key: str | None = None,
) -> TinkerBackend:
    """Construct the appropriate backend.

    ``dry_run=True`` (or a missing SDK / API key) yields the in-process fake so
    development and CI never require live credentials.
    """
    if dry_run:
        from tinker_finetune.tinker_client.fake import FakeTinkerBackend

        return FakeTinkerBackend(base_model=base_model, lora=lora)

    from tinker_finetune.tinker_client.live import LiveTinkerBackend

    return LiveTinkerBackend(base_model=base_model, lora=lora, api_key=api_key)
