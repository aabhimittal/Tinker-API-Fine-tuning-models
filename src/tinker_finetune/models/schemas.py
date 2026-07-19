"""Pydantic schemas shared across the training core, jobs, and the HTTP API."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Chat / data
# ---------------------------------------------------------------------------
class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    role: Role
    content: str


class ChatExample(BaseModel):
    """A single supervised chat example. Loss is applied to assistant turns."""

    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def _needs_an_assistant_turn(self) -> ChatExample:
        if not any(m.role == Role.assistant for m in self.messages):
            raise ValueError("A supervised chat example needs at least one assistant turn.")
        return self


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
class LoRAConfig(BaseModel):
    """Low-rank adapter configuration."""

    rank: int = Field(default=32, ge=1, le=256)
    alpha: float = Field(default=64.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @property
    def scaling(self) -> float:
        """LoRA scaling factor (alpha / rank)."""
        return self.alpha / self.rank


class OptimConfig(BaseModel):
    """Adam optimizer + schedule configuration."""

    learning_rate: float = Field(default=1e-4, gt=0)
    beta1: float = Field(default=0.9, ge=0, lt=1)
    beta2: float = Field(default=0.95, ge=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    lr_schedule: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)


class SFTConfig(BaseModel):
    """Supervised fine-tuning run configuration."""

    base_model: str
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig()
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1)
    max_seq_len: int = Field(default=4096, ge=1)
    pack_sequences: bool = True
    eval_every_steps: int = Field(default=0, ge=0)  # 0 disables periodic eval
    save_every_steps: int = Field(default=0, ge=0)  # 0 => only at the end
    seed: int = 0


class RLConfig(BaseModel):
    """Reinforcement-learning (policy-gradient / importance-sampling) config."""

    base_model: str
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig(learning_rate=5e-5)
    iterations: int = Field(default=100, ge=1)
    group_size: int = Field(default=8, ge=1, description="Samples per prompt (GRPO group).")
    prompts_per_batch: int = Field(default=16, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    kl_coef: float = Field(default=0.0, ge=0, description="KL penalty vs. the reference policy.")
    clip_ratio: float = Field(default=0.2, gt=0, description="PPO-style importance-ratio clip.")
    advantage: Literal["grpo", "reinforce"] = "grpo"
    seed: int = 0


# ---------------------------------------------------------------------------
# Jobs / runs
# ---------------------------------------------------------------------------
class JobType(str, Enum):
    sft = "sft"
    rl = "rl"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class TrainMetrics(BaseModel):
    step: int
    epoch: float = 0.0
    loss: float | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    tokens: int | None = None
    reward_mean: float | None = None  # RL only
    kl: float | None = None  # RL only


class JobRecord(BaseModel):
    """Persisted state for a single training run."""

    id: str
    type: JobType
    status: JobStatus = JobStatus.pending
    base_model: str
    config: dict = Field(default_factory=dict)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    current_step: int = 0
    total_steps: int | None = None
    metrics: list[TrainMetrics] = Field(default_factory=list)
    checkpoint_path: str | None = None
    sampling_model_name: str | None = None
    error: str | None = None

    @property
    def progress(self) -> float:
        if not self.total_steps:
            return 0.0
        return min(1.0, self.current_step / self.total_steps)
