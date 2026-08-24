"""HTTP request/response models for the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tinker_finetune.models.schemas import (
    DPOConfig,
    JobRecord,
    LoRAConfig,
    OptimConfig,
    RLConfig,
    SFTConfig,
)


class ModelSummary(BaseModel):
    name: str
    family: str
    params_b: float
    active_params_b: float
    context_length: int
    license: str
    is_moe: bool
    recommended_lora_rank: int
    recommended_lr: float
    tags: list[str]


class SFTRequest(BaseModel):
    """Start a supervised fine-tuning job."""

    base_model: str
    train_path: str = Field(description="Path to a JSONL chat/prompt-completion dataset.")
    eval_path: str | None = None
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig()
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=8, ge=1)
    max_seq_len: int = Field(default=4096, ge=1)
    pack_sequences: bool = True
    save_every_steps: int = Field(default=0, ge=0)
    eval_every_steps: int = Field(default=0, ge=0)
    seed: int = 0

    def to_config(self) -> SFTConfig:
        return SFTConfig(
            base_model=self.base_model,
            lora=self.lora,
            optim=self.optim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            pack_sequences=self.pack_sequences,
            save_every_steps=self.save_every_steps,
            eval_every_steps=self.eval_every_steps,
            seed=self.seed,
        )


class RLRequest(BaseModel):
    """Start an RL fine-tuning job (GRPO / REINFORCE)."""

    base_model: str
    prompts: list[str] = Field(min_length=1)
    reward: str | None = Field(default=None, description="Registered reward name.")
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig(learning_rate=5e-5)
    iterations: int = Field(default=100, ge=1)
    group_size: int = Field(default=8, ge=1)
    prompts_per_batch: int = Field(default=16, ge=1)
    max_new_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=1.0, gt=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    kl_coef: float = Field(default=0.0, ge=0)
    advantage: str = Field(default="grpo")
    seed: int = 0

    def to_config(self) -> RLConfig:
        return RLConfig(
            base_model=self.base_model,
            lora=self.lora,
            optim=self.optim,
            iterations=self.iterations,
            group_size=self.group_size,
            prompts_per_batch=self.prompts_per_batch,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            kl_coef=self.kl_coef,
            advantage=self.advantage,  # type: ignore[arg-type]
            seed=self.seed,
        )


class DPORequest(BaseModel):
    """Start a Direct Preference Optimization job."""

    base_model: str
    train_path: str = Field(description="Path to a JSONL preference dataset.")
    lora: LoRAConfig = LoRAConfig()
    optim: OptimConfig = OptimConfig(learning_rate=5e-6)
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1)
    max_seq_len: int = Field(default=4096, ge=1)
    beta: float = Field(default=0.1, gt=0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=0.5)
    loss_type: str = Field(default="sigmoid")
    reference_free: bool = False
    save_every_steps: int = Field(default=0, ge=0)
    seed: int = 0

    def to_config(self) -> DPOConfig:
        return DPOConfig(
            base_model=self.base_model,
            lora=self.lora,
            optim=self.optim,
            epochs=self.epochs,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            beta=self.beta,
            label_smoothing=self.label_smoothing,
            loss_type=self.loss_type,  # type: ignore[arg-type]
            reference_free=self.reference_free,
            save_every_steps=self.save_every_steps,
            seed=self.seed,
        )


class JobCreatedResponse(BaseModel):
    job: JobRecord


class SampleRequest(BaseModel):
    base_model: str
    prompt: str
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int | None = None


class SampleResponse(BaseModel):
    text: str
    num_tokens: int
    stop_reason: str


class ValidationFinding(BaseModel):
    code: str
    severity: str
    message: str
    count: int
    examples: list[int] = Field(default_factory=list)


class ValidationReportResponse(BaseModel):
    path: str | None = None
    num_examples: int
    ok: bool
    findings: list[ValidationFinding]
    stats: dict


class FeatureDriftResponse(BaseModel):
    feature: str
    psi: float
    jsd: float
    ks: float | None = None
    severity: str
    detail: str = ""
    baseline: dict = Field(default_factory=dict)
    candidate: dict = Field(default_factory=dict)


class DriftReportResponse(BaseModel):
    severity: str
    baseline_rows: int
    candidate_rows: int
    notes: list[str] = Field(default_factory=list)
    features: list[FeatureDriftResponse]

    @property
    def ok(self) -> bool:
        return self.severity != "error"


class DatasetStats(BaseModel):
    path: str
    num_examples: int
    avg_turns: float
    avg_assistant_tokens: float
    max_tokens: int
