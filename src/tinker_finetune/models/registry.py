"""Registry of fully open-weight base models supported for fine-tuning.

Design principle for this project: *all weights are known upfront*. Every model
here ships with openly published weights under a permissive/open license, so a
run is fully reproducible and auditable — no closed checkpoints, no hidden
adapters. Qwen3 (Apache-2.0) is the default family because the entire dense +
MoE lineup is released openly; the Llama-3.x family is included for coverage.

Each entry records the facts a trainer needs before touching the API:
context length, parameter counts, whether it is a Mixture-of-Experts model, the
license, and recommended LoRA defaults. These are the same base-model names the
Tinker API accepts for ``create_lora_training_client``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """Static, known-upfront metadata for an open-weight base model."""

    name: str  # canonical HF-style id, e.g. "Qwen/Qwen3-8B"
    family: str  # "qwen3", "llama3", ...
    params_b: float  # total parameters in billions
    active_params_b: float  # active params/token (== params_b for dense models)
    context_length: int
    license: str
    is_moe: bool = False
    recommended_lora_rank: int = 32
    recommended_lr: float = 1e-4
    tags: tuple[str, ...] = field(default_factory=tuple)
    modalities: tuple[str, ...] = ("text",)  # input modalities the base model accepts

    @property
    def is_dense(self) -> bool:
        return not self.is_moe

    @property
    def is_multimodal(self) -> bool:
        return tuple(self.modalities) != ("text",)


# ---------------------------------------------------------------------------
# The registry. Ordered roughly small -> large within each family.
# ---------------------------------------------------------------------------
_MODELS: dict[str, ModelInfo] = {
    m.name: m
    for m in [
        # --- Thinking Machines Lab: Inkling (Apache-2.0) ---
        # Inkling is Tinker's own open-weights base model (released 2026-07-15):
        # a 975B-parameter sparse MoE with ~41B active params/token, a 1M-token
        # context, and native multimodal input. This is the flagship the whole
        # Tinker adaptation platform is built around, so it is this project's
        # top-billed open-weight target. All weights are published on the Hub.
        ModelInfo("thinkingmachines/Inkling", "inkling", 975.0, 41.0, 1_000_000,
                  "Apache-2.0", is_moe=True, recommended_lora_rank=64,
                  recommended_lr=3e-5, tags=("moe", "flagship", "multimodal"),
                  modalities=("text", "image", "audio")),
        # NVFP4-quantized checkpoint of the same weights (lower memory footprint).
        ModelInfo("thinkingmachines/Inkling-NVFP4", "inkling", 975.0, 41.0, 1_000_000,
                  "Apache-2.0", is_moe=True, recommended_lora_rank=64,
                  recommended_lr=3e-5, tags=("moe", "quantized", "multimodal"),
                  modalities=("text", "image", "audio")),
        # Inkling-Small (preview): lighter recipe, ~12B active params. Verify the
        # exact Hub id before a live run; kept here as the small Inkling option.
        ModelInfo("thinkingmachines/Inkling-Small", "inkling", 100.0, 12.0, 1_000_000,
                  "Apache-2.0", is_moe=True, recommended_lora_rank=32,
                  recommended_lr=5e-5, tags=("moe", "small", "preview", "multimodal"),
                  modalities=("text", "image", "audio")),
        # --- Qwen3 dense (Apache-2.0) ---
        ModelInfo("Qwen/Qwen3-0.6B", "qwen3", 0.6, 0.6, 32768, "Apache-2.0",
                  recommended_lora_rank=16, recommended_lr=2e-4, tags=("dense", "small")),
        ModelInfo("Qwen/Qwen3-1.7B", "qwen3", 1.7, 1.7, 32768, "Apache-2.0",
                  recommended_lora_rank=16, recommended_lr=2e-4, tags=("dense", "small")),
        ModelInfo("Qwen/Qwen3-4B", "qwen3", 4.0, 4.0, 32768, "Apache-2.0",
                  recommended_lora_rank=32, tags=("dense",)),
        ModelInfo("Qwen/Qwen3-8B", "qwen3", 8.0, 8.0, 32768, "Apache-2.0",
                  recommended_lora_rank=32, tags=("dense", "default")),
        ModelInfo("Qwen/Qwen3-14B", "qwen3", 14.0, 14.0, 32768, "Apache-2.0",
                  recommended_lora_rank=32, recommended_lr=8e-5, tags=("dense",)),
        ModelInfo("Qwen/Qwen3-32B", "qwen3", 32.0, 32.0, 32768, "Apache-2.0",
                  recommended_lora_rank=64, recommended_lr=5e-5, tags=("dense", "large")),
        # --- Qwen3 Mixture-of-Experts (Apache-2.0) ---
        ModelInfo("Qwen/Qwen3-30B-A3B", "qwen3", 30.0, 3.0, 32768, "Apache-2.0",
                  is_moe=True, recommended_lora_rank=32, tags=("moe",)),
        ModelInfo("Qwen/Qwen3-235B-A22B", "qwen3", 235.0, 22.0, 32768, "Apache-2.0",
                  is_moe=True, recommended_lora_rank=64, recommended_lr=4e-5,
                  tags=("moe", "flagship")),
        # --- Meta Llama 3.x (Llama Community License) ---
        ModelInfo("meta-llama/Llama-3.1-8B-Instruct", "llama3", 8.0, 8.0, 131072,
                  "Llama-3.1-Community", recommended_lora_rank=32, tags=("dense",)),
        ModelInfo("meta-llama/Llama-3.1-70B-Instruct", "llama3", 70.0, 70.0, 131072,
                  "Llama-3.1-Community", recommended_lora_rank=64, recommended_lr=5e-5,
                  tags=("dense", "large")),
        ModelInfo("meta-llama/Llama-3.2-1B-Instruct", "llama3", 1.0, 1.0, 131072,
                  "Llama-3.2-Community", recommended_lora_rank=16, recommended_lr=2e-4,
                  tags=("dense", "small")),
        ModelInfo("meta-llama/Llama-3.2-3B-Instruct", "llama3", 3.0, 3.0, 131072,
                  "Llama-3.2-Community", recommended_lora_rank=32, tags=("dense",)),
        ModelInfo("meta-llama/Llama-3.3-70B-Instruct", "llama3", 70.0, 70.0, 131072,
                  "Llama-3.3-Community", recommended_lora_rank=64, recommended_lr=5e-5,
                  tags=("dense", "large")),
    ]
}


def list_models(family: str | None = None) -> list[ModelInfo]:
    """Return all registered models, optionally filtered by family."""
    models = list(_MODELS.values())
    if family:
        models = [m for m in models if m.family == family.lower()]
    return models


def get_model(name: str) -> ModelInfo:
    """Look up a model by canonical name, raising if unknown."""
    try:
        return _MODELS[name]
    except KeyError:
        raise UnknownModelError(name, sorted(_MODELS)) from None


def is_supported(name: str) -> bool:
    return name in _MODELS


class UnknownModelError(KeyError):
    """Raised when a base model is not in the open-weight registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        super().__init__(
            f"Unknown base model {name!r}. This project only fine-tunes fully "
            f"open-weight models. Available: {', '.join(available)}"
        )
