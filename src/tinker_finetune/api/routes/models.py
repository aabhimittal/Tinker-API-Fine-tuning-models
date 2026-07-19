"""Model registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from tinker_finetune.api.schemas import ModelSummary
from tinker_finetune.models.registry import UnknownModelError, get_model, list_models
from tinker_finetune.rewards import list_rewards

router = APIRouter(prefix="/v1/models", tags=["models"])


def _to_summary(m) -> ModelSummary:
    return ModelSummary(
        name=m.name,
        family=m.family,
        params_b=m.params_b,
        active_params_b=m.active_params_b,
        context_length=m.context_length,
        license=m.license,
        is_moe=m.is_moe,
        recommended_lora_rank=m.recommended_lora_rank,
        recommended_lr=m.recommended_lr,
        tags=list(m.tags),
    )


@router.get("", response_model=list[ModelSummary])
def get_models(family: str | None = Query(default=None)) -> list[ModelSummary]:
    """List open-weight base models available for fine-tuning."""
    return [_to_summary(m) for m in list_models(family)]


@router.get("/rewards", response_model=list[str])
def get_rewards() -> list[str]:
    """List registered RL reward functions."""
    return list_rewards()


@router.get("/{model_name:path}", response_model=ModelSummary)
def get_model_detail(model_name: str) -> ModelSummary:
    try:
        return _to_summary(get_model(model_name))
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
