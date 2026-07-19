"""Inference / sampling endpoint.

Samples from a base model (or, in a full deployment, a fine-tuned adapter). In
dry-run mode this returns a deterministic fake completion; with a live Tinker
connection it routes through the sampling client.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tinker_finetune.api.schemas import SampleRequest, SampleResponse
from tinker_finetune.config import get_settings
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.models.registry import UnknownModelError, get_model
from tinker_finetune.models.schemas import LoRAConfig
from tinker_finetune.tinker_client.client import build_backend

router = APIRouter(prefix="/v1/inference", tags=["inference"])


@router.post("/sample", response_model=SampleResponse)
def sample(req: SampleRequest) -> SampleResponse:
    try:
        model = get_model(req.base_model)
    except UnknownModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    settings = get_settings()
    backend = build_backend(
        req.base_model,
        LoRAConfig(rank=model.recommended_lora_rank),
        dry_run=settings.dry_run,
        api_key=settings.tinker_api_key,
    )
    tokenizer = build_tokenizer(req.base_model, prefer_hf=not settings.dry_run)
    prompt_ids = tokenizer.encode(req.prompt, add_special=True)
    result = backend.sample(
        prompt_ids,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
    )
    text = result.text or tokenizer.decode(result.tokens)
    return SampleResponse(
        text=text,
        num_tokens=len(result.tokens),
        stop_reason=result.stop_reason,
    )
