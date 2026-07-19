"""Dataset inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tinker_finetune.api.schemas import DatasetStats
from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.templating import build_supervised_datum
from tinker_finetune.data.tokenization import build_tokenizer

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


@router.get("/inspect", response_model=DatasetStats)
def inspect(path: str, base_model: str | None = None) -> DatasetStats:
    """Validate a JSONL dataset and report tokenization statistics.

    Uses the byte tokenizer in dry-run mode so inspection is instant and needs
    no model download; pass ``base_model`` to use its real tokenizer.
    """
    settings = get_settings()
    try:
        examples = load_chat_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = base_model or settings.default_base_model
    tokenizer = build_tokenizer(model, prefer_hf=not settings.dry_run)

    turns = [len(ex.messages) for ex in examples]
    assistant_tokens: list[int] = []
    max_tokens = 0
    for ex in examples:
        datum = build_supervised_datum(ex, tokenizer, max_seq_len=1_000_000)
        assistant_tokens.append(datum.num_supervised_tokens)
        max_tokens = max(max_tokens, len(datum.input_tokens))

    n = len(examples)
    return DatasetStats(
        path=path,
        num_examples=n,
        avg_turns=round(sum(turns) / n, 2),
        avg_assistant_tokens=round(sum(assistant_tokens) / n, 2),
        max_tokens=max_tokens,
    )
