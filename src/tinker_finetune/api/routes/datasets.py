"""Dataset inspection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tinker_finetune.api.schemas import DatasetStats, DriftReportResponse, ValidationReportResponse
from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.drift import compare_datasets
from tinker_finetune.data.templating import build_supervised_datum
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.data.validation import validate_dataset

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


@router.get("/validate", response_model=ValidationReportResponse)
def validate(
    path: str,
    base_model: str | None = None,
    eval_path: str | None = None,
    max_seq_len: int | None = None,
) -> ValidationReportResponse:
    """Lint a dataset before spending a training budget on it.

    Reports PII, exact/near duplicates, label conflicts, train/eval
    contamination, silent truncation and unicode hazards. A dataset with
    ``ok=false`` has at least one error-severity finding and should not be
    submitted as-is.
    """
    settings = get_settings()
    try:
        examples = load_chat_dataset(path)
        evals = load_chat_dataset(eval_path) if eval_path else None
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = base_model or settings.default_base_model
    report = validate_dataset(
        examples,
        path=path,
        tokenizer=build_tokenizer(model, prefer_hf=not settings.dry_run),
        base_model=model,
        max_seq_len=max_seq_len,
        eval_examples=evals,
    )
    return ValidationReportResponse(**report.to_dict())


@router.get("/drift", response_model=DriftReportResponse)
def drift(path: str, baseline_path: str, num_bins: int = 10, top_k: int = 50) -> DriftReportResponse:
    """Compare a refreshed corpus against the one a model was tuned on.

    Answers the question validation cannot: the new data is well-formed, but is
    it the *same shape*? Length, structure, vocabulary and answer-prefix drift
    are graded on the same info/warning/error scale.
    """
    try:
        baseline = load_chat_dataset(baseline_path)
        candidate = load_chat_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report = compare_datasets(baseline, candidate, num_bins=num_bins, top_k=top_k)
    return DriftReportResponse(**report.to_dict())
