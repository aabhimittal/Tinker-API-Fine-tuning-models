"""Training-job endpoints: submit SFT/RL runs, list, inspect and cancel."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from tinker_finetune.api.schemas import (
    JobCreatedResponse,
    RLRequest,
    SFTRequest,
)
from tinker_finetune.jobs.manager import get_job_manager
from tinker_finetune.models.registry import UnknownModelError, get_model
from tinker_finetune.models.schemas import JobRecord, JobStatus, JobType

router = APIRouter(prefix="/v1/jobs", tags=["training"])


def _validate_model(name: str) -> None:
    try:
        get_model(name)
    except UnknownModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sft", response_model=JobCreatedResponse, status_code=201)
def create_sft_job(req: SFTRequest) -> JobCreatedResponse:
    _validate_model(req.base_model)
    manager = get_job_manager()
    record = manager.submit(
        JobType.sft,
        req.to_config(),
        train_path=req.train_path,
        eval_path=req.eval_path,
    )
    return JobCreatedResponse(job=record)


@router.post("/rl", response_model=JobCreatedResponse, status_code=201)
def create_rl_job(req: RLRequest) -> JobCreatedResponse:
    _validate_model(req.base_model)
    manager = get_job_manager()
    record = manager.submit(
        JobType.rl,
        req.to_config(),
        prompts=req.prompts,
        reward=req.reward,
    )
    return JobCreatedResponse(job=record)


@router.get("", response_model=list[JobRecord])
def list_jobs(
    status: JobStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[JobRecord]:
    return get_job_manager().list(status=status, limit=limit)


@router.get("/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    record = get_job_manager().get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return record


@router.get("/{job_id}/metrics")
def get_job_metrics(job_id: str, tail: int = Query(default=0, ge=0)) -> dict:
    record = get_job_manager().get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    metrics = record.metrics[-tail:] if tail else record.metrics
    return {
        "job_id": job_id,
        "status": record.status,
        "progress": record.progress,
        "current_step": record.current_step,
        "total_steps": record.total_steps,
        "metrics": [m.model_dump() for m in metrics],
    }


@router.post("/{job_id}/cancel", response_model=JobRecord)
def cancel_job(job_id: str) -> JobRecord:
    manager = get_job_manager()
    if not manager.get(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if not manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job already finished.")
    return manager.get(job_id)  # type: ignore[return-value]
