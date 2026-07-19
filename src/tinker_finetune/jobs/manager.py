"""Job manager: create, persist, track and cancel training runs.

Runs execute on a bounded thread pool so multiple SFT/RL jobs can proceed
concurrently up to ``max_concurrent_jobs``. Each job's state is persisted as
JSON under ``artifacts_dir/jobs/<id>.json`` so it survives process restarts
(status is reconciled to ``failed`` for anything left ``running``).
"""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache

from tinker_finetune.config import Settings, get_settings
from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import (
    JobRecord,
    JobStatus,
    JobType,
    RLConfig,
    SFTConfig,
    TrainMetrics,
)

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._jobs: dict[str, JobRecord] = {}
        self._stops: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self.settings.max_concurrent_jobs),
            thread_name_prefix="tf-job",
        )
        self._jobs_dir = self.settings.artifacts_dir / "jobs"
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
        self._reload_from_disk()

    # -- persistence --------------------------------------------------------
    def _path(self, job_id: str) -> str:
        return str(self._jobs_dir / f"{job_id}.json")

    def _persist(self, record: JobRecord) -> None:
        with open(self._path(record.id), "w", encoding="utf-8") as fh:
            json.dump(record.model_dump(mode="json"), fh, indent=2)

    def _reload_from_disk(self) -> None:
        for f in self._jobs_dir.glob("*.json"):
            try:
                record = JobRecord.model_validate_json(f.read_text())
            except Exception as exc:  # pragma: no cover - corrupt file
                log.warning("Skipping unreadable job file %s: %s", f, exc)
                continue
            # A run left "running" from a previous process cannot resume itself.
            if record.status == JobStatus.running:
                record.status = JobStatus.failed
                record.error = "Interrupted by process restart."
                record.finished_at = _utcnow()
                self._persist(record)
            self._jobs[record.id] = record

    # -- creation -----------------------------------------------------------
    def create(self, job_type: JobType, config: SFTConfig | RLConfig) -> JobRecord:
        job_id = uuid.uuid4().hex[:12]
        record = JobRecord(
            id=job_id,
            type=job_type,
            base_model=config.base_model,
            config=config.model_dump(mode="json"),
            created_at=_utcnow(),
        )
        with self._lock:
            self._jobs[job_id] = record
            self._stops[job_id] = threading.Event()
        self._persist(record)
        log.info("Created %s job %s for %s", job_type.value, job_id, config.base_model)
        return record

    def submit(self, job_type: JobType, config: SFTConfig | RLConfig,
               prompts: list[str] | None = None,
               train_path: str | None = None,
               eval_path: str | None = None,
               reward: str | None = None) -> JobRecord:
        """Create a job and schedule it on the pool."""
        record = self.create(job_type, config)
        # Imported here to avoid a circular import at module load.
        from tinker_finetune.jobs.runner import run_job

        self._pool.submit(
            run_job,
            manager=self,
            job_id=record.id,
            config=config,
            prompts=prompts,
            train_path=train_path,
            eval_path=eval_path,
            reward=reward,
        )
        return record

    # -- accessors ----------------------------------------------------------
    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def list(self, *, status: JobStatus | None = None, limit: int = 100) -> list[JobRecord]:
        jobs = sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs[:limit]

    def stop_event(self, job_id: str) -> threading.Event:
        return self._stops.setdefault(job_id, threading.Event())

    def cancel(self, job_id: str) -> bool:
        record = self._jobs.get(job_id)
        if not record or record.status in (JobStatus.succeeded, JobStatus.failed):
            return False
        self.stop_event(job_id).set()
        if record.status == JobStatus.pending:
            self._mark(job_id, JobStatus.cancelled, error="Cancelled before start.")
        return True

    # -- mutation (called by the runner) -----------------------------------
    def _mark(self, job_id: str, status: JobStatus, *, error: str | None = None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = status
            if status == JobStatus.running and not record.started_at:
                record.started_at = _utcnow()
            if status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
                record.finished_at = _utcnow()
            if error:
                record.error = error
            self._persist(record)

    def _on_metrics(self, job_id: str, m: TrainMetrics, *, total_steps: int | None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.metrics.append(m)
            record.current_step = m.step
            if total_steps is not None:
                record.total_steps = total_steps
            # Persist periodically (every 10 steps) to bound IO.
            if m.step % 10 == 0:
                self._persist(record)

    def _set_outputs(self, job_id: str, *, checkpoint: str | None,
                     sampling_model: str | None) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.checkpoint_path = checkpoint
            record.sampling_model_name = sampling_model
            self._persist(record)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


@lru_cache
def get_job_manager() -> JobManager:
    return JobManager()
