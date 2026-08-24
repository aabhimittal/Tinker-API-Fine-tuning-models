"""Checkpoint management.

Delegates the actual weight snapshot to the backend's ``save_state`` and records
a small JSON manifest alongside it so runs are auditable and resumable.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinker_finetune.logging_utils import get_logger
from tinker_finetune.tinker_client.client import TinkerBackend
from tinker_finetune.training.retention import RetentionPlan, RetentionPolicy, apply_retention

log = get_logger(__name__)


class CheckpointManager:
    """Writes and tracks checkpoints for a single run under ``root/<job_id>``."""

    def __init__(
        self,
        root: str | Path,
        job_id: str,
        *,
        retention: RetentionPolicy | None = None,
    ) -> None:
        self.dir = Path(root) / job_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dir / "checkpoints.json"
        self.retention = retention

    def _load_index(self) -> list[dict]:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return []

    def save(self, backend: TinkerBackend, step: int, *, metrics: dict | None = None) -> str:
        """Snapshot weights at ``step`` and append to the manifest. Returns path."""
        ckpt_path = self.dir / f"step-{step:06d}.state"
        saved = backend.save_state(str(ckpt_path))
        index = self._load_index()
        index.append({"step": step, "path": saved, "metrics": metrics or {}})
        self._index_path.write_text(json.dumps(index, indent=2))
        log.info("Saved checkpoint at step %d -> %s", step, saved)
        self.enforce_retention()
        return saved

    def enforce_retention(self, policy: RetentionPolicy | None = None) -> RetentionPlan | None:
        """Garbage-collect old checkpoints; a disk-full run is a lost run."""
        policy = policy or self.retention
        if policy is None:
            return None
        return apply_retention(self._index_path, policy)

    def latest(self) -> dict | None:
        index = self._load_index()
        return index[-1] if index else None

    def resume(self, backend: TinkerBackend) -> int:
        """Load the latest checkpoint into ``backend``; return its step (0 if none)."""
        latest = self.latest()
        if not latest:
            return 0
        backend.load_state(latest["path"])
        log.info("Resumed from %s (step %d)", latest["path"], latest["step"])
        return int(latest["step"])
