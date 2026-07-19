"""Shared pytest fixtures. Forces dry-run mode and an isolated artifacts dir."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch):
    """Point every test at a temp artifacts dir and force offline dry-run."""
    monkeypatch.setenv("TF_DRY_RUN", "true")
    monkeypatch.setenv("TF_ARTIFACTS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("TF_MAX_CONCURRENT_JOBS", "2")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    # Clear cached singletons so the env changes take effect.
    from tinker_finetune import config, jobs

    config.get_settings.cache_clear()
    jobs.manager.get_job_manager.cache_clear()
    importlib.reload  # noqa: B018 - keep reference; reload not needed here
    yield
    config.get_settings.cache_clear()
    jobs.manager.get_job_manager.cache_clear()


@pytest.fixture
def sample_dataset(tmp_path: Path) -> str:
    path = tmp_path / "train.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "hello there"}]}\n'
        '{"prompt": "2+2", "completion": "4"}\n'
    )
    return str(path)
