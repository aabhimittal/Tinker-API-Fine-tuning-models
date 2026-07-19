"""Application configuration.

Settings come from environment variables (prefixed ``TF_``) with sensible
defaults, and can be layered with per-run YAML config for training jobs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide settings, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="TF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- HTTP service ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = False

    # --- Tinker ---
    # Read without the TF_ prefix because the Tinker SDK expects TINKER_API_KEY.
    tinker_api_key: str | None = Field(default=None, alias="TINKER_API_KEY")
    default_base_model: str = "Qwen/Qwen3-8B"

    # When true, no live Tinker connection is made; an in-process fake backend
    # simulates training/sampling so the whole stack runs offline.
    dry_run: bool = True

    # --- Runtime ---
    artifacts_dir: Path = Path("./runs")
    max_concurrent_jobs: int = 2

    # --- Metrics / Weights & Biases ---
    # Set wandb_project to enable W&B logging (requires `pip install wandb`).
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"  # online | offline | disabled

    def ensure_dirs(self) -> None:
        """Create artifact directories if they do not yet exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "jobs").mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "checkpoints").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
