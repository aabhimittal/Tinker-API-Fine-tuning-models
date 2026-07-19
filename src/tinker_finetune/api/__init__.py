"""FastAPI application exposing the fine-tuning backend."""

from tinker_finetune.api.app import create_app

__all__ = ["create_app"]
