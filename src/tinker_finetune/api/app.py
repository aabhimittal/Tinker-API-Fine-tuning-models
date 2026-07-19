"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tinker_finetune import __version__
from tinker_finetune.api.routes import datasets, inference, models, training
from tinker_finetune.config import get_settings
from tinker_finetune.jobs.manager import get_job_manager
from tinker_finetune.logging_utils import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    get_job_manager()  # warm the manager (reloads persisted jobs)
    yield
    get_job_manager().shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="tinker-finetune",
        version=__version__,
        summary="Backend for fine-tuning fully open-weight LLMs with the Tinker API.",
        description=(
            "LoRA / SFT / RL fine-tuning over open-weight models (Qwen3, Llama-3.x). "
            f"Dry-run mode: {'on' if settings.dry_run else 'off'}."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "dry_run": settings.dry_run,
            "default_base_model": settings.default_base_model,
        }

    app.include_router(models.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(inference.router)
    return app


app = create_app()
