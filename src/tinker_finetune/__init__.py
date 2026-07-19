"""tinker-finetune: an extensive backend for fine-tuning fully open-weight LLMs
with the Tinker API.

The package is organized into cohesive layers:

- ``config``            application settings (env + YAML).
- ``models``            open-weight model registry and shared schemas.
- ``data``              dataset loading, chat templating, tokenization, packing.
- ``tinker_client``     a thin, testable wrapper around Tinker's low-level
                        primitives (forward_backward / optim_step / sample /
                        save_state) plus an in-process fake for dry runs.
- ``training``          SFT trainer, LoRA config, RL trainer, optimizer,
                        checkpoint management.
- ``eval``              lightweight evaluation harness.
- ``jobs``              async job manager + worker for long-running runs.
- ``api``               FastAPI application exposing training, models, datasets
                        and inference.
- ``cli``               Typer command-line entrypoint.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
