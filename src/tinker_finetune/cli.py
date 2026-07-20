"""Command-line interface for tinker-finetune.

Examples
--------
List open-weight models::

    tinker-finetune models

Run an SFT job locally (dry-run by default)::

    tinker-finetune sft --base-model Qwen/Qwen3-8B --train data/train.jsonl --epochs 1

Run an RL job::

    tinker-finetune rl --base-model Qwen/Qwen3-8B --prompts data/prompts.txt --reward numeric_match

Serve the HTTP API::

    tinker-finetune serve --port 8000
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tinker_finetune.config import get_settings
from tinker_finetune.logging_utils import configure_logging

app = typer.Typer(add_completion=False, help="Fine-tune open-weight LLMs with the Tinker API.")
console = Console()


@app.callback()
def _root() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)


@app.command()
def models(family: str | None = typer.Option(None, help="Filter by family (qwen3, llama3).")):
    """List open-weight base models in the registry."""
    from tinker_finetune.models.registry import list_models

    table = Table(title="Open-weight base models")
    for col in ("name", "family", "params(B)", "active(B)", "ctx", "MoE", "rank", "license"):
        table.add_column(col)
    for m in list_models(family):
        table.add_row(
            m.name, m.family, f"{m.params_b:g}", f"{m.active_params_b:g}",
            str(m.context_length), "yes" if m.is_moe else "no",
            str(m.recommended_lora_rank), m.license,
        )
    console.print(table)


@app.command()
def inspect(
    dataset: Path = typer.Argument(..., help="Path to a JSONL dataset."),
    base_model: str | None = typer.Option(None),
):
    """Validate a dataset and print tokenization stats."""
    from tinker_finetune.data.datasets import load_chat_dataset
    from tinker_finetune.data.templating import build_supervised_datum
    from tinker_finetune.data.tokenization import build_tokenizer

    settings = get_settings()
    examples = load_chat_dataset(dataset)
    tok = build_tokenizer(base_model or settings.default_base_model, prefer_hf=not settings.dry_run)
    sup = [build_supervised_datum(e, tok, max_seq_len=1_000_000).num_supervised_tokens
           for e in examples]
    console.print(f"[bold]{dataset}[/bold]: {len(examples)} examples, "
                  f"avg assistant tokens = {sum(sup)/len(sup):.1f}")


@app.command()
def sft(
    base_model: str = typer.Option(..., "--base-model", "-m"),
    train: Path = typer.Option(..., "--train", "-t", help="Train JSONL path."),
    eval_path: Path | None = typer.Option(None, "--eval"),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(8),
    lora_rank: int = typer.Option(32),
    lr: float = typer.Option(1e-4),
    max_seq_len: int = typer.Option(4096),
):
    """Run a supervised fine-tuning job synchronously and print the loss curve."""
    from tinker_finetune.data.datasets import load_chat_dataset
    from tinker_finetune.data.tokenization import build_tokenizer
    from tinker_finetune.models.schemas import LoRAConfig, OptimConfig, SFTConfig
    from tinker_finetune.tinker_client.client import build_backend
    from tinker_finetune.training.sft_trainer import SFTTrainer

    settings = get_settings()
    cfg = SFTConfig(
        base_model=base_model,
        lora=LoRAConfig(rank=lora_rank),
        optim=OptimConfig(learning_rate=lr),
        epochs=epochs, batch_size=batch_size, max_seq_len=max_seq_len,
    )
    backend = build_backend(base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    tok = build_tokenizer(base_model, prefer_hf=not settings.dry_run)
    trainer = SFTTrainer(
        backend, tok, cfg,
        on_metrics=lambda m: console.print(
            f"step {m.step:>4} | loss {m.loss:.4f} | lr {m.learning_rate:.2e}"
        ),
    )
    train_examples = load_chat_dataset(train)
    eval_examples = load_chat_dataset(eval_path) if eval_path else None
    history = trainer.train(train_examples, eval_examples)
    console.print(f"[green]Done.[/green] {len(history)} steps, "
                  f"final loss {history[-1].loss:.4f}")


@app.command()
def rl(
    base_model: str = typer.Option(..., "--base-model", "-m"),
    prompts: Path = typer.Option(..., "--prompts", "-p", help="One prompt per line."),
    reward: str = typer.Option(
        "length_target",
        help="Reward: a built-in name, or 'rm:<model>' for a reward-model scorer (RLHF).",
    ),
    iterations: int = typer.Option(20),
    group_size: int = typer.Option(8),
    lora_rank: int = typer.Option(32),
):
    """Run an RL fine-tuning job synchronously."""
    from tinker_finetune.data.tokenization import build_tokenizer
    from tinker_finetune.models.schemas import LoRAConfig, RLConfig
    from tinker_finetune.reward_models import resolve_reward
    from tinker_finetune.tinker_client.client import build_backend
    from tinker_finetune.training.rl_trainer import RLTrainer

    settings = get_settings()
    cfg = RLConfig(
        base_model=base_model, lora=LoRAConfig(rank=lora_rank),
        iterations=iterations, group_size=group_size,
    )
    backend = build_backend(base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    tok = build_tokenizer(base_model, prefer_hf=not settings.dry_run)
    prompt_list = [ln.strip() for ln in Path(prompts).read_text().splitlines() if ln.strip()]
    reward_fn = resolve_reward(reward, dry_run=settings.dry_run, api_key=settings.tinker_api_key)
    trainer = RLTrainer(
        backend, tok, cfg, reward_fn,
        on_metrics=lambda m: console.print(
            f"iter {m.step:>4} | reward {m.reward_mean:.4f} | loss {m.loss:.4f}"
        ),
    )
    history = trainer.train(prompt_list)
    console.print(f"[green]Done.[/green] {len(history)} iterations, "
                  f"final reward {history[-1].reward_mean:.4f}")


@app.command()
def dpo(
    base_model: str = typer.Option(..., "--base-model", "-m"),
    prefs: Path = typer.Option(..., "--prefs", "-d", help="Preference JSONL (prompt/chosen/rejected)."),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(4),
    beta: float = typer.Option(0.1, help="DPO temperature / KL strength."),
    loss_type: str = typer.Option("sigmoid", help="sigmoid | ipo"),
    reference_free: bool = typer.Option(False),
    lora_rank: int = typer.Option(32),
):
    """Run Direct Preference Optimization on a preference dataset."""
    from tinker_finetune.data.preferences import load_preference_dataset
    from tinker_finetune.data.tokenization import build_tokenizer
    from tinker_finetune.models.schemas import DPOConfig, LoRAConfig
    from tinker_finetune.tinker_client.client import build_backend
    from tinker_finetune.training.dpo_trainer import DPOTrainer

    settings = get_settings()
    cfg = DPOConfig(
        base_model=base_model, lora=LoRAConfig(rank=lora_rank),
        epochs=epochs, batch_size=batch_size, beta=beta,
        loss_type=loss_type,  # type: ignore[arg-type]
        reference_free=reference_free,
    )
    backend = build_backend(base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    reference = None if reference_free else build_backend(
        base_model, cfg.lora, dry_run=settings.dry_run, api_key=settings.tinker_api_key)
    tok = build_tokenizer(base_model, prefer_hf=not settings.dry_run)
    trainer = DPOTrainer(
        backend, reference, tok, cfg,
        on_metrics=lambda m: console.print(
            f"step {m.step:>4} | loss {m.loss:.4f} | margin {m.reward_margin:+.4f} "
            f"| acc {m.reward_accuracy:.2f}"
        ),
    )
    history = trainer.train(load_preference_dataset(prefs))
    console.print(f"[green]Done.[/green] {len(history)} steps, "
                  f"final margin {history[-1].reward_margin:+.4f}, "
                  f"acc {history[-1].reward_accuracy:.2f}")


@app.command()
def sample(
    base_model: str = typer.Option(..., "--base-model", "-m"),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    max_new_tokens: int = typer.Option(128),
    temperature: float = typer.Option(0.7),
):
    """Sample a completion from a model."""
    from tinker_finetune.data.tokenization import build_tokenizer
    from tinker_finetune.models.registry import get_model
    from tinker_finetune.models.schemas import LoRAConfig
    from tinker_finetune.tinker_client.client import build_backend

    settings = get_settings()
    info = get_model(base_model)
    backend = build_backend(base_model, LoRAConfig(rank=info.recommended_lora_rank),
                            dry_run=settings.dry_run, api_key=settings.tinker_api_key)
    tok = build_tokenizer(base_model, prefer_hf=not settings.dry_run)
    res = backend.sample(tok.encode(prompt, add_special=True),
                         max_new_tokens=max_new_tokens, temperature=temperature)
    console.print(res.text or tok.decode(res.tokens))


@app.command()
def serve(
    host: str = typer.Option(None),
    port: int = typer.Option(None),
    reload: bool = typer.Option(False),
):
    """Start the FastAPI server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "tinker_finetune.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
