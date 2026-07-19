#!/usr/bin/env python
"""Run an SFT job from a YAML config.

    python scripts/run_sft.py configs/sft_qwen3.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import configure_logging
from tinker_finetune.models.schemas import LoRAConfig, OptimConfig, SFTConfig
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.sft_trainer import SFTTrainer


def main(config_path: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    raw = yaml.safe_load(Path(config_path).read_text())

    train_path = raw.pop("train_path")
    eval_path = raw.pop("eval_path", None)
    lora = LoRAConfig(**raw.pop("lora", {}))
    optim = OptimConfig(**raw.pop("optim", {}))
    cfg = SFTConfig(lora=lora, optim=optim, **raw)

    backend = build_backend(cfg.base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    tok = build_tokenizer(cfg.base_model, prefer_hf=not settings.dry_run)
    trainer = SFTTrainer(
        backend, tok, cfg,
        on_metrics=lambda m: print(f"step {m.step} loss={m.loss:.4f} lr={m.learning_rate:.2e}"),
    )
    train = load_chat_dataset(train_path)
    eval_examples = load_chat_dataset(eval_path) if eval_path else None
    history = trainer.train(train, eval_examples)
    print(f"Done: {len(history)} steps, final loss={history[-1].loss:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
