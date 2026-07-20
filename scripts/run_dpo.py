#!/usr/bin/env python
"""Run a DPO job from a YAML config.

    python scripts/run_dpo.py configs/dpo_inkling.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from tinker_finetune.config import get_settings
from tinker_finetune.data.preferences import load_preference_dataset
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import configure_logging
from tinker_finetune.models.schemas import DPOConfig, LoRAConfig, OptimConfig
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.dpo_trainer import DPOTrainer


def main(config_path: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    raw = yaml.safe_load(Path(config_path).read_text())

    train_path = raw.pop("train_path")
    lora = LoRAConfig(**raw.pop("lora", {}))
    optim = OptimConfig(**raw.pop("optim", {}))
    cfg = DPOConfig(lora=lora, optim=optim, **raw)

    backend = build_backend(cfg.base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    reference = None if cfg.reference_free else build_backend(
        cfg.base_model, cfg.lora, dry_run=settings.dry_run, api_key=settings.tinker_api_key)
    tok = build_tokenizer(cfg.base_model, prefer_hf=not settings.dry_run)
    trainer = DPOTrainer(
        backend, reference, tok, cfg,
        on_metrics=lambda m: print(
            f"step {m.step} loss={m.loss:.4f} margin={m.reward_margin:+.4f} "
            f"acc={m.reward_accuracy:.2f}"
        ),
    )
    history = trainer.train(load_preference_dataset(train_path))
    print(f"Done: {len(history)} steps, final margin={history[-1].reward_margin:+.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
