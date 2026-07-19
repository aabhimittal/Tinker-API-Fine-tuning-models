#!/usr/bin/env python
"""Run an RL job from a YAML config.

    python scripts/run_rl.py configs/rl_qwen3.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from tinker_finetune.config import get_settings
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import configure_logging
from tinker_finetune.models.schemas import LoRAConfig, OptimConfig, RLConfig
from tinker_finetune.rewards import get_reward_fn
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.rl_trainer import RLTrainer


def main(config_path: str) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    raw = yaml.safe_load(Path(config_path).read_text())

    prompts_path = raw.pop("prompts_path")
    reward = raw.pop("reward", None)
    lora = LoRAConfig(**raw.pop("lora", {}))
    optim = OptimConfig(**raw.pop("optim", {}))
    cfg = RLConfig(lora=lora, optim=optim, **raw)

    backend = build_backend(cfg.base_model, cfg.lora, dry_run=settings.dry_run,
                            api_key=settings.tinker_api_key)
    tok = build_tokenizer(cfg.base_model, prefer_hf=not settings.dry_run)
    prompts = [ln.strip() for ln in Path(prompts_path).read_text().splitlines() if ln.strip()]
    trainer = RLTrainer(
        backend, tok, cfg, get_reward_fn(reward),
        on_metrics=lambda m: print(f"iter {m.step} reward={m.reward_mean:.4f} loss={m.loss:.4f}"),
    )
    history = trainer.train(prompts)
    print(f"Done: {len(history)} iterations, final reward={history[-1].reward_mean:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
