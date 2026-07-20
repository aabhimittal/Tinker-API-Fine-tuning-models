#!/usr/bin/env python
"""End-to-end smoke run against a LIVE Tinker API key.

Runs a tiny real fine-tune and a sample to prove the whole pipeline works with
the actual Tinker backend — not the offline simulator.

Preflight (all required):
    export TINKER_API_KEY=...        # your Tinker key
    export TF_DRY_RUN=false          # use the live backend
    pip install tinker               # the SDK (optional extra)

Then:
    python scripts/e2e_live.py --base-model Qwen/Qwen3-8B --train examples/data/sft_sample.jsonl

By default it uses a small base model and 1 epoch to keep the run cheap. Nothing
here runs against the live API unless TF_DRY_RUN is explicitly false; otherwise
it prints the checklist and exits 0 so it is safe to invoke in any environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys

from tinker_finetune.config import get_settings
from tinker_finetune.data.datasets import load_chat_dataset
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.logging_utils import configure_logging, get_logger
from tinker_finetune.models.registry import get_model
from tinker_finetune.models.schemas import LoRAConfig, OptimConfig, SFTConfig
from tinker_finetune.tinker_client.client import build_backend
from tinker_finetune.training.sft_trainer import SFTTrainer

log = get_logger(__name__)


def preflight(settings) -> list[str]:
    problems: list[str] = []
    if settings.dry_run:
        problems.append("TF_DRY_RUN is true — set TF_DRY_RUN=false for a live run.")
    if not settings.tinker_api_key:
        problems.append("TINKER_API_KEY is not set.")
    if importlib.util.find_spec("tinker") is None:
        problems.append("The 'tinker' SDK is not installed (pip install tinker).")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train", default="examples/data/sft_sample.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    # Validate the model is a known open-weight base.
    info = get_model(args.base_model)
    log.info("Target: %s (%s, %.0fB params, %s)", info.name, info.family,
             info.params_b, info.license)

    problems = preflight(settings)
    if problems:
        print("Live run not configured. Resolve these first:")
        for p in problems:
            print(f"  - {p}")
        print("\nThis is expected in offline/CI environments; exiting 0.")
        return 0

    log.info("Preflight OK — running a live fine-tune. This will use real API quota.")
    cfg = SFTConfig(
        base_model=args.base_model,
        lora=LoRAConfig(rank=info.recommended_lora_rank),
        optim=OptimConfig(learning_rate=info.recommended_lr),
        epochs=args.epochs, batch_size=4, max_seq_len=2048,
    )
    backend = build_backend(cfg.base_model, cfg.lora, dry_run=False,
                            api_key=settings.tinker_api_key)
    tok = build_tokenizer(cfg.base_model, prefer_hf=True)
    trainer = SFTTrainer(
        backend, tok, cfg,
        on_metrics=lambda m: log.info("step %d loss=%.4f", m.step, m.loss),
    )
    history = trainer.train(load_chat_dataset(args.train))
    log.info("Training done: %d steps, final loss=%.4f", len(history), history[-1].loss)

    # Sample from the fine-tuned weights.
    res = backend.sample(tok.encode(args.prompt, add_special=True),
                         max_new_tokens=64, temperature=0.7)
    print("\n=== Sample from fine-tuned model ===")
    print(res.text or tok.decode(res.tokens))
    return 0


if __name__ == "__main__":
    sys.exit(main())
