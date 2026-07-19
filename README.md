# tinker-finetune

An extensive, production-shaped backend for fine-tuning **fully open-weight
LLMs** with the [Tinker API](https://thinkingmachines.ai) (Thinking Machines
Lab). LoRA supervised fine-tuning (SFT) and reinforcement learning (GRPO /
REINFORCE) over models whose weights are all known upfront — **Qwen3**
(Apache-2.0) as the default family, **Llama-3.x** supported.

The whole stack — HTTP API, job manager, trainers, CLI — runs **offline by
default** against an in-process simulator, then switches to the live Tinker SDK
by flipping one environment variable. Tests and CI need no credentials.

```
FastAPI ──▶ JobManager ──▶ SFT / RL Trainers ──▶ TinkerBackend
  (api/)       (jobs/)         (training/)      ├─ LiveTinkerBackend  (real SDK)
                                                └─ FakeTinkerBackend  (offline)
```

## Highlights

- **Open weights, enforced.** A registry allow-lists only fully open-weight base
  models; closed checkpoints are rejected at the API boundary. See
  [docs/models.md](docs/models.md).
- **Real Tinker loop.** Trainers reduce to `forward_backward` → `optim_step` →
  `sample` → `save_state`, exactly as the Tinker primitives are used.
- **SFT with correct loss masking.** Chat templating supervises assistant tokens
  only; sequence packing cuts padding waste.
- **RL / RLHF / RLVR.** GRPO group-relative advantages, pluggable reward
  functions, importance-sampling loss, KL control.
- **Async jobs.** Bounded thread pool, on-disk persistence, live metrics,
  cooperative cancel, restart reconciliation.
- **Batteries included.** CLI, Docker, Makefile, GitHub Actions CI, 32 offline
  tests, typed with Pydantic v2.

## Quick start

```bash
pip install -e ".[dev]"          # install (offline dry-run is the default)
make test                        # 32 tests, no credentials needed

tinker-finetune models                                   # list open-weight models
tinker-finetune inspect examples/data/sft_sample.jsonl   # validate a dataset

# Supervised fine-tuning (dry-run simulator)
tinker-finetune sft -m Qwen/Qwen3-8B -t examples/data/sft_sample.jsonl --epochs 3

# RL fine-tuning (GRPO)
tinker-finetune rl -m Qwen/Qwen3-8B -p examples/data/rl_prompts.txt --reward numeric_match

# Serve the HTTP API
make serve            # http://localhost:8000/docs
```

## Going live

```bash
pip install tinker
export TINKER_API_KEY=...   TF_DRY_RUN=false
tinker-finetune sft -m Qwen/Qwen3-8B -t your_data.jsonl
```

Nothing else changes — the same commands now route through the real API.

## Repository layout

```
src/tinker_finetune/
├── config.py            env + settings
├── models/              open-weight registry + shared schemas
├── data/                tokenization · chat templating+masking · datasets · packing
├── tinker_client/       backend protocol · live SDK adapter · offline fake
├── training/            SFT trainer · RL trainer · LR schedules · checkpointing
├── rewards.py           pluggable RL reward registry
├── eval/                held-out loss · exact-match accuracy
├── jobs/                async job manager + runner
├── api/                 FastAPI app + routes (models/datasets/jobs/inference)
└── cli.py               Typer command-line entrypoint
configs/   examples/   scripts/   tests/   docs/   docker/
```

## Documentation

- [Setup](docs/setup.md) — install, env vars, going live.
- [Architecture](docs/architecture.md) — layers and design decisions.
- [Usage](docs/usage.md) — data format, SFT, RL, sampling, HTTP.
- [API reference](docs/api.md) — every endpoint.
- [Models](docs/models.md) — the open-weight registry.

## Status & disclaimer

Dry-run mode is a *simulator* — its loss/reward curves demonstrate the pipeline,
not real model quality. Real training requires a Tinker account and the `tinker`
SDK. Tinker is a product of Thinking Machines Lab; this repository is an
independent client/backend and is not affiliated with them.

## License

MIT — see [LICENSE](LICENSE).
