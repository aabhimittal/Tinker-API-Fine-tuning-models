# tinker-finetune

An extensive, production-shaped backend for fine-tuning **fully open-weight
LLMs** with the [Tinker API](https://thinkingmachines.ai) (Thinking Machines
Lab). Supervised fine-tuning (SFT), reinforcement learning (GRPO / REINFORCE),
**RLHF with a reward-model scorer**, and **DPO / preference training** — over
models whose weights are all known upfront. Top-billed target is
**Inkling** (`thinkingmachines/Inkling`), Tinker's own Apache-2.0 open-weights
base model (975B MoE, 41B active, 1M context, multimodal); **Qwen3** and
**Llama-3.x** are also supported.

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
- **RL / RLHF / RLVR.** GRPO group-relative advantages, importance-sampling
  loss, KL control, and a pluggable **reward-model scorer** (`reward: rm:<model>`)
  that closes the RLHF loop: sample → score with a reward model → optimize.
- **DPO / preference training.** Direct Preference Optimization on chosen/rejected
  pairs against a frozen reference policy — sigmoid + IPO losses, cDPO label
  smoothing, reference-free mode; reports implicit reward margin and accuracy.
- **Dataset validation that gates a run.** `tinker-finetune validate` lints a
  corpus for PII/secrets (Luhn-checked cards, cloud keys, private keys),
  exact/near duplicates via a deterministic MinHash-LSH index, label conflicts,
  train/eval contamination, prompt-destroying truncation, and unicode hazards
  (bidi overrides, lone surrogates, zero-width joiners) — with `--redact-to`
  and a CI-friendly `--fail-on` gate. See [docs/hardening.md](docs/hardening.md).
- **Fault tolerance.** Backend calls compose a client-side token-bucket rate
  limiter, a circuit breaker, and exponential backoff with full jitter that
  honours `Retry-After` — and deliberately never replays a partially applied
  `optim_step`.
- **Budget ceilings.** Token/step/USD accounting with MoE-aware pricing (active
  params, not total), one-shot warnings, and a preflight that rejects a job
  before its first step rather than after its tenth hour.
- **Training guards.** NaN/Inf detection, gradient-explosion limits, loss-spike
  and divergence trend detection, empty-loss-mask detection, and patience-based
  early stopping wired into the SFT loop.
- **Deterministic sharding & resume.** Rendezvous-hashed virtual buckets keep a
  fleet resize from reshuffling the corpus (`W → W+1` moves ~`1/(W+1)` of it),
  epoch order is derived from `(seed, epoch)` so no shuffle state is
  checkpointed, and a resume cursor that was written under a different
  `world_size` is *reported*, never silently misapplied. Weighted, temperature-
  scaled corpus mixtures with explicit exhaustion policies.
- **Checkpoint retention.** Keep-last ∪ keep-best ∪ keep-every ∪ min-age, with
  an atomic manifest rewrite, dry-run previews, and refusal to delete the last
  checkpoint — a disk-full run is a lost run.
- **Distribution drift.** `tinker-finetune drift old.jsonl new.jsonl` scores PSI,
  Jensen-Shannon and KS over length, structure, vocabulary and answer-prefix
  features, catching the style collapse ("Sure! Here's…") that length statistics
  miss.
- **Metrics.** Per-step JSONL logging plus optional **Weights & Biases**
  streaming (`TF_WANDB_PROJECT`).
- **Live end-to-end.** `scripts/e2e_live.py` runs a real fine-tune + sample
  against a live Tinker key, with a safe preflight when unconfigured.
- **Async jobs.** Bounded thread pool, on-disk persistence, live metrics,
  cooperative cancel, restart reconciliation.
- **Batteries included.** CLI, Docker, Makefile, GitHub Actions CI, 300+ offline
  tests, typed with Pydantic v2.

## Quick start

```bash
pip install -e ".[dev]"          # install (offline dry-run is the default)
make test                        # 32 tests, no credentials needed

tinker-finetune models                                   # list open-weight models
tinker-finetune inspect examples/data/sft_sample.jsonl   # validate a dataset

# Lint the dataset before spending anything on it
tinker-finetune validate data/train.jsonl --eval data/eval.jsonl --fail-on warning

# Supervised fine-tuning (dry-run simulator)
tinker-finetune sft -m Qwen/Qwen3-8B -t examples/data/sft_sample.jsonl --epochs 3

# RL fine-tuning (GRPO), with a heuristic or a reward-model scorer (RLHF)
tinker-finetune rl -m Qwen/Qwen3-8B -p examples/data/rl_prompts.txt --reward numeric_match
tinker-finetune rl -m Qwen/Qwen3-8B -p examples/data/rl_prompts.txt --reward rm:thinkingmachines/Inkling-Small

# DPO / preference training on Inkling
tinker-finetune dpo -m thinkingmachines/Inkling -d examples/data/preferences_sample.jsonl

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
├── data/                tokenization · templating+masking · datasets · packing · preferences · validation · sharding · drift
├── tinker_client/       backend protocol · live SDK adapter · offline fake
├── training/            SFT · RL · DPO trainers · LR schedules · checkpointing · guards · retention
├── budget.py            token/step/USD accounting, MoE-aware pricing, ceilings
├── resilience.py        retry · circuit breaker · rate limiter · resilient backend
├── rewards.py           pluggable RL reward registry (heuristics)
├── reward_models.py     RLHF reward-model scorer + reward spec resolver
├── metrics.py           JSONL + Weights & Biases metrics loggers
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
- [RLHF, DPO, metrics & live runs](docs/rlhf_dpo.md) — preference optimization.
- [API reference](docs/api.md) — every endpoint.
- [Models](docs/models.md) — the open-weight registry, incl. Inkling.
- [Hardening](docs/hardening.md) — validation, resilience, budgets, guards, sharding, retention, drift.

## Status & disclaimer

Dry-run mode is a *simulator* — its loss/reward curves demonstrate the pipeline,
not real model quality. Real training requires a Tinker account and the `tinker`
SDK. Tinker is a product of Thinking Machines Lab; this repository is an
independent client/backend and is not affiliated with them.

## License

MIT — see [LICENSE](LICENSE).
