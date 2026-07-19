# Architecture

`tinker-finetune` is layered so the training logic never depends on either the
network transport or the HTTP surface. Every layer talks to the one below
through a small, typed interface.

```
                 ┌─────────────────────────────────────────────┐
   HTTP clients  │  FastAPI  (api/)                             │
   & CLI         │  models · datasets · jobs · inference        │
                 └───────────────┬─────────────────────────────┘
                                 │  submits
                 ┌───────────────▼─────────────────────────────┐
                 │  JobManager + Runner (jobs/)                 │
                 │  thread pool · persistence · cancel/stop     │
                 └───────────────┬─────────────────────────────┘
                                 │  drives
        ┌────────────────────────▼────────────────────────────┐
        │  Trainers (training/)  SFTTrainer · RLTrainer         │
        │  LR schedule · checkpointing · metrics callback       │
        └───────────┬───────────────────────────┬──────────────┘
          data (data/)                  primitives (tinker_client/)
        ┌───────────▼──────────┐       ┌─────────▼──────────────┐
        │ tokenization         │       │ TinkerBackend protocol │
        │ chat templating+mask │       │  ├─ LiveTinkerBackend  │
        │ datasets · packing   │       │  └─ FakeTinkerBackend  │
        └──────────────────────┘       └────────────────────────┘
                                        forward_backward / optim_step
                                        sample / save_state / load_state
```

## Key design decisions

**One backend interface, two implementations.** Nothing imports `tinker`
directly except `tinker_client/live.py`. Everything else depends on the
`TinkerBackend` protocol, so the whole stack runs offline against
`FakeTinkerBackend` — the default `TF_DRY_RUN=true` path — and swaps to the live
SDK by flipping one env var. This keeps tests deterministic and CI credential-free.

**Open weights, known upfront.** `models/registry.py` is an allow-list of fully
open-weight base models (Qwen3 Apache-2.0, Llama-3.x community). The API rejects
any model not in the registry, so a run can never silently depend on a closed
checkpoint. See [models.md](models.md).

**Loss masking is template-agnostic.** `data/templating.py` renders a chat into
segments tagged assistant/non-assistant, then masks loss to assistant tokens
only. The masking works segment-by-segment, so a different chat template does not
change the supervision logic.

**Trainers don't know about jobs or HTTP.** They accept an `on_metrics`
callback and a `should_stop` predicate. The job runner wires those to
persistence and the cooperative cancel event; the CLI wires them to stdout.

## The Tinker training loop

Both trainers reduce to Tinker's core primitives:

| Step | SFT | RL (GRPO) |
|------|-----|-----------|
| generate | — | `sample` a group per prompt |
| score | — | `reward_fn` → group-relative advantages |
| gradient | `forward_backward(cross_entropy)` | `forward_backward(importance_sampling)` |
| update | `optim_step(Adam, lr)` | `optim_step(Adam, lr)` |
| persist | `save_state` + sampler snapshot | `save_state` + sampler snapshot |

See [usage.md](usage.md) for how to run each.
