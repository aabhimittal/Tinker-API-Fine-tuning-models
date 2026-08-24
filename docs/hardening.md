# Industrial hardening

Four subsystems exist for one reason: a fine-tuning run is expensive, long, and
fails quietly. Each is usable on its own and each is covered by its own offline
test suite (`tests/test_validation.py`, `test_resilience.py`, `test_budget.py`,
`test_guards.py`).

## 1. Dataset validation (`data/validation.py`)

Everything that can be caught before the first `forward_backward` is caught
there. `validate_dataset()` returns a graded, aggregated `ValidationReport` — it
never raises on bad data, so one pass reports every defect.

```bash
tinker-finetune validate data/train.jsonl \
    --eval data/eval.jsonl \
    --base-model Qwen/Qwen3-8B \
    --max-seq-len 4096 \
    --redact-to data/train.clean.jsonl \
    --fail-on warning        # exit 1 at this severity or above
```

```http
GET /v1/datasets/validate?path=data/train.jsonl&eval_path=data/eval.jsonl
```

| Code | Severity | What it catches |
| --- | --- | --- |
| `pii_detected` | error | Emails, phones, national ids, **Luhn-verified** card numbers, AWS/GitHub/API keys, JWTs, private-key blocks. Fine-tuned weights memorise rare strings. |
| `bidi_override` | error | Bidirectional overrides — rendered text differs from the tokens actually trained on (trojan-source). |
| `lone_surrogate` / `untokenizable_example` | error | Broken encodings; validation reports them instead of crashing on them. |
| `prompt_truncated_away` | error | Truncation keeps the tail, so the answer survives but its question does not — the model learns to emit answers unprompted. |
| `zero_supervised_tokens` | error | Examples that contribute no gradient at all. |
| `context_overflow` | error | Longer than the base model's real context window (registry-driven). |
| `eval_contamination` | error | Eval rows that also appear in train, exactly or near-exactly. |
| `empty_assistant_turn` | error | Blank targets. |
| `exact_duplicates` | warn/error | Duplicates are undeclared upweighting; ≥20% of the corpus escalates to error. |
| `near_duplicates` | warning | ≥85% Jaccard similarity to an earlier row. |
| `label_conflict` | warning | One prompt, contradictory answers. |
| `response_collapse` | warning | One answer dominating ≥50% of the corpus. |
| `truncation`, `consecutive_same_role`, `system_turn_not_first`, `control_characters`, `zero_width_characters`, `replacement_character` | warning | Structural and encoding hygiene. |
| `unicode_not_nfc`, `length_outliers`, `trailing_user_turn`, `tiny_dataset` | info | Worth knowing, not worth blocking. |

Near-duplicate and contamination search use `NearDuplicateIndex`, a MinHash +
banded-LSH index: candidate generation is sub-quadratic, and hashing is
`blake2b`-based so results are reproducible across processes (Python's builtin
`hash()` is `PYTHONHASHSEED`-salted).

`redact_pii()` / `redact_examples()` replace matches with `[REDACTED:<kind>]`
placeholders rather than deleting them, keeping the sentence structure the model
is learning intact.

## 2. Resilience (`resilience.py`)

Three composable layers wrap any `TinkerBackend`:

```python
backend = ResilientBackend(
    build_backend(...),
    limiter=TokenBucket(rate=5, capacity=10),          # shape our own traffic
    breaker=CircuitBreaker(failure_threshold=5, recovery_timeout=30),
    retry=RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=30),
)
```

- **Retries** use exponential backoff with *full jitter* (`U(0, cap)`) so a
  fleet does not resynchronise after a shared outage, and honour a server
  `Retry-After` hint — bounded by `max_delay`, so a hostile header cannot stall
  a run. Only transient failures qualify: 408/425/429/5xx and transport errors.
  A 400 or 401 fails immediately, because retrying it only burns budget.
- **The circuit breaker** fails fast while a service is down, half-opens after
  `recovery_timeout`, admits a limited number of probes, and re-opens with a
  reset timer on a failed probe. Deterministic 4xx errors never trip it — they
  say the request is wrong, not that the service is unhealthy.
- **`optim_step` and `load_state` are not retried by default.** Replaying a
  weight update that partially applied would silently corrupt training; opt in
  with `retry_mutating=True` only if the backend is idempotent.

Clock and sleep are injectable, so the timing logic is tested deterministically
without real sleeping.

## 3. Budgets (`budget.py`)

```python
budget = BudgetTracker(model=cfg.base_model, max_usd=25.0, max_steps=10_000)
budget.preflight(train_tokens=estimate_sft_tokens(num_examples=n, avg_tokens=t, epochs=e))
SFTTrainer(..., budget=budget).train(examples)
```

`CostModel` prices per million tokens from **active** parameters, so a
975B/41B-active MoE such as Inkling prices like a ~41B dense model rather than a
975B one; explicit per-model prices override the heuristic. Limits are inclusive
(usage exactly at the ceiling is allowed, the next token is not), `warn_at`
fires once per limit, usage is recorded even on the step that trips the ceiling
so post-mortem accounting stays accurate, and `preflight()` rejects a job that
cannot fit *before* it consumes anything.

## 4. Training guards (`training/guards.py`)

Guards are pure observers — they return a `GuardDecision` and never touch the
backend, so they are reusable across SFT, DPO and RL loops.

```python
guards = TrainingGuards(
    numerical=NumericalGuard(max_grad_norm=1e3, spike_patience=3),
    early_stopping=EarlyStopping(patience=3, min_delta=1e-3),
)
SFTTrainer(..., guards=guards).train(train, eval_examples=held_out)
```

| Code | Trigger |
| --- | --- |
| `non_finite_loss` / `non_finite_grad` | NaN/Inf — optimizer state is unrecoverable, so stop at the first occurrence. |
| `grad_explosion` | Gradient norm past `max_grad_norm`. |
| `loss_spike` | Loss above `spike_factor` × running median for `spike_patience` **consecutive** steps (one bad batch is tolerated). |
| `divergence` | Median loss trending up across a window — usually an LR too high for the LoRA rank. |
| `zero_loss` | Loss exactly `0.0` for several steps: an empty loss mask, not a miracle. |
| `early_stop` | Held-out metric stopped improving by more than `min_delta` for `patience` evaluations. |

When a guard trips mid-run the SFT trainer checkpoints, logs the reason, and
returns the history collected so far; `guards.raise_if_tripped()` converts that
into a hard failure for callers that prefer one.
