# HTTP API reference

Base URL: `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and
`/redoc`. All request/response bodies are JSON.

## Meta

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + dry-run flag + version. |

## Models

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/models` | List open-weight base models. Optional `?family=qwen3`. |
| GET | `/v1/models/rewards` | List registered RL reward functions. |
| GET | `/v1/models/{name}` | Metadata for one model (404 if not open-weight). |

## Datasets

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/datasets/inspect?path=...` | Validate a JSONL dataset, return token stats. |

## Training jobs

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/jobs/sft` | Start an SFT job. Returns `201` + job record. |
| POST | `/v1/jobs/rl` | Start an RL job. Set `"reward": "rm:<model>"` for RLHF. |
| POST | `/v1/jobs/dpo` | Start a DPO / preference-optimization job. |
| GET | `/v1/jobs` | List jobs. Optional `?status=` and `?limit=`. |
| GET | `/v1/jobs/{id}` | Full job record incl. metrics history. |
| GET | `/v1/jobs/{id}/metrics` | Metrics + progress. Optional `?tail=N`. |
| POST | `/v1/jobs/{id}/cancel` | Cooperative cancel (sets the stop event). |

### SFT request body

```json
{
  "base_model": "Qwen/Qwen3-8B",
  "train_path": "examples/data/sft_sample.jsonl",
  "eval_path": null,
  "lora": {"rank": 32, "alpha": 64, "dropout": 0.0},
  "optim": {"learning_rate": 1e-4, "lr_schedule": "cosine", "warmup_ratio": 0.03},
  "epochs": 3, "batch_size": 8, "max_seq_len": 4096, "pack_sequences": true
}
```

### RL request body

```json
{
  "base_model": "Qwen/Qwen3-8B",
  "prompts": ["...", "..."],
  "reward": "numeric_match",
  "iterations": 100, "group_size": 8, "prompts_per_batch": 16,
  "advantage": "grpo", "kl_coef": 0.0
}
```

### DPO request body

```json
{
  "base_model": "thinkingmachines/Inkling",
  "train_path": "examples/data/preferences_sample.jsonl",
  "beta": 0.1, "loss_type": "sigmoid", "reference_free": false,
  "epochs": 1, "batch_size": 4
}
```

## Inference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/inference/sample` | Sample a completion from a base model. |

```json
{"base_model": "Qwen/Qwen3-8B", "prompt": "Hello", "max_new_tokens": 128, "temperature": 0.7}
```

## Job lifecycle

`pending → running → {succeeded | failed | cancelled}`. State is persisted to
`TF_ARTIFACTS_DIR/jobs/<id>.json`; a job left `running` when the process dies is
reconciled to `failed` on restart.
