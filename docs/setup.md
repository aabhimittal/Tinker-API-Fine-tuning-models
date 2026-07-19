# Setup

## Requirements

- Python 3.10+
- (Optional) A Tinker account + API key for live training. Without it, the
  backend runs in dry-run mode against an in-process simulator.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # backend + test tooling
```

## Configure

Copy the example env and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `TF_DRY_RUN` | `true` | Use the offline fake backend. Set `false` for live training. |
| `TINKER_API_KEY` | – | Required when `TF_DRY_RUN=false`. |
| `TF_DEFAULT_BASE_MODEL` | `Qwen/Qwen3-8B` | Default open-weight base model. |
| `TF_ARTIFACTS_DIR` | `./runs` | Where job state + checkpoints are written. |
| `TF_MAX_CONCURRENT_JOBS` | `2` | Thread-pool size for concurrent runs. |
| `TF_PORT` | `8000` | API port. |

## Going live with Tinker

1. Install the SDK: `pip install tinker` (or `pip install -e ".[tinker]"`).
2. Export your key: `export TINKER_API_KEY=...`
3. Disable dry-run: `export TF_DRY_RUN=false`

With those set, `LiveTinkerBackend` wraps `tinker.ServiceClient()` and routes
`forward_backward` / `optim_step` / `sample` to the real API. Nothing else in
the codebase changes.

## Verify the install

```bash
make test          # 32 tests, all offline
tinker-finetune models              # list open-weight base models
make serve         # http://localhost:8000/docs
```
