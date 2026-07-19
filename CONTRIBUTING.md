# Contributing

Thanks for your interest in improving `tinker-finetune`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test lint
```

Everything runs offline in dry-run mode (`TF_DRY_RUN=true`), so you never need
Tinker credentials to develop or test.

## Guidelines

- **Keep the backend abstraction clean.** Only `tinker_client/live.py` may
  import `tinker`. Everything else depends on the `TinkerBackend` protocol so it
  stays testable offline.
- **Only open-weight models.** New registry entries must have openly published
  weights and a Tinker-accepted base-model name.
- **Add tests.** New behavior needs a test that passes in dry-run mode.
- **Lint & format** with `make lint` / `make fmt` (ruff) before opening a PR.
- **Type hints** on public functions; `make typecheck` runs mypy.

## Commit / PR

- Small, focused commits with clear messages.
- Fill in the PR template; describe what changed and how you verified it.
- CI (lint + tests on Python 3.10–3.12) must be green.
