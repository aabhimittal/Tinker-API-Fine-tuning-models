# RLHF, DPO, metrics, and live runs

This covers the preference-optimization features layered on top of SFT/RL.
Everything runs offline in dry-run mode; set `TF_DRY_RUN=false` + `TINKER_API_KEY`
to run against the live Tinker API.

## RLHF with a reward-model scorer

RLHF trains a policy against a **reward model** (RM) that scores completions.
Point the RL `reward` at an RM with the `rm:<model>` spec:

```bash
tinker-finetune rl -m thinkingmachines/Inkling \
  -p examples/data/rl_prompts.txt \
  --reward rm:thinkingmachines/Inkling-Small --iterations 100
```

The loop: the RL trainer samples a group of completions per prompt → the reward
model scores each → GRPO turns scores into group-relative advantages → the
policy is optimized toward higher reward. In dry-run mode the RM is a
deterministic `FakeRewardModel`; live, `TinkerRewardModel` scores with a
Tinker-served model. Register a custom RM by implementing the `RewardModel`
protocol (`score(prompt, completion) -> float`) in `reward_models.py`.

YAML: see `configs/rlhf_inkling.yaml`, run with `python scripts/run_rl.py`.

## DPO / preference training

DPO optimizes directly on preference pairs — no reward model, no sampling loop —
against a frozen reference policy:

```
logits = beta * ((logp_pol(chosen) - logp_ref(chosen))
               - (logp_pol(rejected) - logp_ref(rejected)))
loss    = -log_sigmoid(logits)
```

Dataset (`examples/data/preferences_sample.jsonl`), one pair per line:

```json
{"prompt": "a question", "chosen": "good answer", "rejected": "bad answer"}
{"prompt": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}
```

Run it:

```bash
tinker-finetune dpo -m thinkingmachines/Inkling \
  -d examples/data/preferences_sample.jsonl --beta 0.1 --loss-type sigmoid
# or from YAML:
python scripts/run_dpo.py configs/dpo_inkling.yaml
```

Options: `--beta` (KL strength), `--loss-type sigmoid|ipo`, `--reference-free`,
and cDPO label smoothing via the API/config. The trainer reports the implicit
**reward margin** (chosen − rejected) and **reward accuracy** (fraction with
chosen preferred) each step.

## Metrics: JSONL and Weights & Biases

Every job writes per-step metrics to `runs/jobs/<id>/metrics.jsonl`. To also
stream to W&B:

```bash
pip install wandb
export TF_WANDB_PROJECT=my-project   # optional: TF_WANDB_ENTITY, TF_WANDB_MODE
```

When `TF_WANDB_PROJECT` is set the runner logs to W&B automatically; otherwise it
falls back to JSONL. See `metrics.py` (`build_logger`, `WandbLogger`).

## Live end-to-end run

`scripts/e2e_live.py` runs a real fine-tune + sample against the live API:

```bash
export TINKER_API_KEY=...  TF_DRY_RUN=false
pip install tinker
python scripts/e2e_live.py --base-model Qwen/Qwen3-8B --train examples/data/sft_sample.jsonl
```

If any preflight item is missing (key, `TF_DRY_RUN=false`, the SDK) it prints a
checklist and exits 0 — safe to invoke anywhere, including CI.

## HTTP endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/jobs/dpo` | Start a DPO job (preference dataset). |
| POST | `/v1/jobs/rl` | RL job; set `"reward": "rm:<model>"` for RLHF. |

DPO metrics (`reward_margin`, `reward_accuracy`) appear in `/v1/jobs/{id}/metrics`.
