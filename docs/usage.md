# Usage

Everything below works offline in dry-run mode. Set `TF_DRY_RUN=false` +
`TINKER_API_KEY` to run for real against the same commands.

## Data format

JSONL, one example per line, in either shape:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"prompt": "...", "completion": "..."}
```

Only **assistant** tokens receive loss. Inspect a dataset first:

```bash
tinker-finetune inspect examples/data/sft_sample.jsonl
```

## SFT via the CLI

```bash
tinker-finetune sft \
  --base-model Qwen/Qwen3-8B \
  --train examples/data/sft_sample.jsonl \
  --epochs 3 --batch-size 8 --lora-rank 32 --lr 1e-4
```

## SFT via a YAML config

```bash
python scripts/run_sft.py configs/sft_qwen3.yaml
```

## RL (GRPO) via the CLI

```bash
tinker-finetune rl \
  --base-model Qwen/Qwen3-8B \
  --prompts examples/data/rl_prompts.txt \
  --reward numeric_match --iterations 50 --group-size 8
```

Rewards are pluggable — `length_target`, `numeric_match`, `nonempty` ship in
`tinker_finetune/rewards.py`. Register your own (e.g. a reward-model scorer):

```python
from tinker_finetune.rewards import register

@register("my_reward")
def my_reward(prompt: str, completion: str) -> float:
    return score(prompt, completion)
```

## Sampling

```bash
tinker-finetune sample --base-model Qwen/Qwen3-8B --prompt "Hello" --max-new-tokens 64
```

## HTTP API

```bash
make serve   # then open http://localhost:8000/docs
```

Submit an SFT job and poll it:

```bash
curl -s localhost:8000/v1/jobs/sft -H 'content-type: application/json' -d '{
  "base_model": "Qwen/Qwen3-8B",
  "train_path": "examples/data/sft_sample.jsonl",
  "epochs": 2
}' | jq .job.id

curl -s localhost:8000/v1/jobs/<id>/metrics | jq .
```

See [api.md](api.md) for the full endpoint reference.
