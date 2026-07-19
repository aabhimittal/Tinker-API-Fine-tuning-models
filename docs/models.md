# Open-weight model registry

This project only fine-tunes **fully open-weight** base models — every weight is
published and known upfront, so runs are reproducible and auditable. Any model
not in this registry is rejected by the API and CLI.

The default family is **Qwen3** (Apache-2.0): the entire dense + MoE lineup is
released openly. **Llama-3.x** (Meta community license) is included for coverage.

| Model | Family | Params (B) | Active (B) | Context | MoE | LoRA rank | License |
|-------|--------|-----------:|-----------:|--------:|:---:|----------:|---------|
| Qwen/Qwen3-0.6B | qwen3 | 0.6 | 0.6 | 32768 | – | 16 | Apache-2.0 |
| Qwen/Qwen3-1.7B | qwen3 | 1.7 | 1.7 | 32768 | – | 16 | Apache-2.0 |
| Qwen/Qwen3-4B | qwen3 | 4 | 4 | 32768 | – | 32 | Apache-2.0 |
| Qwen/Qwen3-8B | qwen3 | 8 | 8 | 32768 | – | 32 | Apache-2.0 |
| Qwen/Qwen3-14B | qwen3 | 14 | 14 | 32768 | – | 32 | Apache-2.0 |
| Qwen/Qwen3-32B | qwen3 | 32 | 32 | 32768 | – | 64 | Apache-2.0 |
| Qwen/Qwen3-30B-A3B | qwen3 | 30 | 3 | 32768 | ✓ | 32 | Apache-2.0 |
| Qwen/Qwen3-235B-A22B | qwen3 | 235 | 22 | 32768 | ✓ | 64 | Apache-2.0 |
| meta-llama/Llama-3.1-8B-Instruct | llama3 | 8 | 8 | 131072 | – | 32 | Llama-3.1 |
| meta-llama/Llama-3.1-70B-Instruct | llama3 | 70 | 70 | 131072 | – | 64 | Llama-3.1 |
| meta-llama/Llama-3.2-1B-Instruct | llama3 | 1 | 1 | 131072 | – | 16 | Llama-3.2 |
| meta-llama/Llama-3.2-3B-Instruct | llama3 | 3 | 3 | 131072 | – | 32 | Llama-3.2 |
| meta-llama/Llama-3.3-70B-Instruct | llama3 | 70 | 70 | 131072 | – | 64 | Llama-3.3 |

For MoE models, *active* parameters are what each token routes through — the
cost driver — while *total* is what ships. Both are recorded so a trainer can
reason about memory and throughput upfront.

## Adding a model

Append a `ModelInfo(...)` entry to `src/tinker_finetune/models/registry.py`.
Keep it to models with openly published weights and a name the Tinker API's
`create_lora_training_client` accepts.
