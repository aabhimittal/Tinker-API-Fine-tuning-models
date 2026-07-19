"""Deterministic in-process fake of the Tinker backend.

It does not run a real model; it simulates the *shape* of training so the API,
job manager, CLI and tests exercise the full stack offline. Loss decays with an
exponential-plus-noise curve, gradient norm shrinks, and sampling echoes a
deterministic pseudo-completion. All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from tinker_finetune.models.schemas import LoRAConfig, OptimConfig
from tinker_finetune.tinker_client.client import (
    Datum,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
)


class FakeTinkerBackend:
    """Simulated backend with a plausible, deterministic training trajectory."""

    def __init__(self, base_model: str, lora: LoRAConfig, seed: int = 0) -> None:
        self.base_model = base_model
        self.lora = lora
        self._rng = random.Random(seed)
        self._step = 0
        # A base loss level derived from model name so different models differ
        # deterministically but reproducibly.
        self._base_loss = 2.0 + (abs(hash(base_model)) % 100) / 100.0
        self._pending_tokens = 0

    # -- primitives ---------------------------------------------------------
    def forward_backward(
        self, batch: list[Datum], loss_fn: str = "cross_entropy"
    ) -> ForwardBackwardResult:
        self._pending_tokens = sum(d.num_supervised_tokens for d in batch)
        # Exponential decay toward a floor, plus small seeded noise.
        decay = math.exp(-self._step / 40.0)
        noise = self._rng.uniform(-0.03, 0.03)
        loss = 0.4 + (self._base_loss - 0.4) * decay + noise
        loss = max(0.05, loss)
        per_datum = [max(0.05, loss + self._rng.uniform(-0.05, 0.05)) for _ in batch]
        return ForwardBackwardResult(
            loss=loss, per_datum_loss=per_datum, num_tokens=self._pending_tokens
        )

    def logprobs(self, batch: list[Datum]) -> list[float]:
        # Deterministic pseudo-logprobs: more supervised tokens => more negative,
        # nudged by training progress so a "trained" policy looks more confident.
        out: list[float] = []
        for d in batch:
            n = max(1, d.num_supervised_tokens)
            base = -0.8 * n * math.exp(-self._step / 80.0) - 0.2 * n
            # Seeded per-datum jitter keyed by token content for reproducibility.
            key = (sum(d.target_tokens) + n) % 1000
            jitter = (key / 1000.0 - 0.5) * 0.1 * n
            out.append(base + jitter)
        return out

    def optim_step(self, optim: OptimConfig, lr: float) -> OptimStepResult:
        self._step += 1
        grad_norm = max(0.01, optim.max_grad_norm * math.exp(-self._step / 60.0))
        grad_norm += self._rng.uniform(-0.01, 0.01)
        return OptimStepResult(learning_rate=lr, grad_norm=max(0.0, grad_norm))

    def sample(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> SampleResult:
        rng = random.Random(seed if seed is not None else self._rng.random())
        n = rng.randint(1, max(1, max_new_tokens))
        # Deterministic pseudo-tokens in a small vocab band.
        tokens = [rng.randint(10, 2000) for _ in range(n)]
        logprobs = [-abs(rng.gauss(1.0, 0.3)) for _ in tokens]
        stop = "stop" if n < max_new_tokens else "length"
        return SampleResult(
            tokens=tokens,
            text=f"<fake completion of {n} tokens>",
            logprobs=logprobs,
            stop_reason=stop,
        )

    def save_weights_for_sampler(self, name: str) -> str:
        return name

    def save_state(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "base_model": self.base_model,
                    "lora_rank": self.lora.rank,
                    "step": self._step,
                    "fake": True,
                },
                indent=2,
            )
        )
        return str(p)

    def load_state(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self._step = int(data.get("step", 0))
