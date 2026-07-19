"""Live backend that wraps the real Tinker SDK.

Import of ``tinker`` is deferred to construction time so the rest of the
package (and the whole test suite) works without the SDK installed. Install it
with ``pip install tinker`` and set ``TINKER_API_KEY`` to run for real.

The mapping to the SDK follows the public Tinker cookbook API:

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(base_model=..., rank=...)
    fwd_bwd_future = training_client.forward_backward(data, loss_fn="cross_entropy")
    optim_future = training_client.optim_step(tinker.AdamParams(...))
    sampling_client = training_client.save_weights_and_get_sampling_client(name=...)
"""

from __future__ import annotations

from typing import Any

from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import LoRAConfig, OptimConfig
from tinker_finetune.tinker_client.client import (
    Datum,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
)

log = get_logger(__name__)


class LiveTinkerBackend:
    """Adapter from this project's primitives onto the Tinker SDK."""

    def __init__(self, base_model: str, lora: LoRAConfig, api_key: str | None = None) -> None:
        try:
            import tinker  # noqa: F401
        except ImportError as exc:  # pragma: no cover - requires the SDK
            raise RuntimeError(
                "The 'tinker' SDK is not installed. Run `pip install tinker` and set "
                "TINKER_API_KEY, or set TF_DRY_RUN=true to use the offline fake backend."
            ) from exc

        import tinker

        self.base_model = base_model
        self._tinker = tinker
        self._service = tinker.ServiceClient()  # reads TINKER_API_KEY from env
        self._training = self._service.create_lora_training_client(
            base_model=base_model,
            rank=lora.rank,
        )
        self._sampling_client: Any = None
        log.info("Live Tinker training client created for %s (rank=%d)", base_model, lora.rank)

    # -- helpers ------------------------------------------------------------
    def _to_tinker_datum(self, d: Datum) -> Any:
        t = self._tinker
        model_input = t.ModelInput.from_ints(d.input_tokens)
        loss_inputs = {
            "target_tokens": d.target_tokens,
            "weights": d.weights,
        }
        if d.advantages is not None:
            loss_inputs["advantages"] = d.advantages
        return t.Datum(model_input=model_input, loss_fn_inputs=loss_inputs)

    # -- primitives ---------------------------------------------------------
    def forward_backward(
        self, batch: list[Datum], loss_fn: str = "cross_entropy"
    ) -> ForwardBackwardResult:
        data = [self._to_tinker_datum(d) for d in batch]
        future = self._training.forward_backward(data, loss_fn=loss_fn)
        result = future.result()
        per_datum = [float(x) for x in getattr(result, "losses", []) or []]
        total = float(getattr(result, "loss", sum(per_datum) / max(1, len(per_datum))))
        num_tokens = sum(d.num_supervised_tokens for d in batch)
        return ForwardBackwardResult(
            loss=total, per_datum_loss=per_datum, num_tokens=num_tokens
        )

    def optim_step(self, optim: OptimConfig, lr: float) -> OptimStepResult:
        t = self._tinker
        params = t.AdamParams(
            learning_rate=lr,
            beta1=optim.beta1,
            beta2=optim.beta2,
            eps=optim.eps,
            weight_decay=optim.weight_decay,
        )
        future = self._training.optim_step(params)
        result = future.result()
        grad_norm = float(getattr(result, "grad_norm", 0.0) or 0.0)
        return OptimStepResult(learning_rate=lr, grad_norm=grad_norm)

    def sample(
        self,
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> SampleResult:
        t = self._tinker
        if self._sampling_client is None:
            self._sampling_client = self._training.save_weights_and_get_sampling_client(
                name="rollout"
            )
        params = t.SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        model_input = t.ModelInput.from_ints(prompt_tokens)
        future = self._sampling_client.sample(prompt=model_input, sampling_params=params)
        result = future.result()
        tokens = list(getattr(result, "tokens", []) or [])
        return SampleResult(
            tokens=tokens,
            text=getattr(result, "text", ""),
            logprobs=getattr(result, "logprobs", None),
            stop_reason=getattr(result, "stop_reason", "length"),
        )

    def save_weights_for_sampler(self, name: str) -> str:
        self._sampling_client = self._training.save_weights_and_get_sampling_client(name=name)
        return name

    def save_state(self, path: str) -> str:
        future = self._training.save_state(path)
        result = future.result()
        return str(getattr(result, "path", path))

    def load_state(self, path: str) -> None:
        self._training.load_state(path)
