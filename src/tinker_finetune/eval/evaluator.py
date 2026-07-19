"""Evaluation helpers: held-out loss and generation-based exact match."""

from __future__ import annotations

from tinker_finetune.data.datasets import iter_batches
from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.models.schemas import ChatExample, Role
from tinker_finetune.tinker_client.client import TinkerBackend


def evaluate_held_out_loss(
    backend: TinkerBackend,
    tokenizer: Tokenizer,
    examples: list[ChatExample],
    *,
    batch_size: int = 8,
    max_seq_len: int = 4096,
) -> float:
    """Mean cross-entropy over a held-out set (no optimizer step)."""
    batches = list(
        iter_batches(examples, tokenizer, batch_size=batch_size,
                     max_seq_len=max_seq_len, pack=False)
    )
    if not batches:
        return float("nan")
    losses = [backend.forward_backward(b).loss for b in batches]
    return sum(losses) / len(losses)


def exact_match_accuracy(
    backend: TinkerBackend,
    tokenizer: Tokenizer,
    examples: list[ChatExample],
    *,
    max_new_tokens: int = 256,
    normalize: bool = True,
) -> float:
    """Greedy-generate the final assistant turn and compare to the reference.

    A coarse metric (string match), useful for verifiable tasks. For dry runs
    with the fake backend this reflects the simulation, not real quality.
    """

    def norm(s: str) -> str:
        return " ".join(s.lower().split()) if normalize else s

    correct = 0
    counted = 0
    for ex in examples:
        # Take everything up to the last assistant turn as the prompt.
        last_assistant = max(
            i for i, m in enumerate(ex.messages) if m.role == Role.assistant
        )
        reference = ex.messages[last_assistant].content
        prompt_msgs = ex.messages[:last_assistant]
        prompt_text = "\n".join(f"{m.role.value}: {m.content}" for m in prompt_msgs)
        prompt_ids = tokenizer.encode(prompt_text, add_special=True)
        res = backend.sample(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.0)
        gen = res.text or tokenizer.decode(res.tokens)
        counted += 1
        if norm(gen) == norm(reference):
            correct += 1
    return correct / counted if counted else float("nan")
