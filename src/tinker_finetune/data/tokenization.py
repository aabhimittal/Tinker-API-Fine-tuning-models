"""Tokenizer abstraction.

A real tokenizer is loaded from Hugging Face when ``transformers`` is available
and network access permits; otherwise a deterministic byte-level fallback is
used so the pipeline is fully testable offline. Both expose the same tiny
surface used by the data layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tinker_finetune.logging_utils import get_logger

log = get_logger(__name__)


@runtime_checkable
class Tokenizer(Protocol):
    eos_id: int

    def encode(self, text: str, *, add_special: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer. Vocab = 256 bytes + a few specials.

    Not efficient, but exact and dependency-free — ideal for tests and dry runs.
    Token ids: 0..255 are raw bytes; 256 = BOS, 257 = EOS.
    """

    BOS_ID = 256
    EOS_ID = 257

    def __init__(self) -> None:
        self.eos_id = self.EOS_ID

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        if add_special:
            return [self.BOS_ID, *ids, self.EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if i < 256)
        return raw.decode("utf-8", errors="replace")


class HFTokenizer:
    """Wrapper over a Hugging Face fast tokenizer."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.eos_id = int(self._tok.eos_token_id or 0)

    def encode(self, text: str, *, add_special: bool = False) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


def build_tokenizer(model_name: str, *, prefer_hf: bool = True) -> Tokenizer:
    """Return an HF tokenizer for ``model_name`` if possible, else the byte fallback."""
    if prefer_hf:
        try:
            return HFTokenizer(model_name)
        except Exception as exc:  # pragma: no cover - network/optional dep
            log.warning(
                "Falling back to ByteTokenizer for %s (%s: %s)",
                model_name,
                type(exc).__name__,
                exc,
            )
    return ByteTokenizer()
