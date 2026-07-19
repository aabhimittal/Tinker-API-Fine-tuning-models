"""Dataset loading, batching and sequence packing.

Supported input formats (auto-detected):

- **JSONL chat**: one object per line with a ``messages`` array
  (``{"role": ..., "content": ...}``).
- **JSONL prompt/completion**: ``{"prompt": ..., "completion": ...}`` — mapped
  to a two-turn user/assistant conversation.

The loaders yield :class:`ChatExample`; ``iter_batches`` turns them into batches
of :class:`Datum`, optionally packed to reduce padding waste.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tinker_finetune.data.templating import build_supervised_datum
from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.models.schemas import ChatExample, Message, Role
from tinker_finetune.tinker_client.client import Datum


def _row_to_example(row: dict) -> ChatExample:
    if "messages" in row:
        return ChatExample(messages=[Message(**m) for m in row["messages"]])
    if "prompt" in row and "completion" in row:
        return ChatExample(
            messages=[
                Message(role=Role.user, content=str(row["prompt"])),
                Message(role=Role.assistant, content=str(row["completion"])),
            ]
        )
    raise ValueError(
        "Row must contain either 'messages' or both 'prompt' and 'completion'. "
        f"Got keys: {sorted(row)}"
    )


def load_chat_dataset(path: str | Path) -> list[ChatExample]:
    """Load a JSONL dataset from disk into validated chat examples."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    examples: list[ChatExample] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {p}:{lineno}: {exc}") from exc
            examples.append(_row_to_example(row))
    if not examples:
        raise ValueError(f"Dataset {p} contained no examples.")
    return examples


def pack_datums(datums: list[Datum], max_seq_len: int) -> list[Datum]:
    """Greedily concatenate datums into packed sequences up to ``max_seq_len``.

    Reduces padding waste. Masking is preserved because we simply concatenate
    the per-token weights; there is no cross-document loss leakage since packed
    documents are independent next-token streams and boundary predictions carry
    the following document's (correct) targets.
    """
    packed: list[Datum] = []
    cur_in: list[int] = []
    cur_tgt: list[int] = []
    cur_w: list[float] = []

    def flush() -> None:
        nonlocal cur_in, cur_tgt, cur_w
        if cur_in:
            packed.append(Datum(cur_in, cur_tgt, cur_w, metadata={"packed": True}))
            cur_in, cur_tgt, cur_w = [], [], []

    for d in datums:
        if len(d.input_tokens) > max_seq_len:
            # Oversized single doc: emit on its own (already truncated upstream).
            flush()
            packed.append(d)
            continue
        if len(cur_in) + len(d.input_tokens) > max_seq_len:
            flush()
        cur_in.extend(d.input_tokens)
        cur_tgt.extend(d.target_tokens)
        cur_w.extend(d.weights)
    flush()
    return packed


def iter_batches(
    examples: Iterable[ChatExample],
    tokenizer: Tokenizer,
    *,
    batch_size: int,
    max_seq_len: int,
    pack: bool = True,
) -> Iterator[list[Datum]]:
    """Yield batches of :class:`Datum` from chat examples."""
    datums = [
        build_supervised_datum(ex, tokenizer, max_seq_len=max_seq_len) for ex in examples
    ]
    if pack:
        datums = pack_datums(datums, max_seq_len)
    for i in range(0, len(datums), batch_size):
        yield datums[i : i + batch_size]
