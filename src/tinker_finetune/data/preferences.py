"""Preference-pair loading and datum construction for DPO.

Supported JSONL formats (auto-detected):

- **messages + chosen/rejected**::

    {"prompt": [{"role": "user", "content": "..."}], "chosen": "...", "rejected": "..."}

- **flat prompt string**::

    {"prompt": "a question", "chosen": "good answer", "rejected": "bad answer"}

Each example yields two supervised datums (chosen, rejected) sharing the same
prompt, with loss weighted on the response tokens only — identical masking to
SFT so the DPO log-ratios are computed over the response, not the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

from tinker_finetune.data.templating import build_supervised_datum
from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.models.schemas import ChatExample, Message, PreferenceExample, Role
from tinker_finetune.tinker_client.client import Datum


def _row_to_preference(row: dict) -> PreferenceExample:
    if "chosen" not in row or "rejected" not in row:
        raise ValueError(
            f"Preference row needs 'chosen' and 'rejected'. Got keys: {sorted(row)}"
        )
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        messages = [Message(**m) for m in prompt]
    elif isinstance(prompt, str):
        messages = [Message(role=Role.user, content=prompt)]
    else:
        raise ValueError("Preference row 'prompt' must be a string or a messages array.")
    return PreferenceExample(
        prompt=messages, chosen=str(row["chosen"]), rejected=str(row["rejected"])
    )


def load_preference_dataset(path: str | Path) -> list[PreferenceExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preference dataset not found: {p}")
    out: list[PreferenceExample] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {p}:{lineno}: {exc}") from exc
            out.append(_row_to_preference(row))
    if not out:
        raise ValueError(f"Preference dataset {p} contained no examples.")
    return out


def _to_chat(pref: PreferenceExample, response: str) -> ChatExample:
    return ChatExample(
        messages=[*pref.prompt, Message(role=Role.assistant, content=response)]
    )


def build_preference_datums(
    pref: PreferenceExample,
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
) -> tuple[Datum, Datum]:
    """Return ``(chosen_datum, rejected_datum)`` for one preference pair."""
    chosen = build_supervised_datum(_to_chat(pref, pref.chosen), tokenizer,
                                    max_seq_len=max_seq_len)
    rejected = build_supervised_datum(_to_chat(pref, pref.rejected), tokenizer,
                                      max_seq_len=max_seq_len)
    return chosen, rejected
