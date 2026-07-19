"""Chat templating and supervised-datum construction with loss masking.

Renders a conversation into a single token stream and produces a
:class:`~tinker_finetune.tinker_client.client.Datum` where **only assistant
tokens carry loss weight**. Prompt/user/system tokens get weight 0 so the model
is supervised to generate assistant turns, not to parrot the prompt.

A ChatML-style template (``<|im_start|>role ... <|im_end|>``) is used, which
matches Qwen3 and is a close, deterministic stand-in for other families in dry
runs. When a real HF tokenizer with a chat template is present, the live path
can be swapped in; the masking logic here is template-agnostic because it works
segment-by-segment.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinker_finetune.data.tokenization import Tokenizer
from tinker_finetune.models.schemas import ChatExample, Message, Role
from tinker_finetune.tinker_client.client import Datum

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class RenderedChat:
    text: str
    # (segment_text, is_assistant_content) pairs, in order.
    segments: list[tuple[str, bool]]


def render_chat(messages: list[Message], *, add_generation_prompt: bool = False) -> RenderedChat:
    """Render messages to ChatML text, tracking which spans are assistant content."""
    segments: list[tuple[str, bool]] = []
    for msg in messages:
        header = f"{IM_START}{msg.role.value}\n"
        segments.append((header, False))
        if msg.role == Role.assistant:
            # Assistant content + closing tag are supervised.
            segments.append((msg.content, True))
            segments.append((f"{IM_END}\n", True))
        else:
            segments.append((f"{msg.content}{IM_END}\n", False))
    if add_generation_prompt:
        segments.append((f"{IM_START}{Role.assistant.value}\n", False))
    text = "".join(s for s, _ in segments)
    return RenderedChat(text=text, segments=segments)


def build_supervised_datum(
    example: ChatExample,
    tokenizer: Tokenizer,
    *,
    max_seq_len: int,
) -> Datum:
    """Tokenize a chat example into a next-token-prediction datum with masking.

    The sequence is ``tokens[:-1] -> tokens[1:]`` (standard shift). A target
    position is supervised iff its *source* token belongs to an assistant span.
    """
    rendered = render_chat(example.messages)

    tokens: list[int] = []
    supervised_mask: list[bool] = []  # per-token: is this token assistant content?
    for seg_text, is_assistant in rendered.segments:
        seg_ids = tokenizer.encode(seg_text, add_special=False)
        tokens.extend(seg_ids)
        supervised_mask.extend([is_assistant] * len(seg_ids))

    # Terminal EOS closes the final assistant turn and is supervised.
    tokens.append(tokenizer.eos_id)
    supervised_mask.append(True)

    # Truncate (keep the tail, which holds the assistant answer).
    if len(tokens) > max_seq_len:
        tokens = tokens[-max_seq_len:]
        supervised_mask = supervised_mask[-max_seq_len:]

    # Shift for next-token prediction.
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    # A target is supervised if the token we are predicting is assistant content.
    weights = [1.0 if supervised_mask[i + 1] else 0.0 for i in range(len(input_tokens))]

    return Datum(
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        weights=weights,
        metadata={"num_turns": len(example.messages)},
    )
