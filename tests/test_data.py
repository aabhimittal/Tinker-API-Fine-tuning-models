from tinker_finetune.data.datasets import iter_batches, load_chat_dataset, pack_datums
from tinker_finetune.data.templating import build_supervised_datum, render_chat
from tinker_finetune.data.tokenization import ByteTokenizer
from tinker_finetune.models.schemas import ChatExample, Message, Role
from tinker_finetune.tinker_client.client import Datum


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    ids = tok.encode("hello", add_special=True)
    assert ids[0] == ByteTokenizer.BOS_ID and ids[-1] == ByteTokenizer.EOS_ID
    assert tok.decode(ids) == "hello"


def test_supervised_masking_only_assistant_tokens_weighted():
    tok = ByteTokenizer()
    ex = ChatExample(messages=[
        Message(role=Role.user, content="ping"),
        Message(role=Role.assistant, content="pong"),
    ])
    datum = build_supervised_datum(ex, tok, max_seq_len=1000)
    # At least one supervised token, and strictly fewer than total (prompt masked).
    assert 0 < datum.num_supervised_tokens < len(datum.input_tokens)
    # Every weight is 0 or 1.
    assert set(datum.weights) <= {0.0, 1.0}


def test_render_chat_marks_assistant_segments():
    rendered = render_chat([
        Message(role=Role.user, content="q"),
        Message(role=Role.assistant, content="a"),
    ])
    assert any(is_asst for _, is_asst in rendered.segments)
    assert "<|im_start|>assistant" in rendered.text


def test_load_dataset_supports_both_formats(sample_dataset):
    examples = load_chat_dataset(sample_dataset)
    assert len(examples) == 2
    assert all(any(m.role == Role.assistant for m in e.messages) for e in examples)


def test_pack_datums_respects_max_len():
    ds = [Datum([1, 2], [2, 3], [1.0, 1.0]) for _ in range(5)]
    packed = pack_datums(ds, max_seq_len=4)
    assert all(len(d.input_tokens) <= 4 for d in packed)
    # Total tokens preserved.
    assert sum(len(d.input_tokens) for d in packed) == 10


def test_iter_batches_yields_expected_batch_count():
    tok = ByteTokenizer()
    examples = [
        ChatExample(messages=[
            Message(role=Role.user, content=f"q{i}"),
            Message(role=Role.assistant, content=f"a{i}"),
        ])
        for i in range(6)
    ]
    batches = list(iter_batches(examples, tok, batch_size=2, max_seq_len=1000, pack=False))
    assert len(batches) == 3
    assert all(len(b) == 2 for b in batches)
