"""Edge cases for dataset validation, PII redaction and near-duplicate search."""

from __future__ import annotations

import json

import pytest

from tinker_finetune.data.validation import (
    DatasetValidationError,
    NearDuplicateIndex,
    Severity,
    redact_examples,
    redact_pii,
    validate_dataset,
)
from tinker_finetune.models.schemas import ChatExample, Message, Role


def ex(prompt: str, answer: str, *, system: str | None = None) -> ChatExample:
    msgs = [Message(role=Role.system, content=system)] if system else []
    msgs += [Message(role=Role.user, content=prompt), Message(role=Role.assistant, content=answer)]
    return ChatExample(messages=msgs)


def corpus(n: int) -> list[ChatExample]:
    return [ex(f"question {i}", f"answer {i}") for i in range(n)]


# --- PII ------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,kind",
    [
        ("write to jane.doe+tag@sub.example.co.uk please", "email"),
        ("card 4111 1111 1111 1111 on file", "credit_card"),
        ("ssn 123-45-6789", "ssn"),
        ("call 415-555-0100", "phone"),
        ("key AKIAIOSFODNN7EXAMPLE rotated", "aws_access_key"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "github_token"),
        ("use sk-ant-abcdefghijklmnop12345", "api_key"),
        ("host 10.0.0.14 responded", "ipv4"),
        ("-----BEGIN RSA PRIVATE KEY-----\nAAAA\n-----END RSA PRIVATE KEY-----", "private_key"),
    ],
)
def test_redact_pii_detects_each_kind(text: str, kind: str) -> None:
    redacted, kinds = redact_pii(text)
    assert kind in kinds
    assert f"[REDACTED:{kind}]" in redacted


def test_credit_card_requires_luhn() -> None:
    """A 16-digit order number is not a card number; false positives are costly."""
    _, kinds = redact_pii("order 1234567812345678 shipped")
    assert "credit_card" not in kinds
    _, kinds = redact_pii("card 4111-1111-1111-1111")
    assert "credit_card" in kinds


def test_redaction_preserves_sentence_structure() -> None:
    redacted, _ = redact_pii("mail bob@x.com now")
    assert redacted == "mail [REDACTED:email] now"


def test_redact_examples_returns_clean_copy_and_counts() -> None:
    cleaned, counts = redact_examples([ex("mail a@b.com", "sure, a@b.com")])
    assert counts["email"] == 2
    assert "@" not in cleaned[0].messages[1].content
    report = validate_dataset(cleaned)
    assert not report.has("pii_detected")


# --- duplicates / near duplicates ----------------------------------------
def test_exact_duplicates_are_counted_not_listed_per_row() -> None:
    report = validate_dataset(corpus(8) + [ex("question 0", "answer 0")])
    dup = report.by_code("exact_duplicates")
    assert dup is not None and dup.count == 1


def test_duplicates_escalate_to_error_above_twenty_percent() -> None:
    report = validate_dataset([ex("q", "a")] * 10)
    assert report.by_code("exact_duplicates").severity is Severity.error


def test_whitespace_and_case_differences_still_count_as_duplicates() -> None:
    report = validate_dataset(corpus(8) + [ex("QUESTION  0", "Answer\t0")])
    assert report.has("exact_duplicates")


def test_near_duplicate_detection_flags_paraphrase_of_earlier_row() -> None:
    base = "the quick brown fox jumps over the lazy dog in the yard"
    near = "the quick brown fox jumps over the lazy dog in the yards"
    report = validate_dataset(corpus(8) + [ex(base, "ok"), ex(near, "ok")])
    assert report.has("near_duplicates")


def test_near_duplicate_index_is_deterministic_across_instances() -> None:
    """hash() is PYTHONHASHSEED-salted; the index must not be."""
    a, b = NearDuplicateIndex(), NearDuplicateIndex()
    a.add(0, "some reasonably long document about widgets")
    b.add(0, "some reasonably long document about widgets")
    assert a.query("some reasonably long document about widgets") == b.query(
        "some reasonably long document about widgets"
    )


def test_near_duplicate_index_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError):
        NearDuplicateIndex(num_perm=32, bands=5)
    with pytest.raises(ValueError):
        NearDuplicateIndex(threshold=0)


def test_near_duplicate_index_handles_empty_and_short_text() -> None:
    idx = NearDuplicateIndex()
    idx.add(0, "")
    idx.add(1, "ab")
    assert idx.query("") == []
    assert idx.query("ab") == [(1, 1.0)]


# --- structural / signal problems ----------------------------------------
def test_empty_assistant_turn_is_an_error() -> None:
    report = validate_dataset(corpus(8) + [ex("q", "   \n ")])
    assert report.by_code("empty_assistant_turn").severity is Severity.error


def test_label_conflict_when_same_prompt_has_two_answers() -> None:
    report = validate_dataset(corpus(8) + [ex("q", "a"), ex("q", "b")])
    assert report.has("label_conflict")


def test_response_collapse_flags_degenerate_corpus() -> None:
    report = validate_dataset([ex(f"q{i}", "yes") for i in range(10)])
    assert report.has("response_collapse")
    assert report.stats["most_common_response_share"] == 1.0


def test_system_turn_after_start_and_consecutive_roles() -> None:
    messy = ChatExample(
        messages=[
            Message(role=Role.user, content="hi"),
            Message(role=Role.system, content="be terse"),
            Message(role=Role.assistant, content="ok"),
            Message(role=Role.assistant, content="really ok"),
        ]
    )
    report = validate_dataset(corpus(8) + [messy])
    assert report.has("system_turn_not_first")
    assert report.has("consecutive_same_role")


# --- truncation / context -------------------------------------------------
def test_truncation_that_orphans_the_answer_is_an_error() -> None:
    """Tail-keeping truncation can drop the whole question, leaving an
    unconditioned answer that teaches the model to blurt it out."""
    report = validate_dataset([ex("x" * 4000, "the answer")], max_seq_len=16)
    assert report.by_code("prompt_truncated_away").severity is Severity.error


def test_truncation_warning_when_supervision_survives() -> None:
    report = validate_dataset(corpus(8) + [ex("x" * 500, "y" * 500)], max_seq_len=400)
    assert report.has("truncation")
    assert not report.has("zero_supervised_tokens")


def test_context_overflow_is_measured_against_the_real_model_window() -> None:
    small_ctx = validate_dataset([ex("x" * 200_000, "ok")], base_model="Qwen/Qwen3-8B")
    assert small_ctx.has("context_overflow")
    big_ctx = validate_dataset([ex("x" * 200_000, "ok")], base_model="thinkingmachines/Inkling")
    assert not big_ctx.has("context_overflow")


def test_unknown_base_model_does_not_crash_validation() -> None:
    report = validate_dataset(corpus(8), base_model="not/a-real-model")
    assert report.num_examples == 8


# --- unicode hygiene ------------------------------------------------------
def test_bidi_override_is_an_error() -> None:
    report = validate_dataset([ex("transfer ‮ funds", "ok")])
    assert report.by_code("bidi_override").severity is Severity.error


def test_zero_width_and_replacement_characters_are_flagged() -> None:
    report = validate_dataset([ex("hel​lo", "wor�ld")])
    assert report.has("zero_width_characters")
    assert report.has("replacement_character")


def test_lone_surrogate_is_reported_rather_than_crashing() -> None:
    report = validate_dataset([ex("bad \ud800 char", "ok")])
    assert report.by_code("lone_surrogate").severity is Severity.error
    # The tokenizer cannot encode it, but validation still completes.
    assert report.has("untokenizable_example")
    assert report.stats["seq_len_max"] == 0


def test_nfc_mismatch_is_informational() -> None:
    report = validate_dataset(corpus(8) + [ex("café", "ok")])  # e + combining acute
    assert report.by_code("unicode_not_nfc").severity is Severity.info


# --- contamination --------------------------------------------------------
def test_eval_contamination_detects_exact_and_near_overlap() -> None:
    train = corpus(8) + [ex("what is the capital of france", "paris")]
    evals = [ex("what is the capital of france", "paris"), ex("unrelated question here", "no")]
    report = validate_dataset(train, eval_examples=evals)
    finding = report.by_code("eval_contamination")
    assert finding is not None and finding.count == 1
    assert finding.severity is Severity.error


def test_clean_eval_split_is_not_flagged() -> None:
    report = validate_dataset(corpus(8), eval_examples=[ex("totally different thing", "z")])
    assert not report.has("eval_contamination")


# --- report semantics -----------------------------------------------------
def test_empty_dataset_reports_rather_than_raises() -> None:
    report = validate_dataset([])
    assert report.by_code("empty_dataset").severity is Severity.error
    assert not report.ok


def test_raise_for_severity_gates_a_pipeline() -> None:
    report = validate_dataset(corpus(8) + [ex("q", "a@b.com")])
    with pytest.raises(DatasetValidationError):
        report.raise_for_severity(Severity.error)
    clean = validate_dataset(corpus(8))
    clean.raise_for_severity(Severity.error)  # no raise


def test_raise_for_severity_can_be_tightened_to_warnings() -> None:
    report = validate_dataset(corpus(4))  # tiny_dataset warning only
    report.raise_for_severity(Severity.error)
    with pytest.raises(DatasetValidationError):
        report.raise_for_severity(Severity.warning)


def test_findings_are_capped_per_code_but_counts_are_exact() -> None:
    report = validate_dataset([ex(f"q{i}", "  ") for i in range(50)])
    finding = report.by_code("empty_assistant_turn")
    assert finding.count == 50
    assert len(finding.examples) <= 5


def test_report_serialises_to_json() -> None:
    report = validate_dataset(corpus(8))
    assert json.loads(json.dumps(report.to_dict()))["num_examples"] == 8
    assert "supervised_ratio" in report.stats


def test_findings_are_ordered_errors_first() -> None:
    report = validate_dataset(corpus(4) + [ex("q", "a@b.com")])
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1, "info": 2}[s.value])
