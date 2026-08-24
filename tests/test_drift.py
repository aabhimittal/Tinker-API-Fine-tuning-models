"""Edge cases for distribution-drift detection between corpora."""

from __future__ import annotations

import math

import pytest

from tinker_finetune.data.drift import (
    bucketize,
    compare_categorical,
    compare_datasets,
    compare_distributions,
    js_divergence,
    ks_statistic,
    psi,
    quantile_bins,
)
from tinker_finetune.data.validation import Severity
from tinker_finetune.models.schemas import ChatExample, Message, Role


def ex(prompt: str, answer: str) -> ChatExample:
    return ChatExample(
        messages=[Message(role=Role.user, content=prompt), Message(role=Role.assistant, content=answer)]
    )


# --- statistics ---------------------------------------------------------------


def test_psi_is_zero_for_identical_distributions():
    counts = [10, 20, 30, 40]
    assert psi(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_shift():
    base = [50, 30, 20]
    small = psi(base, [45, 33, 22])
    large = psi(base, [5, 5, 90])
    assert 0 < small < 0.1 < large


def test_psi_is_finite_on_disjoint_supports():
    value = psi([100, 0, 0], [0, 0, 100])
    assert math.isfinite(value) and value > 1.0


def test_psi_rejects_misaligned_bins():
    with pytest.raises(ValueError):
        psi([1, 2], [1, 2, 3])


def test_psi_handles_an_empty_side():
    assert psi([1, 2, 3], [0, 0, 0]) == float("inf")
    assert psi([0, 0], [0, 0]) == 0.0
    assert psi([], []) == 0.0


def test_jsd_is_bounded_and_symmetric():
    a, b = [90, 10], [10, 90]
    assert js_divergence(a, a) == pytest.approx(0.0, abs=1e-9)
    assert js_divergence(a, b) == pytest.approx(js_divergence(b, a))
    assert 0.0 <= js_divergence([100, 0], [0, 100]) <= 1.0
    assert js_divergence([100, 0], [0, 100]) > 0.9


def test_ks_detects_a_shift_and_ignores_a_match():
    same = list(range(100))
    assert ks_statistic(same, same) == pytest.approx(0.0)
    assert ks_statistic(same, [x + 1000 for x in same]) == pytest.approx(1.0)
    assert ks_statistic([], []) == 0.0
    assert ks_statistic([1.0], []) == 1.0


def test_quantile_bins_handle_degenerate_baselines():
    # A constant baseline collapses to a single edge at that value, so a
    # candidate that moves off the constant still lands in a different bucket.
    assert quantile_bins([5.0] * 100) == [5.0]
    assert bucketize([5.0, 9.0], quantile_bins([5.0] * 100)) == [1, 1]
    assert len(quantile_bins([1.0, 2.0, 3.0], num_bins=10)) <= 2
    assert len(quantile_bins(list(range(1000)), num_bins=10)) == 9
    with pytest.raises(ValueError):
        quantile_bins([])
    with pytest.raises(ValueError):
        quantile_bins([1.0], num_bins=0)


def test_bucketize_places_values_relative_to_edges():
    assert bucketize([0, 5, 10, 15], [5, 10]) == [2, 1, 1]
    assert bucketize([], [1]) == [0, 0]
    assert sum(bucketize([-1e9, 1e9], [0])) == 2


def test_non_finite_and_non_numeric_values_are_rejected():
    with pytest.raises(ValueError):
        bucketize([float("nan")], [1])
    with pytest.raises(ValueError):
        bucketize([float("inf")], [1])
    with pytest.raises(TypeError):
        bucketize(["12"], [1])
    with pytest.raises(TypeError):
        bucketize([True], [1])


# --- feature comparison -------------------------------------------------------


def test_identical_features_do_not_drift():
    values = [float(i % 40) for i in range(400)]
    drift = compare_distributions("len", values, values)
    assert drift.severity is Severity.info
    assert not drift.drifted
    assert drift.psi == pytest.approx(0.0, abs=1e-9)


def test_a_doubled_length_distribution_is_flagged():
    base = [float(i % 40) + 10 for i in range(400)]
    drift = compare_distributions("len", base, [v * 2 for v in base])
    assert drift.severity is Severity.error
    assert drift.ks > 0.4
    assert drift.candidate_summary["mean"] > drift.baseline_summary["mean"]


def test_small_samples_are_downgraded_not_alarming():
    drift = compare_distributions("len", [1.0, 2.0, 3.0], [100.0, 200.0, 300.0])
    assert drift.severity is Severity.warning
    assert "small sample" in drift.detail


def test_empty_side_is_a_warning_not_a_crash():
    drift = compare_distributions("len", [1.0, 2.0], [])
    assert drift.severity is Severity.warning
    assert drift.ks is None


def test_constant_baseline_does_not_divide_by_zero():
    drift = compare_distributions("len", [7.0] * 200, [7.0] * 200)
    assert drift.psi == pytest.approx(0.0, abs=1e-9)


def test_categorical_drift_reports_unseen_mass():
    base = ["yes"] * 90 + ["no"] * 10
    cand = ["maybe"] * 60 + ["yes"] * 40
    drift = compare_categorical("answer", base, cand)
    assert drift.severity is Severity.error
    assert drift.candidate_summary["unseen_share"] == pytest.approx(0.6)
    assert "unseen" in drift.detail


def test_categorical_drift_on_an_empty_side():
    drift = compare_categorical("answer", [], ["a"])
    assert drift.severity is Severity.warning


# --- corpus-level report ------------------------------------------------------


def test_identical_corpora_report_no_drift():
    corpus = [ex(f"question {i} " * (i % 5 + 1), f"answer {i}") for i in range(200)]
    report = compare_datasets(corpus, list(corpus))
    assert report.severity is Severity.info
    assert report.ok and report.drifted() == []


def test_style_collapse_is_caught_even_when_lengths_match():
    """Every new answer starts 'Sure! Here...' - lengths barely move, style does."""
    base = [ex(f"q{i}", f"answer number {i} here") for i in range(200)]
    cand = [ex(f"q{i}", f"sure here is answer {i}") for i in range(200)]
    report = compare_datasets(base, cand)
    prefix = next(f for f in report.features if f.feature == "answer_prefix")
    assert prefix.severity is Severity.error


def test_length_explosion_is_caught():
    base = [ex("q", "short answer") for _ in range(200)]
    cand = [ex("q", "a much longer answer " * 20) for _ in range(200)]
    report = compare_datasets(base, cand)
    assert report.severity is Severity.error
    assert any(f.feature == "answer_chars" for f in report.drifted())


def test_empty_corpus_is_an_error_not_an_exception():
    report = compare_datasets([], [ex("q", "a")])
    assert report.severity is Severity.error
    assert not report.ok
    assert "empty" in report.notes[0]


def test_size_change_is_noted_without_being_drift():
    base = [ex(f"q{i}", f"a{i}") for i in range(200)]
    report = compare_datasets(base, base[:50])
    assert any("size changed" in note for note in report.notes)


def test_report_serializes_and_renders():
    base = [ex(f"q{i}", f"a{i}") for i in range(60)]
    report = compare_datasets(base, base)
    payload = report.to_dict()
    assert payload["severity"] == "info"
    assert {f["feature"] for f in payload["features"]} == {
        "prompt_chars",
        "answer_chars",
        "turns",
        "answer_prefix",
        "vocabulary",
    }
    assert "drift info" in report.render()


def test_single_turn_vs_multi_turn_structure_drift():
    base = [ex(f"q{i}", f"a{i}") for i in range(200)]
    multi = [
        ChatExample(
            messages=[
                Message(role=Role.system, content="be nice"),
                Message(role=Role.user, content=f"q{i}"),
                Message(role=Role.assistant, content=f"a{i}"),
                Message(role=Role.user, content="more"),
                Message(role=Role.assistant, content="ok"),
            ]
        )
        for i in range(200)
    ]
    report = compare_datasets(base, multi)
    turns = next(f for f in report.features if f.feature == "turns")
    assert turns.severity is Severity.error


# --- surfaces -----------------------------------------------------------------


def write_jsonl(path, answers):
    import json

    path.write_text("".join(json.dumps({"prompt": f"q{i}", "completion": a}) + "\n" for i, a in enumerate(answers)))
    return str(path)


def test_cli_drift_exits_nonzero_on_drift(tmp_path):
    from typer.testing import CliRunner

    from tinker_finetune.cli import app

    base = write_jsonl(tmp_path / "a.jsonl", [f"short answer {i}" for i in range(60)])
    new = write_jsonl(tmp_path / "b.jsonl", ["sure here is a far longer answer " * 8 for _ in range(60)])
    runner = CliRunner()
    assert runner.invoke(app, ["drift", base, new]).exit_code == 1
    assert runner.invoke(app, ["drift", base, base]).exit_code == 0
    assert runner.invoke(app, ["drift", base, base, "--fail-on", "info"]).exit_code == 1


def test_api_drift_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from tinker_finetune.api.app import create_app

    base = write_jsonl(tmp_path / "a.jsonl", [f"short answer {i}" for i in range(60)])
    new = write_jsonl(tmp_path / "b.jsonl", ["sure here is a far longer answer " * 8 for _ in range(60)])
    client = TestClient(create_app())
    body = client.get("/v1/datasets/drift", params={"path": new, "baseline_path": base}).json()
    assert body["severity"] == "error"
    assert {f["feature"] for f in body["features"]} >= {"answer_chars", "vocabulary"}
    missing = client.get("/v1/datasets/drift", params={"path": "nope.jsonl", "baseline_path": base})
    assert missing.status_code == 400
