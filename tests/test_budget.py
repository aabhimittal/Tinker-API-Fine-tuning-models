"""Edge cases for cost estimation and hard budget ceilings."""

from __future__ import annotations

import threading

import pytest

from tinker_finetune.budget import (
    BudgetExceededError,
    BudgetTracker,
    CostModel,
    Usage,
    estimate_sft_tokens,
)


def tracker(**kwargs) -> BudgetTracker:
    kwargs.setdefault("model", "Qwen/Qwen3-8B")
    return BudgetTracker(**kwargs)


# --- cost model -----------------------------------------------------------
def test_moe_is_priced_on_active_parameters_not_total() -> None:
    """A 975B/41B-active model must not price like a 975B dense model."""
    costs = CostModel()
    inkling = costs.train_rate("thinkingmachines/Inkling")
    qwen8b = costs.train_rate("Qwen/Qwen3-8B")
    assert inkling == pytest.approx(qwen8b * 41 / 8)


def test_unknown_model_falls_back_instead_of_raising() -> None:
    costs = CostModel(fallback_active_params_b=8.0)
    assert costs.train_rate("vendor/unreleased-model") == costs.train_rate("Qwen/Qwen3-8B")


def test_explicit_prices_win_over_the_size_heuristic() -> None:
    costs = CostModel(train_per_mtok={"Qwen/Qwen3-8B": 1.0}, sample_per_mtok={"Qwen/Qwen3-8B": 0.1})
    assert costs.cost("Qwen/Qwen3-8B", train_tokens=1_000_000) == pytest.approx(1.0)
    assert costs.cost("Qwen/Qwen3-8B", sample_tokens=1_000_000) == pytest.approx(0.1)


def test_sampling_is_cheaper_than_training_the_same_tokens() -> None:
    costs = CostModel()
    assert costs.sample_rate("Qwen/Qwen3-8B") < costs.train_rate("Qwen/Qwen3-8B")


def test_negative_token_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        CostModel().cost("Qwen/Qwen3-8B", train_tokens=-1)


def test_zero_usage_costs_nothing() -> None:
    assert CostModel().cost("Qwen/Qwen3-8B") == 0.0


def test_estimate_sft_tokens_scales_with_epochs() -> None:
    assert estimate_sft_tokens(num_examples=100, avg_tokens=512, epochs=3) == 153_600
    assert estimate_sft_tokens(num_examples=0, avg_tokens=512, epochs=3) == 0
    with pytest.raises(ValueError):
        estimate_sft_tokens(num_examples=-1, avg_tokens=512, epochs=1)


# --- usage accounting -----------------------------------------------------
def test_usage_adds_component_wise() -> None:
    total = Usage(10, 5, 1, 0.5) + Usage(1, 1, 1, 0.25)
    assert (total.train_tokens, total.sample_tokens, total.steps) == (11, 6, 2)
    assert total.total_tokens == 17
    assert total.cost_usd == pytest.approx(0.75)


def test_unlimited_tracker_never_raises() -> None:
    t = tracker()
    for _ in range(100):
        t.record(train_tokens=10**7, steps=1)
    assert t.fraction_used() == 0.0
    assert t.remaining()["cost_usd"] is None


# --- ceilings -------------------------------------------------------------
def test_hard_token_ceiling_raises_at_the_crossing_step() -> None:
    t = tracker(max_tokens=1000)
    t.record(train_tokens=900)
    with pytest.raises(BudgetExceededError) as exc:
        t.record(train_tokens=200)
    assert exc.value.limit_name == "tokens"


def test_usage_exactly_at_the_limit_is_allowed() -> None:
    t = tracker(max_tokens=1000)
    t.record(train_tokens=1000)  # inclusive boundary
    with pytest.raises(BudgetExceededError):
        t.record(train_tokens=1)


def test_step_ceiling_stops_a_runaway_epoch_count() -> None:
    t = tracker(max_steps=3)
    for _ in range(3):
        t.record(steps=1)
    with pytest.raises(BudgetExceededError):
        t.record(steps=1)


def test_cost_ceiling_uses_the_model_price() -> None:
    cheap = BudgetTracker(model="Qwen/Qwen3-8B", max_usd=1.0)
    dear = BudgetTracker(model="thinkingmachines/Inkling", max_usd=1.0)
    cheap.record(train_tokens=5_000_000)
    with pytest.raises(BudgetExceededError):
        dear.record(train_tokens=5_000_000)


def test_usage_is_still_recorded_when_the_ceiling_trips() -> None:
    """The run is over, but the accounting must remain accurate for reporting."""
    t = tracker(max_tokens=100)
    with pytest.raises(BudgetExceededError):
        t.record(train_tokens=150)
    assert t.usage.train_tokens == 150
    assert t.snapshot()["remaining"]["tokens"] == 0.0


def test_warning_fires_once_per_limit_at_the_threshold() -> None:
    seen: list[str] = []
    t = tracker(max_tokens=100, warn_at=0.8, on_warn=lambda name, used, limit: seen.append(name))
    t.record(train_tokens=50)
    assert seen == []
    t.record(train_tokens=35)   # 85% -> warn
    t.record(train_tokens=10)   # 95% -> already warned
    assert seen == ["tokens"]


def test_preflight_rejects_a_job_that_cannot_fit_before_it_starts() -> None:
    t = tracker(max_usd=0.01)
    with pytest.raises(BudgetExceededError):
        t.preflight(train_tokens=estimate_sft_tokens(num_examples=10_000, avg_tokens=1024, epochs=3))
    assert t.usage.train_tokens == 0  # preflight must not consume budget


def test_preflight_accepts_a_job_that_fits() -> None:
    t = tracker(max_usd=100.0)
    t.preflight(train_tokens=1_000_000, steps=10)


def test_would_exceed_names_the_binding_constraint() -> None:
    t = tracker(max_steps=2, max_tokens=10**9)
    assert t.would_exceed(steps=1) is None
    assert t.would_exceed(steps=3) == "steps"


def test_invalid_limits_are_rejected() -> None:
    for kwargs in ({"max_usd": 0}, {"max_tokens": -5}, {"warn_at": 0}, {"warn_at": 1.5}):
        with pytest.raises(ValueError):
            tracker(**kwargs)


def test_negative_usage_is_rejected() -> None:
    with pytest.raises(ValueError):
        tracker().record(train_tokens=-1)


def test_concurrent_recording_loses_no_tokens() -> None:
    t = tracker()
    def worker() -> None:
        for _ in range(200):
            t.record(train_tokens=1, steps=1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert t.usage.train_tokens == 1600
    assert t.usage.steps == 1600


def test_snapshot_is_json_shaped_for_job_records() -> None:
    t = tracker(max_usd=10.0)
    t.record(train_tokens=1000, sample_tokens=100, steps=1)
    snap = t.snapshot()
    assert snap["usage"]["total_tokens"] == 1100
    assert 0.0 <= snap["fraction_used"] <= 1.0
    assert snap["limits"]["max_usd"] == 10.0
