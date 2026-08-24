"""Edge cases for numerical guards, early stopping, and their SFT integration."""

from __future__ import annotations

import pytest

from tinker_finetune.budget import BudgetExceededError, BudgetTracker
from tinker_finetune.data.tokenization import build_tokenizer
from tinker_finetune.models.schemas import (
    ChatExample,
    LoRAConfig,
    Message,
    OptimConfig,
    Role,
    SFTConfig,
)
from tinker_finetune.tinker_client.fake import FakeTinkerBackend
from tinker_finetune.training.guards import (
    EarlyStopping,
    GuardTripped,
    NumericalGuard,
    TrainingGuards,
)
from tinker_finetune.training.sft_trainer import SFTTrainer


def examples(n: int = 8) -> list[ChatExample]:
    return [
        ChatExample(messages=[
            Message(role=Role.user, content=f"q{i}"),
            Message(role=Role.assistant, content=f"a{i}"),
        ])
        for i in range(n)
    ]


# --- numerical guard ------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_loss_stops_immediately(bad: float) -> None:
    decision = NumericalGuard().observe(bad)
    assert decision.stop and decision.code == "non_finite_loss"


def test_non_finite_gradient_stops_even_with_a_healthy_loss() -> None:
    decision = NumericalGuard().observe(0.5, float("nan"))
    assert decision.code == "non_finite_grad"


def test_gradient_explosion_threshold() -> None:
    guard = NumericalGuard(max_grad_norm=10.0)
    assert not guard.observe(1.0, 9.9)
    assert guard.observe(1.0, 10.1).code == "grad_explosion"


def test_gradient_check_can_be_disabled() -> None:
    assert not NumericalGuard(max_grad_norm=None).observe(1.0, 1e9)


def test_healthy_decreasing_loss_never_trips() -> None:
    guard = NumericalGuard()
    assert not any(guard.observe(1.0 / (i + 1), 0.5) for i in range(200))


def test_noisy_but_stable_loss_does_not_trip() -> None:
    guard = NumericalGuard()
    noisy = [1.0, 1.2, 0.9, 1.1, 0.95, 1.05] * 20
    assert not any(guard.observe(v, 0.4) for v in noisy)


def test_single_spike_is_tolerated_but_a_sustained_one_stops() -> None:
    guard = NumericalGuard(window=8, spike_factor=5.0, spike_patience=3)
    for _ in range(8):
        guard.observe(1.0)
    assert not guard.observe(50.0)   # one bad batch happens
    assert not guard.observe(50.0)
    assert guard.observe(50.0).code == "loss_spike"


def test_spike_counter_resets_after_recovery() -> None:
    guard = NumericalGuard(window=8, spike_factor=5.0, spike_patience=3)
    for _ in range(8):
        guard.observe(1.0)
    guard.observe(50.0)
    guard.observe(1.0)  # recovered
    assert not guard.observe(50.0)


def test_divergence_is_detected_from_the_trend_not_a_single_step() -> None:
    guard = NumericalGuard(window=5, min_steps_before_divergence=10, divergence_factor=1.5)
    decisions = [guard.observe(1.0 + 0.1 * i) for i in range(30)]
    assert any(d.code == "divergence" for d in decisions)


def test_exactly_zero_loss_is_treated_as_an_empty_loss_mask() -> None:
    guard = NumericalGuard(zero_loss_patience=3)
    assert not guard.observe(0.0)
    assert not guard.observe(0.0)
    assert guard.observe(0.0).code == "zero_loss"


def test_a_single_zero_loss_step_is_allowed() -> None:
    guard = NumericalGuard(zero_loss_patience=3)
    guard.observe(0.0)
    guard.observe(0.01)
    assert not guard.observe(0.0)


def test_missing_loss_is_ignored_rather_than_treated_as_zero() -> None:
    guard = NumericalGuard(zero_loss_patience=2)
    assert not guard.observe(None)
    assert not guard.observe(None)


# --- early stopping -------------------------------------------------------
def test_early_stopping_fires_after_patience_is_exhausted() -> None:
    es = EarlyStopping(patience=2)
    assert not es.observe(1.0, step=1)
    assert not es.observe(1.1, step=2)
    assert not es.observe(1.2, step=3)
    assert es.observe(1.3, step=4).code == "early_stop"


def test_min_delta_rejects_noise_sized_improvements() -> None:
    es = EarlyStopping(patience=1, min_delta=0.01)
    es.observe(1.0)
    assert not es.observe(0.9999)  # improvement below min_delta: patience burns
    assert es.observe(0.9998).stop


def test_best_value_and_step_are_tracked_for_checkpoint_selection() -> None:
    es = EarlyStopping(patience=5)
    for value, step in [(1.0, 10), (0.5, 20), (0.7, 30)]:
        es.observe(value, step=step)
    assert es.best == 0.5 and es.best_step == 20


def test_max_mode_tracks_increasing_metrics() -> None:
    es = EarlyStopping(patience=1, mode="max")
    es.observe(0.5)
    es.observe(0.9)
    assert es.best == 0.9
    assert not es.observe(0.8)
    assert es.observe(0.7).stop


def test_zero_patience_stops_on_the_first_non_improvement() -> None:
    es = EarlyStopping(patience=0)
    es.observe(1.0)
    assert es.observe(1.0).stop


def test_non_finite_eval_metric_stops() -> None:
    assert EarlyStopping().observe(float("nan")).code == "non_finite_eval"


def test_invalid_early_stopping_configuration_is_rejected() -> None:
    for kwargs in ({"mode": "sideways"}, {"patience": -1}, {"min_delta": -0.1}):
        with pytest.raises(ValueError):
            EarlyStopping(**kwargs)


# --- composite ------------------------------------------------------------
def test_composite_records_the_trip_and_can_raise() -> None:
    guards = TrainingGuards()
    guards.observe_train(loss=float("nan"))
    assert guards.tripped.code == "non_finite_loss"
    with pytest.raises(GuardTripped):
        guards.raise_if_tripped()


def test_composite_with_guards_disabled_is_a_no_op() -> None:
    guards = TrainingGuards(numerical=None)
    assert not guards.observe_train(loss=float("nan"))
    assert not guards.observe_eval(999.0)  # no early stopping configured
    guards.raise_if_tripped()


# --- trainer integration --------------------------------------------------
def _trainer(**kwargs) -> SFTTrainer:
    config = SFTConfig(
        base_model="Qwen/Qwen3-8B",
        train_path="unused.jsonl",
        epochs=1,
        batch_size=2,
        max_seq_len=128,
        optim=OptimConfig(),
    )
    backend = FakeTinkerBackend(config.base_model, LoRAConfig())
    return SFTTrainer(backend, build_tokenizer("byte", prefer_hf=False), config, **kwargs)


def test_trainer_halts_when_a_guard_trips() -> None:
    class AlwaysTrip(NumericalGuard):
        def observe(self, loss, grad_norm=None):  # type: ignore[override]
            from tinker_finetune.training.guards import GuardDecision

            return GuardDecision.halt("test_trip", "stop now")

    guards = TrainingGuards(numerical=AlwaysTrip())
    history = _trainer(guards=guards).train(examples(8))
    assert len(history) == 1
    assert guards.tripped.code == "test_trip"


def test_trainer_runs_to_completion_with_healthy_guards() -> None:
    guards = TrainingGuards()
    history = _trainer(guards=guards).train(examples(8))
    assert len(history) == 4  # 8 examples / batch_size 2
    assert guards.tripped is None


def test_trainer_enforces_the_budget_mid_run() -> None:
    budget = BudgetTracker(model="Qwen/Qwen3-8B", max_steps=2)
    with pytest.raises(BudgetExceededError):
        _trainer(budget=budget).train(examples(8))
    assert budget.usage.steps == 3  # the step that crossed the ceiling is counted


def test_trainer_records_token_usage_when_it_completes() -> None:
    budget = BudgetTracker(model="Qwen/Qwen3-8B")
    _trainer(budget=budget).train(examples(8))
    assert budget.usage.steps == 4
    assert budget.usage.train_tokens > 0
    assert budget.usage.cost_usd > 0
