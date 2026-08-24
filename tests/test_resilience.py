"""Edge cases for retries, circuit breaking and rate limiting.

Every test drives an injected clock/sleep, so the suite exercises real timing
logic without sleeping for real.
"""

from __future__ import annotations

import random
import threading

import pytest

from tinker_finetune.models.schemas import LoRAConfig, OptimConfig
from tinker_finetune.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ResilientBackend,
    RetryPolicy,
    TokenBucket,
    is_retryable,
    retry_after_seconds,
)
from tinker_finetune.tinker_client.fake import FakeTinkerBackend


class Clock:
    """Manual clock whose ``sleep`` advances time - the pairing acquire() needs."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class HttpError(Exception):
    def __init__(self, status_code: int, retry_after: float | None = None) -> None:
        self.status_code = status_code
        if retry_after is not None:
            self.retry_after = retry_after
        super().__init__(f"HTTP {status_code}")


def no_retry_policy(**kwargs) -> RetryPolicy:
    kwargs.setdefault("jitter", False)
    kwargs.setdefault("sleep", lambda _s: None)
    return RetryPolicy(**kwargs)


# --- retry classification -------------------------------------------------
@pytest.mark.parametrize("status,expected", [(429, True), (503, True), (500, True), (408, True),
                                             (400, False), (401, False), (404, False), (422, False)])
def test_status_codes_are_classified(status: int, expected: bool) -> None:
    assert is_retryable(HttpError(status)) is expected


def test_transport_errors_are_retryable_but_bugs_are_not() -> None:
    assert is_retryable(TimeoutError("timed out"))
    assert is_retryable(ConnectionError("connection reset by peer"))
    assert is_retryable(RuntimeError("Rate limit exceeded"))
    assert not is_retryable(TypeError("batch must be a list"))
    assert not is_retryable(ValueError("input_tokens length mismatch"))


def test_retry_after_hint_is_read_from_attribute_and_headers() -> None:
    assert retry_after_seconds(HttpError(429, retry_after=7)) == 7.0

    class WithHeaders(Exception):
        response = type("R", (), {"headers": {"Retry-After": "3"}})()

    assert retry_after_seconds(WithHeaders()) == 3.0
    assert retry_after_seconds(ValueError("nope")) is None


# --- retry behaviour ------------------------------------------------------
def test_succeeds_after_transient_failures() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise HttpError(503)
        return "ok"

    assert no_retry_policy(max_attempts=5).call(flaky) == "ok"
    assert calls["n"] == 3


def test_non_retryable_error_fails_on_first_attempt() -> None:
    calls = {"n": 0}

    def broken() -> None:
        calls["n"] += 1
        raise HttpError(400)

    with pytest.raises(HttpError):
        no_retry_policy(max_attempts=5).call(broken)
    assert calls["n"] == 1


def test_last_error_is_raised_not_swallowed() -> None:
    with pytest.raises(HttpError) as exc:
        no_retry_policy(max_attempts=2).call(lambda: (_ for _ in ()).throw(HttpError(503)))
    assert exc.value.status_code == 503


def test_max_attempts_of_one_means_no_retry() -> None:
    calls = {"n": 0}

    def always_fail() -> None:
        calls["n"] += 1
        raise HttpError(503)

    with pytest.raises(HttpError):
        no_retry_policy(max_attempts=1).call(always_fail)
    assert calls["n"] == 1


def test_backoff_is_exponential_and_capped() -> None:
    policy = no_retry_policy(max_attempts=8, base_delay=1.0, multiplier=2.0, max_delay=8.0)
    assert list(policy.delays()) == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0]


def test_full_jitter_stays_within_the_cap_and_varies() -> None:
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, rng=random.Random(0), sleep=lambda _s: None)
    samples = [policy.delay_for(3) for _ in range(50)]
    assert all(0.0 <= d <= 4.0 for d in samples)
    assert len(set(samples)) > 1  # jitter actually jitters


def test_retry_after_hint_overrides_a_shorter_backoff() -> None:
    policy = no_retry_policy(base_delay=0.1, max_delay=60.0)
    assert policy.delay_for(1, hint=30.0) == 30.0
    # ...but never beyond max_delay, so a hostile header cannot stall a run.
    assert no_retry_policy(base_delay=0.1, max_delay=5.0).delay_for(1, hint=3600.0) == 5.0


def test_keyboard_interrupt_is_never_retried() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        no_retry_policy(max_attempts=5, should_retry=lambda _e: True).call(interrupted)


def test_invalid_policy_configuration_is_rejected() -> None:
    for kwargs in ({"max_attempts": 0}, {"base_delay": -1}, {"multiplier": 0.5}):
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)


# --- circuit breaker ------------------------------------------------------
def test_breaker_opens_after_threshold_and_fails_fast() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0, clock=clock)
    for _ in range(3):
        with pytest.raises(HttpError):
            breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    assert breaker.state is CircuitState.open
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "never runs")


def test_breaker_half_opens_after_recovery_timeout_then_closes() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)
    with pytest.raises(HttpError):
        breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    clock.now += 10.0
    assert breaker.state is CircuitState.half_open
    assert breaker.call(lambda: "probe ok") == "probe ok"
    assert breaker.state is CircuitState.closed


def test_failed_probe_reopens_the_circuit_and_resets_the_timer() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0, clock=clock)
    with pytest.raises(HttpError):
        breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    clock.now += 10.0
    with pytest.raises(HttpError):
        breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    assert breaker.state is CircuitState.open
    clock.now += 9.0
    assert breaker.state is CircuitState.open  # timer restarted, not resumed


def test_half_open_admits_only_the_configured_number_of_probes() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0,
                             half_open_max_calls=1, success_threshold=2, clock=clock)
    with pytest.raises(HttpError):
        breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    clock.now += 1.0
    breaker.call(lambda: "probe")
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "second probe rejected")


def test_deterministic_client_errors_do_not_trip_the_breaker() -> None:
    """A 400 says the request is wrong, not that the service is down."""
    breaker = CircuitBreaker(failure_threshold=2, clock=Clock())
    for _ in range(5):
        with pytest.raises(HttpError):
            breaker.call(lambda: (_ for _ in ()).throw(HttpError(400)))
    assert breaker.state is CircuitState.closed


def test_success_resets_the_consecutive_failure_count() -> None:
    breaker = CircuitBreaker(failure_threshold=3, clock=Clock())
    for _ in range(2):
        with pytest.raises(HttpError):
            breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    breaker.call(lambda: "ok")
    for _ in range(2):
        with pytest.raises(HttpError):
            breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
    assert breaker.state is CircuitState.closed


def test_breaker_is_thread_safe_under_concurrent_failures() -> None:
    breaker = CircuitBreaker(failure_threshold=50, clock=Clock())
    def hammer() -> None:
        for _ in range(20):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(HttpError(503)))
            except (HttpError, CircuitOpenError):
                pass
    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert breaker.state is CircuitState.open


# --- token bucket ---------------------------------------------------------
def test_bucket_allows_a_burst_then_throttles() -> None:
    clock = Clock()
    bucket = TokenBucket(rate=2.0, capacity=5.0, clock=clock, sleep=clock.sleep)
    assert all(bucket.try_acquire() for _ in range(5))
    assert not bucket.try_acquire()


def test_bucket_refills_at_the_configured_rate() -> None:
    clock = Clock()
    bucket = TokenBucket(rate=2.0, capacity=5.0, clock=clock, sleep=clock.sleep)
    bucket.try_acquire(5)
    clock.now += 1.0
    assert bucket.tokens == pytest.approx(2.0)
    clock.now += 100.0
    assert bucket.tokens == 5.0  # never over capacity


def test_acquire_waits_exactly_as_long_as_needed() -> None:
    clock = Clock()
    bucket = TokenBucket(rate=4.0, capacity=1.0, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    waited = bucket.acquire()
    assert waited == pytest.approx(0.25)
    assert clock.now == pytest.approx(0.25)


def test_acquire_honours_a_timeout_instead_of_blocking_forever() -> None:
    clock = Clock()
    bucket = TokenBucket(rate=1.0, capacity=1.0, clock=clock, sleep=clock.sleep)
    bucket.acquire()
    with pytest.raises(TimeoutError):
        bucket.acquire(timeout=0.5)


def test_request_larger_than_capacity_is_rejected_not_deadlocked() -> None:
    bucket = TokenBucket(rate=1.0, capacity=2.0, clock=Clock())
    with pytest.raises(ValueError):
        bucket.try_acquire(3)


def test_frozen_clock_with_fake_sleep_raises_instead_of_spinning() -> None:
    bucket = TokenBucket(rate=1.0, capacity=1.0, clock=lambda: 0.0, sleep=lambda _s: None)
    bucket.acquire()
    with pytest.raises(RuntimeError):
        bucket.acquire()


def test_invalid_bucket_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=0)
    with pytest.raises(ValueError):
        TokenBucket(rate=1, capacity=0)


# --- ResilientBackend -----------------------------------------------------
class FlakyBackend(FakeTinkerBackend):
    """Fake backend that fails the first ``fail_times`` calls of each method."""

    def __init__(self, fail_times: int = 0, status: int = 503) -> None:
        super().__init__("Qwen/Qwen3-8B", LoRAConfig())
        self.fail_times = fail_times
        self.status = status
        self.seen: dict[str, int] = {}

    def _maybe_fail(self, name: str) -> None:
        self.seen[name] = self.seen.get(name, 0) + 1
        if self.seen[name] <= self.fail_times:
            raise HttpError(self.status)

    def forward_backward(self, batch, loss_fn="cross_entropy"):
        self._maybe_fail("forward_backward")
        return super().forward_backward(batch, loss_fn)

    def optim_step(self, optim, lr):
        self._maybe_fail("optim_step")
        return super().optim_step(optim, lr)


def one_batch():
    from tinker_finetune.tinker_client.client import Datum

    return [Datum(input_tokens=[1, 2, 3], target_tokens=[2, 3, 4], weights=[0.0, 1.0, 1.0])]


def test_resilient_backend_retries_forward_backward() -> None:
    inner = FlakyBackend(fail_times=2)
    backend = ResilientBackend(inner, retry=no_retry_policy(max_attempts=4),
                               breaker=CircuitBreaker(failure_threshold=10, clock=Clock()))
    assert backend.forward_backward(one_batch()).num_tokens > 0
    assert inner.seen["forward_backward"] == 3
    assert backend.stats["retries"] == 2


def test_optimizer_steps_are_not_retried_by_default() -> None:
    """Replaying a partially applied weight update would corrupt training."""
    inner = FlakyBackend(fail_times=1)
    backend = ResilientBackend(inner, retry=no_retry_policy(max_attempts=4),
                               breaker=CircuitBreaker(failure_threshold=10, clock=Clock()))
    with pytest.raises(HttpError):
        backend.optim_step(OptimConfig(), 1e-4)
    assert inner.seen["optim_step"] == 1


def test_mutating_calls_can_opt_in_to_retries() -> None:
    inner = FlakyBackend(fail_times=1)
    backend = ResilientBackend(inner, retry=no_retry_policy(max_attempts=4), retry_mutating=True,
                               breaker=CircuitBreaker(failure_threshold=10, clock=Clock()))
    assert backend.optim_step(OptimConfig(), 1e-4) is not None
    assert inner.seen["optim_step"] == 2


def test_open_circuit_short_circuits_further_calls() -> None:
    clock = Clock()
    inner = FlakyBackend(fail_times=99)
    backend = ResilientBackend(
        inner,
        retry=no_retry_policy(max_attempts=2),
        breaker=CircuitBreaker(failure_threshold=2, recovery_timeout=60.0, clock=clock),
    )
    with pytest.raises(HttpError):
        backend.forward_backward(one_batch())
    before = inner.seen["forward_backward"]
    with pytest.raises(CircuitOpenError):
        backend.forward_backward(one_batch())
    assert inner.seen["forward_backward"] == before  # never reached the service
    assert backend.stats["rejected"] == 1


def test_rate_limiter_shapes_calls_through_the_wrapper() -> None:
    clock = Clock()
    backend = ResilientBackend(
        FlakyBackend(),
        retry=no_retry_policy(max_attempts=1),
        breaker=CircuitBreaker(clock=clock),
        limiter=TokenBucket(rate=1.0, capacity=1.0, clock=clock, sleep=clock.sleep),
    )
    backend.forward_backward(one_batch())
    backend.forward_backward(one_batch())
    assert clock.now == pytest.approx(1.0)


def test_wrapper_passes_through_unknown_attributes() -> None:
    inner = FlakyBackend()
    backend = ResilientBackend(inner, retry=no_retry_policy(max_attempts=1))
    assert backend.base_model == inner.base_model
