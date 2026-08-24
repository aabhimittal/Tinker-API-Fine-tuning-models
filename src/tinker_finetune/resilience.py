"""Fault-tolerance primitives for talking to a remote training service.

A fine-tuning run is a long-lived conversation with a service that will,
occasionally, rate-limit you, drop a connection, or return a 503 mid-epoch.
Losing an hour of gradient accumulation to a transient error is the single
most expensive failure mode in production fine-tuning, so every backend call
goes through three composable layers:

``TokenBucket``      client-side rate limiting, so we shape our own traffic
                     instead of discovering the server's limit the hard way.
``CircuitBreaker``   stop hammering a service that is already down; fail fast
                     for ``recovery_timeout`` seconds, then probe.
``RetryPolicy``      exponential backoff with full jitter, honouring
                     ``Retry-After`` when the error carries one.

All three take injectable ``clock``/``sleep`` callables so tests exercise the
timing logic deterministically, with no real sleeping.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from tinker_finetune.logging_utils import get_logger
from tinker_finetune.models.schemas import OptimConfig
from tinker_finetune.tinker_client.client import (
    Datum,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
    TinkerBackend,
)

log = get_logger(__name__)

T = TypeVar("T")

__all__ = [
    "RetryPolicy",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "TokenBucket",
    "ResilientBackend",
    "is_retryable",
    "retry_after_seconds",
]

# Status codes worth retrying: rate limit, and the transient 5xx family.
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRY_MARKERS = (
    "timeout", "timed out", "connection reset", "connection aborted",
    "temporarily unavailable", "too many requests", "service unavailable",
    "broken pipe", "econnreset", "rate limit",
)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    """Heuristic: transport hiccups and 429/5xx are retryable, 4xx is not.

    A 400 (malformed batch) or 401 (bad key) will fail identically forever;
    retrying those just burns the budget and delays the real error.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = _status_of(exc)
    if status is not None:
        return status in _RETRY_STATUS
    text = str(exc).casefold()
    return any(marker in text for marker in _RETRY_MARKERS)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a server-provided ``Retry-After`` hint, if present."""
    for attr in ("retry_after", "retry_after_seconds"):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if isinstance(headers, dict):
        raw = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
    return None


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter.

    Full jitter (``U(0, capped_delay)``) rather than fixed backoff is what
    stops a fleet of workers from synchronising into a thundering herd after a
    shared outage.
    """

    max_attempts: int = 5
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 30.0
    jitter: bool = True
    should_retry: Callable[[BaseException], bool] = is_retryable
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be non-negative.")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1.")

    def delay_for(self, attempt: int, *, hint: float | None = None) -> float:
        """Delay before retry ``attempt`` (1-based). ``hint`` wins if larger."""
        capped = min(self.base_delay * (self.multiplier ** (attempt - 1)), self.max_delay)
        delay = self.rng.uniform(0.0, capped) if self.jitter else capped
        if hint is not None:
            delay = max(delay, min(hint, self.max_delay))
        return delay

    def delays(self) -> Iterator[float]:
        for attempt in range(1, self.max_attempts):
            yield self.delay_for(attempt)

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        last: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if isinstance(exc, (KeyboardInterrupt, SystemExit)) or not self.should_retry(exc):
                    raise
                last = exc
                if attempt == self.max_attempts:
                    break
                delay = self.delay_for(attempt, hint=retry_after_seconds(exc))
                log.warning(
                    "Attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt, self.max_attempts, exc.__class__.__name__, delay,
                )
                self.sleep(delay)
        assert last is not None
        raise last


class CircuitState(str, Enum):
    closed = "closed"      # normal operation
    open = "open"          # failing fast
    half_open = "half_open"  # probing recovery


class CircuitOpenError(RuntimeError):
    """Raised instead of calling through an open circuit."""

    def __init__(self, retry_in: float) -> None:
        self.retry_in = retry_in
        super().__init__(f"Circuit is open; retry in {retry_in:.1f}s.")


class CircuitBreaker:
    """Trip after ``failure_threshold`` consecutive failures.

    While open, calls fail immediately (no queueing behind a dead service).
    After ``recovery_timeout`` a limited number of probe calls are admitted;
    ``success_threshold`` consecutive successes close it again, and a single
    failure re-opens it with the timer reset.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 1,
        half_open_max_calls: int = 1,
        should_count: Callable[[BaseException], bool] = is_retryable,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1 or success_threshold < 1 or half_open_max_calls < 1:
            raise ValueError("thresholds must be >= 1.")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls
        self.should_count = should_count
        self._clock = clock
        self._lock = threading.RLock()
        self._state = CircuitState.closed
        self._failures = 0
        self._successes = 0
        self._probes = 0
        self._opened_at = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if self._state is CircuitState.open and self._clock() - self._opened_at >= self.recovery_timeout:
            self._state = CircuitState.half_open
            self._probes = 0
            self._successes = 0

    def _open(self) -> None:
        self._state = CircuitState.open
        self._opened_at = self._clock()
        self._probes = 0
        self._successes = 0

    def _before_call(self) -> None:
        with self._lock:
            self._maybe_half_open()
            if self._state is CircuitState.open:
                raise CircuitOpenError(max(0.0, self.recovery_timeout - (self._clock() - self._opened_at)))
            if self._state is CircuitState.half_open:
                if self._probes >= self.half_open_max_calls:
                    raise CircuitOpenError(self.recovery_timeout)
                self._probes += 1

    def record_success(self) -> None:
        with self._lock:
            if self._state is CircuitState.half_open:
                self._successes += 1
                if self._successes >= self.success_threshold:
                    self._state = CircuitState.closed
                    self._failures = 0
                    self._successes = 0
                    self._probes = 0
            else:
                self._failures = 0

    def record_failure(self, exc: BaseException | None = None) -> None:
        with self._lock:
            if exc is not None and not self.should_count(exc):
                return  # a deterministic 400 says nothing about service health
            if self._state is CircuitState.half_open:
                self._failures += 1
                self._open()
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open()

    def reset(self) -> None:
        with self._lock:
            self._state = CircuitState.closed
            self._failures = self._successes = self._probes = 0

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        self._before_call()
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
            self.record_failure(exc)
            raise
        self.record_success()
        return result


class TokenBucket:
    """Classic token bucket: sustained ``rate``/s with bursts up to ``capacity``."""

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0.")
        self.rate = rate
        self.capacity = float(capacity if capacity is not None else rate)
        if self.capacity <= 0:
            raise ValueError("capacity must be > 0.")
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = now

    @property
    def tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def try_acquire(self, amount: float = 1.0) -> bool:
        if amount <= 0:
            raise ValueError("amount must be > 0.")
        if amount > self.capacity:
            raise ValueError(f"amount {amount} exceeds bucket capacity {self.capacity}.")
        with self._lock:
            self._refill()
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    def acquire(self, amount: float = 1.0, *, timeout: float | None = None) -> float:
        """Block until ``amount`` tokens are available; returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if amount > self.capacity:
                    raise ValueError(f"amount {amount} exceeds bucket capacity {self.capacity}.")
                if self._tokens >= amount:
                    self._tokens -= amount
                    return waited
                needed = (amount - self._tokens) / self.rate
            if timeout is not None and waited + needed > timeout:
                raise TimeoutError(f"Rate limit wait would exceed timeout ({timeout}s).")
            before = self._clock()
            self._sleep(needed)
            waited += needed
            if needed > 0 and self._clock() <= before:
                # A fake sleep paired with a frozen clock would spin forever.
                raise RuntimeError(
                    "TokenBucket clock did not advance across sleep(); clock and sleep must agree."
                )


class ResilientBackend:
    """Wraps any :class:`TinkerBackend` with rate limiting, breaker and retries.

    Ordering matters: the limiter shapes traffic *before* a call is attempted,
    the breaker decides whether attempting is worthwhile at all, and retries
    sit innermost so each attempt re-checks both. Weight-mutating calls
    (``optim_step``, ``load_state``) are **not** retried by default: replaying
    an optimizer step that partially applied would silently corrupt training.
    """

    def __init__(
        self,
        inner: TinkerBackend,
        *,
        retry: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        limiter: TokenBucket | None = None,
        retry_mutating: bool = False,
    ) -> None:
        self.inner = inner
        self.retry = retry or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        self.limiter = limiter
        self.retry_mutating = retry_mutating
        self.stats: dict[str, int] = {"calls": 0, "retries": 0, "rejected": 0}

    # -- plumbing --------------------------------------------------------
    def _guarded(self, fn: Callable[..., T], *args: Any, retriable: bool = True, **kwargs: Any) -> T:
        if self.limiter is not None:
            self.limiter.acquire()
        self.stats["calls"] += 1
        attempts = 0

        def _once() -> T:
            nonlocal attempts
            attempts += 1
            return self.breaker.call(fn, *args, **kwargs)

        try:
            if retriable:
                return self.retry.call(_once)
            return _once()
        except CircuitOpenError:
            self.stats["rejected"] += 1
            raise
        finally:
            self.stats["retries"] += max(0, attempts - 1)

    # -- TinkerBackend protocol -----------------------------------------
    def forward_backward(
        self, batch: list[Datum], loss_fn: str = "cross_entropy"
    ) -> ForwardBackwardResult:
        return self._guarded(self.inner.forward_backward, batch, loss_fn)

    def logprobs(self, batch: list[Datum]) -> list[float]:
        return self._guarded(self.inner.logprobs, batch)

    def optim_step(self, optim: OptimConfig, lr: float) -> OptimStepResult:
        return self._guarded(self.inner.optim_step, optim, lr, retriable=self.retry_mutating)

    def sample(self, prompt_tokens: list[int], **kwargs: Any) -> SampleResult:
        return self._guarded(self.inner.sample, prompt_tokens, **kwargs)

    def save_weights_for_sampler(self, name: str) -> str:
        return self._guarded(self.inner.save_weights_for_sampler, name)

    def save_state(self, path: str) -> str:
        return self._guarded(self.inner.save_state, path)

    def load_state(self, path: str) -> None:
        return self._guarded(self.inner.load_state, path, retriable=self.retry_mutating)

    # Pass through anything the protocol does not name (e.g. base_model, lora).
    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


def wrap_resilient(backend: TinkerBackend, **kwargs: Any) -> ResilientBackend:
    """Convenience factory mirroring ``build_backend`` call sites."""
    return ResilientBackend(backend, **kwargs)
