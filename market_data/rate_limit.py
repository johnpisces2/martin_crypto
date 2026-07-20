"""Process-wide, provider-aware API request throttling."""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime


@dataclass(frozen=True)
class RateLimitPolicy:
    """Weighted request budget for one API provider."""

    limit: float
    window_seconds: float
    min_interval: float = 0.0
    max_backoff: float = 60.0

    def __post_init__(self):
        if not math.isfinite(self.limit) or self.limit <= 0:
            raise ValueError("rate-limit budget must be greater than zero")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("rate-limit window must be greater than zero")
        if not math.isfinite(self.min_interval) or self.min_interval < 0:
            raise ValueError("rate-limit minimum interval cannot be negative")
        if not math.isfinite(self.max_backoff) or self.max_backoff <= 0:
            raise ValueError("rate-limit maximum backoff must be greater than zero")


# Limits are deliberately below provider ceilings. Binance values are request
# weight rather than raw call counts; endpoint callers supply their own weight.
DEFAULT_POLICIES = {
    "alpaca": RateLimitPolicy(180, 60.0, min_interval=0.05, max_backoff=65.0),
    "binance": RateLimitPolicy(1_000, 60.0, min_interval=0.01, max_backoff=120.0),
    "coingecko": RateLimitPolicy(20, 60.0, min_interval=1.0, max_backoff=65.0),
}
DEFAULT_POLICY = RateLimitPolicy(120, 60.0, min_interval=0.1, max_backoff=60.0)


def _provider_name(value) -> str:
    name = str(value or "default").strip().lower()
    return name or "default"


def _environment_token(provider: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in provider.upper())


def _environment_float(environ, name: str, default: float, *, allow_zero=False) -> float:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def policy_from_environment(provider: str, base: RateLimitPolicy, environ=None):
    """Apply ``MARTIN_RATE_LIMIT_<PROVIDER>_*`` overrides to a policy."""
    env = os.environ if environ is None else environ
    prefix = f"MARTIN_RATE_LIMIT_{_environment_token(provider)}"
    return RateLimitPolicy(
        limit=_environment_float(env, f"{prefix}_LIMIT", base.limit),
        window_seconds=_environment_float(
            env, f"{prefix}_WINDOW", base.window_seconds
        ),
        min_interval=_environment_float(
            env, f"{prefix}_MIN_INTERVAL", base.min_interval, allow_zero=True
        ),
        max_backoff=_environment_float(
            env, f"{prefix}_MAX_BACKOFF", base.max_backoff
        ),
    )


@dataclass
class _ProviderState:
    policy: RateLimitPolicy
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: deque = field(default_factory=deque)
    used_weight: float = 0.0
    next_request_at: float = 0.0
    blocked_until: float = 0.0


class RateLimiterRegistry:
    """One registry shared by every worker, with isolated provider budgets."""

    def __init__(
        self,
        policies=None,
        *,
        clock=None,
        wall_clock=None,
        sleeper=None,
        environ=None,
    ):
        self._policies = dict(DEFAULT_POLICIES)
        if policies:
            self._policies.update(
                {_provider_name(name): policy for name, policy in policies.items()}
            )
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._sleeper = sleeper or time.sleep
        self._environ = os.environ if environ is None else environ
        self._states = {}
        self._states_lock = threading.Lock()

    def _state(self, provider) -> _ProviderState:
        name = _provider_name(provider)
        with self._states_lock:
            state = self._states.get(name)
            if state is None:
                base = self._policies.get(name, DEFAULT_POLICY)
                state = _ProviderState(
                    policy=policy_from_environment(name, base, self._environ)
                )
                self._states[name] = state
            return state

    def policy(self, provider) -> RateLimitPolicy:
        return self._state(provider).policy

    @staticmethod
    def _prune(state: _ProviderState, now: float):
        cutoff = now - state.policy.window_seconds
        while state.events and state.events[0][0] <= cutoff:
            _, expired_weight = state.events.popleft()
            state.used_weight = max(0.0, state.used_weight - expired_weight)

    def acquire(self, provider, weight: float = 1.0):
        """Wait until a weighted request is allowed, then reserve its budget."""
        request_weight = float(weight)
        if not math.isfinite(request_weight) or request_weight <= 0:
            raise ValueError("request weight must be greater than zero")
        state = self._state(provider)
        if request_weight > state.policy.limit:
            raise ValueError(
                f"request weight {request_weight:g} exceeds {_provider_name(provider)} "
                f"budget {state.policy.limit:g}"
            )

        while True:
            with state.lock:
                now = self._clock()
                self._prune(state, now)
                ready_at = max(state.blocked_until, state.next_request_at)
                excess = state.used_weight + request_weight - state.policy.limit
                if excess > 1e-12:
                    released = 0.0
                    for timestamp, event_weight in state.events:
                        released += event_weight
                        if released + 1e-12 >= excess:
                            ready_at = max(
                                ready_at,
                                timestamp + state.policy.window_seconds,
                            )
                            break
                delay = ready_at - now
                if delay <= 1e-9:
                    state.events.append((now, request_weight))
                    state.used_weight += request_weight
                    state.next_request_at = (
                        now + state.policy.min_interval * request_weight
                    )
                    return
            self._sleeper(max(delay, 0.001))

    def defer(self, provider, seconds: float) -> float:
        """Pause all current and future workers for one provider."""
        state = self._state(provider)
        delay = max(0.0, float(seconds))
        with state.lock:
            state.blocked_until = max(
                state.blocked_until, self._clock() + delay
            )
        return delay

    @staticmethod
    def _header(headers, name):
        if not headers:
            return None
        value = headers.get(name)
        if value is not None:
            return value
        lower_name = name.lower()
        for key, item in headers.items():
            if str(key).lower() == lower_name:
                return item
        return None

    def _retry_after(self, response) -> float | None:
        headers = getattr(response, "headers", {}) or {}
        value = self._header(headers, "Retry-After")
        if value is not None:
            try:
                parsed = float(value)
                if math.isfinite(parsed) and parsed >= 0:
                    return parsed
            except (TypeError, ValueError):
                try:
                    parsed_date = parsedate_to_datetime(str(value))
                    return max(0.0, parsed_date.timestamp() - self._wall_clock())
                except (TypeError, ValueError, OverflowError):
                    pass

        reset_value = self._header(headers, "X-RateLimit-Reset")
        if reset_value is not None:
            try:
                reset = float(reset_value)
                if math.isfinite(reset):
                    if reset > 10_000_000:
                        return max(0.0, reset - self._wall_clock() + 0.1)
                    return max(0.0, reset)
            except (TypeError, ValueError):
                pass
        return None

    def retry_delay(self, provider, response=None, attempt: int = 0) -> float:
        policy = self.policy(provider)
        header_delay = self._retry_after(response) if response is not None else None
        if header_delay is None:
            # A headerless 429 is safer to retry in the provider's next full
            # accounting window instead of exhausting retries in this one.
            delay = min(
                max(2 ** max(0, int(attempt)), policy.window_seconds),
                policy.max_backoff,
            )
        else:
            delay = min(max(header_delay, 0.25), policy.max_backoff)
        return float(delay)

    def defer_from_response(self, provider, response=None, attempt: int = 0) -> float:
        delay = self.retry_delay(provider, response=response, attempt=attempt)
        return self.defer(provider, delay)

    def observe_response(self, provider, response):
        """Honor a provider reset header when its remaining budget reaches zero."""
        headers = getattr(response, "headers", {}) or {}
        remaining = self._header(headers, "X-RateLimit-Remaining")
        try:
            exhausted = float(remaining) <= 0
        except (TypeError, ValueError):
            exhausted = False
        if exhausted:
            reset_delay = self._retry_after(response)
            if reset_delay is not None and reset_delay > 0:
                self.defer(provider, min(reset_delay, self.policy(provider).max_backoff))


GLOBAL_RATE_LIMITER = RateLimiterRegistry()


def acquire(provider, weight: float = 1.0):
    return GLOBAL_RATE_LIMITER.acquire(provider, weight)


def defer(provider, seconds: float) -> float:
    return GLOBAL_RATE_LIMITER.defer(provider, seconds)


def retry_delay(provider, response=None, attempt: int = 0) -> float:
    return GLOBAL_RATE_LIMITER.retry_delay(provider, response, attempt)


def defer_from_response(provider, response=None, attempt: int = 0) -> float:
    return GLOBAL_RATE_LIMITER.defer_from_response(provider, response, attempt)


def observe_response(provider, response):
    return GLOBAL_RATE_LIMITER.observe_response(provider, response)
