"""Tests for the ``RetryPolicy`` and ``_apply_retry_policy`` helper.

Concrete UX win: a 401 bad API key used to retry 3 times (wasting 7+
seconds) before surfacing. With a policy, non-retryable status codes
abort immediately.
"""

from __future__ import annotations

import asyncio
import random
import time

import pytest

from mmar_carl import OpenAIClientConfig, RetryPolicy
from mmar_carl.llm import _apply_retry_policy


class FakeAPIError(Exception):
    """Mock OpenAI-SDK-style exception with a .status_code attribute."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.status_code = status


class FakeHttpxStatusError(Exception):
    """Mock httpx-style exception with a .response.status_code attribute."""

    def __init__(self, status: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status}")
        self.response = type("Response", (), {"status_code": status})()


# ---------------------------------------------------------------------------
# RetryPolicy model
# ---------------------------------------------------------------------------


class TestRetryPolicyDefaults:
    def test_default_max_attempts(self) -> None:
        assert RetryPolicy().max_attempts == 3

    def test_default_retry_on_status_includes_transient(self) -> None:
        p = RetryPolicy()
        for code in (429, 500, 502, 503, 504):
            assert code in p.retry_on_status

    def test_default_excludes_auth_errors(self) -> None:
        """Critical: defaults must NOT retry 401/403/404."""
        p = RetryPolicy()
        for code in (401, 403, 404, 422):
            assert code not in p.retry_on_status

    def test_default_backoff_is_exponential(self) -> None:
        assert RetryPolicy().backoff == "exponential"

    def test_default_jitter_is_on(self) -> None:
        assert RetryPolicy().jitter is True


class TestRetryPolicyValidation:
    def test_max_attempts_min_is_one(self) -> None:
        with pytest.raises(Exception):
            RetryPolicy(max_attempts=0)

    def test_max_attempts_capped_at_twenty(self) -> None:
        with pytest.raises(Exception):
            RetryPolicy(max_attempts=21)

    def test_initial_delay_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            RetryPolicy(initial_delay_s=0)

    def test_max_delay_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            RetryPolicy(max_delay_s=0)


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------


class TestIsRetryable:
    def test_retryable_status_in_list(self) -> None:
        assert RetryPolicy().is_retryable(FakeAPIError(500)) is True

    def test_non_retryable_status_excluded(self) -> None:
        assert RetryPolicy().is_retryable(FakeAPIError(401)) is False

    def test_httpx_style_status_extracted_from_response(self) -> None:
        assert RetryPolicy().is_retryable(FakeHttpxStatusError(503)) is True
        assert RetryPolicy().is_retryable(FakeHttpxStatusError(401)) is False

    def test_no_status_info_retried_conservatively(self) -> None:
        """ConnectionError / TimeoutError have no .status_code → retry."""
        assert RetryPolicy().is_retryable(ConnectionError("blip")) is True
        assert RetryPolicy().is_retryable(asyncio.TimeoutError()) is True

    def test_custom_retry_on_status_list(self) -> None:
        """Users can override defaults — e.g. only retry 503."""
        p = RetryPolicy(retry_on_status=[503])
        assert p.is_retryable(FakeAPIError(503)) is True
        assert p.is_retryable(FakeAPIError(500)) is False
        assert p.is_retryable(FakeAPIError(429)) is False

    def test_invalid_status_value_retried_conservatively(self) -> None:
        """If status_code is non-numeric (rare), treat as transient."""
        class WeirdError(Exception):
            status_code = "not-a-number"

        assert RetryPolicy().is_retryable(WeirdError()) is True


# ---------------------------------------------------------------------------
# compute_delay
# ---------------------------------------------------------------------------


class TestComputeDelay:
    def test_exponential_no_jitter(self) -> None:
        p = RetryPolicy(
            initial_delay_s=2.0, max_delay_s=100.0,
            backoff="exponential", jitter=False,
        )
        assert p.compute_delay(0) == 2.0
        assert p.compute_delay(1) == 4.0
        assert p.compute_delay(2) == 8.0
        assert p.compute_delay(3) == 16.0

    def test_exponential_capped_at_max_delay(self) -> None:
        p = RetryPolicy(
            initial_delay_s=2.0, max_delay_s=10.0,
            backoff="exponential", jitter=False,
        )
        assert p.compute_delay(3) == 10.0  # would be 16, capped to 10
        assert p.compute_delay(10) == 10.0

    def test_constant_no_jitter(self) -> None:
        p = RetryPolicy(
            initial_delay_s=2.5, backoff="constant", jitter=False,
        )
        assert p.compute_delay(0) == 2.5
        assert p.compute_delay(5) == 2.5
        assert p.compute_delay(100) == 2.5

    def test_jitter_scales_delay_to_50_to_100_pct(self) -> None:
        """With jitter, delays land in [0.5*base, 1.0*base) range."""
        p = RetryPolicy(initial_delay_s=10.0, backoff="constant", jitter=True)
        rng = random.Random(42)
        # Probe many samples
        for _ in range(20):
            d = p.compute_delay(0, rng=rng)
            assert 5.0 <= d < 10.0

    def test_jitter_off_returns_exact_value(self) -> None:
        p = RetryPolicy(initial_delay_s=5.0, backoff="constant", jitter=False)
        rng = random.Random(0)
        assert p.compute_delay(0, rng=rng) == 5.0


# ---------------------------------------------------------------------------
# _apply_retry_policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApplyRetryPolicy:
    async def test_non_retryable_aborts_immediately(self) -> None:
        """401 should fail after exactly 1 attempt."""
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            raise FakeAPIError(401)

        policy = RetryPolicy(max_attempts=5, initial_delay_s=0.01)
        with pytest.raises(FakeAPIError):
            await _apply_retry_policy(coro, policy)
        assert calls["n"] == 1

    async def test_retryable_eventually_succeeds(self) -> None:
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeAPIError(500)
            return "ok"

        policy = RetryPolicy(max_attempts=5, initial_delay_s=0.01, jitter=False)
        result = await _apply_retry_policy(coro, policy)
        assert result == "ok"
        assert calls["n"] == 3

    async def test_exhausts_max_attempts(self) -> None:
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            raise FakeAPIError(500)

        policy = RetryPolicy(max_attempts=3, initial_delay_s=0.01, jitter=False)
        with pytest.raises(FakeAPIError):
            await _apply_retry_policy(coro, policy)
        assert calls["n"] == 3

    async def test_network_error_retries_without_status(self) -> None:
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            if calls["n"] < 2:
                raise ConnectionError("blip")
            return "recovered"

        policy = RetryPolicy(max_attempts=3, initial_delay_s=0.01)
        result = await _apply_retry_policy(coro, policy)
        assert result == "recovered"
        assert calls["n"] == 2

    async def test_succeeds_on_first_attempt(self) -> None:
        """Happy path: no retries needed."""
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            return "fast"

        policy = RetryPolicy(max_attempts=5, initial_delay_s=10.0)
        # Should NOT sleep (delay is 10s but we never enter the retry loop)
        start = time.perf_counter()
        result = await _apply_retry_policy(coro, policy)
        elapsed = time.perf_counter() - start
        assert result == "fast"
        assert calls["n"] == 1
        assert elapsed < 0.5  # well under the 10s delay

    async def test_backoff_actually_delays(self) -> None:
        """With initial_delay_s=0.1 and 3 failed attempts, total elapsed
        should reflect at least 1 * 0.1 + 1 * 0.2 ≈ 0.3s (2 retries)."""
        calls = {"n": 0}

        async def coro():
            calls["n"] += 1
            raise FakeAPIError(500)

        policy = RetryPolicy(
            max_attempts=3, initial_delay_s=0.1, jitter=False, backoff="exponential",
        )
        start = time.perf_counter()
        with pytest.raises(FakeAPIError):
            await _apply_retry_policy(coro, policy)
        elapsed = time.perf_counter() - start
        # 3 attempts, 2 backoff sleeps: 0.1 + 0.2 = 0.3s minimum
        assert elapsed >= 0.25
        assert calls["n"] == 3


# ---------------------------------------------------------------------------
# OpenAIClientConfig integration
# ---------------------------------------------------------------------------


class TestOpenAIClientConfigIntegration:
    def test_retry_policy_defaults_to_none(self) -> None:
        config = OpenAIClientConfig(api_key="x", model="m")
        assert config.retry_policy is None

    def test_retry_policy_can_be_set(self) -> None:
        policy = RetryPolicy(max_attempts=5, backoff="constant")
        config = OpenAIClientConfig(api_key="x", model="m", retry_policy=policy)
        assert config.retry_policy is not None
        assert config.retry_policy.max_attempts == 5
        assert config.retry_policy.backoff == "constant"

    def test_retry_policy_round_trips_via_model_dump(self) -> None:
        config = OpenAIClientConfig(
            api_key="x",
            model="m",
            retry_policy=RetryPolicy(max_attempts=4, jitter=False),
        )
        dumped = config.model_dump()
        assert "retry_policy" in dumped
        # And can be rehydrated
        rehydrated = OpenAIClientConfig.model_validate(dumped)
        assert rehydrated.retry_policy is not None
        assert rehydrated.retry_policy.max_attempts == 4
        assert rehydrated.retry_policy.jitter is False
