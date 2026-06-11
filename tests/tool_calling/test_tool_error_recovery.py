"""
Tests for ToolErrorRecovery — error recovery config for ToolStepConfig.

Covers:
- ToolErrorRecovery model fields and defaults
- retry_max: retries on exception up to N times
- retry_delay: waits between retries (mocked asyncio.sleep)
- on_exception: fallback tool called after all retries fail
- on_timeout: fallback tool called when primary times out
- No recovery when error_recovery=None (original behavior preserved)
- Fallback tool name appears in history entry
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, call

from mmar_carl import (
    Language,
    ReasoningChain,
    ReasoningContext,
    LLMClientBase,
)
from mmar_carl.models.steps import ToolStepDescription
from mmar_carl.models.config import ToolStepConfig, ToolErrorRecovery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLMClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _make_context() -> ReasoningContext:
    return ReasoningContext(
        outer_context="test",
        api=_MockLLMClient(),
        model="test",
        language=Language.ENGLISH,
    )


def _make_chain(config: ToolStepConfig) -> ReasoningChain:
    return ReasoningChain(
        steps=[ToolStepDescription(number=1, title="Step", config=config)]
    )


# ---------------------------------------------------------------------------
# ToolErrorRecovery model
# ---------------------------------------------------------------------------


class TestToolErrorRecoveryModel:
    def test_defaults(self):
        rec = ToolErrorRecovery()
        assert rec.retry_max == 0
        assert rec.retry_delay == 0.0
        assert rec.on_timeout is None
        assert rec.on_exception is None

    def test_custom_values(self):
        rec = ToolErrorRecovery(
            retry_max=3,
            retry_delay=1.5,
            on_timeout="fallback_a",
            on_exception="fallback_b",
        )
        assert rec.retry_max == 3
        assert rec.retry_delay == 1.5
        assert rec.on_timeout == "fallback_a"
        assert rec.on_exception == "fallback_b"

    def test_retry_max_must_be_non_negative(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ToolErrorRecovery(retry_max=-1)

    def test_retry_delay_must_be_non_negative(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ToolErrorRecovery(retry_delay=-0.1)

    def test_tool_step_config_has_error_recovery_field(self):
        cfg = ToolStepConfig(tool_name="t", input_mapping={})
        assert cfg.error_recovery is None

    def test_tool_step_config_accepts_error_recovery(self):
        rec = ToolErrorRecovery(retry_max=2)
        cfg = ToolStepConfig(
            tool_name="t",
            input_mapping={},
            error_recovery=rec,
        )
        assert cfg.error_recovery is rec


# ---------------------------------------------------------------------------
# Retry on exception
# ---------------------------------------------------------------------------


class TestRetryOnException:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt_no_retry_needed(self):
        call_count = [0]

        def reliable_tool() -> str:
            call_count[0] += 1
            return "ok"

        cfg = ToolStepConfig(
            tool_name="reliable",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=2),
        )
        ctx = _make_context()
        ctx.register_tool("reliable", reliable_tool)
        result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_retries_on_exception_and_succeeds(self):
        call_count = [0]

        def flaky_tool() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("temporary error")
            return "eventually ok"

        cfg = ToolStepConfig(
            tool_name="flaky",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=3),
        )
        ctx = _make_context()
        ctx.register_tool("flaky", flaky_tool)
        result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        assert result.step_results[0].result == "eventually ok"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_fails_after_all_retries_exhausted(self):
        call_count = [0]

        def always_fails() -> str:
            call_count[0] += 1
            raise RuntimeError(f"fail #{call_count[0]}")

        cfg = ToolStepConfig(
            tool_name="always_fails",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=2),
        )
        ctx = _make_context()
        ctx.register_tool("always_fails", always_fails)
        result = await _make_chain(cfg).execute_async(ctx)

        assert not result.success
        assert call_count[0] == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_when_retry_max_zero(self):
        call_count = [0]

        def fails_once() -> str:
            call_count[0] += 1
            raise RuntimeError("error")

        cfg = ToolStepConfig(
            tool_name="fails_once",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=0),
        )
        ctx = _make_context()
        ctx.register_tool("fails_once", fails_once)
        result = await _make_chain(cfg).execute_async(ctx)

        assert not result.success
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_no_recovery_when_error_recovery_is_none(self):
        """Without error_recovery, original behavior: fail immediately."""
        call_count = [0]

        def always_fails() -> str:
            call_count[0] += 1
            raise RuntimeError("fail")

        cfg = ToolStepConfig(tool_name="t", input_mapping={})  # no error_recovery
        ctx = _make_context()
        ctx.register_tool("t", always_fails)
        result = await _make_chain(cfg).execute_async(ctx)

        assert not result.success
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Retry delay
# ---------------------------------------------------------------------------


class TestRetryDelay:
    @pytest.mark.asyncio
    async def test_sleep_called_between_retries(self):
        call_count = [0]

        def fails_twice() -> str:
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("fail")
            return "ok"

        cfg = ToolStepConfig(
            tool_name="fails_twice",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=2, retry_delay=0.5),
        )
        ctx = _make_context()
        ctx.register_tool("fails_twice", fails_twice)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        # sleep called twice (between attempt 1→2 and 2→3)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([call(0.5), call(0.5)])

    @pytest.mark.asyncio
    async def test_no_sleep_when_retry_delay_zero(self):
        call_count = [0]

        def fails_once() -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("fail")
            return "ok"

        cfg = ToolStepConfig(
            tool_name="fails_once",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=1, retry_delay=0.0),
        )
        ctx = _make_context()
        ctx.register_tool("fails_once", fails_once)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Fallback on exception
# ---------------------------------------------------------------------------


class TestFallbackOnException:
    @pytest.mark.asyncio
    async def test_on_exception_fallback_called_after_primary_fails(self):
        def primary() -> str:
            raise RuntimeError("primary down")

        def fallback() -> str:
            return "fallback result"

        cfg = ToolStepConfig(
            tool_name="primary",
            input_mapping={},
            error_recovery=ToolErrorRecovery(on_exception="fallback"),
        )
        ctx = _make_context()
        ctx.register_tool("primary", primary)
        ctx.register_tool("fallback", fallback)
        result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        assert result.step_results[0].result == "fallback result"
        # Fallback tool name appears in history
        assert "TOOL: fallback" in result.history[0]

    @pytest.mark.asyncio
    async def test_on_exception_not_called_when_primary_succeeds(self):
        fallback_called = [False]

        def primary() -> str:
            return "primary ok"

        def fallback() -> str:
            fallback_called[0] = True
            return "fallback result"

        cfg = ToolStepConfig(
            tool_name="primary",
            input_mapping={},
            error_recovery=ToolErrorRecovery(on_exception="fallback"),
        )
        ctx = _make_context()
        ctx.register_tool("primary", primary)
        ctx.register_tool("fallback", fallback)
        result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        assert not fallback_called[0]

    @pytest.mark.asyncio
    async def test_on_exception_after_retries_exhausted(self):
        attempts = [0]

        def flaky() -> str:
            attempts[0] += 1
            raise RuntimeError("always fails")

        def fallback() -> str:
            return "saved"

        cfg = ToolStepConfig(
            tool_name="flaky",
            input_mapping={},
            error_recovery=ToolErrorRecovery(retry_max=2, on_exception="fallback"),
        )
        ctx = _make_context()
        ctx.register_tool("flaky", flaky)
        ctx.register_tool("fallback", fallback)
        result = await _make_chain(cfg).execute_async(ctx)

        # Primary tried 3 times, then fallback used
        assert result.success
        assert attempts[0] == 3
        assert result.step_results[0].result == "saved"


# ---------------------------------------------------------------------------
# Fallback on timeout
# ---------------------------------------------------------------------------


class TestFallbackOnTimeout:
    @pytest.mark.asyncio
    async def test_on_timeout_fallback_called_when_primary_times_out(self):
        async def slow_tool() -> str:
            await asyncio.sleep(999)
            return "never"

        def fast_fallback() -> str:
            return "fallback saved"

        cfg = ToolStepConfig(
            tool_name="slow_tool",
            input_mapping={},
            timeout=0.01,  # very short timeout
            error_recovery=ToolErrorRecovery(on_timeout="fast_fallback"),
        )
        ctx = _make_context()
        ctx.register_tool("slow_tool", slow_tool)
        ctx.register_tool("fast_fallback", fast_fallback)
        result = await _make_chain(cfg).execute_async(ctx)

        assert result.success
        assert result.step_results[0].result == "fallback saved"
        assert "TOOL: fast_fallback" in result.history[0]

    @pytest.mark.asyncio
    async def test_on_timeout_not_triggered_for_exception(self):
        """on_timeout fallback should NOT fire for regular exceptions (only timeouts)."""
        fallback_called = [False]

        def raises_error() -> str:
            raise RuntimeError("not a timeout")

        def timeout_fallback() -> str:
            fallback_called[0] = True
            return "should not appear"

        cfg = ToolStepConfig(
            tool_name="raises_error",
            input_mapping={},
            error_recovery=ToolErrorRecovery(on_timeout="timeout_fallback"),
        )
        ctx = _make_context()
        ctx.register_tool("raises_error", raises_error)
        ctx.register_tool("timeout_fallback", timeout_fallback)
        result = await _make_chain(cfg).execute_async(ctx)

        # No on_exception configured, so it should fail (not use on_timeout fallback)
        assert not result.success
        assert not fallback_called[0]
