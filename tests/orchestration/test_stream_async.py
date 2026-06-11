"""Tests for streaming-first chain execution mode.

``chain.stream_async(ctx)`` is an async generator that yields each
``StepExecutionResult`` as soon as its step finishes, then yields the
final ``ReasoningResult`` last. Lets UIs show partial progress instead
of blocking on the whole chain.
"""

from __future__ import annotations

import asyncio

from mmar_carl import (
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    ToolStepDescription,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


class _FakeClient(LLMClientBase):
    """Deterministic in-memory LLM used to drive the streaming loop."""

    def __init__(self, *, sleep: float = 0.0) -> None:
        self._sleep = sleep

    @property
    def model_name(self) -> str:
        return "fake"

    async def get_response(self, prompt: str) -> str:
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        if self._sleep:
            await asyncio.sleep(self._sleep)
        return "ok"


def _linear_chain(n: int) -> ReasoningChain:
    steps = [LLMStepDescription(number=1, title="step1", aim="x")]
    for i in range(2, n + 1):
        steps.append(LLMStepDescription(
            number=i, title=f"step{i}", aim="x", dependencies=[i - 1],
        ))
    return ReasoningChain(steps=steps)


def _ctx(sleep: float = 0.0) -> ReasoningContext:
    return ReasoningContext(outer_context="N/A", api=_FakeClient(sleep=sleep))


# ---------------------------------------------------------------------------
# Basic streaming behaviour
# ---------------------------------------------------------------------------


class TestBasicStreaming:
    async def test_yields_one_item_per_step_plus_final_result(self) -> None:
        chain = _linear_chain(3)
        items = [item async for item in chain.stream_async(_ctx())]
        assert len(items) == 4  # 3 steps + 1 final
        assert all(
            isinstance(i, StepExecutionResult) for i in items[:-1]
        )
        assert isinstance(items[-1], ReasoningResult)

    async def test_step_results_yielded_in_order_for_serial_chain(self) -> None:
        chain = _linear_chain(3)
        step_results = []
        async for item in chain.stream_async(_ctx()):
            if isinstance(item, StepExecutionResult):
                step_results.append(item.step_number)
        # Linear chain → completion order matches definition order.
        assert step_results == [1, 2, 3]

    async def test_final_result_aggregates_all_steps(self) -> None:
        chain = _linear_chain(3)
        final: ReasoningResult | None = None
        async for item in chain.stream_async(_ctx()):
            if isinstance(item, ReasoningResult):
                final = item
        assert final is not None
        assert final.success is True
        assert len(final.step_results) == 3

    async def test_single_step_chain_yields_two_items(self) -> None:
        chain = _linear_chain(1)
        items = [item async for item in chain.stream_async(_ctx())]
        assert len(items) == 2
        assert isinstance(items[0], StepExecutionResult)
        assert isinstance(items[1], ReasoningResult)


# ---------------------------------------------------------------------------
# Backward compatibility: callback wiring
# ---------------------------------------------------------------------------


class TestCallbackChaining:
    async def test_user_on_step_complete_still_fires(self) -> None:
        """The streaming wrapper must not clobber an existing
        ``context.on_step_complete`` callback."""
        seen: list[int] = []

        def my_cb(sr: StepExecutionResult) -> None:
            seen.append(sr.step_number)

        chain = _linear_chain(2)
        ctx = _ctx()
        ctx.on_step_complete = my_cb
        async for _ in chain.stream_async(ctx):
            pass
        assert seen == [1, 2]

    async def test_callback_is_restored_after_streaming(self) -> None:
        """Once ``stream_async`` finishes (or aborts), the original
        ``on_step_complete`` reference must be reinstated so the
        context is reusable."""
        def my_cb(sr: StepExecutionResult) -> None:
            pass

        chain = _linear_chain(1)
        ctx = _ctx()
        ctx.on_step_complete = my_cb
        async for _ in chain.stream_async(ctx):
            pass
        assert ctx.on_step_complete is my_cb

    async def test_user_callback_failure_does_not_break_stream(self) -> None:
        """A buggy user callback must not propagate — matches the
        executor's existing 'callbacks can't kill the chain' contract."""

        def boom(_sr: StepExecutionResult) -> None:
            raise RuntimeError("bad callback")

        chain = _linear_chain(2)
        ctx = _ctx()
        ctx.on_step_complete = boom
        items = [item async for item in chain.stream_async(ctx)]
        # All 2 steps + final result still arrived
        assert len(items) == 3


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    async def test_step_failure_still_terminates_with_reasoning_result(self) -> None:
        """If a step fails, the chain still ends — the final
        ReasoningResult has success=False but the stream is well-
        formed."""
        def boom() -> str:
            raise RuntimeError("tool crashed")

        chain = ReasoningChain(steps=[
            ToolStepDescription(
                number=1, title="boom",
                config=ToolStepConfig(tool_name="boom"),
            ),
        ])
        ctx = _ctx()
        ctx.register_tool("boom", boom)

        items = [item async for item in chain.stream_async(ctx)]
        assert len(items) == 2
        step = items[0]
        final = items[1]
        assert isinstance(step, StepExecutionResult)
        assert isinstance(final, ReasoningResult)
        assert step.success is False
        assert final.success is False


# ---------------------------------------------------------------------------
# Real ordering: stream items arrive before execute_async returns
# ---------------------------------------------------------------------------


class TestEagerStreaming:
    async def test_first_step_result_arrives_before_chain_finishes(self) -> None:
        """The whole point of streaming: step 1 should be inspectable
        before the chain's final step finishes.

        Use a delayed FakeClient and observe that the gap between
        receiving the first step result and the ReasoningResult is
        at least one ``sleep`` cycle — proving the consumer didn't
        block on the whole chain.
        """
        chain = _linear_chain(2)
        ctx = _ctx(sleep=0.05)

        loop = asyncio.get_event_loop()
        first_step_at: float | None = None
        final_at: float | None = None
        async for item in chain.stream_async(ctx):
            now = loop.time()
            if isinstance(item, StepExecutionResult) and first_step_at is None:
                first_step_at = now
            if isinstance(item, ReasoningResult):
                final_at = now

        assert first_step_at is not None and final_at is not None
        # Step 2 sleeps 0.05 s after step 1 finishes — gap must be ≥ ~0.04 s.
        assert (final_at - first_step_at) >= 0.03


# ---------------------------------------------------------------------------
# Context reuse — the wrapper must not leak callback state
# ---------------------------------------------------------------------------


class TestContextSafetyForReuse:
    async def test_second_run_after_streaming_uses_default_callback(self) -> None:
        chain = _linear_chain(1)
        ctx = _ctx()
        assert ctx.on_step_complete is None
        async for _ in chain.stream_async(ctx):
            pass
        # After streaming, callback must be back to its prior None state.
        assert ctx.on_step_complete is None
        # And a follow-up plain execute_async still works.
        result = await chain.execute_async(ctx)
        assert result.success


# ---------------------------------------------------------------------------
# Generator cancellation cleans up
# ---------------------------------------------------------------------------


class TestCancellationCleanup:
    async def test_break_inside_consumer_does_not_hang(self) -> None:
        """If the consumer breaks out early, the underlying chain
        execution should be cancelled cleanly (not deadlock)."""
        chain = _linear_chain(3)
        ctx = _ctx(sleep=0.02)
        # Break after the very first step result.
        count = 0
        async for item in chain.stream_async(ctx):
            count += 1
            if isinstance(item, StepExecutionResult):
                break
        # We got at least one item; we did NOT have to wait for the
        # whole chain.
        assert count >= 1
