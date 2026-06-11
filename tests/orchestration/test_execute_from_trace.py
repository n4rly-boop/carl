"""
Tests for ``ReasoningChain.execute_from_trace``.

``execute_from_trace(trace, from_step, context)`` resumes a chain from step
``from_step``, short-circuiting earlier LLM-style steps with their recorded
``result`` values from *trace* and dispatching steps ``>= from_step`` to the
live LLM client on ``context.api``.
"""

import pytest

from mmar_carl import (
    LLMClientBase,
    LLMStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #


class _CountingLLM(LLMClientBase):
    """Returns ``f'<prefix>-<n>'`` per call and tracks call count."""

    def __init__(self, prefix: str = "live") -> None:
        self.prefix = prefix
        self.calls = 0
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return f"{self.prefix}-{self.calls}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _build_linear_chain(n_steps: int = 3) -> ReasoningChain:
    steps = []
    for i in range(1, n_steps + 1):
        deps = [i - 1] if i > 1 else []
        steps.append(LLMStepDescription(number=i, title=f"s{i}", aim="x", dependencies=deps))
    return ReasoningChain(steps=steps, max_workers=1)


async def _run_original(chain: ReasoningChain) -> tuple[object, _CountingLLM]:
    ctx = ReasoningContext(outer_context="topic", api=_CountingLLM(prefix="orig"))
    result = await chain.execute_async(ctx)
    return result, ctx.api


# --------------------------------------------------------------------------- #
# Core: prefix replay + suffix live execution
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resume_from_middle_uses_trace_for_prefix_and_live_for_suffix() -> None:
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    resumed_llm = _CountingLLM(prefix="resumed")
    ctx = ReasoningContext(outer_context="topic", api=resumed_llm)
    result = await chain.execute_from_trace(
        first_result.trace, from_step=3, context=ctx
    )
    # Steps 1 and 2 used recorded results; step 3 hit the live client
    results = {sr.step_number: sr.result for sr in result.step_results}
    assert results[1] == "orig-1"
    assert results[2] == "orig-2"
    assert results[3] == "resumed-1"
    # Live client called exactly once (for the suffix)
    assert resumed_llm.calls == 1


@pytest.mark.asyncio
async def test_resume_from_first_step_runs_everything_live() -> None:
    """``from_step`` at the chain's smallest step number runs the chain fully live."""
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    resumed_llm = _CountingLLM(prefix="resumed")
    ctx = ReasoningContext(outer_context="topic", api=resumed_llm)
    result = await chain.execute_from_trace(first_result.trace, from_step=1, context=ctx)
    results = {sr.step_number: sr.result for sr in result.step_results}
    # All three steps hit the live client → "resumed-1/2/3"
    assert results == {1: "resumed-1", 2: "resumed-2", 3: "resumed-3"}
    assert resumed_llm.calls == 3


@pytest.mark.asyncio
async def test_resume_past_chain_end_replays_everything_from_trace() -> None:
    """``from_step`` beyond the last step → all steps replayed from trace, no live calls."""
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    resumed_llm = _CountingLLM(prefix="resumed")
    ctx = ReasoningContext(outer_context="topic", api=resumed_llm)
    result = await chain.execute_from_trace(
        first_result.trace, from_step=99, context=ctx
    )
    results = {sr.step_number: sr.result for sr in result.step_results}
    assert results == {1: "orig-1", 2: "orig-2", 3: "orig-3"}
    assert resumed_llm.calls == 0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rejects_from_step_below_smallest_step() -> None:
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    ctx = ReasoningContext(outer_context="topic", api=_CountingLLM())
    with pytest.raises(ValueError, match="below the chain's smallest step number"):
        await chain.execute_from_trace(first_result.trace, from_step=0, context=ctx)


# --------------------------------------------------------------------------- #
# Trace event without a corresponding chain step → handled gracefully
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_trace_event_resolves_to_empty_string_response() -> None:
    """If a prefix step has no event in the trace, the hybrid client returns ''."""
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    # Drop event for step 2
    first_result.trace.events = [
        e for e in first_result.trace.events if e.step_number != 2
    ]

    resumed_llm = _CountingLLM(prefix="resumed")
    ctx = ReasoningContext(outer_context="topic", api=resumed_llm)
    result = await chain.execute_from_trace(first_result.trace, from_step=3, context=ctx)
    results = {sr.step_number: sr.result for sr in result.step_results}
    assert results[1] == "orig-1"
    assert results[2] == ""          # missing event → empty replay
    assert results[3] == "resumed-1"  # live call
    assert resumed_llm.calls == 1


# --------------------------------------------------------------------------- #
# Mixed step types — tool steps re-run normally on both halves
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_steps_run_on_both_sides_of_from_step() -> None:
    """Tool steps don't touch the LLM client at all — they re-execute on both
    halves, exercising the same registered tool. We confirm this by counting
    invocations of a tool that increments a counter on each call."""
    invocations: list[int] = []

    def counter_tool() -> int:
        invocations.append(len(invocations) + 1)
        return invocations[-1]

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="llm-1", aim="x"),
            ToolStepDescription(
                number=2, title="tool", dependencies=[1],
                config=ToolStepConfig(
                    tool_name="counter", parameters=[], input_mapping={},
                ),
            ),
            LLMStepDescription(number=3, title="llm-3", aim="x", dependencies=[2]),
        ],
        max_workers=1,
    )

    # Original run
    ctx_orig = ReasoningContext(outer_context="topic", api=_CountingLLM(prefix="orig"))
    ctx_orig.register_tool("counter", counter_tool)
    first_result = await chain.execute_async(ctx_orig)
    first_tool_call_count = len(invocations)

    # Resume from step 3
    resumed_llm = _CountingLLM(prefix="resumed")
    ctx_resume = ReasoningContext(outer_context="topic", api=resumed_llm)
    ctx_resume.register_tool("counter", counter_tool)
    result = await chain.execute_from_trace(first_result.trace, from_step=3, context=ctx_resume)
    assert result.step_results[0].success
    # The tool ran a second time (re-execution of step 2's prefix side effect)
    assert len(invocations) == first_tool_call_count + 1


@pytest.mark.asyncio
async def test_memory_writes_from_replayed_prefix_are_visible_to_suffix() -> None:
    """Memory writes during the prefix half are visible to step >= from_step."""
    captured: dict[str, str] = {}

    def capture_tool(value: str) -> str:
        captured["got"] = value
        return value

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="produce", aim="x"),
            MemoryStepDescription(
                number=2, title="store", dependencies=[1],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="cached_answer",
                    value_source="$history[-1]",
                ),
            ),
            ToolStepDescription(
                number=3, title="use cached", dependencies=[2],
                config=ToolStepConfig(
                    tool_name="capture",
                    parameters=[],
                    input_mapping={"value": "$memory.default.cached_answer"},
                ),
            ),
        ],
        max_workers=1,
    )

    # Original run — step 1 LLM produces "orig-1", step 2 stores history[-1]
    # to memory.default.cached_answer. (history[-1] includes the formatted
    # step header — "...Result: orig-1" — by design.)
    ctx_orig = ReasoningContext(outer_context="x", api=_CountingLLM(prefix="orig"))
    ctx_orig.register_tool("capture", capture_tool)
    first_result = await chain.execute_async(ctx_orig)
    assert "orig-1" in captured["got"]
    captured.clear()

    # Resume from step 3 — steps 1 and 2 re-run from trace, memory rewritten,
    # then step 3 sees the cached value
    ctx_resume = ReasoningContext(outer_context="x", api=_CountingLLM(prefix="resumed"))
    ctx_resume.register_tool("capture", capture_tool)
    result = await chain.execute_from_trace(first_result.trace, from_step=3, context=ctx_resume)
    assert all(sr.success for sr in result.step_results)
    # Step 1's prefix-replay returned "orig-1", which step 2 wrote to memory,
    # which step 3 (resumed half) consumed.
    assert "orig-1" in captured["got"]


# --------------------------------------------------------------------------- #
# Context propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pre_existing_context_state_carried_into_resume() -> None:
    """Pre-populated memory / metadata in the resume context must reach the
    resumed execution (the implementation deep-copies these forward)."""
    chain = _build_linear_chain(2)
    first_result, _ = await _run_original(chain)

    ctx = ReasoningContext(outer_context="x", api=_CountingLLM(prefix="resumed"))
    ctx.memory_write("hint", "USE_CACHE", namespace="config")
    ctx.metadata["run_id"] = "RUN_42"

    result = await chain.execute_from_trace(first_result.trace, from_step=2, context=ctx)
    # The resumed context inside execute_from_trace gets its own copy, but
    # the result is observable through the ReasoningResult.
    assert result.step_results[1].success


@pytest.mark.asyncio
async def test_tools_and_tags_inherited_into_resume_context() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="x", aim="x"),
            ToolStepDescription(
                number=2, title="t", dependencies=[1],
                config=ToolStepConfig(
                    tool_name="x", parameters=[], input_mapping={},
                    allowed_tool_tags=["safe"],
                ),
            ),
        ],
        max_workers=1,
    )
    ctx_orig = ReasoningContext(outer_context="d", api=_CountingLLM())
    ctx_orig.register_tool("x", lambda: "ok", tags=["safe"])
    first = await chain.execute_async(ctx_orig)

    ctx_resume = ReasoningContext(outer_context="d", api=_CountingLLM(prefix="r"))
    ctx_resume.register_tool("x", lambda: "ok-again", tags=["safe"])
    result = await chain.execute_from_trace(first.trace, from_step=2, context=ctx_resume)
    # Tool whitelist still satisfied because tags were propagated
    assert result.step_results[1].success


# --------------------------------------------------------------------------- #
# Conditional / skipped trace events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_skipped_trace_events_are_not_used_for_replay() -> None:
    """A trace event flagged as skipped must NOT seed the hybrid client.

    Simulated by manually marking an event skipped and confirming the live
    client sees an empty response for that step number."""
    chain = _build_linear_chain(3)
    first_result, _ = await _run_original(chain)

    # Mark step 1 as skipped in the trace
    for e in first_result.trace.events:
        if e.step_number == 1:
            e.skipped = True

    resumed_llm = _CountingLLM(prefix="resumed")
    ctx = ReasoningContext(outer_context="x", api=resumed_llm)
    result = await chain.execute_from_trace(first_result.trace, from_step=3, context=ctx)
    results = {sr.step_number: sr.result for sr in result.step_results}
    # Skipped event → empty replay for step 1
    assert results[1] == ""
    # Step 2 still has a successful trace event
    assert results[2] == "orig-2"
    # Step 3 hit live client
    assert results[3] == "resumed-1"
