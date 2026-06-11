"""Tests for partial-output recovery after ``ExecutionCancelledError``.

Pre-fix: ``context.cancel()`` raised ``ExecutionCancelledError`` with no
way to recover the ``ReasoningResult`` — users had to manually inspect
``context.history`` or step-result dicts.

Post-fix:
1. ``ExecutionCancelledError`` carries ``.result: ReasoningResult``.
2. ``ReasoningResult.partial_outputs: dict[int, str]`` lists every
   successful step's text output.
3. ``ReasoningResult.get_partial_final_output()`` returns the
   highest-numbered successful step's output (recovery shorthand).
"""

from __future__ import annotations

import asyncio

import pytest

from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.executor import ExecutionCancelledError
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


# ---------------------------------------------------------------------------
# Direct ReasoningResult tests
# ---------------------------------------------------------------------------


def _step(
    number: int,
    *,
    title: str = "s",
    result: str = "ok",
    success: bool = True,
    skipped: bool = False,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=StepType.TOOL,
        result=result,
        success=success,
        skipped=skipped,
    )


def _make_result(*steps: StepExecutionResult) -> ReasoningResult:
    return ReasoningResult(success=True, history=[], step_results=list(steps))


class TestPartialOutputsProperty:
    def test_empty_chain(self) -> None:
        assert _make_result().partial_outputs == {}

    def test_only_successful_steps_included(self) -> None:
        result = _make_result(
            _step(1, result="A"),
            _step(2, result="B", success=False),
            _step(3, result="C"),
        )
        assert result.partial_outputs == {1: "A", 3: "C"}

    def test_skipped_steps_excluded(self) -> None:
        result = _make_result(
            _step(1, result="A"),
            _step(2, result="", skipped=True, success=True),
            _step(3, result="C"),
        )
        assert result.partial_outputs == {1: "A", 3: "C"}

    def test_empty_string_outputs_still_included(self) -> None:
        """A successful step with empty output should still appear — useful
        for confirming the step ran."""
        result = _make_result(_step(1, result=""))
        assert result.partial_outputs == {1: ""}

    def test_keyed_by_step_number_not_position(self) -> None:
        result = _make_result(
            _step(7, result="seven"),
            _step(11, result="eleven"),
        )
        assert result.partial_outputs == {7: "seven", 11: "eleven"}


class TestGetPartialFinalOutput:
    def test_none_when_no_successful_steps(self) -> None:
        result = _make_result(_step(1, success=False))
        assert result.get_partial_final_output() is None

    def test_returns_highest_numbered_step_output(self) -> None:
        result = _make_result(
            _step(1, result="A"),
            _step(2, result="B"),
            _step(3, result="C"),
        )
        assert result.get_partial_final_output() == "C"

    def test_skips_failed_step_to_find_latest_successful(self) -> None:
        result = _make_result(
            _step(1, result="A"),
            _step(2, result="B"),
            _step(3, result="", success=False),  # failed
        )
        assert result.get_partial_final_output() == "B"

    def test_skips_skipped_step(self) -> None:
        result = _make_result(
            _step(1, result="A"),
            _step(2, result="", skipped=True),
            _step(3, result="C"),
        )
        assert result.get_partial_final_output() == "C"

    def test_uses_step_number_not_list_position(self) -> None:
        """If steps are out of order in the list, find the actual highest."""
        result = _make_result(
            _step(5, result="five"),
            _step(2, result="two"),
            _step(7, result="seven"),
        )
        assert result.get_partial_final_output() == "seven"


# ---------------------------------------------------------------------------
# End-to-end: cancel mid-chain, recover via exception
# ---------------------------------------------------------------------------


def _make_three_step_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit-a", config=ToolStepConfig(tool_name="emit_a")
            ),
            ToolStepDescription(
                number=2,
                title="emit-b-and-cancel",
                dependencies=[1],
                config=ToolStepConfig(tool_name="emit_b_and_cancel"),
            ),
            ToolStepDescription(
                number=3,
                title="should-not-run",
                dependencies=[2],
                config=ToolStepConfig(tool_name="never"),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_cancelled_error_carries_partial_result() -> None:
    chain = _make_three_step_chain()
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    holder = {"ctx": ctx}

    def emit_a() -> str:
        return "A"

    def emit_b_and_cancel() -> str:
        holder["ctx"].cancel()
        return "B"

    ctx.register_tool("emit_a", emit_a)
    ctx.register_tool("emit_b_and_cancel", emit_b_and_cancel)
    ctx.register_tool("never", lambda: "should not run")

    with pytest.raises(ExecutionCancelledError) as exc_info:
        await chain.execute_async(ctx)

    # Exception carries the partial result
    assert exc_info.value.result is not None
    partial = exc_info.value.result.partial_outputs
    assert partial == {1: "A", 2: "B"}

    # And the recovery accessor returns the latest output
    assert exc_info.value.result.get_partial_final_output() == "B"


@pytest.mark.asyncio
async def test_normal_completion_also_exposes_partial_outputs() -> None:
    """The property is also usable on a normally-completed run — every
    successful step appears."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="a", config=ToolStepConfig(tool_name="emit")
            ),
            ToolStepDescription(
                number=2, title="b", dependencies=[1], config=ToolStepConfig(tool_name="emit")
            ),
        ],
    )
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("emit", lambda: "done")

    result = await chain.execute_async(ctx)
    assert result.partial_outputs == {1: "done", 2: "done"}
    assert result.get_partial_final_output() == "done"


@pytest.mark.asyncio
async def test_pre_execution_cancel_leaves_empty_partial() -> None:
    """If cancellation is set BEFORE execution starts, the executor's
    reset_cancellation() clears it (existing behaviour) — the chain
    completes normally and partial_outputs reflects the full run."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="a", config=ToolStepConfig(tool_name="emit")
            ),
        ],
    )
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("emit", lambda: "ok")
    ctx.cancel()  # pre-set — but executor clears this at start

    result = await chain.execute_async(ctx)
    assert result.success is True
    assert result.partial_outputs == {1: "ok"}


def test_execution_cancelled_error_default_result_is_none() -> None:
    """When constructed without ``result=`` (e.g. by user code), the
    attribute is None — no crash."""
    exc = ExecutionCancelledError("manual raise")
    assert exc.result is None
    assert str(exc) == "manual raise"


# ---------------------------------------------------------------------------
# Sync wrapper sanity
# ---------------------------------------------------------------------------


def test_partial_outputs_via_asyncio_run() -> None:
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="a", config=ToolStepConfig(tool_name="emit")
            ),
        ],
    )
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("emit", lambda: "ok")
    result = asyncio.run(chain.execute_async(ctx))
    assert result.partial_outputs == {1: "ok"}
