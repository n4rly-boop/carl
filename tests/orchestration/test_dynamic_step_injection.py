"""
Tests for dynamic step injection at runtime.

A step's :class:`StepExecutionResult` may carry ``injected_steps`` —
additional :class:`StepDescription` instances to splice into the running
DAG. The executor renumbers each one to a fresh integer, validates that
its dependencies reference already-registered step numbers (no forward
references), and registers it as a new :class:`ExecutionNode` so subsequent
batches pick it up.

A chain-level ``max_injections`` budget guard prevents runaway injection
loops.
"""

from typing import Any, Optional

import pytest

from mmar_carl import (
    LLMClientBase,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import StepExecutionResult
from mmar_carl.step_executors import (
    StepExecutorBase,
    get_executor,
    register_executor,
)


# --------------------------------------------------------------------------- #
# Helpers — a custom executor for the LLM step type that *injects* a follow-on
# step on first invocation, then falls through to the normal LLM behaviour on
# subsequent calls.
# --------------------------------------------------------------------------- #


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


class _InjectingExecutor(StepExecutorBase):
    """Executor used in place of the standard LLM executor for one test step.

    Produces a configurable list of injected steps on the first call, and a
    plain success result thereafter.
    """

    def __init__(
        self,
        steps_to_inject: list[Any],
        *,
        only_first_call: bool = True,
    ) -> None:
        self._steps_to_inject = list(steps_to_inject)
        self._only_first_call = only_first_call
        self._calls = 0

    async def execute(
        self,
        step: Any,
        context: ReasoningContext,
        prompt_template: Optional[Any] = None,
    ) -> StepExecutionResult:
        self._calls += 1
        inject = []
        if not self._only_first_call or self._calls == 1:
            inject = list(self._steps_to_inject)
        history_entry = f"Step {step.number}. {step.title}\nResult: injected\n"
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.LLM,
            result=f"call-{self._calls}",
            success=True,
            updated_history=list(context.history) + [history_entry],
            injected_steps=inject,
        )


def _install_injector(
    steps_to_inject: list[Any], *, only_first_call: bool = True
) -> StepExecutorBase:
    """Swap in an injecting executor for ``StepType.LLM`` and return the
    original for the caller to restore afterwards."""
    original = get_executor(StepType.LLM)
    register_executor(
        StepType.LLM, _InjectingExecutor(steps_to_inject, only_first_call=only_first_call)
    )
    return original


def _restore_llm_executor(original: StepExecutorBase) -> None:
    register_executor(StepType.LLM, original)


# --------------------------------------------------------------------------- #
# Field plumbing
# --------------------------------------------------------------------------- #


def test_step_execution_result_accepts_injected_steps_field() -> None:
    inj = LLMStepDescription(number=1, title="x", aim="x")
    r = StepExecutionResult(
        step_number=1, step_title="x", step_type=StepType.LLM, result="r", success=True,
        injected_steps=[inj],
    )
    assert len(r.injected_steps) == 1
    assert r.injected_steps[0].number == 1


def test_step_execution_result_default_injected_steps_is_empty_list() -> None:
    r = StepExecutionResult(
        step_number=1, step_title="x", step_type=StepType.LLM, result="r", success=True,
    )
    assert r.injected_steps == []


# --------------------------------------------------------------------------- #
# Happy path: one injected step runs in the next batch
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_injected_step_runs_after_parent_in_same_chain() -> None:
    followup = LLMStepDescription(number=999, title="injected", aim="extra")
    original = _install_injector([followup])
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
        # Both the root step and the injected follow-on should appear in results
        assert len(result.step_results) == 2
        titles = [sr.step_title for sr in result.step_results]
        assert "root" in titles
        assert "injected" in titles
    finally:
        _restore_llm_executor(original)


@pytest.mark.asyncio
async def test_injected_step_renumbered_above_existing() -> None:
    """The original ``number`` field on the injected step is overwritten with
    ``max_existing + 1`` so collisions with declared step numbers are impossible."""
    followup = LLMStepDescription(number=5, title="renumber-me", aim="x")
    original = _install_injector([followup], only_first_call=True)
    try:
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="a", aim="x"),
                LLMStepDescription(number=2, title="b", aim="x", dependencies=[1]),
                LLMStepDescription(number=3, title="c", aim="x", dependencies=[2]),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
        # 3 declared + 1 injected = 4 step results
        assert len(result.step_results) == 4
        renumbered = [sr for sr in result.step_results if sr.step_title == "renumber-me"]
        assert len(renumbered) == 1
        # Injected step got 4 (max declared was 3)
        assert renumbered[0].step_number == 4
    finally:
        _restore_llm_executor(original)


@pytest.mark.asyncio
async def test_multiple_injected_steps_in_one_result() -> None:
    inj1 = LLMStepDescription(number=99, title="inj1", aim="x")
    inj2 = LLMStepDescription(number=99, title="inj2", aim="x")
    original = _install_injector([inj1, inj2])
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
        titles = [sr.step_title for sr in result.step_results]
        assert "root" in titles
        assert "inj1" in titles
        assert "inj2" in titles
        # Two injected steps must have distinct fresh numbers
        injected_numbers = {sr.step_number for sr in result.step_results if sr.step_title.startswith("inj")}
        assert len(injected_numbers) == 2
        assert all(n > 1 for n in injected_numbers)
    finally:
        _restore_llm_executor(original)


# --------------------------------------------------------------------------- #
# Dependency wiring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_injected_step_can_declare_dependency_on_existing_step() -> None:
    """An injected step whose ``dependencies=[1]`` waits for step 1 to finish."""
    followup = LLMStepDescription(
        number=999, title="injected", aim="x", dependencies=[1]
    )
    original = _install_injector([followup])
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
        assert len(result.step_results) == 2
    finally:
        _restore_llm_executor(original)


@pytest.mark.asyncio
async def test_injected_step_with_unknown_dependency_raises() -> None:
    """Forward references to unregistered step numbers are rejected."""
    followup = LLMStepDescription(
        number=999, title="injected", aim="x", dependencies=[42],  # 42 doesn't exist
    )
    original = _install_injector([followup])
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        with pytest.raises(ValueError, match="unknown step number 42"):
            await chain.execute_async(ctx)
    finally:
        _restore_llm_executor(original)


# --------------------------------------------------------------------------- #
# Budget guard
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_injections_budget_exhausted_raises() -> None:
    """Injecting more than ``max_injections`` total steps fails the chain."""
    # Inject 5 every call; max_injections=3 ⇒ overflow on the first batch
    inj_list = [
        LLMStepDescription(number=99, title=f"inj{i}", aim="x") for i in range(5)
    ]
    original = _install_injector(inj_list, only_first_call=True)
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
            max_injections=3,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        with pytest.raises(ValueError, match="max_injections budget"):
            await chain.execute_async(ctx)
    finally:
        _restore_llm_executor(original)


@pytest.mark.asyncio
async def test_max_injections_budget_accumulates_across_batches() -> None:
    """The budget is per-execution, not per-batch — injections accumulate."""
    inj_list = [LLMStepDescription(number=99, title="inj", aim="x")]
    # Inject 1 step every time → with 2 declared steps each producing 1 injection,
    # we'd want 2 injections total. Set max_injections=1 to overflow on batch 2.
    original = _install_injector(inj_list, only_first_call=False)
    try:
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="a", aim="x"),
                LLMStepDescription(number=2, title="b", aim="x", dependencies=[1]),
            ],
            max_workers=1,
            max_injections=1,  # only 1 injection allowed
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        with pytest.raises(ValueError, match="max_injections budget"):
            await chain.execute_async(ctx)
    finally:
        _restore_llm_executor(original)


@pytest.mark.asyncio
async def test_max_injections_budget_reset_between_runs() -> None:
    """Calling ``execute_async`` again on the same chain resets the counter."""
    followup = LLMStepDescription(number=999, title="injected", aim="x")
    original = _install_injector([followup], only_first_call=True)
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
            max_injections=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result1 = await chain.execute_async(ctx)
        assert len(result1.step_results) == 2
        # The injector's `_calls` keeps growing but we control it via only_first_call.
        # Reset its internal counter to allow a second injection round.
        execs = [
            e for e in get_executor.__globals__["_EXECUTORS"].values()
            if isinstance(e, _InjectingExecutor)
        ]
        assert len(execs) == 1
        execs[0]._calls = 0  # reset so the next chain run also injects
        ctx2 = ReasoningContext(outer_context="x", api=_StubLLM())
        result2 = await chain.execute_async(ctx2)
        assert len(result2.step_results) == 2
    finally:
        _restore_llm_executor(original)


# --------------------------------------------------------------------------- #
# Validation: malformed injected_steps entries
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_invalid_injected_step_object_raises_typed_error() -> None:
    """An ``injected_steps`` entry without ``.number`` / ``.dependencies`` is rejected."""

    class _NotAStep:
        pass

    original = _install_injector([_NotAStep()])
    try:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="root", aim="x")],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        with pytest.raises(ValueError, match="invalid injected_steps entry"):
            await chain.execute_async(ctx)
    finally:
        _restore_llm_executor(original)


# --------------------------------------------------------------------------- #
# Default budget = 50 — never triggers for sane chains
# --------------------------------------------------------------------------- #


def test_default_max_injections_is_50() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="x", aim="x")],
        max_workers=1,
    )
    assert chain.max_injections == 50
    assert chain.executor.max_injections == 50


@pytest.mark.asyncio
async def test_no_injection_when_field_empty() -> None:
    """A normal step that doesn't populate ``injected_steps`` runs as usual."""
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="root", aim="x")],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_StubLLM())
    result = await chain.execute_async(ctx)
    # Only the declared step ran
    assert len(result.step_results) == 1
    assert result.step_results[0].step_title == "root"


# --------------------------------------------------------------------------- #
# Tool step doesn't inject by default
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_step_default_does_not_inject() -> None:
    """Regression guard: existing step executors never accidentally inject."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="run",
                config=ToolStepConfig(tool_name="t", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_StubLLM())
    ctx.register_tool("t", lambda: "ok")
    result = await chain.execute_async(ctx)
    assert len(result.step_results) == 1
    assert result.step_results[0].injected_steps == []
