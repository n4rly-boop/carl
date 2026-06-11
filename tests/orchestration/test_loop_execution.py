"""
Tests for loop/cycle support in the DAG executor and ChainBuilder helpers.

Covers:
- LoopConfig model fields and defaults
- loop_back_to / loop_config fields on StepDescriptionBase
- Basic loop: steps re-executed when condition is truthy
- Loop exits when condition becomes falsy
- Budget guard (max_iterations)
- Always-loop (no condition_key) stops at max_iterations
- Multi-step loop body
- Memory state accumulates across loop iterations
- Loop iteration history recorded in metadata
- Loop followed by downstream steps
- ChainBuilder.add_while_loop / add_until_loop helpers
- negate_condition (until semantics)
"""

import pytest

from mmar_carl import (
    ChainBuilder,
    Language,
    LLMClientBase,
    LoopConfig,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLMClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "llm ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _make_context(memory=None) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context="test",
        api=_MockLLMClient(),
        model="test",
        language=Language.ENGLISH,
    )
    if memory:
        for ns, pairs in memory.items():
            for key, val in pairs.items():
                ctx.memory_write(key, val, namespace=ns)
    return ctx


def _tool_step(number: int, tool_name: str, deps=None, loop_back_to=None, loop_config=None) -> ToolStepDescription:
    return ToolStepDescription(
        number=number,
        title=f"Step {number}",
        dependencies=deps or [],
        config=ToolStepConfig(tool_name=tool_name, input_mapping={}),
        loop_back_to=loop_back_to,
        loop_config=loop_config,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.condition_key == ""
        assert cfg.max_iterations == 10

    def test_custom_values(self):
        cfg = LoopConfig(condition_key="$memory.loop.go", max_iterations=5)
        assert cfg.condition_key == "$memory.loop.go"
        assert cfg.max_iterations == 5

    def test_max_iterations_must_be_positive(self):
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            LoopConfig(max_iterations=0)

    def test_loop_fields_on_step(self):
        step = _tool_step(1, "worker", loop_back_to=1, loop_config=LoopConfig(max_iterations=3))
        assert step.loop_back_to == 1
        assert step.loop_config is not None
        assert step.loop_config.max_iterations == 3

    def test_loop_fields_default_none(self):
        step = _tool_step(1, "worker")
        assert step.loop_back_to is None
        assert step.loop_config is None


# ---------------------------------------------------------------------------
# Basic loop: step re-executed when condition is truthy
# ---------------------------------------------------------------------------


class TestBasicLoop:
    @pytest.mark.asyncio
    async def test_single_step_loop_runs_multiple_times(self):
        """A 1-step loop with empty condition runs max_iterations+1 times total."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return f"iteration {call_count[0]}"

        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=2),
                )
            ]
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        # Initial run + 2 re-executions = 3 total
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_loop_exits_when_condition_false_from_start(self):
        """If condition is already False, the step runs exactly once (no loop)."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "done"

        ctx = _make_context(memory={"loop": {"go": False}})
        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="$memory.loop.go", max_iterations=5),
                )
            ]
        )
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_loop_exits_when_condition_truthy_memory(self):
        """condition_key pointing to truthy memory value causes looping (budget stops it)."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        ctx = _make_context(memory={"loop": {"go": True}})
        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="$memory.loop.go", max_iterations=2),
                )
            ]
        )
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        # Condition stays True (snapshot merge keeps it True), so budget stops at max_iterations+1
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# Budget guard (max_iterations)
# ---------------------------------------------------------------------------


class TestBudgetGuard:
    @pytest.mark.asyncio
    async def test_loop_stops_at_max_iterations(self):
        """With always-truthy condition, loop runs exactly max_iterations+1 times total."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        # condition_key="" means always loop → budget guard kicks in after max_iterations
        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=3),
                )
            ]
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        # initial run + 3 re-executions = 4 total calls
        assert call_count[0] == 4

    @pytest.mark.asyncio
    async def test_max_iterations_one(self):
        """max_iterations=1 → initial run + exactly 1 re-execution."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=1),
                )
            ]
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        await chain.execute_async(ctx)
        assert call_count[0] == 2


# ---------------------------------------------------------------------------
# No loop_config: no looping
# ---------------------------------------------------------------------------


class TestNoLoop:
    @pytest.mark.asyncio
    async def test_step_without_loop_back_to_runs_once(self):
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "once"

        chain = ReasoningChain(steps=[_tool_step(1, "worker")])
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Multi-step loop body
# ---------------------------------------------------------------------------


class TestMultiStepLoop:
    @pytest.mark.asyncio
    async def test_two_step_loop_body(self):
        """Steps 1-2 form the loop body; both steps re-execute on each iteration."""
        calls = {"step1": 0, "step2": 0}

        def step1():
            calls["step1"] += 1
            return "s1"

        def step2():
            calls["step2"] += 1
            return "s2"

        ctx = _make_context()
        ctx.register_tool("step1", step1)
        ctx.register_tool("step2", step2)

        chain = ReasoningChain(
            steps=[
                _tool_step(1, "step1"),
                _tool_step(
                    2, "step2",
                    deps=[1],
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=1),
                ),
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # Initial run + 1 re-execution = 2 of each step
        assert calls["step1"] == 2
        assert calls["step2"] == 2

    @pytest.mark.asyncio
    async def test_loop_body_executes_n_times(self):
        """Loop body runs exactly max_iterations+1 times (budget guard as exit)."""
        call_count = [0]

        def accumulator():
            call_count[0] += 1
            return f"iter {call_count[0]}"

        ctx = _make_context()
        ctx.register_tool("acc", accumulator)
        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "acc",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=4),
                )
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 5  # initial + 4 re-executions


# ---------------------------------------------------------------------------
# Loop iteration history in metadata
# ---------------------------------------------------------------------------


class TestLoopIterationHistory:
    @pytest.mark.asyncio
    async def test_iteration_history_recorded(self):
        """loop_iteration_history is populated in chain metadata after looping."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "worker",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=2),
                )
            ]
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        await chain.execute_async(ctx)
        history = ctx.metadata.get("loop_iteration_history", {})
        # Key is "1-1" for loop_back_to=1, tail=1
        assert "1-1" in history
        assert len(history["1-1"]) == 2  # 2 re-executions recorded


# ---------------------------------------------------------------------------
# Downstream steps after a loop
# ---------------------------------------------------------------------------


class TestDownstreamAfterLoop:
    @pytest.mark.asyncio
    async def test_step_after_loop_runs_once(self):
        """A step that depends on the loop tail runs exactly once after the loop exits."""
        calls = {"looper": 0, "after": 0}

        def looper():
            calls["looper"] += 1
            return "loop"

        def after():
            calls["after"] += 1
            return "after"

        ctx = _make_context()
        ctx.register_tool("looper", looper)
        ctx.register_tool("after", after)

        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "looper",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="", max_iterations=2),
                ),
                _tool_step(2, "after", deps=[1]),
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert calls["looper"] == 3  # initial + 2 re-executions
        assert calls["after"] == 1  # runs only once after loop exits

    @pytest.mark.asyncio
    async def test_result_of_final_loop_iteration_visible_to_downstream(self):  # noqa: E501
        """The downstream step sees the result of the last loop iteration in history."""
        iteration = [0]
        seen_history = []

        def looper():
            iteration[0] += 1
            if iteration[0] >= 2:
                ctx.memory_write("go", False, namespace="loop")
            return f"iter-{iteration[0]}"

        def reader():
            seen_history.extend(ctx.history)
            return "read"

        ctx = _make_context(memory={"loop": {"go": True}})
        ctx.register_tool("looper", looper)
        ctx.register_tool("reader", reader)

        chain = ReasoningChain(
            steps=[
                _tool_step(
                    1, "looper",
                    loop_back_to=1,
                    loop_config=LoopConfig(condition_key="$memory.loop.go", max_iterations=5),
                ),
                _tool_step(2, "reader", deps=[1]),
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # The final iteration's result should be in history seen by step 2
        combined = "\n".join(seen_history)
        assert "iter-2" in combined


# ---------------------------------------------------------------------------
# ChainBuilder.add_while_loop / add_until_loop
# ---------------------------------------------------------------------------


class TestChainBuilderLoopHelpers:
    @pytest.mark.asyncio
    async def test_add_while_loop_basic(self):
        """add_while_loop runs body steps the expected number of times."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        chain = (
            ChainBuilder()
            .add_while_loop(
                [ToolStepDescription(number=1, title="Worker", config=ToolStepConfig(tool_name="worker", input_mapping={}))],
                condition_key="",  # always loop
                max_iterations=2,
            )
            .build()
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 3  # initial + 2 re-executions

    @pytest.mark.asyncio
    async def test_add_until_loop_basic(self):
        """add_until_loop stops when condition becomes truthy."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        # condition_key="" → always falsy → loop forever → budget stops it
        chain = (
            ChainBuilder()
            .add_until_loop(
                [ToolStepDescription(number=1, title="Worker", config=ToolStepConfig(tool_name="worker", input_mapping={}))],
                condition_key="$memory.done.flag",  # reads False (no such key → None → falsy) → loops
                max_iterations=2,
            )
            .build()
        )
        ctx = _make_context()
        ctx.register_tool("worker", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 3  # budget guard stops it

    @pytest.mark.asyncio
    async def test_add_while_loop_auto_numbers_steps(self):
        """Builder renumbers body steps starting from next available number."""
        body = [ToolStepDescription(number=99, title="Step", config=ToolStepConfig(tool_name="w", input_mapping={}))]
        chain = ChainBuilder().add_while_loop(body, condition_key="", max_iterations=1).build()
        assert chain.steps[0].number == 1  # renumbered from 99 → 1

    @pytest.mark.asyncio
    async def test_add_while_loop_loop_back_to_set(self):
        """Builder sets loop_back_to on the last body step."""
        step1 = ToolStepDescription(number=1, title="S1", config=ToolStepConfig(tool_name="t", input_mapping={}))
        step2 = ToolStepDescription(number=2, title="S2", config=ToolStepConfig(tool_name="t", input_mapping={}))
        chain = ChainBuilder().add_while_loop([step1, step2], condition_key="", max_iterations=1).build()
        assert chain.steps[-1].loop_back_to == 1
        assert chain.steps[-1].loop_config is not None
        assert chain.steps[-1].loop_config.negate_condition is False

    @pytest.mark.asyncio
    async def test_add_until_loop_negate_condition_set(self):
        """add_until_loop uses negate_condition=True in LoopConfig."""
        body = [ToolStepDescription(number=1, title="S", config=ToolStepConfig(tool_name="t", input_mapping={}))]
        chain = ChainBuilder().add_until_loop(body, condition_key="$memory.x.y", max_iterations=5).build()
        last = chain.steps[-1]
        assert last.loop_config is not None
        assert last.loop_config.negate_condition is True

    @pytest.mark.asyncio
    async def test_negate_condition_exits_when_truthy(self):
        """until-loop exits when memory value becomes truthy (condition is negated)."""
        call_count = [0]

        def worker():
            call_count[0] += 1
            return "ok"

        # condition_key points to value that's initially missing (falsy) → loop continues
        # We use max_iterations=1 to limit
        chain = (
            ChainBuilder()
            .add_until_loop(
                [ToolStepDescription(number=1, title="W", config=ToolStepConfig(tool_name="w", input_mapping={}))],
                condition_key="$memory.done.flag",
                max_iterations=3,
            )
            .build()
        )
        ctx = _make_context(memory={"done": {"flag": True}})  # already truthy → no looping
        ctx.register_tool("w", worker)
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1  # exits immediately (condition truthy → NOT truthy → don't loop)
