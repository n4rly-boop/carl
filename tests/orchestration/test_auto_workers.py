"""
Tests for max_workers="auto" parallel batch size auto-tuning.

Auto mode runs each batch with exactly len(batch) concurrent coroutines:
- A batch of 1 step runs sequentially (no task overhead).
- A batch of N parallel steps runs all N concurrently.
- Different batches in the same chain can have different concurrency levels.

This is safer than a fixed max_workers that either under-parallelises (too low)
or wastes goroutine slots (too high).
"""

import pytest

from mmar_carl import ReasoningChain, ReasoningContext
from mmar_carl.executor import DAGExecutor
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.steps import LLMStepDescription, ToolStepDescription
from mmar_carl.models.config import ToolStepConfig


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return "mock"


def make_context() -> ReasoningContext:
    return ReasoningContext(outer_context="test", api=MockClient(), model="mock")


# ---------------------------------------------------------------------------
# DAGExecutor unit tests
# ---------------------------------------------------------------------------


class TestDAGExecutorAutoWorkers:
    def test_auto_string_accepted(self):
        executor = DAGExecutor(max_workers="auto")
        assert executor.max_workers == "auto"

    def test_int_accepted(self):
        executor = DAGExecutor(max_workers=4)
        assert executor.max_workers == 4

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="max_workers must be"):
            DAGExecutor(max_workers="unlimited")  # type: ignore[arg-type]

    def test_default_is_one(self):
        executor = DAGExecutor()
        assert executor.max_workers == 1


# ---------------------------------------------------------------------------
# ReasoningChain integration
# ---------------------------------------------------------------------------


class TestReasoningChainAutoWorkers:
    def test_auto_accepted_by_chain(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="a")],
            max_workers="auto",
        )
        assert chain.max_workers == "auto"

    def test_int_accepted_by_chain(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="a")],
            max_workers=5,
        )
        assert chain.max_workers == 5

    def test_default_is_three(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="a")],
        )
        assert chain.max_workers == 3

    @pytest.mark.asyncio
    async def test_auto_single_step_succeeds(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="Step", aim="do something")],
            max_workers="auto",
        )
        result = await chain.execute_async(make_context())
        assert result.success

    @pytest.mark.asyncio
    async def test_auto_sequential_chain_succeeds(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="a"),
                LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
                LLMStepDescription(number=3, title="C", aim="c", dependencies=[2]),
            ],
            max_workers="auto",
        )
        result = await chain.execute_async(make_context())
        assert result.success
        assert len(result.history) == 3

    @pytest.mark.asyncio
    async def test_auto_parallel_fan_out_succeeds(self):
        """Auto mode should run 3 parallel steps all at once."""
        execution_order: list[str] = []

        def make_tool(name: str):
            def tool():
                execution_order.append(name)
                return f"result-{name}"
            return tool

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Branch A",
                    config=ToolStepConfig(tool_name="tool_a", input_mapping={}),
                ),
                ToolStepDescription(
                    number=2,
                    title="Branch B",
                    config=ToolStepConfig(tool_name="tool_b", input_mapping={}),
                ),
                ToolStepDescription(
                    number=3,
                    title="Branch C",
                    config=ToolStepConfig(tool_name="tool_c", input_mapping={}),
                ),
                ToolStepDescription(
                    number=4,
                    title="Fan-in",
                    dependencies=[1, 2, 3],
                    config=ToolStepConfig(tool_name="fanin", input_mapping={}),
                ),
            ],
            max_workers="auto",
        )
        ctx = make_context()
        ctx.register_tool("tool_a", make_tool("a"))
        ctx.register_tool("tool_b", make_tool("b"))
        ctx.register_tool("tool_c", make_tool("c"))
        ctx.register_tool("fanin", lambda: "combined")

        result = await chain.execute_async(ctx)
        assert result.success
        assert len(result.history) == 4
        # All three parallel branches must have executed
        assert "a" in execution_order
        assert "b" in execution_order
        assert "c" in execution_order

    @pytest.mark.asyncio
    async def test_auto_profiling_batch_indices(self):
        """With auto mode, batch indices should still be correct."""
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="P1", aim="p1"),
                LLMStepDescription(number=2, title="P2", aim="p2"),
                LLMStepDescription(number=3, title="Sequential", aim="seq", dependencies=[1, 2]),
            ],
            max_workers="auto",
        )
        result = await chain.execute_async(make_context())
        assert result.success
        by_num = {sr.step_number: sr for sr in result.step_results}
        assert by_num[1].profiling["batch_index"] == 0
        assert by_num[2].profiling["batch_index"] == 0
        assert by_num[3].profiling["batch_index"] == 1

    @pytest.mark.asyncio
    async def test_auto_serialization_round_trip(self):
        """max_workers='auto' survives to_dict / from_dict."""
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="a")],
            max_workers="auto",
        )
        data = chain.to_dict()
        assert data["max_workers"] == "auto"
        restored = ReasoningChain.from_dict(data)
        assert restored.max_workers == "auto"


# ---------------------------------------------------------------------------
# ChainBuilder integration
# ---------------------------------------------------------------------------


class TestChainBuilderAutoWorkers:
    def test_builder_with_max_workers_auto(self):
        from mmar_carl import ChainBuilder

        builder = ChainBuilder()
        returned = builder.with_max_workers("auto")
        assert returned is builder  # fluent interface
        assert builder.max_workers == "auto"

    def test_builder_with_max_workers_int(self):
        from mmar_carl import ChainBuilder

        builder = ChainBuilder()
        builder.with_max_workers(8)
        assert builder.max_workers == 8

    @pytest.mark.asyncio
    async def test_builder_auto_chain_executes(self):
        """Chain built with auto max_workers via ChainBuilder runs correctly."""
        from mmar_carl import ChainBuilder

        builder = ChainBuilder()
        builder.with_max_workers("auto")
        builder.steps = [
            LLMStepDescription(number=1, title="A", aim="a"),
            LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
        ]
        chain = ReasoningChain(steps=builder.steps, max_workers=builder.max_workers)
        result = await chain.execute_async(make_context())
        assert result.success
        assert chain.max_workers == "auto"
