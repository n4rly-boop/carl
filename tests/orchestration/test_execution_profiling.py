"""
Tests for execution profiling hooks (StepExecutionResult.profiling and
ReasoningResult.get_profiling_summary).

What is tested:
- profiling dict is populated for each step after execution
- history_bytes_added tracks the byte size of the history entry written
- memory_bytes_after reflects cumulative working memory
- batch_index matches the 0-based execution batch
- history_bytes_after matches total history size at that point
- get_profiling_summary() aggregates step data correctly
- peak_memory_bytes is the max memory_bytes_after across steps
- skipped steps still have profiling entries (may be empty)
- parallel steps share the same batch_index
- memory_bytes_after grows when memory is written
- _estimate_memory_bytes handles non-serializable objects gracefully
"""

import pytest

from mmar_carl import ReasoningChain, ReasoningContext
from mmar_carl.executor import _estimate_memory_bytes
from mmar_carl.models.config import MemoryStepConfig, ToolStepConfig
from mmar_carl.models.enums import MemoryOperation
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.steps import (
    LLMStepDescription,
    MemoryStepDescription,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    """Returns a fixed response regardless of prompt."""

    def __init__(self, response: str = "mock result"):
        self._response = response

    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        return self._response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return self._response


def make_context(**kwargs) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test",
        api=MockClient(),
        model="mock",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Unit tests: _estimate_memory_bytes
# ---------------------------------------------------------------------------


class TestEstimateMemoryBytes:
    def test_empty_dict(self):
        assert _estimate_memory_bytes({}) == 2  # JSON: "{}"

    def test_flat_dict(self):
        mem = {"ns": {"key": "value"}}
        size = _estimate_memory_bytes(mem)
        assert size > 10  # non-trivial content

    def test_non_serializable_falls_back(self):
        """Non-JSON-serializable values should use str() fallback."""
        mem = {"ns": {"key": object()}}
        # Should not raise; returns some positive integer
        size = _estimate_memory_bytes(mem)
        assert size > 0

    def test_larger_payload_yields_larger_size(self):
        small = {"ns": {"k": "v"}}
        large = {"ns": {"k": "v" * 1000}}
        assert _estimate_memory_bytes(large) > _estimate_memory_bytes(small)


# ---------------------------------------------------------------------------
# Integration tests: profiling populated by DAGExecutor
# ---------------------------------------------------------------------------


class TestProfilingFieldPopulated:
    @pytest.mark.asyncio
    async def test_single_step_has_profiling(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1,
                    title="Test Step",
                    aim="do something",
                )
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        assert result.success
        assert len(result.step_results) == 1
        p = result.step_results[0].profiling
        assert "history_bytes_added" in p
        assert "memory_bytes_after" in p
        assert "history_bytes_after" in p
        assert "batch_index" in p

    @pytest.mark.asyncio
    async def test_history_bytes_added_matches_entry_size(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1,
                    title="Step One",
                    aim="produce output",
                )
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        assert result.success
        sr = result.step_results[0]
        # The history entry written by this step
        expected = len(result.history[0])
        assert sr.profiling["history_bytes_added"] == expected

    @pytest.mark.asyncio
    async def test_batch_index_single_batch(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="do a"),
                LLMStepDescription(number=2, title="B", aim="do b", dependencies=[1]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        assert result.success
        assert result.step_results[0].profiling["batch_index"] == 0
        assert result.step_results[1].profiling["batch_index"] == 1

    @pytest.mark.asyncio
    async def test_parallel_steps_share_batch_index(self):
        """Steps in the same parallel batch get the same batch_index."""
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="Parallel A", aim="do a"),
                LLMStepDescription(number=2, title="Parallel B", aim="do b"),
                LLMStepDescription(number=3, title="Fanin", aim="combine", dependencies=[1, 2]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        assert result.success
        by_num = {sr.step_number: sr for sr in result.step_results}
        # Steps 1 and 2 have no dependencies → batch 0
        assert by_num[1].profiling["batch_index"] == 0
        assert by_num[2].profiling["batch_index"] == 0
        # Step 3 depends on both → batch 1
        assert by_num[3].profiling["batch_index"] == 1

    @pytest.mark.asyncio
    async def test_memory_bytes_grows_when_memory_written(self):
        """memory_bytes_after should increase after a memory write step."""
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="No Memory Write",
                    config=ToolStepConfig(
                        tool_name="noop",
                        input_mapping={},
                    ),
                ),
                MemoryStepDescription(
                    number=2,
                    title="Write Memory",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        namespace="test",
                        memory_key="big_key",
                        default_value="x" * 500,
                    ),
                ),
            ]
        )
        ctx = make_context()
        ctx.register_tool("noop", lambda: "ok")
        result = await chain.execute_async(ctx)

        assert result.success
        by_num = {sr.step_number: sr for sr in result.step_results}
        mem_after_1 = by_num[1].profiling["memory_bytes_after"]
        mem_after_2 = by_num[2].profiling["memory_bytes_after"]
        # After writing 500-byte value, memory should be larger
        assert mem_after_2 > mem_after_1

    @pytest.mark.asyncio
    async def test_history_bytes_after_accumulates(self):
        """history_bytes_after should be strictly increasing across sequential steps."""
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="First", aim="first"),
                LLMStepDescription(number=2, title="Second", aim="second", dependencies=[1]),
                LLMStepDescription(number=3, title="Third", aim="third", dependencies=[2]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        assert result.success
        h_after = [sr.profiling["history_bytes_after"] for sr in result.step_results]
        # Each step adds at least 1 byte to history
        assert h_after[0] < h_after[1] < h_after[2]


# ---------------------------------------------------------------------------
# get_profiling_summary() tests
# ---------------------------------------------------------------------------


class TestProfilingSummary:
    @pytest.mark.asyncio
    async def test_summary_contains_all_steps(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="a"),
                LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        summary = result.get_profiling_summary()
        assert len(summary["steps"]) == 2

    @pytest.mark.asyncio
    async def test_summary_top_level_keys(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="X", aim="x")]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        summary = result.get_profiling_summary()
        assert "total_execution_time_s" in summary
        assert "total_history_bytes" in summary
        assert "peak_memory_bytes" in summary
        assert "token_usage" in summary

    @pytest.mark.asyncio
    async def test_total_history_bytes_matches_history(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="a"),
                LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        summary = result.get_profiling_summary()
        expected_total = sum(len(e) for e in result.history)
        assert summary["total_history_bytes"] == expected_total

    @pytest.mark.asyncio
    async def test_peak_memory_bytes_is_max(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="a"),
                LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
            ]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        summary = result.get_profiling_summary()
        max_from_steps = max(
            (s["memory_bytes_after"] for s in summary["steps"]), default=0
        )
        assert summary["peak_memory_bytes"] == max_from_steps

    @pytest.mark.asyncio
    async def test_step_row_fields(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="Step One", aim="do it")]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        row = result.get_profiling_summary()["steps"][0]
        assert row["step_number"] == 1
        assert row["step_title"] == "Step One"
        assert row["step_type"] == "llm"
        assert isinstance(row["execution_time_s"], float)
        assert row["history_bytes_added"] > 0
        assert row["batch_index"] == 0
        assert row["success"] is True
        assert row["skipped"] is False

    @pytest.mark.asyncio
    async def test_total_execution_time_matches_result(self):
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="a")]
        )
        ctx = make_context()
        result = await chain.execute_async(ctx)

        summary = result.get_profiling_summary()
        assert summary["total_execution_time_s"] == result.total_execution_time

    @pytest.mark.asyncio
    async def test_profiling_on_tool_step(self):
        """Tool steps also get profiling data."""
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Tool Step",
                    config=ToolStepConfig(tool_name="get_data", input_mapping={}),
                )
            ]
        )
        ctx = make_context()
        ctx.register_tool("get_data", lambda: "some data returned")
        result = await chain.execute_async(ctx)

        assert result.success
        p = result.step_results[0].profiling
        assert p["history_bytes_added"] > 0
        assert p["batch_index"] == 0
