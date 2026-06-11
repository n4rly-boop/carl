"""
Tests for ChainTestHarness — deterministic unit testing for CARL chains.

Covers:
- set_step_response / LLM mock per step number
- set_tool_response / tool mock
- default_response fallback
- assert_step_called / assert_step_not_called
- assert_history_contains / assert_history_not_contains
- assert_memory_contains / assert_memory_equals
- assert_succeeded / assert_failed
- get_step_result
- result and context properties
- multiple steps with dependencies
- conditional branch skipping
- pre-populated memory input via run(memory=...)
"""

import pytest

from mmar_carl import (
    Language,
    ReasoningChain,
    ChainTestHarness,
)
from mmar_carl.models.steps import (
    LLMStepDescription,
    ToolStepDescription,
    MemoryStepDescription,
    ConditionalStepDescription,
)
from mmar_carl.models.config import (
    ToolStepConfig,
    MemoryStepConfig,
    ConditionalStepConfig,
    ConditionalBranch,
)
from mmar_carl.models.enums import MemoryOperation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simple_llm_chain(*step_count: int) -> ReasoningChain:
    """Create a linear LLM chain with N steps."""
    steps = [
        LLMStepDescription(
            number=i + 1,
            title=f"Step {i + 1}",
            aim="Do something",
            dependencies=[i] if i > 0 else [],
        )
        for i in range(step_count[0] if step_count else 1)
    ]
    return ReasoningChain(steps=steps)


# ---------------------------------------------------------------------------
# Basic mock LLM response
# ---------------------------------------------------------------------------


class TestStepResponse:
    @pytest.mark.asyncio
    async def test_set_step_response_used_in_history(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "mocked output")
        result = await harness.run("context")
        assert result.success
        assert any("mocked output" in e for e in result.history)

    @pytest.mark.asyncio
    async def test_default_response_when_no_step_response_set(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain, default_response="fallback")
        result = await harness.run("context")
        assert result.success
        assert any("fallback" in e for e in result.history)

    @pytest.mark.asyncio
    async def test_per_step_responses_independent(self):
        chain = simple_llm_chain(2)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "first answer")
        harness.set_step_response(2, "second answer")
        result = await harness.run("ctx")
        assert result.success
        assert any("first answer" in e for e in result.history)
        assert any("second answer" in e for e in result.history)

    @pytest.mark.asyncio
    async def test_chained_set_step_response_returns_self(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        returned = harness.set_step_response(1, "x")
        assert returned is harness


# ---------------------------------------------------------------------------
# Tool mock
# ---------------------------------------------------------------------------


class TestToolResponse:
    @pytest.mark.asyncio
    async def test_tool_response_returned_in_step(self):
        step = ToolStepDescription(
            number=1,
            title="Lookup",
            config=ToolStepConfig(tool_name="lookup", input_mapping={}),
        )
        chain = ReasoningChain(steps=[step])
        harness = ChainTestHarness(chain)
        harness.set_tool_response("lookup", "tool result value")
        result = await harness.run("ctx")
        assert result.success
        assert any("tool result value" in e for e in result.history)

    @pytest.mark.asyncio
    async def test_tool_response_dict_returned(self):
        step = ToolStepDescription(
            number=1,
            title="Fetch",
            config=ToolStepConfig(tool_name="fetch", input_mapping={}),
        )
        chain = ReasoningChain(steps=[step])
        harness = ChainTestHarness(chain)
        harness.set_tool_response("fetch", {"key": "val"})
        result = await harness.run("")
        assert result.success
        step_result = result.get_step_result(1)
        assert step_result is not None
        assert "key" in step_result.result
        assert "val" in step_result.result

    @pytest.mark.asyncio
    async def test_chained_set_tool_response_returns_self(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        returned = harness.set_tool_response("t", 42)
        assert returned is harness


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


class TestAssertions:
    @pytest.mark.asyncio
    async def test_assert_step_called_passes_for_run_step(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        harness.assert_step_called(1)

    @pytest.mark.asyncio
    async def test_assert_step_called_fails_for_unrun_step(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        with pytest.raises(AssertionError, match="Step 99 was not called"):
            harness.assert_step_called(99)

    @pytest.mark.asyncio
    async def test_assert_step_not_called_passes_for_unrun_step(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        harness.assert_step_not_called(99)

    @pytest.mark.asyncio
    async def test_assert_step_not_called_fails_for_run_step(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        with pytest.raises(AssertionError):
            harness.assert_step_not_called(1)

    @pytest.mark.asyncio
    async def test_assert_history_contains_passes(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "expected phrase")
        await harness.run("")
        harness.assert_history_contains("expected phrase")

    @pytest.mark.asyncio
    async def test_assert_history_contains_fails_when_absent(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "something else")
        await harness.run("")
        with pytest.raises(AssertionError, match="missing phrase"):
            harness.assert_history_contains("missing phrase")

    @pytest.mark.asyncio
    async def test_assert_history_not_contains_passes(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "only this")
        await harness.run("")
        harness.assert_history_not_contains("not here")

    @pytest.mark.asyncio
    async def test_assert_succeeded_passes(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        harness.assert_succeeded()

    @pytest.mark.asyncio
    async def test_assert_failed_passes_on_failure(self):
        # A tool step with an unregistered tool will fail
        step = ToolStepDescription(
            number=1,
            title="Broken",
            config=ToolStepConfig(tool_name="nonexistent_tool", input_mapping={}),
        )
        chain = ReasoningChain(steps=[step])
        harness = ChainTestHarness(chain)
        result = await harness.run("")
        assert not result.success
        harness.assert_failed()

    @pytest.mark.asyncio
    async def test_assert_succeeded_fails_on_failure(self):
        step = ToolStepDescription(
            number=1,
            title="Broken",
            config=ToolStepConfig(tool_name="nonexistent_tool", input_mapping={}),
        )
        chain = ReasoningChain(steps=[step])
        harness = ChainTestHarness(chain)
        await harness.run("")
        with pytest.raises(AssertionError):
            harness.assert_succeeded()


# ---------------------------------------------------------------------------
# Memory assertions
# ---------------------------------------------------------------------------


class TestMemoryAssertions:
    @pytest.mark.asyncio
    async def test_assert_memory_contains_written_by_step(self):
        steps = [
            ToolStepDescription(
                number=1,
                title="Produce",
                config=ToolStepConfig(tool_name="produce", input_mapping={}),
            ),
            MemoryStepDescription(
                number=2,
                title="Store",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    namespace="output",
                    memory_key="result",
                    value_source="$history[-1]",
                ),
                dependencies=[1],
            ),
        ]
        chain = ReasoningChain(steps=steps)
        harness = ChainTestHarness(chain)
        harness.set_tool_response("produce", "stored_content")
        await harness.run("")
        harness.assert_memory_contains("output", "result", contains="stored_content")

    @pytest.mark.asyncio
    async def test_assert_memory_equals(self):
        steps = [
            ToolStepDescription(
                number=1,
                title="Produce",
                config=ToolStepConfig(tool_name="num_tool", input_mapping={}),
            ),
            MemoryStepDescription(
                number=2,
                title="Store",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    namespace="ns",
                    memory_key="x",
                    value_source="$history[-1]",
                ),
                dependencies=[1],
            ),
        ]
        chain = ReasoningChain(steps=steps)
        harness = ChainTestHarness(chain)
        harness.set_tool_response("num_tool", 42)
        await harness.run("")
        # value_source="$history[-1]" stores the full history entry string
        harness.assert_memory_contains("ns", "x", contains="42")

    @pytest.mark.asyncio
    async def test_memory_contains_fails_when_absent(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("")
        with pytest.raises(AssertionError, match="None or missing"):
            harness.assert_memory_contains("nonexistent", "key", contains="val")

    @pytest.mark.asyncio
    async def test_run_with_pre_populated_memory(self):
        """Memory pre-populated via run(memory=...) is accessible to steps."""
        step = MemoryStepDescription(
            number=1,
            title="Read",
            config=MemoryStepConfig(
                operation=MemoryOperation.READ,
                namespace="input",
                memory_key="data",
            ),
        )
        chain = ReasoningChain(steps=[step])
        harness = ChainTestHarness(chain)
        result = await harness.run("", memory={"input": {"data": "seeded_value"}})
        assert result.success
        harness.assert_history_contains("seeded_value")


# ---------------------------------------------------------------------------
# get_step_result and properties
# ---------------------------------------------------------------------------


class TestResultProperties:
    @pytest.mark.asyncio
    async def test_get_step_result_returns_correct_step(self):
        chain = simple_llm_chain(2)
        harness = ChainTestHarness(chain)
        harness.set_step_response(1, "step1_out")
        harness.set_step_response(2, "step2_out")
        await harness.run("")
        sr = harness.get_step_result(2)
        assert sr is not None
        assert sr.step_number == 2
        assert sr.result == "step2_out"

    @pytest.mark.asyncio
    async def test_result_property_available_after_run(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        result = await harness.run("")
        assert harness.result is result

    @pytest.mark.asyncio
    async def test_context_property_available_after_run(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        await harness.run("my input")
        assert harness.context is not None
        assert harness.context.outer_context == "my input"

    def test_result_and_context_none_before_run(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain)
        assert harness.result is None
        assert harness.context is None


# ---------------------------------------------------------------------------
# Conditional branch skipping
# ---------------------------------------------------------------------------


class TestConditionalBranchSkipping:
    @pytest.mark.asyncio
    async def test_skipped_branch_not_called(self):
        steps = [
            ToolStepDescription(
                number=1,
                title="Classify",
                config=ToolStepConfig(tool_name="classify", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route",
                config=ConditionalStepConfig(
                    branches=[
                        ConditionalBranch(condition="'A' in value", next_step=3),
                    ],
                    default_step=4,
                    condition_context_key="$history[-1]",
                ),
                dependencies=[1],
            ),
            LLMStepDescription(
                number=3,
                title="Branch A",
                aim="branch A",
                dependencies=[2],
            ),
            LLMStepDescription(
                number=4,
                title="Branch B",
                aim="branch B",
                dependencies=[2],
            ),
        ]
        chain = ReasoningChain(steps=steps)
        harness = ChainTestHarness(chain)
        # Classify returns 'A' → step 3 runs, step 4 skipped
        harness.set_tool_response("classify", "A")
        harness.set_step_response(3, "branch A output")
        result = await harness.run("ctx")
        assert result.success
        harness.assert_step_called(3)
        harness.assert_step_not_called(4)

    @pytest.mark.asyncio
    async def test_language_russian_works(self):
        chain = simple_llm_chain(1)
        harness = ChainTestHarness(chain, language=Language.RUSSIAN)
        harness.set_step_response(1, "результат")
        result = await harness.run("контекст")
        assert result.success
        assert any("результат" in e for e in result.history)
