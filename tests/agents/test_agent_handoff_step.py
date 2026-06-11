"""
Tests for AgentHandoffStep — explicit sub-chain delegation.

Covers:
- AgentHandoffStepConfig model fields and defaults
- AgentHandoffStepDescription model fields
- Successful sub-chain execution with result written to parent memory
- Input mapping resolves parent context references into sub-chain memory
- Tools inherited from parent (inherit_tools=True)
- Tools NOT inherited (inherit_tools=False)
- propagate_failure=True: sub-chain failure fails parent step
- propagate_failure=False: sub-chain failure → step still succeeds
- Timeout enforcement (asyncio.TimeoutError wrapped)
- Missing sub_chain raises graceful error
- Integration: parent chain uses sub-chain result from memory
"""

import asyncio
import pytest

from mmar_carl import (
    AgentHandoffStepConfig,
    AgentHandoffStepDescription,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.steps import ToolStepDescription
from mmar_carl.models.config import ToolStepConfig


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


def _simple_chain(tool_name: str) -> ReasoningChain:
    """Build a minimal chain with one tool step."""
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Do work",
                config=ToolStepConfig(tool_name=tool_name, input_mapping={}),
            )
        ]
    )


def _handoff_step(
    sub_chain: ReasoningChain,
    config: AgentHandoffStepConfig,
    number: int = 1,
) -> AgentHandoffStepDescription:
    return AgentHandoffStepDescription(
        number=number,
        title="Handoff",
        sub_chain=sub_chain,
        config=config,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TestAgentHandoffStepConfig:
    def test_defaults(self):
        cfg = AgentHandoffStepConfig()
        assert cfg.input_mapping == {}
        assert cfg.output_memory_key == ""
        assert cfg.output_namespace == "handoff"
        assert cfg.propagate_failure is True
        assert cfg.inherit_tools is True
        assert cfg.timeout is None

    def test_custom_values(self):
        cfg = AgentHandoffStepConfig(
            input_mapping={"input.topic": "$memory.input.topic"},
            output_memory_key="result",
            output_namespace="research",
            propagate_failure=False,
            inherit_tools=False,
            timeout=30.0,
        )
        assert cfg.input_mapping == {"input.topic": "$memory.input.topic"}
        assert cfg.output_memory_key == "result"
        assert cfg.output_namespace == "research"
        assert cfg.propagate_failure is False
        assert cfg.inherit_tools is False
        assert cfg.timeout == 30.0

    def test_step_type(self):
        from mmar_carl.models.enums import StepType
        step = AgentHandoffStepDescription(
            number=1,
            title="Delegate",
            sub_chain=_simple_chain("t"),
            config=AgentHandoffStepConfig(),
        )
        assert step.step_type == StepType.AGENT_HANDOFF


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------


class TestSuccessfulHandoff:
    @pytest.mark.asyncio
    async def test_sub_chain_runs_and_result_in_result_data(self):
        sub = _simple_chain("worker")
        chain = ReasoningChain(steps=[_handoff_step(sub, AgentHandoffStepConfig())])
        ctx = _make_context()
        ctx.register_tool("worker", lambda: "worker result")
        result = await chain.execute_async(ctx)
        assert result.success
        rd = result.step_results[0].result_data
        assert rd["sub_chain_success"] is True
        assert rd["steps_executed"] == 1

    @pytest.mark.asyncio
    async def test_result_written_to_parent_memory(self):
        sub = _simple_chain("researcher")
        config = AgentHandoffStepConfig(
            output_memory_key="answer",
            output_namespace="research",
        )
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("researcher", lambda: "research findings")
        result = await chain.execute_async(ctx)
        assert result.success
        # The sub-chain result is stored in parent memory
        stored = ctx.memory_read("answer", namespace="research")
        assert stored is not None
        assert "research findings" in str(stored)

    @pytest.mark.asyncio
    async def test_no_output_written_when_key_empty(self):
        sub = _simple_chain("worker")
        config = AgentHandoffStepConfig(output_memory_key="")
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("worker", lambda: "out")
        await chain.execute_async(ctx)
        # Nothing written to handoff namespace
        assert ctx.memory_read("", namespace="handoff") is None

    @pytest.mark.asyncio
    async def test_history_entry_added(self):
        sub = _simple_chain("worker")
        chain = ReasoningChain(steps=[_handoff_step(sub, AgentHandoffStepConfig())])
        ctx = _make_context()
        ctx.register_tool("worker", lambda: "done")
        result = await chain.execute_async(ctx)
        assert result.success
        history_text = "\n".join(result.history)
        assert "[HANDOFF]" in history_text


# ---------------------------------------------------------------------------
# Input mapping
# ---------------------------------------------------------------------------


class TestInputMapping:
    @pytest.mark.asyncio
    async def test_namespaced_key_written_to_sub_memory(self):
        """input.topic → sub-chain memory namespace 'input', key 'topic'."""

        def worker():
            return "done"

        # We need to inspect what the sub-chain context has.
        # Wrap the tool to record the sub-chain context state.
        # Since tools don't receive context, use a closure with a spy sub-chain.

        sub_ctx_capture = {}

        async def spy_execute(ctx):
            sub_ctx_capture["memory"] = dict(ctx.memory)
            return await _simple_chain("worker").execute_async(ctx)

        sub = _simple_chain("worker")
        original_execute = sub.execute_async

        async def patched_execute(ctx):
            sub_ctx_capture["topic"] = ctx.memory_read("topic", namespace="input")
            return await original_execute(ctx)

        sub.execute_async = patched_execute

        config = AgentHandoffStepConfig(
            input_mapping={"input.topic": "$memory.src.query"},
        )
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context(memory={"src": {"query": "AI research"}})
        ctx.register_tool("worker", worker)
        await chain.execute_async(ctx)
        assert sub_ctx_capture.get("topic") == "AI research"

    @pytest.mark.asyncio
    async def test_bare_key_uses_input_namespace(self):
        """A key without a dot is written to 'input' namespace."""
        sub_ctx_capture = {}

        sub = _simple_chain("worker")
        orig = sub.execute_async

        async def patched(ctx):
            sub_ctx_capture["val"] = ctx.memory_read("mykey", namespace="input")
            return await orig(ctx)

        sub.execute_async = patched

        config = AgentHandoffStepConfig(input_mapping={"mykey": "'hello'"})
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("worker", lambda: "ok")
        await chain.execute_async(ctx)
        assert sub_ctx_capture.get("val") == "hello"

    @pytest.mark.asyncio
    async def test_literal_value_in_mapping(self):
        """Quoted string literals resolved from parent input_mapping."""
        sub_ctx_capture = {}
        sub = _simple_chain("worker")
        orig = sub.execute_async

        async def patched(ctx):
            sub_ctx_capture["lang"] = ctx.memory_read("lang", namespace="input")
            return await orig(ctx)

        sub.execute_async = patched
        config = AgentHandoffStepConfig(input_mapping={"input.lang": "'english'"})
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("worker", lambda: "ok")
        await chain.execute_async(ctx)
        assert sub_ctx_capture.get("lang") == "english"


# ---------------------------------------------------------------------------
# Tool inheritance
# ---------------------------------------------------------------------------


class TestToolInheritance:
    @pytest.mark.asyncio
    async def test_inherit_tools_true(self):
        """Sub-chain can call parent-registered tools."""
        sub = _simple_chain("shared_tool")
        config = AgentHandoffStepConfig(inherit_tools=True)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("shared_tool", lambda: "from parent")
        result = await chain.execute_async(ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_inherit_tools_false_sub_chain_fails(self):
        """Sub-chain without tools fails if it needs a tool."""
        sub = _simple_chain("parent_only_tool")
        config = AgentHandoffStepConfig(inherit_tools=False, propagate_failure=True)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("parent_only_tool", lambda: "result")
        result = await chain.execute_async(ctx)
        # Sub-chain can't find the tool → fails → parent step fails
        assert not result.success


# ---------------------------------------------------------------------------
# propagate_failure
# ---------------------------------------------------------------------------


class TestPropagateFailure:
    @pytest.mark.asyncio
    async def test_propagate_true_fails_parent(self):
        """Sub-chain failure propagates to parent step by default."""
        sub = _simple_chain("bad_tool")
        config = AgentHandoffStepConfig(propagate_failure=True)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()

        def bad_tool():
            raise RuntimeError("intentional failure")

        ctx.register_tool("bad_tool", bad_tool)
        result = await chain.execute_async(ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_propagate_false_parent_succeeds(self):
        """Sub-chain failure does NOT fail parent when propagate_failure=False."""
        sub = _simple_chain("bad_tool")
        config = AgentHandoffStepConfig(propagate_failure=False)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()

        def bad_tool():
            raise RuntimeError("intentional failure")

        ctx.register_tool("bad_tool", bad_tool)
        result = await chain.execute_async(ctx)
        assert result.success
        rd = result.step_results[0].result_data
        assert rd["sub_chain_success"] is False

    @pytest.mark.asyncio
    async def test_propagate_false_result_data_has_sub_result(self):
        """result_data['sub_result'] is a ReasoningResult even on sub-chain failure."""
        from mmar_carl.models.results import ReasoningResult

        sub = _simple_chain("bad_tool")
        config = AgentHandoffStepConfig(propagate_failure=False)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()

        def bad_tool():
            raise RuntimeError("fail")

        ctx.register_tool("bad_tool", bad_tool)
        result = await chain.execute_async(ctx)
        rd = result.step_results[0].result_data
        assert isinstance(rd["sub_result"], ReasoningResult)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_fails_step(self):
        """Sub-chain timeout causes step failure."""
        async def slow():
            await asyncio.sleep(999)
            return "never"

        sub = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Slow",
                    config=ToolStepConfig(tool_name="slow", input_mapping={}),
                )
            ]
        )
        config = AgentHandoffStepConfig(timeout=0.01)
        chain = ReasoningChain(steps=[_handoff_step(sub, config)])
        ctx = _make_context()
        ctx.register_tool("slow", slow)
        result = await chain.execute_async(ctx)
        assert not result.success
        assert "timed out" in result.step_results[0].error_message.lower()


# ---------------------------------------------------------------------------
# Missing sub_chain
# ---------------------------------------------------------------------------


class TestMissingSubChain:
    @pytest.mark.asyncio
    async def test_none_sub_chain_fails_gracefully(self):
        """sub_chain=None gives a clear error."""

        # Build the description with sub_chain set, then unset it at runtime
        step = AgentHandoffStepDescription(
            number=1,
            title="Delegate",
            sub_chain=_simple_chain("t"),  # set to satisfy Field(...)
            config=AgentHandoffStepConfig(),
        )
        step.sub_chain = None  # unset at runtime

        chain = ReasoningChain(steps=[step])
        ctx = _make_context()
        result = await chain.execute_async(ctx)
        assert not result.success
        assert "sub_chain" in result.step_results[0].error_message.lower()


# ---------------------------------------------------------------------------
# Integration: parent chain uses sub-chain result
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_parent_chain_reads_handoff_result_from_memory(self):
        """
        Chain: handoff step writes result to memory → tool step reads it.
        """

        # Sub-chain produces "research data"
        sub = _simple_chain("researcher")

        handoff = AgentHandoffStepDescription(
            number=1,
            title="Research agent",
            sub_chain=sub,
            config=AgentHandoffStepConfig(
                output_memory_key="data",
                output_namespace="research",
            ),
        )
        # Step 2: read from memory and produce final answer
        reader = ToolStepDescription(
            number=2,
            title="Combine results",
            dependencies=[1],
            config=ToolStepConfig(tool_name="combiner", input_mapping={}),
        )

        chain = ReasoningChain(steps=[handoff, reader])
        ctx = _make_context()
        ctx.register_tool("researcher", lambda: "research data")
        ctx.register_tool("combiner", lambda: "combined answer")
        result = await chain.execute_async(ctx)
        assert result.success
        assert result.step_results[1].result == "combined answer"
        # Sub-chain result is in memory
        assert ctx.memory_read("data", namespace="research") is not None

    @pytest.mark.asyncio
    async def test_nested_handoff_two_levels(self):
        """A handoff step inside a handoff step — two levels of delegation."""
        inner_sub = _simple_chain("inner_worker")
        inner_handoff = AgentHandoffStepDescription(
            number=1,
            title="Inner handoff",
            sub_chain=inner_sub,
            config=AgentHandoffStepConfig(
                output_memory_key="inner_result",
                output_namespace="inner",
            ),
        )
        outer_sub = ReasoningChain(steps=[inner_handoff])

        outer_handoff = AgentHandoffStepDescription(
            number=1,
            title="Outer handoff",
            sub_chain=outer_sub,
            config=AgentHandoffStepConfig(
                output_memory_key="outer_result",
                output_namespace="outer",
            ),
        )
        chain = ReasoningChain(steps=[outer_handoff])
        ctx = _make_context()
        ctx.register_tool("inner_worker", lambda: "deep result")
        result = await chain.execute_async(ctx)
        assert result.success
