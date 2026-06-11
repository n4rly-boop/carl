"""
Tests for HumanInputStep feature.

Covers:
  - HumanInputStepConfig defaults and validation
  - HumanInputStepExecutor: no callback → fallback value immediately
  - HumanInputStepExecutor: sync callback resolved before await
  - HumanInputStepExecutor: async callback, provide_human_input resolves future
  - HumanInputStepExecutor: timeout → fallback value
  - HumanInputStepExecutor: output_memory_key written
  - HumanInputStepExecutor: Russian language history entry
  - ReasoningContext.provide_human_input: resolves pending future
  - ReasoningContext.provide_human_input: returns False when no pending future
  - Integration: HumanInputStep in a full chain execution
"""

import asyncio

import pytest

from mmar_carl import (
    HumanInputStepConfig,
    HumanInputStepDescription,
    HumanInputStepExecutor,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.enums import StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_step(
    prompt="Enter your feedback:",
    timeout=None,
    fallback_value="[no input]",
    output_memory_key=None,
):
    """Return a minimal HumanInputStepDescription."""
    return HumanInputStepDescription(
        number=1,
        title="Wait for user",
        config=HumanInputStepConfig(
            prompt=prompt,
            timeout=timeout,
            fallback_value=fallback_value,
            output_memory_key=output_memory_key,
        ),
    )


def make_context(language=Language.ENGLISH):
    """Return a real ReasoningContext with a mock LLM client."""

    class MockClient(LLMClientBase):
        async def get_response(self, prompt): return "ok"
        async def get_response_with_retries(self, prompt, retries=3): return "ok"

    return ReasoningContext(
        outer_context="test",
        api=MockClient(),
        language=language,
    )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestHumanInputStepConfig:
    def test_defaults(self):
        config = HumanInputStepConfig()
        assert config.prompt == "Please provide input:"
        assert config.timeout is None
        assert config.fallback_value == "[no input provided]"
        assert config.output_memory_key is None

    def test_custom_values(self):
        config = HumanInputStepConfig(
            prompt="Review?",
            timeout=30.0,
            fallback_value="skip",
            output_memory_key="review",
        )
        assert config.prompt == "Review?"
        assert config.timeout == 30.0
        assert config.fallback_value == "skip"
        assert config.output_memory_key == "review"

    def test_step_type(self):
        step = make_step()
        assert step.step_type == StepType.HUMAN_INPUT

    def test_step_config_property(self):
        step = make_step(prompt="Hello?")
        assert step.step_config.prompt == "Hello?"


# ---------------------------------------------------------------------------
# Executor: no callback (non-interactive / test mode)
# ---------------------------------------------------------------------------


class TestHumanInputExecutorNoCallback:
    @pytest.mark.asyncio
    async def test_returns_fallback_when_no_callback(self):
        """Without on_human_input_requested, step immediately uses fallback_value."""
        step = make_step(fallback_value="default answer")
        ctx = make_context()
        executor = HumanInputStepExecutor()

        result = await executor.execute(step, ctx)

        assert result.success is True
        assert result.result == "default answer"
        assert result.step_type == StepType.HUMAN_INPUT

    @pytest.mark.asyncio
    async def test_no_callback_history_entry(self):
        step = make_step(fallback_value="auto")
        ctx = make_context()

        result = await executor_execute(step, ctx)

        assert any("auto" in h for h in result.updated_history)

    @pytest.mark.asyncio
    async def test_timed_out_false_when_no_callback(self):
        """No callback means immediate fallback — not a timeout."""
        step = make_step()
        ctx = make_context()
        executor = HumanInputStepExecutor()

        result = await executor.execute(step, ctx)
        assert result.result_data.get("timed_out") is False


async def executor_execute(step, ctx):
    return await HumanInputStepExecutor().execute(step, ctx)


# ---------------------------------------------------------------------------
# Executor: synchronous callback that resolves future before await
# ---------------------------------------------------------------------------


class TestHumanInputExecutorSyncCallback:
    @pytest.mark.asyncio
    async def test_sync_callback_provides_input(self):
        """Callback sets result on the future synchronously before the await fires."""
        step = make_step(prompt="What's your name?")
        ctx = make_context()

        def on_input(prompt, future):
            # Resolve immediately (sync) — simulates a CLI prompt that happens
            # to execute before the coroutine resumes.
            future.set_result("Alice")

        ctx.on_human_input_requested = on_input

        executor = HumanInputStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success is True
        assert result.result == "Alice"
        assert result.result_data.get("timed_out") is False

    @pytest.mark.asyncio
    async def test_human_input_future_cleared_after_execution(self):
        """_human_input_future is None after execution completes."""
        step = make_step()
        ctx = make_context()

        ctx.on_human_input_requested = lambda p, f: f.set_result("done")

        executor = HumanInputStepExecutor()
        await executor.execute(step, ctx)

        assert ctx._human_input_future is None


# ---------------------------------------------------------------------------
# Executor: async callback via provide_human_input
# ---------------------------------------------------------------------------


class TestHumanInputExecutorAsyncCallback:
    @pytest.mark.asyncio
    async def test_async_provide_human_input(self):
        """
        Callback schedules provide_human_input in the event loop; executor awaits it.
        """
        step = make_step()
        ctx = make_context()

        async def async_provider(prompt, future):
            # Simulate async I/O (e.g., reading from a websocket)
            await asyncio.sleep(0)
            ctx.provide_human_input("async value")

        ctx.on_human_input_requested = async_provider

        executor = HumanInputStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success is True
        assert result.result == "async value"


# ---------------------------------------------------------------------------
# Executor: timeout
# ---------------------------------------------------------------------------


class TestHumanInputExecutorTimeout:
    @pytest.mark.asyncio
    async def test_timeout_uses_fallback_value(self):
        """When timeout elapses before input is provided, fallback_value is used."""
        step = make_step(timeout=0.05, fallback_value="timed out!")
        ctx = make_context()

        # Callback intentionally never resolves the future
        ctx.on_human_input_requested = lambda p, f: None

        executor = HumanInputStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success is True
        assert result.result == "timed out!"
        assert result.result_data.get("timed_out") is True

    @pytest.mark.asyncio
    async def test_timeout_future_cleared(self):
        """_human_input_future is None after timeout too."""
        step = make_step(timeout=0.05)
        ctx = make_context()
        ctx.on_human_input_requested = lambda p, f: None

        executor = HumanInputStepExecutor()
        await executor.execute(step, ctx)

        assert ctx._human_input_future is None


# ---------------------------------------------------------------------------
# Output memory key
# ---------------------------------------------------------------------------


class TestHumanInputMemoryKey:
    @pytest.mark.asyncio
    async def test_output_memory_key_written(self):
        """When output_memory_key is set, input is written to memory.human_input.<key>."""
        step = make_step(output_memory_key="feedback")
        ctx = make_context()
        ctx.on_human_input_requested = lambda p, f: f.set_result("great work")

        executor = HumanInputStepExecutor()
        await executor.execute(step, ctx)

        value = ctx.memory_read("feedback", namespace="human_input")
        assert value == "great work"

    @pytest.mark.asyncio
    async def test_no_memory_write_without_output_key(self):
        """Without output_memory_key, no memory write happens."""
        step = make_step()  # no output_memory_key
        ctx = make_context()
        ctx.on_human_input_requested = lambda p, f: f.set_result("value")

        executor = HumanInputStepExecutor()
        await executor.execute(step, ctx)

        # human_input namespace should not be populated
        assert ctx.memory.get("human_input") is None


# ---------------------------------------------------------------------------
# Language support
# ---------------------------------------------------------------------------


class TestHumanInputLanguage:
    @pytest.mark.asyncio
    async def test_russian_history_entry(self):
        """Russian language produces Cyrillic history entry."""
        step = make_step(fallback_value="без ответа")
        ctx = make_context(language=Language.RUSSIAN)

        executor = HumanInputStepExecutor()
        result = await executor.execute(step, ctx)

        assert any("Шаг" in h or "Ввод" in h for h in result.updated_history)

    @pytest.mark.asyncio
    async def test_english_history_entry(self):
        """English language produces English history entry."""
        step = make_step(fallback_value="default")
        ctx = make_context(language=Language.ENGLISH)

        executor = HumanInputStepExecutor()
        result = await executor.execute(step, ctx)

        assert any("Step" in h or "Human input" in h for h in result.updated_history)


# ---------------------------------------------------------------------------
# provide_human_input on context
# ---------------------------------------------------------------------------


class TestProvideHumanInput:
    @pytest.mark.asyncio
    async def test_provide_human_input_resolves_future(self):
        ctx = make_context()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        ctx._human_input_future = future

        result = ctx.provide_human_input("hello")

        assert result is True
        assert future.result() == "hello"

    def test_provide_human_input_no_future_returns_false(self):
        ctx = make_context()
        # No pending future
        result = ctx.provide_human_input("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_provide_human_input_done_future_returns_false(self):
        ctx = make_context()
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        future.set_result("already done")
        ctx._human_input_future = future

        result = ctx.provide_human_input("ignored")
        assert result is False


# ---------------------------------------------------------------------------
# Integration: HumanInputStep in a real chain
# ---------------------------------------------------------------------------


class TestHumanInputChainIntegration:
    @pytest.mark.asyncio
    async def test_human_input_step_in_chain(self):
        """
        Chain with a HumanInputStep: callback is registered and resolves the future.
        """
        from mmar_carl.models.config import ToolStepConfig
        from mmar_carl.models.steps import ToolStepDescription

        received_feedback = {}

        def record_feedback(feedback: str) -> str:
            received_feedback["value"] = feedback
            return f"Recorded: {feedback}"

        chain = ReasoningChain(
            steps=[
                HumanInputStepDescription(
                    number=1,
                    title="Get Feedback",
                    config=HumanInputStepConfig(
                        prompt="Enter feedback:",
                        output_memory_key="feedback",
                    ),
                ),
                ToolStepDescription(
                    number=2,
                    title="Record Feedback",
                    dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="record_feedback",
                        input_mapping={"feedback": "$memory.human_input.feedback"},
                    ),
                ),
            ]
        )

        class MockClient(LLMClientBase):
            async def get_response(self, prompt): return "ok"
            async def get_response_with_retries(self, prompt, retries=3): return "ok"

        ctx = ReasoningContext(
            outer_context="integration test",
            api=MockClient(),
            language=Language.ENGLISH,
        )
        ctx.register_tool("record_feedback", record_feedback)
        ctx.on_human_input_requested = lambda p, f: f.set_result("user provided value")

        result = await chain.execute_async(ctx)

        assert result.success, result.get_failed_steps()
        assert received_feedback.get("value") == "user provided value"
        assert ctx.memory_read("feedback", namespace="human_input") == "user provided value"

    @pytest.mark.asyncio
    async def test_human_input_step_serialization(self):
        """HumanInputStepDescription serializes step_type correctly."""
        step = HumanInputStepDescription(
            number=1,
            title="Get Feedback",
            config=HumanInputStepConfig(prompt="Review?", timeout=60.0),
        )
        data = step.model_dump()
        assert data["step_type"] == "human_input"
        assert data["config"]["prompt"] == "Review?"
        assert data["config"]["timeout"] == 60.0
