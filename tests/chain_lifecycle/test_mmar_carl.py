"""
Tests for mmar-carl library with mocked LLMHub.

This test suite covers the main functionality of the CARL library
including chain execution, LLM client integration, and model behavior.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from mmar_carl._optional_deps import check_vector_search_available

# Skip tests that expect missing vector-search deps when they ARE installed
VECTOR_SEARCH_AVAILABLE = check_vector_search_available()

from mmar_carl import (  # noqa: E402
    ChainBuilder,
    ContextQuery,
    ContextSearchConfig,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    StepDescription,  # Kept for backward compat tests
    StepType,
    ToolStepConfig,
    MemoryStepConfig,
    MemoryOperation,
    TransformStepConfig,
    ConditionalStepConfig,
    ConditionalBranch,
    ExecutionMode,
    LLMStepConfig,
    LLMStepDescription,
    ToolStepDescription,
    MemoryStepDescription,
    TransformStepDescription,
    ConditionalStepDescription,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    StepDescriptionBase,
    MCPStepDescription,
    AnyStepDescription,
    create_step,
    ToolParameter,
    MCPStepConfig,
    MCPServerConfig,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
)


class TestStepDescription:
    """Test StepDescription model."""

    def test_step_creation(self):
        """Test creating a step with all required fields."""
        step = LLMStepDescription(
            number=1,
            title="Test Step",
            aim="Test aim",
            reasoning_questions="Test questions",
            stage_action="Test action",
            example_reasoning="Test example",
        )
        assert step.number == 1
        assert step.title == "Test Step"
        assert step.dependencies == []
        assert step.step_context_queries == []

    def test_step_with_dependencies(self):
        """Test creating a step with dependencies."""
        step = LLMStepDescription(
            number=2,
            title="Test Step 2",
            aim="Test aim 2",
            reasoning_questions="Test questions 2",
            stage_action="Test action 2",
            example_reasoning="Test example 2",
            dependencies=[1],
            step_context_queries=["test query", "sample context"],
        )
        assert step.depends_on(1)
        assert not step.depends_on(3)
        assert step.has_dependencies()

    def test_without_dependencies(self):
        """Test checking dependencies on step without them."""
        step = LLMStepDescription(
            number=1,
            title="Test Step",
            aim="Test aim",
            reasoning_questions="Test questions",
            stage_action="Test action",
            example_reasoning="Test example",
        )
        assert not step.has_dependencies()
        assert not step.depends_on(1)


class TestReasoningContext:
    """Test ReasoningContext model."""

    def test_context_creation(self):
        """Test creating a reasoning context."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            retry_max=2,
            language=Language.ENGLISH,
        )
        assert context.outer_context == "Test data"
        assert context.api == mock_api
        assert context.model == "test_key"
        assert context.retry_max == 2
        assert context.language == Language.ENGLISH
        assert context.history == []
        # Test that LLM client is created automatically
        assert context.llm_client is not None
        assert isinstance(context.llm_client, LLMClientBase)

    def test_history_management(self):
        """Test adding and retrieving history."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )
        context.add_to_history("First step result")
        context.add_to_history("Second step result")

        assert len(context.history) == 2
        assert context.history[0] == "First step result"
        assert context.history[1] == "Second step result"
        assert context.get_current_history() == "First step result\nSecond step result"

    def test_system_prompt_field(self):
        """Test system prompt field in ReasoningContext."""
        mock_api = MockLLMClient("Test response")

        # Test with empty system prompt (default)
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )
        assert context.system_prompt == ""

        # Test with custom system prompt
        system_prompt = "You are an expert analyst."
        context_with_prompt = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            system_prompt=system_prompt,
        )
        assert context_with_prompt.system_prompt == system_prompt

    def test_self_critic_evaluator_registry(self):
        """Test self-critic evaluator registry methods on ReasoningContext."""

        class DummyEvaluator(SelfCriticEvaluatorBase):
            async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
                _ = step, candidate, base_prompt, context, llm_client, retries
                return SelfCriticDecision(verdict="APPROVE", review_text="ok")

        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )
        evaluator = DummyEvaluator()

        context.register_self_critic_evaluator("dummy", evaluator)
        assert context.get_self_critic_evaluator("dummy") is evaluator
        assert "dummy" in context.list_self_critic_evaluators()


class MockLLMClient(LLMClientBase):
    """Mock LLM client for testing."""

    def __init__(self, response_text="Mock response"):
        self.response_text = response_text
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        """Mock async method that returns predefined response."""
        self.call_count += 1
        return f"Response to: {prompt[:50]}..."

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock async method that returns predefined response."""
        self.call_count += 1
        return f"Response to: {prompt[:50]}... (retries={retries})"


class ModeAwareMockLLMClient(LLMClientBase):
    """Mock LLM client for FAST + SELF_CRITIC execution mode tests."""

    def __init__(self):
        self.prompts: list[str] = []
        self._generation_round = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        res = await self._get_response_with_retries(prompt, retries)
        return res

    async def _get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        self.prompts.append(prompt)
        lowered = prompt.lower()

        if "strict reviewer of an llm answer" in lowered:
            if "draft-v2" in lowered:
                return '{"verdict":"APPROVE","review":"Looks good"}'
            return '{"verdict":"DISAPPROVE","review":"Need clearer final answer"}'

        if "regenerate the same task output with higher quality" in lowered:
            self._generation_round += 1
            return f"draft-v{self._generation_round}"

        self._generation_round = 1
        return "draft-v1"


class InvalidJsonCriticMockLLMClient(ModeAwareMockLLMClient):
    """Mock client where built-in LLM critic returns non-JSON output."""

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        lowered = prompt.lower()
        if "strict reviewer of an llm answer" in lowered:
            return "VERDICT: DISAPPROVE\nREVIEW: Not JSON"
        return await super().get_response_with_retries(prompt, retries=retries)


class AlwaysApproveEvaluator(SelfCriticEvaluatorBase):
    """Custom evaluator that always approves."""

    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        _ = step, candidate, base_prompt, context, llm_client, retries
        return SelfCriticDecision(
            verdict="APPROVE",
            review_text="Approved by custom evaluator.",
            metadata={"llm_calls": 0},
        )


class DisapproveOnceEvaluator(SelfCriticEvaluatorBase):
    """Custom evaluator that disapproves first round, then approves."""

    def __init__(self):
        self.calls = 0

    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        _ = step, candidate, base_prompt, context, llm_client, retries
        self.calls += 1
        if self.calls == 1:
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="First pass rejected.",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="APPROVE",
            review_text="Second pass approved.",
            metadata={"llm_calls": 0},
        )


class AlwaysErrorEvaluator(SelfCriticEvaluatorBase):
    """Custom evaluator that always raises an exception."""

    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        _ = step, candidate, base_prompt, context, llm_client, retries
        raise RuntimeError("Evaluator crashed")


class DisapproveNoReviewThenApproveEvaluator(SelfCriticEvaluatorBase):
    """Custom evaluator that disapproves without review once, then approves."""

    def __init__(self):
        self.calls = 0

    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        _ = step, candidate, base_prompt, context, llm_client, retries
        self.calls += 1
        if self.calls == 1:
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="APPROVE",
            review_text="Approved on second pass.",
            metadata={"llm_calls": 0},
        )


class TestLLMHub:
    """Test LLMHub usage."""

    @pytest.mark.asyncio
    async def test_accessor_async(self):
        """Test getting response when llm returns async value."""

        # Create async mock
        async def mock_async_response(prompt: str, retries: int = 3):
            return f"Async response to: {prompt[:50]}..."

        mock_llm = AsyncMock()
        mock_llm.get_response_with_retries = mock_async_response

        mock_accessor = MagicMock()
        mock_accessor.__getitem__ = MagicMock(return_value=mock_llm)

        client = mock_accessor["test_key"]
        result = client.get_response_with_retries("Test prompt", retries=1)

        # If the result is a coroutine, await it
        if hasattr(result, "__await__") or hasattr(result, "__aiter__"):
            result = await result

        assert "Async response to: Test prompt" in result


class TestReasoningChain:
    """Test ReasoningChain functionality."""

    @pytest.fixture
    def sample_steps(self):
        """Create sample reasoning steps for testing."""
        return [
            LLMStepDescription(
                number=1,
                title="First Step",
                aim="Analyze initial data",
                reasoning_questions="What does the data show?",
                stage_action="Analyze the data",
                example_reasoning="Data analysis example",
            ),
            LLMStepDescription(
                number=2,
                title="Second Step",
                aim="Build on first analysis",
                reasoning_questions="What conclusions can we draw?",
                stage_action="Draw conclusions",
                example_reasoning="Conclusion example",
                dependencies=[1],
            ),
        ]

    @pytest.fixture
    def mock_llm_client_for_chain(self):
        """Create mock API for chain tests."""
        return MockLLMClient("Mock LLM response")

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock API for tests."""
        return MockLLMClient("Mock response")

    def test_chain_creation(self, sample_steps):
        """Test creating a reasoning chain."""
        chain = ReasoningChain(steps=sample_steps, max_workers=2)
        assert len(chain.steps) == 2
        assert chain.max_workers == 2
        assert chain.enable_progress is False  # Default value

    def test_chain_creation_with_progress(self, sample_steps):
        """Test creating a reasoning chain with progress enabled."""
        chain = ReasoningChain(steps=sample_steps, max_workers=1, enable_progress=True)
        assert chain.enable_progress is True

    def test_get_execution_plan(self, sample_steps):
        """Test getting execution plan."""
        chain = ReasoningChain(steps=sample_steps, max_workers=2)
        plan = chain.get_execution_plan()
        assert plan["total_steps"] == 2
        assert plan["max_workers"] == 2
        assert len(plan["execution_levels"]) >= 1
        # Check that both steps are included
        all_steps_in_levels = []
        for level in plan["execution_levels"]:
            all_steps_in_levels.extend(level["steps"])
        assert set(all_steps_in_levels) == {1, 2}

    def test_get_step_dependencies(self, sample_steps):
        """Test getting step dependencies."""
        chain = ReasoningChain(steps=sample_steps, max_workers=2)
        deps = chain.get_step_dependencies()
        assert deps[1] == []  # Step 1 has no dependencies
        assert deps[2] == [1]  # Step 2 depends on step 1

    @pytest.mark.asyncio
    async def test_chain_execution_mocked(self, sample_steps, mock_llm_client_for_chain):
        """Test chain execution with mocked LLM client."""
        # Create context with mock API (LLM client created automatically)
        context = ReasoningContext(
            outer_context="Test data for analysis",
            api=mock_llm_client_for_chain,
            model="test_key",
            retry_max=1,
            language=Language.RUSSIAN,
        )

        # Create chain
        chain = ReasoningChain(steps=sample_steps, max_workers=1, enable_progress=False)

        # Execute chain
        result = await chain.execute_async(context)

        # Verify result
        assert result.success
        assert len(result.step_results) == 2
        assert all(step.success for step in result.step_results)
        assert len(result.history) == 2

        # Verify LLM client was called
        assert mock_llm_client_for_chain.call_count == 2

    @pytest.mark.asyncio
    async def test_chain_execution_with_mock_llm_client(self, sample_steps, mock_llm_client):
        """Test chain execution with mocked LLM client."""
        # Create context
        context = ReasoningContext(
            outer_context="Period,EBITDA,SALES_REVENUE\n2023-Q1,1000000,5000000",
            api=mock_llm_client,
            model="test_key",
            retry_max=1,
            language=Language.RUSSIAN,
        )

        # Create chain
        chain = ReasoningChain(steps=sample_steps, max_workers=1, enable_progress=False)

        # Execute chain
        result = await chain.execute_async(context)

        # Verify result
        assert result.success
        assert len(result.step_results) == 2
        assert all(step.success for step in result.step_results)

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution with independent steps."""
        # Create independent steps (no dependencies)
        independent_steps = [
            LLMStepDescription(
                number=1,
                title="Independent Step 1",
                aim="Analyze data aspect 1",
                reasoning_questions="What does aspect 1 show?",
                stage_action="Analyze aspect 1",
                example_reasoning="Analysis example 1",
            ),
            LLMStepDescription(
                number=2,
                title="Independent Step 2",
                aim="Analyze data aspect 2",
                reasoning_questions="What does aspect 2 show?",
                stage_action="Analyze aspect 2",
                example_reasoning="Analysis example 2",
            ),
        ]

        # Mock API
        mock_api = MockLLMClient("Mock response")

        # Create context
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            retry_max=1,
        )

        # Create chain with parallel execution
        chain = ReasoningChain(steps=independent_steps, max_workers=2, enable_progress=False)

        # Execute chain
        result = await chain.execute_async(context)

        # Verify result
        assert result.success
        assert len(result.step_results) == 2

        # Check execution statistics
        metadata = result.metadata.get("execution_stats", {})
        assert metadata.get("parallel_batches", 0) >= 1

    @pytest.mark.asyncio
    async def test_chain_execution_error_handling(self, sample_steps):
        """Test error handling in chain execution."""

        # Override the method to raise an exception
        class MockLLMClient(LLMClientBase):
            async def get_response_with_retries(self, prompt: str, retries: int = 3):
                raise Exception("LLM error")

            async def get_response(self, prompt: str, retries: int = 3):
                raise Exception("LLM error")

        mock_api = MockLLMClient()

        # Create context
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            retry_max=1,
        )

        # Create chain
        chain = ReasoningChain(steps=sample_steps, max_workers=1, enable_progress=False)

        # Execute chain - should fail
        result = await chain.execute_async(context)

        # Verify failure handling
        assert not result.success
        assert len(result.get_failed_steps()) > 0
        assert any(step.error_message and "LLM error" in step.error_message for step in result.get_failed_steps())


class TestExecutionModes:
    """Test LLM execution modes (fast + self_critic)."""

    def _build_single_step_chain(self, mode: ExecutionMode) -> ReasoningChain:
        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(execution_mode=mode),
        )
        return ReasoningChain(steps=[step], max_workers=1)

    def test_execution_mode_enum_values(self):
        """Test execution mode enum values."""
        assert ExecutionMode.FAST == "fast"
        assert ExecutionMode.SELF_CRITIC == "self_critic"

    def test_replan_mode_is_rejected(self):
        """Legacy REPLAN value should be rejected by LLMStepConfig validation."""
        with pytest.raises(ValidationError):
            LLMStepConfig(execution_mode="replan")

    def test_fast_mode_uses_single_llm_call(self):
        """FAST mode should execute one direct LLM call."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
        chain = self._build_single_step_chain(ExecutionMode.FAST)

        result = chain.execute(context)

        assert result.success
        assert result.get_final_output() == "draft-v1"
        assert len(client.prompts) == 1
        details = context.metadata["execution_mode_details"]["1"]
        assert details["execution_mode"] == ExecutionMode.FAST.value
        assert details["llm_calls"] == 1
        assert details["evaluator_decisions"] == []

    def test_self_critic_mode_revises_response(self):
        """SELF_CRITIC mode should critique and revise the initial draft."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(
            outer_context="input",
            api=client,
            model="unused",
            language=Language.ENGLISH,
        )
        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)

        assert result.success, result
        assert result.get_final_output() == "draft-v2"
        assert any("strict reviewer of an llm answer" in prompt.lower() for prompt in client.prompts)
        assert any("regenerate the same task output" in prompt.lower() for prompt in client.prompts)

        details = context.metadata["execution_mode_details"]["1"]
        assert details["execution_mode"] == ExecutionMode.SELF_CRITIC.value
        assert details["llm_calls"] >= 3

    def test_self_critic_mode_accepts_custom_llm_instruction(self):
        """SELF_CRITIC built-in llm evaluator should include custom instruction in prompt."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
        custom_instruction = "Prioritize factual consistency and call out missing mitigation owners."
        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
                self_critic_instruction=custom_instruction,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)

        assert result.success
        critic_prompts = [p for p in client.prompts if "strict reviewer of an llm answer" in p.lower()]
        assert critic_prompts
        assert any(custom_instruction.lower() in p.lower() for p in critic_prompts)

    def test_self_critic_invalid_llm_json_adds_feedback_to_regeneration(self):
        """Invalid built-in llm critic output should disapprove and add parse feedback."""
        client = InvalidJsonCriticMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)

        assert result.success
        regen_prompts = [
            p for p in client.prompts if "regenerate the same task output with higher quality" in p.lower()
        ]
        assert regen_prompts
        assert "Evaluator response is not valid JSON." in regen_prompts[0]

    def test_self_critic_with_multiple_evaluators_all_must_approve(self):
        """SELF_CRITIC should regenerate when any evaluator disapproves."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(
            outer_context="input",
            api=client,
            model="unused",
            language=Language.ENGLISH,
        )
        context.register_self_critic_evaluator("custom_approve", AlwaysApproveEvaluator())
        disapprove_once = DisapproveOnceEvaluator()
        context.register_self_critic_evaluator("custom_disapprove_once", disapprove_once)

        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["custom_approve", "custom_disapprove_once"],
                self_critic_max_revisions=2,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)

        assert result.success
        assert result.get_final_output() == "draft-v2"
        assert disapprove_once.calls == 2

        details = context.metadata["execution_mode_details"]["1"]
        assert details["execution_mode"] == ExecutionMode.SELF_CRITIC.value
        assert details["rounds"] == 2
        assert details["evaluator_policy"] == "all_must_approve"

    def test_self_critic_static_feedback_for_function_evaluator(self):
        """Configured static feedback should be appended when function evaluator disapproves."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
        evaluator = DisapproveNoReviewThenApproveEvaluator()
        context.register_self_critic_evaluator("custom_no_review", evaluator)

        static_feedback = "Always include explicit risk mitigation and ownership."
        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["custom_no_review"],
                self_critic_max_revisions=1,
                self_critic_disapprove_feedback={"custom_no_review": static_feedback},
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)

        assert result.success
        assert evaluator.calls == 2
        regen_prompts = [
            p for p in client.prompts if "regenerate the same task output with higher quality" in p.lower()
        ]
        assert regen_prompts
        assert static_feedback in regen_prompts[0]

    def test_self_critic_missing_evaluator_fails_step(self):
        """Missing evaluator in chain should fail step with explicit error."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)

        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["missing_evaluator"],
                self_critic_max_revisions=1,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)
        assert not result.success
        assert "not registered" in (result.get_failed_steps()[0].error_message or "")

    def test_self_critic_evaluator_exception_fails_step(self):
        """Evaluator exceptions should fail step explicitly."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
        context.register_self_critic_evaluator("always_error", AlwaysErrorEvaluator())

        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["always_error"],
                self_critic_max_revisions=1,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)
        assert not result.success
        assert "Evaluator crashed" in (result.get_failed_steps()[0].error_message or "")

    def test_self_critic_max_revision_cap_adds_warning(self):
        """When max revisions is reached, step succeeds with quality warning metadata."""
        client = ModeAwareMockLLMClient()
        context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)

        step = LLMStepDescription(
            number=1,
            title="Mode Step",
            aim="Produce an answer",
            reasoning_questions="What is the answer?",
            stage_action="Generate answer",
            example_reasoning="Example reasoning",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],  # llm evaluator disapproves draft-v1
                self_critic_max_revisions=0,
            ),
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        result = chain.execute(context)
        assert result.success
        assert result.get_final_output() == "draft-v1"
        details = context.metadata["execution_mode_details"]["1"]
        assert "quality_warning" in details

    def test_chain_builder_execution_mode_shortcut(self):
        """ChainBuilder should set llm_config.execution_mode from shortcut argument."""
        chain = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Builder Mode Step",
                aim="Produce an answer",
                reasoning_questions="What is the answer?",
                stage_action="Generate answer",
                example_reasoning="Example reasoning",
                execution_mode=ExecutionMode.SELF_CRITIC,
            )
            .build()
        )

        assert chain.steps[0].llm_config is not None
        assert chain.steps[0].llm_config.execution_mode == ExecutionMode.SELF_CRITIC


class TestLanguage:
    """Test Language enum."""

    def test_language_values(self):
        """Test language enum values."""
        assert Language.RUSSIAN == "ru"
        assert Language.ENGLISH == "en"


class TestContextSearchConfig:
    """Test ContextSearchConfig model."""

    def test_default_config(self):
        """Test default configuration uses substring search."""
        config = ContextSearchConfig()
        assert config.strategy == "substring"
        strategy = config.get_strategy()
        assert strategy.__class__.__name__ == "SubstringSearchStrategy"

    @pytest.mark.skipif(VECTOR_SEARCH_AVAILABLE, reason="Vector search dependencies are installed")
    def test_vector_config(self):
        """Test vector search configuration."""
        config = ContextSearchConfig(
            strategy="vector",
            vector_config={"embedding_model": "all-MiniLM-L6-v2", "similarity_threshold": 0.8, "max_results": 3},
        )
        assert config.strategy == "vector"
        # Should raise ImportError without vector-search dependencies
        with pytest.raises(ImportError) as exc_info:
            strategy = config.get_strategy()  # noqa: F841
        assert "pip install 'mmar-carl[vector-search]'" in str(exc_info.value)

    def test_substring_config(self):
        """Test substring search configuration."""
        config = ContextSearchConfig(
            strategy="substring",
            substring_config={"case_sensitive": True, "min_word_length": 3, "max_matches_per_query": 5},
        )
        assert config.strategy == "substring"
        strategy = config.get_strategy()
        assert strategy.__class__.__name__ == "SubstringSearchStrategy"
        assert strategy.case_sensitive is True
        assert strategy.min_word_length == 3
        assert strategy.max_matches_per_query == 5


class TestSearchStrategies:
    """Test search strategy implementations."""

    def test_substring_search_basic(self):
        """Test basic substring search functionality."""
        from mmar_carl.models import SubstringSearchStrategy

        strategy = SubstringSearchStrategy()
        context = "Revenue: 1000\nProfit: 200\nGrowth: 15%"
        queries = ["revenue", "growth"]

        result = strategy.extract_context(context, queries)

        assert "revenue" in result.lower()
        assert "growth" in result.lower()
        assert "1000" in result

    def test_substring_search_case_sensitive(self):
        """Test case-sensitive substring search."""
        from mmar_carl.models import SubstringSearchStrategy

        strategy = SubstringSearchStrategy(case_sensitive=True)
        context = "Revenue: 1000\nrevenue: 500"
        queries = ["Revenue"]

        result = strategy.extract_context(context, queries)

        assert "1000" in result
        assert "500" not in result

    def test_substring_search_empty_queries(self):
        """Test substring search with no queries."""
        from mmar_carl.models import SubstringSearchStrategy

        strategy = SubstringSearchStrategy()
        result = strategy.extract_context("Some context", [])

        assert "No specific context queries defined" in result

    @pytest.mark.skipif(VECTOR_SEARCH_AVAILABLE, reason="Vector search dependencies are installed")
    def test_vector_search_fallback(self):
        """Test vector search raises ImportError without dependencies."""
        from mmar_carl.models import VectorSearchStrategy

        # Should raise ImportError at initialization without vector-search dependencies
        with pytest.raises(ImportError) as exc_info:
            strategy = VectorSearchStrategy()  # noqa: F841

        assert "pip install 'mmar-carl[vector-search]'" in str(exc_info.value)


class TestReasoningChainSearch:
    """Test ReasoningChain with different search configurations."""

    def test_chain_with_default_search(self):
        """Test chain creation with default search configuration."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]
        chain = ReasoningChain(steps=steps)

        # Should have default substring search
        assert chain.prompt_template.search_config.strategy == "substring"

    def test_chain_with_vector_search(self):
        """Test chain creation with vector search configuration."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]
        search_config = ContextSearchConfig(strategy="vector", vector_config={"similarity_threshold": 0.9})

        chain = ReasoningChain(steps=steps, search_config=search_config)

        assert chain.prompt_template.search_config.strategy == "vector"
        assert chain.prompt_template.search_config.vector_config["similarity_threshold"] == 0.9

    def test_chain_builder_with_search_config(self):
        """Test ChainBuilder with search configuration."""
        search_config = ContextSearchConfig(strategy="substring", substring_config={"case_sensitive": True})

        chain = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
            .with_search_config(search_config)
            .build()
        )

        assert chain.prompt_template.search_config.strategy == "substring"
        assert chain.prompt_template.search_config.substring_config["case_sensitive"] is True


class TestPromptTemplate:
    """Test PromptTemplate model."""

    def test_system_prompt_in_english(self):
        """Test system prompt inclusion in English prompts."""
        from mmar_carl.models import PromptTemplate

        template = PromptTemplate()
        system_prompt = "You are an expert financial analyst."

        # Test without system prompt
        prompt_no_sp = template.format_chain_prompt(
            outer_context="Revenue: 1000", current_task="Analyze revenue", language=Language.ENGLISH, system_prompt=""
        )
        assert "System Instructions:" not in prompt_no_sp

        # Test with system prompt
        prompt_with_sp = template.format_chain_prompt(
            outer_context="Revenue: 1000",
            current_task="Analyze revenue",
            language=Language.ENGLISH,
            system_prompt=system_prompt,
        )
        assert "System Instructions:" in prompt_with_sp
        assert system_prompt in prompt_with_sp

    def test_system_prompt_in_russian(self):
        """Test system prompt inclusion in Russian prompts."""
        from mmar_carl.models import PromptTemplate

        template = PromptTemplate()
        system_prompt = "Вы экспертный финансовый аналитик."

        # Test with Russian system prompt
        prompt_with_sp = template.format_chain_prompt(
            outer_context="Выручка: 1000",
            current_task="Проанализируйте выручку",
            language=Language.RUSSIAN,
            system_prompt=system_prompt,
        )
        assert "Системные инструкции:" in prompt_with_sp
        assert system_prompt in prompt_with_sp

    def test_system_prompt_integration(self):
        """Test system prompt integration with full prompt flow."""
        from mmar_carl.models import PromptTemplate

        template = PromptTemplate()
        system_prompt = "You are an expert data analyst."
        outer_context = "Revenue: 1000\nCost: 800"

        # Generate step prompt
        step_prompt = template.format_step_prompt(
            step=LLMStepDescription(
                number=1,
                title="Analysis",
                aim="Analyze data",
                reasoning_questions="What are the trends?",
                stage_action="Extract insights",
                example_reasoning="Data analysis reveals patterns",
            ),
            outer_context=outer_context,
            language=Language.ENGLISH,
        )

        # Generate full chain prompt with system prompt
        full_prompt = template.format_chain_prompt(
            outer_context=outer_context,
            current_task=step_prompt,
            language=Language.ENGLISH,
            system_prompt=system_prompt,
        )

        # Verify system prompt is included at the beginning
        assert full_prompt.startswith("System Instructions:")
        assert system_prompt in full_prompt
        assert "Data for analysis:" in full_prompt  # Should include chain template


class TestContextQuery:
    """Test ContextQuery model."""

    def test_context_query_creation(self):
        """Test creating a context query with search strategy override."""
        query = ContextQuery(
            query="financial performance",
            search_strategy="vector",
            search_config={"similarity_threshold": 0.8, "max_results": 3},
        )
        assert query.query == "financial performance"
        assert query.search_strategy == "vector"
        assert query.search_config["similarity_threshold"] == 0.8

    def test_context_query_string_behavior(self):
        """Test that ContextQuery can be used as string."""
        query = ContextQuery(query="test query")
        assert str(query) == "test query"

    def test_context_query_defaults(self):
        """Test context query with default values."""
        query = ContextQuery(query="test")
        assert query.search_strategy is None
        assert query.search_config is None


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_simple_financial_analysis(self):
        """Test a simple financial analysis scenario."""
        # Create financial analysis steps
        steps = [
            LLMStepDescription(
                number=1,
                title="Анализ выручки",
                aim="Проанализировать динамику выручки",
                reasoning_questions="Как изменилась выручка за период?",
                stage_action="Рассчитать изменение выручки",
                example_reasoning="Если выручка растет, это положительный сигнал",
            ),
            LLMStepDescription(
                number=2,
                title="Анализ прибыли",
                aim="Проанализировать динамику прибыли",
                reasoning_questions="Как изменилась прибыль за период?",
                stage_action="Рассчитать изменение прибыли",
                example_reasoning="Если прибыль растет, это положительный сигнал",
                dependencies=[1],
            ),
        ]

        # Mock API
        mock_api = MockLLMClient("Финансовый анализ выполнен успешно")

        # Create context with financial data
        financial_data = """Период,SALES_REVENUE,NET_INCOME
2023-Q1,1000000,200000
2023-Q2,1100000,250000"""

        context = ReasoningContext(
            outer_context=financial_data,
            api=mock_api,
            model="test_key",
            retry_max=2,
            language=Language.RUSSIAN,
        )

        # Create and execute chain
        chain = ReasoningChain(steps=steps, max_workers=2, enable_progress=False)
        result = await chain.execute_async(context)

        # Verify successful execution
        assert result.success
        assert len(result.step_results) == 2
        assert result.get_final_output()

    @pytest.mark.asyncio
    async def test_mixed_search_strategies(self):
        """Test a chain with mixed search strategies per query."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Mixed Search Analysis",
                aim="Test mixed search strategies",
                reasoning_questions="How do different search strategies perform?",
                stage_action="Analyze with mixed search",
                example_reasoning="Mixed search provides comprehensive results",
                step_context_queries=[
                    "EBITDA",  # Uses chain default
                    ContextQuery(query="revenue patterns", search_strategy="substring"),
                    ContextQuery(
                        query="NET_INCOME", search_strategy="substring", search_config={"case_sensitive": True}
                    ),
                ],
            )
        ]

        # Mock API
        mock_api = MockLLMClient("Mixed search analysis completed")

        # Create context
        context = ReasoningContext(
            outer_context="EBITDA,1000000\nRevenue,2000000\nNET_INCOME,500000",
            api=mock_api,
            model="test_key",
            retry_max=1,
            language=Language.ENGLISH,
        )

        # Create and execute chain
        chain = ReasoningChain(
            steps=steps, search_config=ContextSearchConfig(strategy="substring"), max_workers=1, enable_progress=False
        )

        result = await chain.execute_async(context)

        # Verify execution succeeded with mixed search strategies
        assert result.success
        assert len(result.step_results) == 1


class TestReasoningResult:
    """Test ReasoningResult model methods."""

    def test_get_final_output_russian_step(self):
        """Test get_final_output() extracts clean content from Russian step headers."""
        from mmar_carl.models import ReasoningResult, StepExecutionResult

        # Create a result with Russian step header
        step_results = [
            StepExecutionResult(
                step_number=1,
                step_title="First Step",
                result="First result content",
                success=True,
            ),
            StepExecutionResult(
                step_number=4,
                step_title="Форматирование финального рецепта",
                result="Вот правильно отформатированный рецепт без заголовков",
                success=True,
            ),
        ]

        # Create history as it would be generated by the executor
        history = [
            "Шаг 1. First Step\nРезультат: First result content\n",
            "Шаг 4. Форматирование финального рецепта\nРезультат: Вот правильно отформатированный рецепт без заголовков\n",
        ]

        result = ReasoningResult(
            success=True,
            history=history,
            step_results=step_results,
            total_execution_time=1.0,
        )

        # Test that get_final_output returns clean content without headers
        final_output = result.get_final_output()
        assert final_output == "Вот правильно отформатированный рецепт без заголовков"
        assert "Шаг 4" not in final_output
        assert "Результат:" not in final_output

    def test_get_final_output_english_step(self):
        """Test get_final_output() extracts clean content from English step headers."""
        from mmar_carl.models import ReasoningResult, StepExecutionResult

        # Create a result with English step header
        step_results = [
            StepExecutionResult(
                step_number=1,
                step_title="First Step",
                result="First result content",
                success=True,
            ),
            StepExecutionResult(
                step_number=2,
                step_title="Final Analysis",
                result="Here is the final analysis without technical headers",
                success=True,
            ),
        ]

        # Create history as it would be generated by the executor
        history = [
            "Step 1. First Step\nResult: First result content\n",
            "Step 2. Final Analysis\nResult: Here is the final analysis without technical headers\n",
        ]

        result = ReasoningResult(
            success=True,
            history=history,
            step_results=step_results,
            total_execution_time=1.0,
        )

        # Test that get_final_output returns clean content without headers
        final_output = result.get_final_output()
        assert final_output == "Here is the final analysis without technical headers"
        assert "Step 2" not in final_output
        assert "Result:" not in final_output

    def test_get_final_output_empty_history(self):
        """Test get_final_output() with empty history."""
        from mmar_carl.models import ReasoningResult

        result = ReasoningResult(success=True, history=[], step_results=[], total_execution_time=0.0)

        final_output = result.get_final_output()
        assert final_output == ""

    def test_get_final_output_non_standard_format(self):
        """Test get_final_output() with non-standard history format."""
        from mmar_carl.models import ReasoningResult

        # Test with history that doesn't follow the expected format
        history = [
            "Some custom format",
            "Another custom entry without step headers",
        ]

        result = ReasoningResult(success=True, history=history, step_results=[], total_execution_time=0.0)

        final_output = result.get_final_output()
        assert final_output == "Another custom entry without step headers"


# =============================================================================
# Tests for Extended Step Types
# =============================================================================


class TestStepTypes:
    """Test StepType enum and step type configurations."""

    def test_step_type_enum_values(self):
        """Test StepType enum has expected values."""
        assert StepType.LLM == "llm"
        assert StepType.TOOL == "tool"
        assert StepType.MCP == "mcp"
        assert StepType.MEMORY == "memory"
        assert StepType.TRANSFORM == "transform"
        assert StepType.CONDITIONAL == "conditional"

    def test_llm_step_default(self):
        """Test that LLM is the default step type."""
        step = LLMStepDescription(
            number=1,
            title="Test Step",
            aim="Test aim",
            reasoning_questions="Test questions",
            stage_action="Test action",
            example_reasoning="Test example",
        )
        assert step.step_type == StepType.LLM
        assert step.is_llm_step()

    def test_tool_step_creation(self):
        """Test creating a TOOL step."""
        step = ToolStepDescription(
            number=1,
            title="Tool Step",
            config=ToolStepConfig(
                tool_name="my_tool",
                tool_description="A test tool",
                input_mapping={"arg1": "$history[-1]"},
                timeout=10.0,
            ),
        )
        assert step.step_type == StepType.TOOL
        assert step.is_tool_step()
        assert step.config.tool_name == "my_tool"
        assert step.config.timeout == 10.0

    def test_memory_step_creation(self):
        """Test creating a MEMORY step."""
        step = MemoryStepDescription(
            number=1,
            title="Memory Step",
            config=MemoryStepConfig(
                operation=MemoryOperation.WRITE,
                memory_key="result",
                value_source="$history[-1]",
                namespace="default",
            ),
        )
        assert step.step_type == StepType.MEMORY
        assert step.is_memory_step()
        assert step.config.operation == MemoryOperation.WRITE

    def test_transform_step_creation(self):
        """Test creating a TRANSFORM step."""
        step = TransformStepDescription(
            number=1,
            title="Transform Step",
            config=TransformStepConfig(
                transform_type="extract",
                input_key="$history[-1]",
                expression=r"\d+",
            ),
        )
        assert step.step_type == StepType.TRANSFORM
        assert step.is_transform_step()
        assert step.config.transform_type == "extract"

    def test_conditional_step_creation(self):
        """Test creating a CONDITIONAL step."""
        step = ConditionalStepDescription(
            number=1,
            title="Conditional Step",
            config=ConditionalStepConfig(
                branches=[
                    ConditionalBranch(condition="contains:success", next_step=2),
                    ConditionalBranch(condition="contains:error", next_step=3),
                ],
                default_step=4,
            ),
        )
        assert step.step_type == StepType.CONDITIONAL
        assert step.is_conditional_step()
        assert len(step.config.branches) == 2

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_invalid_step_config_raises_error(self):
        """Test that invalid step config raises validation error."""
        with pytest.raises(ValueError, match="TOOL steps require ToolStepConfig"):
            StepDescription(
                number=1,
                title="Invalid Tool Step",
                step_type=StepType.TOOL,
                step_config=MemoryStepConfig(  # Wrong config type
                    operation=MemoryOperation.READ,
                    memory_key="test",
                ),
            )


class TestChainBuilderStepTypes:
    """Test ChainBuilder methods for different step types."""

    def test_add_tool_step(self):
        """Test adding a tool step via ChainBuilder."""
        chain = (
            ChainBuilder()
            .add_tool_step(
                number=1,
                title="Fetch Data",
                tool_name="fetch_data",
                input_mapping={"url": "$metadata.data_url"},
                timeout=30.0,
            )
            .build()
        )

        assert len(chain.steps) == 1
        assert chain.steps[0].step_type == StepType.TOOL
        assert chain.steps[0].step_config.tool_name == "fetch_data"

    def test_add_memory_step(self):
        """Test adding a memory step via ChainBuilder."""
        chain = (
            ChainBuilder()
            .add_memory_step(
                number=1,
                title="Store Result",
                operation="write",
                memory_key="analysis_result",
                value_source="$history[-1]",
            )
            .build()
        )

        assert len(chain.steps) == 1
        assert chain.steps[0].step_type == StepType.MEMORY
        assert chain.steps[0].step_config.memory_key == "analysis_result"

    def test_add_transform_step(self):
        """Test adding a transform step via ChainBuilder."""
        chain = (
            ChainBuilder()
            .add_transform_step(
                number=1,
                title="Extract Numbers",
                transform_type="extract",
                expression=r"\d+",
            )
            .build()
        )

        assert len(chain.steps) == 1
        assert chain.steps[0].step_type == StepType.TRANSFORM
        assert chain.steps[0].step_config.expression == r"\d+"

    def test_add_conditional_step(self):
        """Test adding a conditional step via ChainBuilder."""
        chain = (
            ChainBuilder()
            .add_conditional_step(
                number=1,
                title="Branch Decision",
                branches=[
                    ("contains:approve", 2),
                    ("contains:reject", 3),
                ],
                default_step=4,
            )
            .build()
        )

        assert len(chain.steps) == 1
        assert chain.steps[0].step_type == StepType.CONDITIONAL
        assert len(chain.steps[0].step_config.branches) == 2

    def test_mixed_step_types_chain(self):
        """Test building a chain with mixed step types."""
        chain = (
            ChainBuilder()
            .add_step(
                number=1,
                title="LLM Analysis",
                aim="Analyze input data",
                reasoning_questions="What patterns exist?",
                stage_action="Analyze the data",
                example_reasoning="Example analysis",
            )
            .add_memory_step(
                number=2,
                title="Store Analysis",
                operation="write",
                memory_key="analysis",
                value_source="$history[-1]",
                dependencies=[1],
            )
            .add_tool_step(
                number=3,
                title="Fetch Additional Data",
                tool_name="fetch_data",
                input_mapping={"query": "$memory.default.analysis"},
                dependencies=[2],
            )
            .build()
        )

        assert len(chain.steps) == 3
        assert chain.steps[0].step_type == StepType.LLM
        assert chain.steps[1].step_type == StepType.MEMORY
        assert chain.steps[2].step_type == StepType.TOOL


class TestMemoryContext:
    """Test ReasoningContext memory operations."""

    def test_memory_write_and_read(self):
        """Test writing and reading from context memory."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        # Write to memory
        context.memory_write("test_key", {"value": 123})

        # Read from memory
        result = context.memory_read("test_key")
        assert result == {"value": 123}

    def test_memory_namespaces(self):
        """Test memory namespaces isolation."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        # Write to different namespaces
        context.memory_write("key", "value1", namespace="ns1")
        context.memory_write("key", "value2", namespace="ns2")

        # Read from each namespace
        assert context.memory_read("key", namespace="ns1") == "value1"
        assert context.memory_read("key", namespace="ns2") == "value2"

    def test_memory_append(self):
        """Test appending to memory list."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        # Append to a new key (creates list)
        context.memory_append("items", "first")
        context.memory_append("items", "second")

        # Read list
        result = context.memory_read("items")
        assert result == ["first", "second"]

    def test_memory_delete(self):
        """Test deleting from memory."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        # Write and delete
        context.memory_write("temp", "value")
        assert context.memory_delete("temp") is True
        assert context.memory_read("temp") is None
        assert context.memory_delete("nonexistent") is False

    def test_memory_list_keys(self):
        """Test listing memory keys."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        context.memory_write("key1", "value1")
        context.memory_write("key2", "value2")

        keys = context.memory_list()
        assert set(keys) == {"key1", "key2"}


class TestToolRegistry:
    """Test ReasoningContext tool registry."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving tools."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        # Define a simple tool
        def my_tool(arg1, arg2):
            return f"{arg1} + {arg2}"

        # Register tool
        context.register_tool("my_tool", my_tool)

        # Get tool
        tool = context.get_tool("my_tool")
        assert tool is not None
        assert tool("a", "b") == "a + b"

    def test_has_tool(self):
        """Test checking if tool exists."""
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
        )

        assert not context.has_tool("nonexistent")

        context.register_tool("my_tool", lambda: None)
        assert context.has_tool("my_tool")


class TestChainSerialization:
    """Test chain JSON serialization/deserialization."""

    # Serialization may trigger deprecation warnings from pydantic validation
    # of legacy StepDescription during roundtrip
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_to_dict_and_from_dict(self):
        """Test serializing and deserializing chain to/from dict."""
        # Create a chain
        original = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Analysis Step",
                aim="Analyze data",
                reasoning_questions="What patterns exist?",
                stage_action="Analyze",
                example_reasoning="Example",
            )
            .with_max_workers(2)
            .with_metadata(name="test_chain")
            .build()
        )

        # Serialize to dict
        data = original.to_dict()

        assert data["format_version"] == ReasoningChain.FORMAT_VERSION
        assert "carl_version" in data
        assert data["max_workers"] == 2
        assert data["metadata"]["name"] == "test_chain"
        assert len(data["steps"]) == 1

        # Deserialize from dict
        restored = ReasoningChain.from_dict(data)

        assert restored.max_workers == 2
        assert restored.metadata["name"] == "test_chain"
        assert len(restored.steps) == 1
        assert restored.steps[0].title == "Analysis Step"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_to_json_and_from_json(self):
        """Test serializing and deserializing chain to/from JSON."""
        original = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
            .build()
        )

        # Serialize to JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)
        assert "Test Step" in json_str

        # Deserialize from JSON
        restored = ReasoningChain.from_json(json_str)
        assert len(restored.steps) == 1
        assert restored.steps[0].aim == "Test aim"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_serialize_chain_with_tool_steps(self):
        """Test serializing chain with tool steps."""
        original = (
            ChainBuilder()
            .add_tool_step(
                number=1,
                title="Fetch Data",
                tool_name="fetch_api",
                input_mapping={"url": "$metadata.api_url"},
                timeout=30.0,
            )
            .build()
        )

        # Serialize and restore
        data = original.to_dict()
        restored = ReasoningChain.from_dict(data)

        assert restored.steps[0].step_type == StepType.TOOL
        assert restored.steps[0].step_config.tool_name == "fetch_api"
        assert restored.steps[0].step_config.timeout == 30.0

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_serialize_chain_with_memory_steps(self):
        """Test serializing chain with memory steps."""
        original = (
            ChainBuilder()
            .add_memory_step(
                number=1,
                title="Store Result",
                operation="write",
                memory_key="result",
                value_source="$history[-1]",
            )
            .build()
        )

        # Serialize and restore
        data = original.to_dict()
        restored = ReasoningChain.from_dict(data)

        assert restored.steps[0].step_type == StepType.MEMORY
        assert restored.steps[0].step_config.operation == MemoryOperation.WRITE

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_serialize_mixed_step_chain(self):
        """Test serializing chain with mixed step types."""
        original = (
            ChainBuilder()
            .add_step(
                number=1,
                title="LLM Step",
                aim="Analyze",
                reasoning_questions="Questions",
                stage_action="Action",
                example_reasoning="Example",
            )
            .add_tool_step(
                number=2,
                title="Tool Step",
                tool_name="my_tool",
                dependencies=[1],
            )
            .add_memory_step(
                number=3,
                title="Memory Step",
                operation="write",
                memory_key="key",
                dependencies=[2],
            )
            .build()
        )

        # Serialize and restore
        json_str = original.to_json()
        restored = ReasoningChain.from_json(json_str)

        assert len(restored.steps) == 3
        assert restored.steps[0].step_type == StepType.LLM
        assert restored.steps[1].step_type == StepType.TOOL
        assert restored.steps[2].step_type == StepType.MEMORY


class TestToolStepExecution:
    """Test tool step execution."""

    @pytest.mark.asyncio
    async def test_tool_step_execution(self):
        """Test executing a chain with a tool step."""

        # Define a simple tool
        def calculate_sum(a, b):
            return a + b

        # Create chain with tool step
        chain = (
            ChainBuilder()
            .add_tool_step(
                number=1,
                title="Calculate",
                tool_name="calculate_sum",
                input_mapping={"a": "$metadata.value_a", "b": "$metadata.value_b"},
            )
            .build()
        )

        # Create context
        mock_api = MockLLMClient("Test response")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            metadata={"value_a": 5, "value_b": 3},
        )

        # Register the tool
        context.register_tool("calculate_sum", calculate_sum)

        # Execute chain
        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 1
        assert result.step_results[0].step_type == StepType.TOOL
        assert result.step_results[0].result_data == 8


class TestMemoryStepExecution:
    """Test memory step execution."""

    @pytest.mark.asyncio
    async def test_memory_write_and_read_execution(self):
        """Test executing memory write and read steps."""
        # Create chain with memory steps
        chain = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Generate",
                aim="Generate a value",
                reasoning_questions="What value?",
                stage_action="Generate",
                example_reasoning="Value: 42",
            )
            .add_memory_step(
                number=2,
                title="Store Value",
                operation="write",
                memory_key="stored_value",
                value_source="$history[-1]",
                dependencies=[1],
            )
            .add_memory_step(
                number=3,
                title="Read Value",
                operation="read",
                memory_key="stored_value",
                dependencies=[2],
            )
            .with_max_workers(1)
            .build()
        )

        # Create context
        mock_api = MockLLMClient("Generated value is 42")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            language=Language.ENGLISH,
        )

        # Execute chain
        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 3

        # Check memory step types
        assert result.step_results[1].step_type == StepType.MEMORY
        assert result.step_results[2].step_type == StepType.MEMORY


class TestTransformStepExecution:
    """Test transform step execution."""

    @pytest.mark.asyncio
    async def test_extract_transform(self):
        """Test extract transformation."""
        chain = (
            ChainBuilder()
            .add_step(
                number=1,
                title="Generate Numbers",
                aim="Generate text with numbers",
                reasoning_questions="What numbers?",
                stage_action="Generate",
                example_reasoning="The values are 10, 20, and 30.",
            )
            .add_transform_step(
                number=2,
                title="Extract Numbers",
                transform_type="extract",
                input_key="$history[-1]",
                expression=r"\d+",
                dependencies=[1],
            )
            .with_max_workers(1)
            .build()
        )

        mock_api = MockLLMClient("The values are 100, 200, and 300.")
        context = ReasoningContext(
            outer_context="Test data",
            api=mock_api,
            model="test_key",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        assert result.step_results[1].step_type == StepType.TRANSFORM
        # The extract should find numbers from the history entry


class TestTypedStepDescriptions:
    """Test the new typed step description classes."""

    def test_llm_step_description_creation(self):
        """Test creating an LLMStepDescription."""
        step = LLMStepDescription(
            number=1,
            title="Analysis Step",
            aim="Analyze the data",
            reasoning_questions="What patterns exist?",
            stage_action="Extract insights",
            example_reasoning="Pattern analysis reveals trends",
        )

        assert step.number == 1
        assert step.title == "Analysis Step"
        assert step.aim == "Analyze the data"
        assert step.step_type == StepType.LLM
        assert step.is_llm_step()
        assert not step.is_tool_step()

    def test_llm_step_requires_aim(self):
        """Test that LLMStepDescription requires aim field."""
        with pytest.raises(ValueError, match="LLM steps require 'aim'"):
            LLMStepDescription(
                number=1,
                title="Missing Aim Step",
            )

    def test_tool_step_description_creation(self):
        """Test creating a ToolStepDescription."""
        config = ToolStepConfig(
            tool_name="calculate_sum",
            input_mapping={"a": "$metadata.x", "b": "$metadata.y"},
            parameters=[
                ToolParameter(name="a", type="int", required=True),
                ToolParameter(name="b", type="int", required=True),
            ],
        )
        step = ToolStepDescription(
            number=2,
            title="Calculate Sum",
            config=config,
            dependencies=[1],
        )

        assert step.number == 2
        assert step.title == "Calculate Sum"
        assert step.step_type == StepType.TOOL
        assert step.is_tool_step()
        assert step.config.tool_name == "calculate_sum"
        assert step.dependencies == [1]

    def test_mcp_step_description_creation(self):
        """Test creating an MCPStepDescription."""
        config = MCPStepConfig(
            server=MCPServerConfig(server_name="test_server", command="test_cmd"),
            tool_name="list_tools",
        )
        step = MCPStepDescription(
            number=3,
            title="MCP Call",
            config=config,
        )

        assert step.step_type == StepType.MCP
        assert step.is_mcp_step()
        assert step.config.tool_name == "list_tools"

    def test_memory_step_description_creation(self):
        """Test creating a MemoryStepDescription."""
        config = MemoryStepConfig(
            operation=MemoryOperation.WRITE,
            memory_key="stored_value",
            value_source="$history[-1]",
            namespace="test",
        )
        step = MemoryStepDescription(
            number=4,
            title="Store Value",
            config=config,
        )

        assert step.step_type == StepType.MEMORY
        assert step.is_memory_step()
        assert step.config.memory_key == "stored_value"
        assert step.config.namespace == "test"

    def test_transform_step_description_creation(self):
        """Test creating a TransformStepDescription."""
        config = TransformStepConfig(
            transform_type="extract",
            input_key="$history[-1]",
            expression=r"\d+",
        )
        step = TransformStepDescription(
            number=5,
            title="Extract Numbers",
            config=config,
        )

        assert step.step_type == StepType.TRANSFORM
        assert step.is_transform_step()
        assert step.config.transform_type == "extract"

    def test_conditional_step_description_creation(self):
        """Test creating a ConditionalStepDescription."""
        config = ConditionalStepConfig(
            condition_context_key="$metadata.status",
            branches=[
                ConditionalBranch(condition="success", next_step=10),
                ConditionalBranch(condition="failure", next_step=20),
            ],
            default_step=30,
        )
        step = ConditionalStepDescription(
            number=6,
            title="Check Status",
            config=config,
        )

        assert step.step_type == StepType.CONDITIONAL
        assert step.is_conditional_step()
        assert len(step.config.branches) == 2

    def test_create_step_factory_llm(self):
        """Test create_step factory for LLM steps."""
        step = create_step(
            number=1,
            title="Analysis",
            step_type=StepType.LLM,
            aim="Analyze data",
            reasoning_questions="What to do?",
            stage_action="Analyze",
            example_reasoning="Example",
        )

        assert isinstance(step, LLMStepDescription)
        assert step.step_type == StepType.LLM
        assert step.aim == "Analyze data"

    def test_create_step_factory_tool(self):
        """Test create_step factory for tool steps."""
        config = ToolStepConfig(
            tool_name="my_tool",
            input_mapping={"x": "$metadata.x"},
        )
        step = create_step(
            number=2,
            title="Tool Step",
            step_type=StepType.TOOL,
            config=config,
        )

        assert isinstance(step, ToolStepDescription)
        assert step.step_type == StepType.TOOL
        assert step.config.tool_name == "my_tool"

    def test_create_step_factory_memory(self):
        """Test create_step factory for memory steps."""
        config = MemoryStepConfig(
            operation=MemoryOperation.READ,
            memory_key="stored_key",
        )
        step = create_step(
            number=3,
            title="Read Memory",
            step_type=StepType.MEMORY,
            config=config,
        )

        assert isinstance(step, MemoryStepDescription)
        assert step.step_type == StepType.MEMORY

    def test_step_description_base_is_abstract(self):
        """Test that StepDescriptionBase cannot be instantiated directly."""
        # StepDescriptionBase.step_type is abstract
        with pytest.raises(NotImplementedError):
            base = StepDescriptionBase(number=1, title="Test")
            _ = base.step_type

    def test_get_llm_field_method(self):
        """Test get_llm_field helper method."""
        llm_step = LLMStepDescription(
            number=1,
            title="Test",
            aim="Test aim",
        )
        tool_step = ToolStepDescription(
            number=2,
            title="Tool",
            config=ToolStepConfig(tool_name="test"),
        )

        # LLM step has the field
        assert llm_step.get_llm_field("aim") == "Test aim"

        # Tool step doesn't have the field - returns default
        assert tool_step.get_llm_field("aim") == ""
        assert tool_step.get_llm_field("aim", "default_value") == "default_value"

    def test_typed_steps_in_chain(self):
        """Test that typed step descriptions work in ReasoningChain."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Analyze",
                aim="Analyze data",
                reasoning_questions="What patterns?",
                stage_action="Extract",
                example_reasoning="Patterns found",
            ),
            ToolStepDescription(
                number=2,
                title="Calculate",
                config=ToolStepConfig(
                    tool_name="sum",
                    input_mapping={"a": "$metadata.a", "b": "$metadata.b"},
                ),
                dependencies=[1],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=2)
        assert len(chain.steps) == 2
        assert chain.steps[0].step_type == StepType.LLM
        assert chain.steps[1].step_type == StepType.TOOL

    def test_any_step_description_union(self):
        """Test that AnyStepDescription accepts all step types."""
        from typing import get_args

        # Check the union contains all expected types
        from mmar_carl.models.steps import AgentSkillStepDescription

        from mmar_carl.models.steps import EvaluationStepDescription

        from mmar_carl.models.steps import (
            AgentHandoffStepDescription,
            HumanInputStepDescription,
            ParallelSamplingStepDescription,
            ToolDiscoveryStepDescription,
        )

        expected_types = {
            LLMStepDescription,
            ToolStepDescription,
            MCPStepDescription,
            MemoryStepDescription,
            TransformStepDescription,
            ConditionalStepDescription,
            StructuredOutputStepDescription,
            AgentSkillStepDescription,
            EvaluationStepDescription,
            AgentHandoffStepDescription,
            HumanInputStepDescription,
            ParallelSamplingStepDescription,
            ToolDiscoveryStepDescription,
        }

        union_args = set(get_args(AnyStepDescription))
        assert expected_types == union_args

    def test_step_dependencies_methods(self):
        """Test dependency-related methods on typed steps."""
        step1 = LLMStepDescription(number=1, title="Step 1", aim="First step")
        step2 = ToolStepDescription(
            number=2,
            title="Step 2",
            config=ToolStepConfig(tool_name="test"),
            dependencies=[1],
        )

        assert not step1.has_dependencies()
        assert step2.has_dependencies()
        assert step2.depends_on(1)
        assert not step2.depends_on(3)


class _StructuredLLMHub:
    def __init__(self, payload: str):
        self.payload = payload

    async def get_response_with_retries(self, prompt: str, retries: int = 3):
        return self.payload


class TestStructuredOutputStep:
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_legacy_step_validation_for_structured_output(self):
        step = StepDescription(
            number=1,
            title="Extract order",
            step_type=StepType.STRUCTURED_OUTPUT,
            step_config=StructuredOutputStepConfig(
                input_source="$outer_context",
                output_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
            ),
        )

        assert step.step_type == StepType.STRUCTURED_OUTPUT
        assert isinstance(step.step_config, StructuredOutputStepConfig)

    def test_structured_output_config_from_pydantic_model(self):
        class OrderModel(BaseModel):
            order_id: str
            amount: float

        config = StructuredOutputStepConfig.from_pydantic_model(OrderModel)

        assert config.schema_name == "OrderModel"
        assert config.output_schema["type"] == "object"
        assert "order_id" in config.output_schema["properties"]

    def test_structured_output_result_can_be_used_by_connected_steps(self):
        structured_step = StructuredOutputStepDescription(
            number=1,
            title="Extract metrics",
            config=StructuredOutputStepConfig(
                input_source="$outer_context",
                output_schema={
                    "type": "object",
                    "properties": {"score": {"type": "integer"}, "label": {"type": "string"}},
                    "required": ["score", "label"],
                },
                instruction="Extract score and label from input",
            ),
        )

        tool_step = ToolStepDescription(
            number=2,
            title="Build summary",
            dependencies=[1],
            config=ToolStepConfig(
                tool_name="summarize",
                input_mapping={"score": "$steps.1.result_data.score", "label": "$steps.1.result_data.label"},
            ),
        )

        chain = ReasoningChain(steps=[structured_step, tool_step], max_workers=1)
        context = ReasoningContext(
            outer_context="Score is 7 and label is good",
            api=_StructuredLLMHub('{"score": 7, "label": "good"}'),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("summarize", lambda score, label: f"{label}:{score}")

        result = chain.execute(context)
        assert result.success, result
        assert result.step_results[0].step_type == StepType.STRUCTURED_OUTPUT
        assert result.step_results[0].result_data == {"score": 7, "label": "good"}
        assert result.step_results[1].result_data == "good:7"


# ============================================================================
# LangFuse Tracing Tests
# ============================================================================


class TestLangFuseTracing:
    """Test LangFuse tracing integration."""

    def test_noop_span_when_disabled(self):
        """Test that no-op span is returned when LangFuse is disabled."""
        from mmar_carl.tracing import _NoOpSpan, is_langfuse_enabled

        # Ensure LangFuse is disabled
        assert not is_langfuse_enabled()

        noop = _NoOpSpan()
        assert noop.id == "noop"
        assert noop.trace_id == "noop-trace"

        # Test that methods don't raise errors
        child = noop.start_span(name="child")
        assert isinstance(child, _NoOpSpan)

        gen = noop.start_generation(name="gen")
        assert isinstance(gen, _NoOpSpan)

        # Test update and end don't raise
        noop.update(input="test")
        noop.end(output="test")
        noop.update_trace(metadata={"test": "value"})

    def test_create_chain_trace_returns_noop_when_disabled(self):
        """Test that create_chain_trace returns no-op when disabled."""
        from mmar_carl.tracing import create_chain_trace

        trace = create_chain_trace(
            chain_name="Test Chain",
            context_preview="test context",
            total_steps=3,
            language="en",
        )

        assert trace.id == "noop"
        assert trace.trace_id == "noop-trace"

    def test_noop_span_has_context_managers(self):
        """Test that no-op span provides working context managers."""
        from mmar_carl.tracing import _NoOpSpan

        noop = _NoOpSpan()

        # Test context managers work
        with noop.start_as_current_span(name="test"):
            pass

        with noop.start_as_current_generation(name="test"):
            pass

        with noop.start_as_current_observation(name="test"):
            pass

    def test_noop_span_update_end_pattern(self):
        """Test that update+end pattern works for no-op span."""
        from mmar_carl.tracing import _NoOpSpan

        noop = _NoOpSpan()

        # Test the update->end pattern used in executor
        generation = noop.start_observation(name="test", as_type="generation")
        generation.update(output="test output")
        generation.end()

        # Test span update->end pattern
        span = noop.start_span(name="test")
        span.update(output={"success": True})
        span.end()


class TestLangFuseTracingMocked:
    """Test LangFuse tracing with mocked client."""

    def test_trace_and_span_creation_with_mock(self, monkeypatch):
        """Test trace and span creation with mocked Langfuse client."""
        from mmar_carl import (
            LLMStepDescription,
            ReasoningChain,
            ReasoningContext,
            Language,
        )

        # Mock the Langfuse client
        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "test-trace-id"
        mock_span.trace_id = "test-trace-id"
        mock_span.start_span = MagicMock(return_value=mock_span)
        mock_span.start_generation = MagicMock(return_value=mock_span)
        mock_span.update = MagicMock()
        mock_span.update_trace = MagicMock()
        mock_client.start_span = MagicMock(return_value=mock_span)

        # Mock _get_client to return our mock
        import mmar_carl.tracing as tracing_module

        original_get_client = tracing_module._get_client
        tracing_module._get_client = MagicMock(return_value=mock_client)
        # Also mock is_langfuse_enabled to return True so _get_client doesn't return None early
        original_is_enabled = tracing_module.is_langfuse_enabled
        tracing_module.is_langfuse_enabled = MagicMock(return_value=True)

        try:
            steps = [
                LLMStepDescription(
                    number=1,
                    title="Test Step",
                    aim="Test aim",
                    reasoning_questions="Test questions",
                    stage_action="Test action",
                    example_reasoning="Test example",
                )
            ]

            chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Test Trace")
            context = ReasoningContext(
                outer_context="test data",
                api=MockLLMClient("Test response"),
                model="test",
                language=Language.ENGLISH,
            )

            result = chain.execute(context)

            # Verify that client.start_span() was called to create the trace
            mock_client.start_span.assert_called_once()
            call_kwargs = mock_client.start_span.call_args
            assert call_kwargs[1]["name"] == "Test Trace"
            # Verify trace has input with outer_context
            assert "outer_context" in call_kwargs[1]["input"]
            # Verify span was created for the step
            assert mock_span.start_span.called or result.success
            # Verify trace output was updated
            assert mock_span.update.called

        finally:
            tracing_module._get_client = original_get_client
            tracing_module.is_langfuse_enabled = original_is_enabled

    def test_trace_with_session_id(self, monkeypatch):
        """Test that session_id is passed to LangFuse trace."""
        from mmar_carl import (
            LLMStepDescription,
            ReasoningChain,
            ReasoningContext,
            Language,
        )

        mock_client = MagicMock()
        mock_span = MagicMock()
        mock_span.id = "test-trace-id"
        mock_span.trace_id = "test-trace-id"
        mock_span.start_span = MagicMock(return_value=mock_span)
        mock_span.start_generation = MagicMock(return_value=mock_span)
        mock_span.update_trace = MagicMock()
        mock_client.start_span = MagicMock(return_value=mock_span)

        import mmar_carl.tracing as tracing_module

        original_get_client = tracing_module._get_client
        tracing_module._get_client = MagicMock(return_value=mock_client)

        try:
            steps = [
                LLMStepDescription(
                    number=1,
                    title="Test Step",
                    aim="Test aim",
                    reasoning_questions="Test questions",
                    stage_action="Test action",
                    example_reasoning="Test example",
                )
            ]

            chain = ReasoningChain(
                steps=steps,
                max_workers=1,
                trace_name="Session Test",
                session_id="test-session-123",
            )
            context = ReasoningContext(
                outer_context="test data",
                api=MockLLMClient("Test response"),
                model="test",
                language=Language.ENGLISH,
            )

            result = chain.execute(context)  # noqa: F841

            # Verify session_id was passed to trace via update_trace
            mock_span.update_trace.assert_called_once_with(session_id="test-session-123")
            # Verify the span was created with the correct name
            call_kwargs = mock_client.start_span.call_args
            assert call_kwargs[1]["name"] == "Session Test"

        finally:
            tracing_module._get_client = original_get_client

    def test_flush_with_mock(self, monkeypatch):
        """Test that flush calls the client's flush method."""
        import mmar_carl.tracing as tracing_module

        mock_client = MagicMock()
        original_get_client = tracing_module._get_client
        tracing_module._get_client = MagicMock(return_value=mock_client)

        try:
            tracing_module.flush()
            mock_client.flush.assert_called_once()
        finally:
            tracing_module._get_client = original_get_client

    def test_flush_with_no_client(self, monkeypatch):
        """Test that flush doesn't raise when client is None."""
        import mmar_carl.tracing as tracing_module

        original_get_client = tracing_module._get_client
        tracing_module._get_client = MagicMock(return_value=None)

        try:
            # Should not raise
            tracing_module.flush()
        finally:
            tracing_module._get_client = original_get_client
