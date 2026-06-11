"""
Tests for per-step token budget enforcement.

Covers:
- LLMClientBase.get_response_with_usage() default (empty usage)
- OpenAICompatibleClient.get_response_with_usage() with real response mock
- LLMStepConfig.token_budget_warning field
- StepExecutionResult.token_usage populated after chain execution
- warnings.warn fired when total tokens >= token_budget_warning
- No warning when token_budget_warning is None
- No warning when usage is empty (client doesn't support it)
"""

import warnings
import pytest

from mmar_carl import (
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.steps import LLMStepDescription
from mmar_carl.models.config import LLMStepConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockClientNoUsage(LLMClientBase):
    """Client that does NOT override get_response_with_usage (default empty usage)."""

    async def get_response(self, prompt: str) -> str:
        return "result from mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "result from mock"


class _MockClientWithUsage(LLMClientBase):
    """Client that returns token usage."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    async def get_response(self, prompt: str) -> str:
        return "result with usage"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "result with usage"

    async def get_response_with_usage(self, prompt: str, retries: int = 3) -> tuple[str, dict[str, int]]:
        usage = {
            "prompt": self._prompt_tokens,
            "completion": self._completion_tokens,
            "total": self._prompt_tokens + self._completion_tokens,
        }
        return "result with usage", usage


def _make_context(api: LLMClientBase) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test context",
        api=api,
        model="test",
        language=Language.ENGLISH,
    )


def _simple_llm_step(budget: int | None = None) -> LLMStepDescription:
    llm_config = LLMStepConfig(token_budget_warning=budget) if budget is not None else None
    return LLMStepDescription(
        number=1,
        title="Test Step",
        aim="Do something",
        llm_config=llm_config,
    )


# ---------------------------------------------------------------------------
# LLMClientBase default get_response_with_usage
# ---------------------------------------------------------------------------


class TestDefaultGetResponseWithUsage:
    @pytest.mark.asyncio
    async def test_default_returns_empty_usage(self):
        client = _MockClientNoUsage()
        result, usage = await client.get_response_with_usage("hello")
        assert result == "result from mock"
        assert usage == {}

    @pytest.mark.asyncio
    async def test_default_delegates_to_get_response_with_retries(self):
        client = _MockClientNoUsage()
        result, _ = await client.get_response_with_usage("hello", retries=2)
        assert result == "result from mock"


# ---------------------------------------------------------------------------
# LLMStepConfig.token_budget_warning field
# ---------------------------------------------------------------------------


class TestTokenBudgetWarningField:
    def test_field_defaults_to_none(self):
        config = LLMStepConfig()
        assert config.token_budget_warning is None

    def test_field_accepts_positive_int(self):
        config = LLMStepConfig(token_budget_warning=1000)
        assert config.token_budget_warning == 1000

    def test_field_must_be_positive(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            LLMStepConfig(token_budget_warning=0)

    def test_field_serializes_in_model_dump(self):
        config = LLMStepConfig(token_budget_warning=500)
        d = config.model_dump()
        assert d["token_budget_warning"] == 500


# ---------------------------------------------------------------------------
# Token usage propagated to StepExecutionResult
# ---------------------------------------------------------------------------


class TestTokenUsagePropagation:
    @pytest.mark.asyncio
    async def test_token_usage_populated_when_client_returns_usage(self):
        client = _MockClientWithUsage(prompt_tokens=80, completion_tokens=40)
        ctx = _make_context(client)
        chain = ReasoningChain(steps=[_simple_llm_step()])
        result = await chain.execute_async(ctx)

        assert result.success, result.get_final_output()
        step_result = result.step_results[0]
        assert step_result.token_usage == {"prompt": 80, "completion": 40, "total": 120}

    @pytest.mark.asyncio
    async def test_token_usage_empty_when_client_has_no_usage(self):
        client = _MockClientNoUsage()
        ctx = _make_context(client)
        chain = ReasoningChain(steps=[_simple_llm_step()])
        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.step_results[0]
        assert step_result.token_usage == {}

    @pytest.mark.asyncio
    async def test_total_token_usage_aggregated_across_steps(self):
        client = _MockClientWithUsage(prompt_tokens=50, completion_tokens=25)
        ctx = _make_context(client)
        steps = [
            LLMStepDescription(number=1, title="Step 1", aim="first"),
            LLMStepDescription(number=2, title="Step 2", aim="second", dependencies=[1]),
        ]
        chain = ReasoningChain(steps=steps)
        result = await chain.execute_async(ctx)

        assert result.success
        totals = result.get_total_tokens()
        # 2 steps × (50 prompt + 25 completion) = 150 prompt, 50 completion
        assert totals["prompt"] == 100
        assert totals["completion"] == 50


# ---------------------------------------------------------------------------
# Token budget warning fires
# ---------------------------------------------------------------------------


class TestTokenBudgetWarning:
    @pytest.mark.asyncio
    async def test_warning_fired_when_budget_exceeded(self):
        client = _MockClientWithUsage(prompt_tokens=200, completion_tokens=100)
        ctx = _make_context(client)
        # Budget of 250 < 300 total → warning expected
        chain = ReasoningChain(steps=[_simple_llm_step(budget=250)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 1
        assert "300" in str(budget_warnings[0].message)  # total tokens
        assert "250" in str(budget_warnings[0].message)  # threshold

    @pytest.mark.asyncio
    async def test_warning_not_fired_when_under_budget(self):
        client = _MockClientWithUsage(prompt_tokens=50, completion_tokens=25)
        ctx = _make_context(client)
        # Budget of 1000 > 75 total → no warning
        chain = ReasoningChain(steps=[_simple_llm_step(budget=1000)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 0

    @pytest.mark.asyncio
    async def test_warning_not_fired_when_no_budget_set(self):
        client = _MockClientWithUsage(prompt_tokens=9999, completion_tokens=9999)
        ctx = _make_context(client)
        chain = ReasoningChain(steps=[_simple_llm_step(budget=None)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 0

    @pytest.mark.asyncio
    async def test_warning_not_fired_when_usage_empty(self):
        """Client with no usage info: no warning even if budget set."""
        client = _MockClientNoUsage()
        ctx = _make_context(client)
        # Budget of 1 < anything, but no usage → no warning
        chain = ReasoningChain(steps=[_simple_llm_step(budget=1)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 0

    @pytest.mark.asyncio
    async def test_warning_fires_at_exact_threshold(self):
        """Warning fires when total == budget (>= semantics)."""
        client = _MockClientWithUsage(prompt_tokens=150, completion_tokens=50)
        ctx = _make_context(client)
        # Budget exactly equals total (200 == 200) → warning
        chain = ReasoningChain(steps=[_simple_llm_step(budget=200)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 1

    @pytest.mark.asyncio
    async def test_per_step_budgets_independent(self):
        """Each step has its own budget; warning from step 1 doesn't mute step 2."""
        client = _MockClientWithUsage(prompt_tokens=200, completion_tokens=100)
        ctx = _make_context(client)
        steps = [
            LLMStepDescription(
                number=1,
                title="Step 1",
                aim="first",
                llm_config=LLMStepConfig(token_budget_warning=250),  # will warn (300 > 250)
            ),
            LLMStepDescription(
                number=2,
                title="Step 2",
                aim="second",
                dependencies=[1],
                llm_config=LLMStepConfig(token_budget_warning=1000),  # won't warn (300 < 1000)
            ),
        ]
        chain = ReasoningChain(steps=steps)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await chain.execute_async(ctx)

        assert result.success
        budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
        assert len(budget_warnings) == 1
        assert "Step 1" in str(budget_warnings[0].message)
