"""
Tests for EvaluationStep — inline quality gate.

Covers:
- EvalFailAction enum values
- EvaluationStepConfig model fields and validation
- Rule-based evaluation (pattern shortcuts + simpleeval expressions)
- LLM-based evaluation (mock judge, verdict parsing)
- on_fail=CONTINUE: chain continues even when criteria fail
- on_fail=ABORT: step marked as failure, chain stops
- on_fail=RETRY_WITH_FEEDBACK: LLM asked to improve; re-evaluated; replaces history entry
- ChainTestHarness integration: full chain with evaluation gate
"""

import pytest

from mmar_carl import (
    EvalFailAction,
    EvaluationStepConfig,
    EvaluationStepDescription,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.steps import ToolStepDescription
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.testing import ChainTestHarness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PassLLMClient(LLMClientBase):
    """Always returns 'Overall: PASS'."""

    async def get_response(self, prompt: str) -> str:
        return "1: PASS — ok\nOverall: PASS"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class _FailLLMClient(LLMClientBase):
    """Always returns 'Overall: FAIL' with a critique."""

    async def get_response(self, prompt: str) -> str:
        return "1: FAIL — missing\nOverall: FAIL\nCritique: Response lacks detail."

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class _RetryLLMClient(LLMClientBase):
    """
    First call (tool/prior step) returns 'short'.
    First eval call returns FAIL.
    Improvement call returns a longer text.
    Second eval call returns PASS.
    """

    def __init__(self):
        self._calls = 0

    async def get_response(self, prompt: str) -> str:
        self._calls += 1
        if "evaluator" in prompt.lower() or "assess" in prompt.lower():
            # Judge call
            return "1: FAIL — too short\nOverall: FAIL\nCritique: Need more words."
        if "improve" in prompt.lower() or "original response" in prompt.lower():
            # Improvement call
            return "This is a much longer and more detailed response with sufficient content."
        return "short"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _make_context(api=None) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test",
        api=api or _PassLLMClient(),
        model="test",
        language=Language.ENGLISH,
    )


def _eval_step(config: EvaluationStepConfig, deps=None) -> EvaluationStepDescription:
    return EvaluationStepDescription(
        number=2,
        title="Quality Gate",
        dependencies=deps or [1],
        config=config,
    )


def _tool_step(number=1, tool_name="producer") -> ToolStepDescription:
    return ToolStepDescription(
        number=number,
        title="Producer",
        config=ToolStepConfig(tool_name=tool_name, input_mapping={}),
    )


# ---------------------------------------------------------------------------
# EvalFailAction enum
# ---------------------------------------------------------------------------


class TestEvalFailAction:
    def test_values_exist(self):
        assert EvalFailAction.CONTINUE == "continue"
        assert EvalFailAction.ABORT == "abort"
        assert EvalFailAction.RETRY_WITH_FEEDBACK == "retry_with_feedback"


# ---------------------------------------------------------------------------
# EvaluationStepConfig model
# ---------------------------------------------------------------------------


class TestEvaluationStepConfig:
    def test_required_fields(self):
        cfg = EvaluationStepConfig(evaluates_step=3, criteria=["len(value) > 0"])
        assert cfg.evaluates_step == 3
        assert cfg.criteria == ["len(value) > 0"]

    def test_defaults(self):
        cfg = EvaluationStepConfig(evaluates_step=1, criteria=["nonempty"])
        assert cfg.on_fail == EvalFailAction.CONTINUE
        assert cfg.evaluation_method == "rule"
        assert cfg.input_source == "$history[-1]"
        assert cfg.max_retries == 1

    def test_criteria_must_be_nonempty(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            EvaluationStepConfig(evaluates_step=1, criteria=[])

    def test_max_retries_non_negative(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            EvaluationStepConfig(evaluates_step=1, criteria=["x"], max_retries=-1)


# ---------------------------------------------------------------------------
# Rule evaluation — pattern shortcuts
# ---------------------------------------------------------------------------


class TestRuleEvaluation:
    @pytest.mark.asyncio
    async def test_min_words_passes(self):
        text = "one two three four five"
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:5"],
                    evaluation_method="rule",
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: text)
        result = await chain.execute_async(ctx)
        assert result.success
        eval_result = result.step_results[1]
        assert "PASS" in eval_result.result

    @pytest.mark.asyncio
    async def test_min_words_fails_then_continue(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:100"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.CONTINUE,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "too short")
        result = await chain.execute_async(ctx)
        # Chain still succeeds (CONTINUE)
        assert result.success

    @pytest.mark.asyncio
    async def test_contains_criterion(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["contains:hello"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "Say hello world")
        result = await chain.execute_async(ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_contains_criterion_fail_aborts(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["contains:MISSING"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "nothing here")
        result = await chain.execute_async(ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_simpleeval_expression(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["len(value) >= 3"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "abc")
        result = await chain.execute_async(ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_multiple_criteria_all_must_pass(self):
        """Both criteria must hold for the evaluation to PASS."""
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:2", "contains:hello"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "hello world")
        result = await chain.execute_async(ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_one_failing_criterion_causes_fail(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:2", "contains:MISSING"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "hello world")
        result = await chain.execute_async(ctx)
        assert not result.success


# ---------------------------------------------------------------------------
# on_fail=ABORT
# ---------------------------------------------------------------------------


class TestOnFailAbort:
    @pytest.mark.asyncio
    async def test_abort_returns_failure(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:999"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "short")
        result = await chain.execute_async(ctx)
        assert not result.success
        failed = result.get_failed_steps()
        assert len(failed) == 1
        assert failed[0].step_number == 2


# ---------------------------------------------------------------------------
# on_fail=CONTINUE
# ---------------------------------------------------------------------------


class TestOnFailContinue:
    @pytest.mark.asyncio
    async def test_chain_continues_after_fail(self):
        """Evaluation fails but chain keeps going; step 3 still runs."""
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:999"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.CONTINUE,
                )),
                ToolStepDescription(
                    number=3,
                    title="Downstream",
                    dependencies=[2],
                    config=ToolStepConfig(tool_name="downstream", input_mapping={}),
                ),
            ]
        )
        ctx = _make_context()
        ctx.register_tool("producer", lambda: "short")
        ctx.register_tool("downstream", lambda: "downstream ok")
        result = await chain.execute_async(ctx)
        assert result.success
        assert result.step_results[2].result == "downstream ok"


# ---------------------------------------------------------------------------
# LLM evaluation
# ---------------------------------------------------------------------------


class TestLLMEvaluation:
    @pytest.mark.asyncio
    async def test_llm_pass_verdict(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["The response is informative"],
                    evaluation_method="llm",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context(api=_PassLLMClient())
        ctx.register_tool("producer", lambda: "informative text here")
        result = await chain.execute_async(ctx)
        assert result.success
        assert "PASS" in result.step_results[1].result

    @pytest.mark.asyncio
    async def test_llm_fail_verdict_aborts(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["The response must be at least 200 words"],
                    evaluation_method="llm",
                    on_fail=EvalFailAction.ABORT,
                )),
            ]
        )
        ctx = _make_context(api=_FailLLMClient())
        ctx.register_tool("producer", lambda: "too short")
        result = await chain.execute_async(ctx)
        assert not result.success

    @pytest.mark.asyncio
    async def test_llm_fail_verdict_critique_in_result_data(self):
        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["detailed"],
                    evaluation_method="llm",
                    on_fail=EvalFailAction.CONTINUE,
                )),
            ]
        )
        ctx = _make_context(api=_FailLLMClient())
        ctx.register_tool("producer", lambda: "short")
        result = await chain.execute_async(ctx)
        assert result.success  # CONTINUE
        rd = result.step_results[1].result_data
        assert rd.get("verdict") == "FAIL"
        assert "detail" in rd.get("critique", "").lower()


# ---------------------------------------------------------------------------
# on_fail=RETRY_WITH_FEEDBACK
# ---------------------------------------------------------------------------


class TestRetryWithFeedback:
    @pytest.mark.asyncio
    async def test_retry_improves_and_passes(self):
        """
        Initial output is short → fails rule → retry called → improved text
        is long enough → passes rule → PASS verdict.
        """
        improvement = "word " * 20  # 20 words, passes min_words:10

        call_counts = {"eval": 0, "improve": 0}

        class _Client(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                if "assess" in prompt.lower() or "evaluator" in prompt.lower():
                    call_counts["eval"] += 1
                    if call_counts["eval"] == 1:
                        return "1: FAIL — too short\nOverall: FAIL\nCritique: Too short."
                    return "1: PASS\nOverall: PASS"
                call_counts["improve"] += 1
                return improvement

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:10"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
                    max_retries=1,
                )),
            ]
        )
        ctx = _make_context(api=_Client())
        ctx.register_tool("producer", lambda: "short text")
        result = await chain.execute_async(ctx)
        assert result.success
        assert "PASS" in result.step_results[1].result

    @pytest.mark.asyncio
    async def test_retry_exhausted_falls_through(self):
        """After max_retries, evaluation step continues (doesn't abort)."""

        class _AlwaysFailClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                return improvement  # improved text still short

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

        improvement = "still short"

        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:100"],
                    evaluation_method="rule",
                    on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
                    max_retries=2,
                )),
            ]
        )
        ctx = _make_context(api=_AlwaysFailClient())
        ctx.register_tool("producer", lambda: "short")
        result = await chain.execute_async(ctx)
        # After exhausting retries RETRY_WITH_FEEDBACK falls through → chain continues
        assert result.success

    @pytest.mark.asyncio
    async def test_retry_stores_improved_in_result_data(self):
        """Improved response is stored in result_data['improved_response']."""
        improved = "vastly improved response text here for the test"

        class _ImprovingClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                if "original response" in prompt.lower():
                    return improved
                return "1: FAIL\nOverall: FAIL\nCritique: Too short."

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

        chain = ReasoningChain(
            steps=[
                _tool_step(),
                _eval_step(EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:99"],  # will still fail after improvement
                    evaluation_method="rule",
                    on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
                    max_retries=1,
                )),
            ]
        )
        ctx = _make_context(api=_ImprovingClient())
        ctx.register_tool("producer", lambda: "short")
        result = await chain.execute_async(ctx)
        assert result.success
        # Improved text captured in result_data
        rd = result.step_results[1].result_data
        assert "improved_response" in rd
        assert rd["improved_response"] == improved


# ---------------------------------------------------------------------------
# ChainTestHarness integration
# ---------------------------------------------------------------------------


class TestChainTestHarnessIntegration:
    @pytest.mark.asyncio
    async def test_harness_passes_rule_evaluation(self):
        """ChainTestHarness with a rule gate that passes."""
        from mmar_carl import ChainBuilder

        chain = (
            ChainBuilder()
            .add_tool_step(1, "Producer", tool_name="source", input_mapping={})
            .build()
        )
        # Add evaluation step manually
        chain.steps.append(EvaluationStepDescription(
            number=2,
            title="Gate",
            dependencies=[1],
            config=EvaluationStepConfig(
                evaluates_step=1,
                criteria=["contains:hello"],
                evaluation_method="rule",
                on_fail=EvalFailAction.ABORT,
            ),
        ))

        harness = ChainTestHarness(chain)
        harness.set_tool_response("source", "say hello world")
        await harness.run("")
        harness.assert_succeeded()

    @pytest.mark.asyncio
    async def test_step_type_is_evaluation(self):
        step = EvaluationStepDescription(
            number=1,
            title="Gate",
            config=EvaluationStepConfig(evaluates_step=0, criteria=["len(value) > 0"]),
        )
        from mmar_carl.models.enums import StepType
        assert step.step_type == StepType.EVALUATION
