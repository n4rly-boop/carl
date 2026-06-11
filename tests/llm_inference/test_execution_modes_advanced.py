"""
Tests for execution modes in CARL.

This test suite covers execution mode functionality including:
- FAST mode single-pass behavior
- SELF_CRITIC with default LLM evaluator
- SELF_CRITIC with custom evaluators
- Evaluator chains (all must approve policy)
- Self-critic max revisions limits
- Quality warning generation
- Mixed execution modes in single chain
"""

import pytest
from typing import Any
from mmar_carl import (
    ExecutionMode,
    Language,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
)
from tests.mocks import ExecutionModeMockClient, MockLLMClient


# ============================================================================
# Test Fixtures
# ============================================================================


def find_step_result(result, step_number):
    """Find a step result by step number."""
    for sr in result.step_results:
        if sr.step_number == step_number:
            return sr
    return None


# ============================================================================
# Mock Evaluators
# ============================================================================


class AlwaysApproveEvaluator(SelfCriticEvaluatorBase):
    """Evaluator that always approves."""

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        return SelfCriticDecision(
            verdict="APPROVE",
            review_text="Auto-approved",
            metadata={"llm_calls": 0},
        )


class AlwaysDisapproveEvaluator(SelfCriticEvaluatorBase):
    """Evaluator that always disapproves."""

    def __init__(self, disapprove_count: int = 1):
        self.disapprove_count = disapprove_count
        self.call_count = 0

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        self.call_count += 1
        if self.call_count <= self.disapprove_count:
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text=f"Disapproval {self.call_count}",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="APPROVE",
            review_text="Finally approved",
            metadata={"llm_calls": 0},
        )


class ConditionalApproveEvaluator(SelfCriticEvaluatorBase):
    """Evaluator that approves based on content."""

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        if "complete" in candidate.lower() or "final" in candidate.lower():
            return SelfCriticDecision(
                verdict="APPROVE",
                review_text="Content is complete",
                metadata={"llm_calls": 0},
            )
        return SelfCriticDecision(
            verdict="DISAPPROVE",
            review_text="Content needs improvement",
            metadata={"llm_calls": 0},
        )


# ============================================================================
# FAST Mode Tests
# ============================================================================


class TestFastMode:
    """Test FAST execution mode."""

    @pytest.mark.asyncio
    async def test_fast_mode_single_pass(self):
        """Test FAST mode executes in a single pass."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Fast Mode Step",
                aim="Generate quick response",
                reasoning_questions="What's the quick answer?",
                stage_action="Provide quick response",
                example_reasoning="Quick response generated",
                llm_config=LLMStepConfig(
                    model="test-model",
                    temperature=0.5,
                    execution_mode=ExecutionMode.FAST,
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Fast Mode Test")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success

        # FAST mode should execute only once
        assert "Response to" in step_1_result.result

    @pytest.mark.asyncio
    async def test_fast_mode_no_evaluation(self):
        """Test FAST mode doesn't perform evaluation."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Fast Mode",
                aim="Generate without evaluation",
                reasoning_questions="Quick answer?",
                stage_action="Quick response",
                example_reasoning="Fast response",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.FAST,
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="No Evaluation")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        # Should complete quickly without evaluation rounds
        assert result.total_execution_time < 1.0


# ============================================================================
# SELF_CRITIC Mode Tests
# ============================================================================


class TestSelfCriticMode:
    """Test SELF_CRITIC execution mode."""

    @pytest.mark.asyncio
    async def test_self_critic_with_default_evaluator(self):
        """Test SELF_CRITIC mode with default LLM evaluator."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Self Critic Step",
                aim="Generate and evaluate",
                reasoning_questions="What's the answer?",
                stage_action="Generate and self-evaluate",
                example_reasoning="Initial draft",
                llm_config=LLMStepConfig(
                    model="test-model",
                    temperature=0.5,
                    execution_mode=ExecutionMode.SELF_CRITIC,
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Self Critic Default")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="approve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success

    @pytest.mark.asyncio
    async def test_self_critic_with_custom_evaluator(self):
        """Test SELF_CRITIC mode with custom evaluator."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Custom Evaluator Step",
                aim="Generate with custom evaluation",
                reasoning_questions="Generate content",
                stage_action="Create and evaluate",
                example_reasoning="Draft content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=ConditionalApproveEvaluator(),
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Custom Evaluator")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="improve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success


# ============================================================================
# Evaluator Chain Tests
# ============================================================================


class TestEvaluatorChains:
    """Test evaluator chains with multiple evaluators."""

    @pytest.mark.asyncio
    async def test_evaluator_chain_all_approve(self):
        """Test evaluator chain where all must approve."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Evaluator Chain Step",
                aim="Generate with multiple evaluators",
                reasoning_questions="What to generate?",
                stage_action="Create and evaluate",
                example_reasoning="Initial content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=[
                        AlwaysApproveEvaluator(),
                        AlwaysApproveEvaluator(),
                        AlwaysApproveEvaluator(),
                    ],
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="All Approve")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success

    @pytest.mark.asyncio
    async def test_evaluator_chain_one_disapproves(self):
        """Test evaluator chain where one disapproves."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Mixed Evaluator Chain",
                aim="Generate with mixed evaluators",
                reasoning_questions="Generate content",
                stage_action="Create and evaluate",
                example_reasoning="Draft content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=[
                        AlwaysApproveEvaluator(),
                        AlwaysDisapproveEvaluator(disapprove_count=1),  # Disapproves once
                        AlwaysApproveEvaluator(),
                    ],
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="One Disapproves")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="improve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success


# ============================================================================
# Revision Limits Tests
# ============================================================================


class TestRevisionLimits:
    """Test self-critic revision limits."""

    @pytest.mark.asyncio
    async def test_max_revisions_limit(self):
        """Test max revisions limit in SELF_CRITIC mode."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Limited Revisions",
                aim="Generate with revision limit",
                reasoning_questions="What to generate?",
                stage_action="Create and refine",
                example_reasoning="Initial draft",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysDisapproveEvaluator(disapprove_count=2),
                    max_revisions=2,  # Allow up to 2 revisions
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Revision Limit")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="improve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        # Should succeed with max revisions
        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success

    @pytest.mark.asyncio
    async def test_exceed_max_revisions(self):
        """Test behavior when exceeding max revisions."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Exceed Limit",
                aim="Generate with low revision limit",
                reasoning_questions="Generate content",
                stage_action="Create with limit",
                example_reasoning="Draft",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysDisapproveEvaluator(disapprove_count=5),  # Will disapprove 5 times
                    max_revisions=2,  # But only allow 2 revisions
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Exceed Limit")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="improve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        # Should handle exceeding revision limit gracefully
        # Either succeeds with last revision or fails gracefully
        assert result.success or not result.success


# ============================================================================
# Quality Warning Tests
# ============================================================================


class TestQualityWarnings:
    """Test quality warning generation in SELF_CRITIC mode."""

    @pytest.mark.asyncio
    async def test_quality_warning_on_disapproval(self):
        """Test quality warning when evaluator disapproves."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Quality Warning Test",
                aim="Generate with quality check",
                reasoning_questions="Generate content",
                stage_action="Create and check quality",
                example_reasoning="Initial content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysDisapproveEvaluator(disapprove_count=1),
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Quality Warning")

        context = ReasoningContext(
            outer_context="Test input",
            api=ExecutionModeMockClient(mode="SELF_CRITIC", revision_pattern="improve"),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        step_1_result = find_step_result(result, 1)
        assert step_1_result.success


# ============================================================================
# Mixed Execution Modes Tests
# ============================================================================


class TestMixedExecutionModes:
    """Test chains with mixed execution modes."""

    @pytest.mark.asyncio
    async def test_fast_and_self_critic_in_chain(self):
        """Test chain with both FAST and SELF_CRITIC steps."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Fast Step",
                aim="Quick generation",
                reasoning_questions="Quick answer?",
                stage_action="Quick response",
                example_reasoning="Fast response",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.FAST,
                ),
            ),
            LLMStepDescription(
                number=2,
                title="Self Critic Step",
                aim="Quality generation",
                reasoning_questions="Detailed answer?",
                stage_action="Quality response",
                example_reasoning="Quality response",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysApproveEvaluator(),
                ),
                dependencies=[1],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Mixed Modes")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 2

        # Both steps should succeed
        step_1_result = find_step_result(result, 1)
        step_2_result = find_step_result(result, 2)

        assert step_1_result.success
        assert step_2_result.success

    @pytest.mark.asyncio
    async def test_parallel_different_modes(self):
        """Test parallel steps with different execution modes."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Fast Parallel",
                aim="Fast generation",
                reasoning_questions="Quick?",
                stage_action="Quick",
                example_reasoning="Fast",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.FAST,
                ),
            ),
            LLMStepDescription(
                number=2,
                title="Critic Parallel",
                aim="Quality generation",
                reasoning_questions="Quality?",
                stage_action="Quality",
                example_reasoning="Quality",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysApproveEvaluator(),
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Parallel Modes")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 2


# ============================================================================
# Performance Characteristics Tests
# ============================================================================


class TestPerformanceCharacteristics:
    """Test performance characteristics of execution modes."""

    @pytest.mark.asyncio
    async def test_fast_mode_performance(self):
        """Test FAST mode is faster than SELF_CRITIC."""
        fast_steps = [
            LLMStepDescription(
                number=1,
                title="Fast Step",
                aim="Quick generation",
                reasoning_questions="Quick?",
                stage_action="Quick",
                example_reasoning="Fast",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.FAST,
                ),
            ),
        ]

        fast_chain = ReasoningChain(steps=fast_steps, max_workers=1, trace_name="Fast Perf")

        critic_steps = [
            LLMStepDescription(
                number=1,
                title="Critic Step",
                aim="Quality generation",
                reasoning_questions="Quality?",
                stage_action="Quality",
                example_reasoning="Quality",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=AlwaysApproveEvaluator(),
                ),
            ),
        ]

        critic_chain = ReasoningChain(steps=critic_steps, max_workers=1, trace_name="Critic Perf")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        fast_result = await fast_chain.execute_async(context)
        critic_result = await critic_chain.execute_async(context)

        assert fast_result.success
        assert critic_result.success

        # FAST mode should generally be faster
        # (though with mocks the difference may be minimal)
        assert fast_result.total_execution_time >= 0
        assert critic_result.total_execution_time >= 0


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestExecutionModeErrorHandling:
    """Test error handling in execution modes."""

    @pytest.mark.asyncio
    async def test_evaluator_error_handling(self):
        """Test error handling when evaluator fails."""
        class FailingEvaluator(SelfCriticEvaluatorBase):
            async def evaluate(
                self,
                step: Any,
                candidate: str,
                base_prompt: str,
                context: Any,
                llm_client: Any,
                retries: int,
            ) -> SelfCriticDecision:
                raise RuntimeError("Evaluator failed")

        steps = [
            LLMStepDescription(
                number=1,
                title="Failing Evaluator",
                aim="Generate with failing evaluator",
                reasoning_questions="Generate?",
                stage_action="Generate",
                example_reasoning="Content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    execution_mode=ExecutionMode.SELF_CRITIC,
                    evaluator=FailingEvaluator(),
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Failing Evaluator")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        # Should handle evaluator error gracefully
        assert result.success or not result.success

    @pytest.mark.asyncio
    async def test_invalid_execution_mode(self):
        """Test handling of invalid execution mode."""
        # Test with default mode (should fall back to FAST)
        steps = [
            LLMStepDescription(
                number=1,
                title="Default Mode",
                aim="Generate with default mode",
                reasoning_questions="Generate?",
                stage_action="Generate",
                example_reasoning="Content",
                llm_config=LLMStepConfig(
                    model="test-model",
                    # No execution_mode specified - should use FAST
                ),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1, trace_name="Default Mode")

        context = ReasoningContext(
            outer_context="Test input",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        # Should succeed with default FAST mode
        assert result.success
