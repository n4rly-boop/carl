"""Focused tests for chain-level RE-PLAN policy."""

from __future__ import annotations

from typing import Any

import pytest

from mmar_carl import (
    ExecutionMode,
    Language,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanBudgetConfig,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanVerdict,
    ReasoningChain,
    ReasoningContext,
    RuleBasedReplanCheckerConfig,
)
from mmar_carl.chain import ChainBuilder
from mmar_carl.models.replan import LLMReplanCheckerConfig
from mmar_carl.replan import CheckerVote, LLMReplanChecker, aggregate_replan_votes


class SequenceMockLLMClient(LLMClientBase):
    """Simple deterministic mock for step generations."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        self.calls += 1
        self.prompts.append(prompt)
        if not self.responses:
            return ""
        if self.calls <= len(self.responses):
            return self.responses[self.calls - 1]
        return self.responses[-1]


class MixedModeReplanMockLLMClient(LLMClientBase):
    """Mock that supports FAST + SELF_CRITIC flows with a RE-PLAN retry."""

    def __init__(self):
        self.prompts: list[str] = []
        self.non_critic_calls = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        _ = retries
        self.prompts.append(prompt)
        lowered = prompt.lower()

        if "strict reviewer of an llm answer" in lowered:
            if "draft-v1" in lowered:
                return '{"verdict":"DISAPPROVE","review":"Need clearer answer"}'
            return '{"verdict":"APPROVE","review":"Looks good"}'

        if "regenerate the same task output with higher quality" in lowered:
            return "draft-v2"

        self.non_critic_calls += 1
        if self.non_critic_calls == 1:
            return "bad-fast"
        if self.non_critic_calls == 2:
            return "good-fast"
        if self.non_critic_calls == 3:
            return "draft-v1"
        return "draft-v2"


class AlwaysRetryChecker(ReplanCheckerBase):
    """Custom checker for loop/budget tests."""

    async def evaluate(self, checker_input: ReplanCheckerInput, context: Any) -> ReplanVerdict:
        _ = checker_input, context
        return ReplanVerdict(
            action=ReplanAction.RETRY_CURRENT_STEP,
            reason="Always retry",
            confidence=1.0,
        )


def _single_llm_step_chain(mode: ExecutionMode = ExecutionMode.FAST) -> ReasoningChain:
    step = LLMStepDescription(
        number=1,
        title="Mode Step",
        aim="Produce answer",
        reasoning_questions="What is the answer?",
        stage_action="Generate",
        example_reasoning="Example",
        llm_config=LLMStepConfig(execution_mode=mode, self_critic_evaluators=["llm"], self_critic_max_revisions=1),
    )
    return ReasoningChain(steps=[step], max_workers=1)


def test_backward_compatibility_when_replan_disabled():
    client = SequenceMockLLMClient(["draft-v1"])
    context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)
    chain = _single_llm_step_chain(ExecutionMode.FAST)

    result = chain.execute(context)

    assert result.success
    assert result.get_final_output() == "draft-v1"
    assert result.replan_events == []
    assert result.metadata.get("replan", {}).get("enabled") in {None, False}


def test_single_checker_replan_retry_current_step():
    client = SequenceMockLLMClient(["draft-v1", "draft-v2"])
    context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="result_guard",
                result_substrings=["draft-v1"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
                feedback_on_match=["Provide a clearer final answer."],
            )
        ],
    )
    chain = ReasoningChain(steps=_single_llm_step_chain().steps, max_workers=1, replan_policy=policy)

    result = chain.execute(context)

    assert result.success
    assert result.get_final_output() == "draft-v2"
    assert result.metadata["replan"]["chain_replans"] >= 1
    assert any(event.final_action == ReplanAction.RETRY_CURRENT_STEP for event in result.replan_events)


def test_aggregation_strategies_any_all_kofn_and_mandatory():
    votes = [
        CheckerVote(checker_name="a", verdict=ReplanVerdict(action=ReplanAction.RETRY_CURRENT_STEP, reason="a")),
        CheckerVote(checker_name="b", verdict=ReplanVerdict(action=ReplanAction.CONTINUE, reason="b")),
        CheckerVote(checker_name="c", verdict=ReplanVerdict(action=ReplanAction.REPLAN_FROM_CHECKPOINT, reason="c")),
    ]

    any_result = aggregate_replan_votes(votes, ReplanAggregationConfig(strategy=ReplanAggregationStrategy.ANY))
    assert any_result.triggered

    all_result = aggregate_replan_votes(votes, ReplanAggregationConfig(strategy=ReplanAggregationStrategy.ALL))
    assert not all_result.triggered

    kofn_result = aggregate_replan_votes(
        votes,
        ReplanAggregationConfig(strategy=ReplanAggregationStrategy.K_OF_N, k=2),
    )
    assert kofn_result.triggered

    mandatory_result = aggregate_replan_votes(
        votes,
        ReplanAggregationConfig(
            strategy=ReplanAggregationStrategy.MANDATORY_PLUS_K_OF_REST,
            mandatory_checkers=["a"],
            k=1,
        ),
    )
    assert mandatory_result.triggered


def test_rollback_to_named_checkpoint():
    calls = {"step1": 0, "step2": 0, "step3": 0}

    def step1_tool() -> str:
        calls["step1"] += 1
        return "checkpoint-ready"

    def step2_tool() -> str:
        calls["step2"] += 1
        if calls["step2"] == 1:
            return "bad-pass"
        return "good-pass"

    def step3_tool(value: str) -> str:
        calls["step3"] += 1
        return f"final:{value}"

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="bad_pass_guard",
                result_substrings=["bad-pass"],
                action_on_match=ReplanAction.REPLAN_FROM_CHECKPOINT,
                rollback_target_on_match=ReplanRollbackTarget(
                    target_type=ReplanTargetType.NAMED_CHECKPOINT,
                    checkpoint_name="cp1",
                ),
                feedback_on_match=["Regenerate with corrected value."],
            )
        ],
    )

    chain = (
        ChainBuilder()
        .add_tool_step(number=1, title="Checkpoint", tool_name="step1", checkpoint=True, checkpoint_name="cp1")
        .add_tool_step(number=2, title="Flaky", tool_name="step2", dependencies=[1])
        .add_tool_step(number=3, title="Finalize", tool_name="step3", dependencies=[2], input_mapping={"value": "$steps.2.result"})
        .with_replan_policy(policy)
        .build()
    )

    context = ReasoningContext(outer_context="input", api=SequenceMockLLMClient(["unused"]), model="unused")
    context.register_tool("step1", step1_tool)
    context.register_tool("step2", step2_tool)
    context.register_tool("step3", step3_tool)

    result = chain.execute(context)

    assert result.success
    assert calls["step1"] == 1
    assert calls["step2"] == 2
    assert calls["step3"] == 1
    assert result.get_final_output() == "final:good-pass"
    assert any(
        event.rollback_target and event.rollback_target.target_type == ReplanTargetType.NAMED_CHECKPOINT
        for event in result.replan_events
    )


def test_budget_exhaustion_fails_gracefully():
    client = SequenceMockLLMClient(["loop", "loop", "loop"])
    context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="loop_guard",
                result_substrings=["loop"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            )
        ],
        budgets=ReplanBudgetConfig(max_replans_per_chain=1, max_replans_per_step=1),
    )

    chain = ReasoningChain(steps=_single_llm_step_chain().steps, max_workers=1, replan_policy=policy)
    result = chain.execute(context)

    assert not result.success
    assert result.metadata["replan"]["failed"] is True
    assert "max_replans_per_chain" in result.metadata["replan"]["failure_reason"]
    assert any("RE-PLAN failure" in (step.error_message or "") for step in result.get_failed_steps())


@pytest.mark.asyncio
async def test_llm_checker_structured_verdict_parsing():
    class StructuredCheckerClient(LLMClientBase):
        async def get_response(self, prompt: str) -> str:
            return await self.get_response_with_retries(prompt, retries=1)

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            _ = prompt, retries
            return (
                '{"action":"replan_from_checkpoint",'
                '"reason":"Need rollback",'
                '"confidence":0.88,'
                '"suggested_target":{"target_type":"chain_start"},'
                '"regeneration_hints":["Use cleaner assumptions"]}'
            )

    checker = LLMReplanChecker(LLMReplanCheckerConfig(name="llm_structured"))
    context = ReasoningContext(outer_context="input", api=StructuredCheckerClient(), model="unused")
    checker_input = ReplanCheckerInput(
        step_number=1,
        step_title="Step",
        step_type=LLMStepDescription(
            number=1,
            title="Tmp",
            aim="A",
            reasoning_questions="Q",
            stage_action="S",
            example_reasoning="E",
        ).step_type,
        step_success=True,
        step_result="candidate",
    )

    verdict = await checker.evaluate(checker_input, context)

    assert verdict.action == ReplanAction.REPLAN_FROM_CHECKPOINT
    assert verdict.suggested_target is not None
    assert verdict.suggested_target.target_type == ReplanTargetType.CHAIN_START
    assert verdict.confidence == pytest.approx(0.88)


def test_replan_interacts_with_fast_and_self_critic_modes():
    client = MixedModeReplanMockLLMClient()
    context = ReasoningContext(outer_context="input", api=client, model="unused", language=Language.ENGLISH)

    steps = [
        LLMStepDescription(
            number=1,
            title="Fast step",
            aim="Initial answer",
            reasoning_questions="Question",
            stage_action="Act",
            example_reasoning="Example",
            llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
        ),
        LLMStepDescription(
            number=2,
            title="Self critic step",
            dependencies=[1],
            aim="Improve answer",
            reasoning_questions="Question",
            stage_action="Act",
            example_reasoning="Example",
            llm_config=LLMStepConfig(
                execution_mode=ExecutionMode.SELF_CRITIC,
                self_critic_evaluators=["llm"],
                self_critic_max_revisions=1,
            ),
        ),
    ]

    policy = ReplanPolicy(
        enabled=True,
        checkers=[
            RuleBasedReplanCheckerConfig(
                name="fast_guard",
                result_substrings=["bad-fast"],
                action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            )
        ],
    )

    chain = ReasoningChain(steps=steps, max_workers=1, replan_policy=policy)
    result = chain.execute(context)

    assert result.success, result
    assert result.get_final_output() == "draft-v2"
    assert context.metadata["execution_mode_details"]["1"]["execution_mode"] == ExecutionMode.FAST.value
    assert context.metadata["execution_mode_details"]["2"]["execution_mode"] == ExecutionMode.SELF_CRITIC.value
    assert any(event.final_action == ReplanAction.RETRY_CURRENT_STEP for event in result.replan_events)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_serialization_roundtrip_preserves_replan_policy_and_checkpoint_fields():
    policy = ReplanPolicy(
        enabled=True,
        checkers=[RuleBasedReplanCheckerConfig(name="guard", result_substrings=["x"])],
    )
    chain = (
        ChainBuilder()
        .add_step(
            number=1,
            title="Checkpoint Step",
            aim="Analyze",
            reasoning_questions="Q",
            stage_action="Act",
            example_reasoning="E",
            checkpoint=True,
            checkpoint_name="cp-main",
            replan_enabled=True,
        )
        .with_replan_policy(policy)
        .build()
    )

    payload = chain.to_dict()
    restored = ReasoningChain.from_dict(payload)

    assert payload["replan_policy"]["enabled"] is True
    assert restored.replan_policy is not None
    assert restored.replan_policy.enabled is True
    assert restored.steps[0].checkpoint is True
    assert restored.steps[0].checkpoint_name == "cp-main"
    assert restored.steps[0].replan_enabled is True
