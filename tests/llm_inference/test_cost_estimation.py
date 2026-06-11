"""
Tests for ``chain.estimate_cost``.

The estimator is a dry-run analyser: walks each step, classifies it as
LLM-calling or not, computes a chars/char_per_token token estimate, and
applies optional pricing. No LLM calls are made.
"""

import pytest

from mmar_carl import (
    CostEstimate,
    ExecutionMode,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepCostEstimate,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
    ToolStepConfig,
    ToolStepDescription,
    TransformStepConfig,
    TransformStepDescription,
)
from mmar_carl.models import (
    EvaluationStepConfig,
    EvaluationStepDescription,
    ParallelSamplingAggregation,
    ParallelSamplingStepConfig,
    ParallelSamplingStepDescription,
)


class _NoopLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return ""

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return ""


def _ctx(outer: str = "x" * 400) -> ReasoningContext:
    return ReasoningContext(outer_context=outer, api=_NoopLLM())


# --------------------------------------------------------------------------- #
# Basic classification: LLM vs non-LLM
# --------------------------------------------------------------------------- #


def test_returns_one_row_per_step() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x"),
            ToolStepDescription(
                number=2,
                title="b",
                config=ToolStepConfig(tool_name="n", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert isinstance(est, CostEstimate)
    assert [s.step_number for s in est.steps] == [1, 2]


def test_non_llm_steps_have_zero_tokens_and_calls_llm_false() -> None:
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="t",
                config=ToolStepConfig(tool_name="n", parameters=[], input_mapping={}),
            ),
            MemoryStepDescription(
                number=2,
                title="m",
                config=MemoryStepConfig(operation=MemoryOperation.WRITE, memory_key="k",
                                        value_source="'\"v\"'"),
            ),
            TransformStepDescription(
                number=3,
                title="x",
                config=TransformStepConfig(transform_type="extract", expression="x"),
            ),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert all(not row.calls_llm for row in est.steps)
    assert est.total_input_tokens == 0
    assert est.total_output_tokens == 0
    assert est.total_cost_usd == 0.0
    assert est.llm_call_steps == []


def test_llm_step_classified_correctly() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].calls_llm is True
    assert est.steps[0].estimated_calls == 1
    assert est.steps[0].input_tokens > 0
    assert est.steps[0].output_tokens > 0


def test_structured_output_step_classified_as_llm() -> None:
    chain = ReasoningChain(
        steps=[
            StructuredOutputStepDescription(
                number=1,
                title="schema",
                aim="extract",
                config=StructuredOutputStepConfig(
                    output_schema={"type": "object", "properties": {"k": {"type": "string"}}}
                ),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].calls_llm is True
    assert est.steps[0].estimated_calls == 1


# --------------------------------------------------------------------------- #
# Token math
# --------------------------------------------------------------------------- #


def test_max_tokens_drives_output_token_count() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="a", aim="x",
                llm_config=LLMStepConfig(max_tokens=256),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].output_tokens == 256


def test_default_output_tokens_when_max_tokens_absent() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx(), default_output_tokens=128)
    assert est.steps[0].output_tokens == 128


def test_char_per_token_changes_input_estimate() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    ctx = _ctx(outer="x" * 1000)
    est_4 = chain.estimate_cost(ctx, char_per_token=4)
    est_2 = chain.estimate_cost(ctx, char_per_token=2)
    # Halving char_per_token roughly doubles input tokens (more tokens per chars)
    assert est_2.steps[0].input_tokens > est_4.steps[0].input_tokens


def test_rejects_invalid_char_per_token() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    with pytest.raises(ValueError, match="char_per_token"):
        chain.estimate_cost(_ctx(), char_per_token=0)


def test_rejects_negative_default_output_tokens() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    with pytest.raises(ValueError, match="default_output_tokens"):
        chain.estimate_cost(_ctx(), default_output_tokens=-1)


# --------------------------------------------------------------------------- #
# Model resolution mirrors runtime precedence
# --------------------------------------------------------------------------- #


def test_per_step_model_wins_over_chain_default() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="a", aim="x",
                llm_config=LLMStepConfig(model="step-model"),
            )
        ],
        default_llm_config=LLMStepConfig(model="chain-default"),
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].model == "step-model"


def test_chain_default_used_when_step_unset() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        default_llm_config=LLMStepConfig(model="chain-default"),
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].model == "chain-default"


def test_context_model_used_when_no_chain_default_or_per_step() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_NoopLLM(), model="ctx-model")
    est = chain.estimate_cost(ctx)
    assert est.steps[0].model == "ctx-model"


def test_default_sentinel_model_treated_as_unresolved() -> None:
    """`ReasoningContext.model` defaults to 'default' — that's not a real model name."""
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_NoopLLM())  # default model='default'
    est = chain.estimate_cost(ctx)
    assert est.steps[0].model is None


# --------------------------------------------------------------------------- #
# Pricing application
# --------------------------------------------------------------------------- #


def test_pricing_applied_per_model() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="cheap", aim="x",
                llm_config=LLMStepConfig(model="mini", max_tokens=1000),
            ),
            LLMStepDescription(
                number=2, title="pricey", aim="x",
                llm_config=LLMStepConfig(model="big", max_tokens=1000),
                dependencies=[1],
            ),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(
        _ctx(),
        pricing={"mini": (0.001, 0.002), "big": (0.01, 0.03)},
    )
    s1, s2 = est.steps
    # s2 should cost ~10x more than s1 on output alone (output rate 0.03 vs 0.002)
    assert s2.output_cost_usd > s1.output_cost_usd * 5
    assert est.total_cost_usd == round(s1.total_cost_usd + s2.total_cost_usd, 6)


def test_pricing_missing_flagged_and_zeroed() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="a", aim="x",
                llm_config=LLMStepConfig(model="unknown-model"),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx(), pricing={"other-model": (0.001, 0.001)})
    row = est.steps[0]
    assert row.pricing_missing is True
    assert row.total_cost_usd == 0.0


def test_no_pricing_dict_means_zero_cost_no_flag() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    row = est.steps[0]
    assert row.input_tokens > 0
    assert row.output_tokens > 0
    assert row.input_cost_usd == 0.0
    assert row.output_cost_usd == 0.0
    assert row.pricing_missing is False


# --------------------------------------------------------------------------- #
# Multi-call step types
# --------------------------------------------------------------------------- #


def test_parallel_sampling_step_counts_n_samples_calls() -> None:
    chain = ReasoningChain(
        steps=[
            ParallelSamplingStepDescription(
                number=1,
                title="sample",
                base_step=LLMStepDescription(number=2, title="base", aim="x"),
                config=ParallelSamplingStepConfig(
                    n_samples=5, aggregation=ParallelSamplingAggregation.MAJORITY_VOTE,
                ),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    row = est.steps[0]
    assert row.calls_llm
    assert row.estimated_calls == 5
    assert "parallel sampling" in row.note


def test_parallel_sampling_with_llm_judge_adds_one_call() -> None:
    chain = ReasoningChain(
        steps=[
            ParallelSamplingStepDescription(
                number=1,
                title="sample",
                base_step=LLMStepDescription(number=2, title="base", aim="x"),
                config=ParallelSamplingStepConfig(
                    n_samples=3, aggregation=ParallelSamplingAggregation.LLM_JUDGE,
                ),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    row = est.steps[0]
    assert row.estimated_calls == 4  # 3 samples + 1 judge
    assert "judge" in row.note


def test_rule_based_evaluation_does_not_count_as_llm_call() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x"),
            EvaluationStepDescription(
                number=2,
                title="judge",
                config=EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["non-empty"],
                    evaluation_method="rule",
                ),
                dependencies=[1],
            ),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    eval_row = est.steps[1]
    assert eval_row.calls_llm is False
    assert "rule-based" in eval_row.note


def test_llm_evaluation_counts_max_retries() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x"),
            EvaluationStepDescription(
                number=2,
                title="judge",
                config=EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["non-empty"],
                    evaluation_method="llm",
                    max_retries=2,
                ),
                dependencies=[1],
            ),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    eval_row = est.steps[1]
    assert eval_row.calls_llm is True
    assert eval_row.estimated_calls == 1 + 2


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #


def test_totals_match_sum_of_rows() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x",
                               llm_config=LLMStepConfig(model="m1", max_tokens=100)),
            LLMStepDescription(number=2, title="b", aim="y",
                               llm_config=LLMStepConfig(model="m2", max_tokens=200),
                               dependencies=[1]),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(
        _ctx(),
        pricing={"m1": (0.001, 0.002), "m2": (0.005, 0.01)},
    )
    assert est.total_input_tokens == sum(s.input_tokens for s in est.steps)
    assert est.total_output_tokens == sum(s.output_tokens for s in est.steps)
    assert est.total_cost_usd == round(sum(s.total_cost_usd for s in est.steps), 6)
    assert est.total_tokens == est.total_input_tokens + est.total_output_tokens


def test_format_table_includes_each_step_and_total_line() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="alpha", aim="x"),
            ToolStepDescription(
                number=2,
                title="beta",
                config=ToolStepConfig(tool_name="n", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=1,
    )
    table = chain.estimate_cost(_ctx()).format_table()
    assert "alpha" in table
    assert "beta" in table
    assert "TOTAL" in table


# --------------------------------------------------------------------------- #
# Row type
# --------------------------------------------------------------------------- #


def test_step_cost_estimate_is_pydantic_and_serialises() -> None:
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="a", aim="x")],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    row = est.steps[0]
    assert isinstance(row, StepCostEstimate)
    dumped = row.model_dump()
    assert dumped["step_number"] == 1
    assert dumped["calls_llm"] is True


def test_history_proxy_increases_downstream_input_estimates() -> None:
    """Each LLM-style step should see a larger context than the one before it
    because the accumulated-history proxy grows."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x",
                               llm_config=LLMStepConfig(max_tokens=400)),
            LLMStepDescription(number=2, title="b", aim="x",
                               llm_config=LLMStepConfig(max_tokens=400),
                               dependencies=[1]),
            LLMStepDescription(number=3, title="c", aim="x",
                               llm_config=LLMStepConfig(max_tokens=400),
                               dependencies=[2]),
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].input_tokens < est.steps[1].input_tokens < est.steps[2].input_tokens


# --------------------------------------------------------------------------- #
# Self-critic / step group interaction
# --------------------------------------------------------------------------- #


def test_execution_mode_does_not_inflate_call_count_for_self_critic() -> None:
    """SELF_CRITIC mode does multiple LLM calls at runtime, but the simple
    estimator deliberately stays at 1 call per LLM step. Document this via a
    test so the behaviour is intentional."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="a", aim="x",
                llm_config=LLMStepConfig(execution_mode=ExecutionMode.SELF_CRITIC),
            )
        ],
        max_workers=1,
    )
    est = chain.estimate_cost(_ctx())
    assert est.steps[0].estimated_calls == 1
