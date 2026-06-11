"""Tests for ``GenerationStats.population_metrics`` — per-individual
runtime breakdown (score, wall_time_s, total_tokens, llm_calls).

Surfaces the speed/quality trade-off: a chain variant that scores high
but takes 3× longer than the rest of the population is now visible
in the history directly.
"""

from __future__ import annotations

import asyncio

import pytest

from mmar_carl import (
    ChainEvolver,
    DataCase,
    IndividualMetrics,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    ToolStepConfig,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstMetric(MetricBase):
    @property
    def name(self) -> str:
        return "const"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 0.7


def _tool_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit", config=ToolStepConfig(tool_name="emit")
            ),
        ],
    )


def _ctx_factory(case: DataCase) -> ReasoningContext:
    ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
    ctx.register_tool("emit", lambda: "ok")
    return ctx


# ---------------------------------------------------------------------------
# IndividualMetrics model
# ---------------------------------------------------------------------------


class TestIndividualMetricsModel:
    def test_default_fields(self) -> None:
        m = IndividualMetrics(score=0.5)
        assert m.score == 0.5
        assert m.wall_time_s == 0.0
        assert m.total_tokens == 0
        assert m.llm_calls == 0

    def test_explicit_fields(self) -> None:
        m = IndividualMetrics(score=0.5, wall_time_s=2.5, total_tokens=400, llm_calls=2)
        assert m.score == 0.5
        assert m.wall_time_s == 2.5
        assert m.total_tokens == 400
        assert m.llm_calls == 2


# ---------------------------------------------------------------------------
# GenerationStats wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_population_metrics_paired_by_index() -> None:
    """``population_metrics[i].score == population_scores[i]``."""
    ev = ChainEvolver(
        _tool_chain(),
        SimpleDataset([DataCase(input="x"), DataCase(input="y")]),
        _ConstMetric(),
        population_size=3,
        generations=1,
        smoke_check=False,
    )
    result = await ev.evolve(_ctx_factory)
    stats = result.history[0]
    assert len(stats.population_metrics) == len(stats.population_scores)
    for score, metrics in zip(stats.population_scores, stats.population_metrics):
        assert isinstance(metrics, IndividualMetrics)
        assert metrics.score == score


@pytest.mark.asyncio
async def test_population_metrics_record_wall_time() -> None:
    """Non-zero wall times when the chain actually runs."""
    ev = ChainEvolver(
        _tool_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=2,
        generations=1,
        smoke_check=False,
    )
    result = await ev.evolve(_ctx_factory)
    stats = result.history[0]
    for m in stats.population_metrics:
        # Even a no-op tool step measurably takes some time
        assert m.wall_time_s >= 0


@pytest.mark.asyncio
async def test_tool_only_chain_records_zero_tokens_and_no_llm_calls() -> None:
    """Tool-only chains don't record token usage — metrics should reflect that."""
    ev = ChainEvolver(
        _tool_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=2,
        generations=1,
        smoke_check=False,
    )
    result = await ev.evolve(_ctx_factory)
    for m in result.history[0].population_metrics:
        assert m.total_tokens == 0
        assert m.llm_calls == 0


@pytest.mark.asyncio
async def test_population_metrics_sorted_with_scores() -> None:
    """``population_scores`` is sorted descending by design; ``population_metrics``
    must follow the same ordering (paired by index)."""

    # Use a stateful metric that returns DIFFERENT scores per individual
    class _AscendingMetric(MetricBase):
        def __init__(self) -> None:
            self._n = 0

        @property
        def name(self) -> str:
            return "asc"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            self._n += 1
            return float(self._n)

    ev = ChainEvolver(
        _tool_chain(),
        SimpleDataset([DataCase(input="x")]),
        _AscendingMetric(),
        population_size=3,
        generations=1,
        smoke_check=False,
    )
    result = await ev.evolve(_ctx_factory)
    stats = result.history[0]
    # Descending
    assert stats.population_scores == sorted(stats.population_scores, reverse=True)
    # And metrics match
    for score, metrics in zip(stats.population_scores, stats.population_metrics):
        assert metrics.score == score


@pytest.mark.asyncio
async def test_failed_individual_records_minus_inf_metrics() -> None:
    """When ``_evaluate`` blows up entirely (rare — DatasetEvaluator catches
    per-case exceptions), the individual gets score=-inf in IndividualMetrics."""

    # Force _evaluate to raise via a malformed dataset that breaks DatasetEvaluator
    class _BrokenDataset(SimpleDataset):
        def __iter__(self):
            raise RuntimeError("intentional dataset failure")

    ev = ChainEvolver(
        _tool_chain(),
        _BrokenDataset([]),
        _ConstMetric(),
        population_size=2,
        generations=1,
        smoke_check=False,
    )
    result = await ev.evolve(_ctx_factory)
    stats = result.history[0]
    # All -inf because every evaluation crashed at iteration time
    for m in stats.population_metrics:
        assert m.score == float("-inf")


# ---------------------------------------------------------------------------
# DatasetEvaluator side: new fields on CaseEvaluationResult
# ---------------------------------------------------------------------------


def test_case_evaluation_result_default_token_usage_empty() -> None:
    from mmar_carl.models.dataset import CaseEvaluationResult

    r = CaseEvaluationResult(
        case=DataCase(input="x"),
        score=0.5,
        chain_output="ok",
        success=True,
    )
    assert r.token_usage == {}
    assert r.llm_calls == 0


def test_case_evaluation_result_accepts_token_usage() -> None:
    from mmar_carl.models.dataset import CaseEvaluationResult

    r = CaseEvaluationResult(
        case=DataCase(input="x"),
        score=0.5,
        chain_output="ok",
        success=True,
        token_usage={"prompt": 100, "completion": 50, "total": 150},
        llm_calls=2,
    )
    assert r.token_usage["total"] == 150
    assert r.llm_calls == 2


def test_smoke_with_asyncio_run() -> None:
    """Sanity: the whole stack also works under asyncio.run."""
    ev = ChainEvolver(
        _tool_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=1,
        generations=1,
        smoke_check=False,
    )
    result = asyncio.run(ev.evolve(_ctx_factory))
    stats = result.history[0]
    assert len(stats.population_metrics) == 1
    assert stats.population_metrics[0].score == 0.7
