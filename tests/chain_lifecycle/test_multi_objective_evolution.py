"""Tests for multi-objective ChainEvolver.

Accepts ``metric: MetricBase | list[MetricBase]`` and an optional
``fitness_fn`` so users can trade off competing objectives (e.g.
``score - 0.1 * cost_usd``). Single-metric calls keep the legacy
``DatasetEvaluator``-based path; multi-metric calls drive the chain
once per case and apply every metric to the same ``ReasoningResult``.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ChainEvolver,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.metrics import MetricBase
from mmar_carl.models.dataset import DataCase, SimpleDataset
from mmar_carl.models.llm_client_base import LLMClientBase


class _FakeClient(LLMClientBase):
    """Deterministic in-memory client used to drive the evolver loop."""

    @property
    def model_name(self) -> str:
        return "fake"

    async def get_response(self, prompt: str) -> str:
        return "the answer is 42"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "the answer is 42"


class _AlwaysScore(MetricBase):
    """Metric that always returns a fixed score (useful for tests)."""

    def __init__(self, name: str, value: float) -> None:
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        return self._value


def _make_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="Solve", aim="Answer."),
    ])


def _make_dataset(n: int = 1) -> SimpleDataset:
    return SimpleDataset([
        DataCase(input=f"q{i}", label=f"c{i}") for i in range(n)
    ])


def _factory(case: DataCase) -> ReasoningContext:
    return ReasoningContext(outer_context=case.input, api=_FakeClient())


# ---------------------------------------------------------------------------
# Constructor: validates input + normalises metric
# ---------------------------------------------------------------------------


class TestConstructorAcceptsBoth:
    def test_single_metric_legacy_path(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=_AlwaysScore("acc", 0.5),
            population_size=1, generations=1, smoke_check=False,
        )
        assert len(ev.metrics) == 1
        assert ev.metric.name == "acc"

    def test_list_metric_multi_objective_path(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=[_AlwaysScore("acc", 0.5), _AlwaysScore("speed", 0.8)],
            population_size=1, generations=1, smoke_check=False,
        )
        assert len(ev.metrics) == 2
        # ``self.metric`` aliases the first one for legacy callsites
        assert ev.metric.name == "acc"

    def test_empty_metric_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ChainEvolver(
                base_chain=_make_chain(), dataset=_make_dataset(),
                metric=[],
                population_size=1, generations=1, smoke_check=False,
            )

    def test_duplicate_metric_names_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate metric name"):
            ChainEvolver(
                base_chain=_make_chain(), dataset=_make_dataset(),
                metric=[_AlwaysScore("acc", 0.5), _AlwaysScore("acc", 0.8)],
                population_size=1, generations=1, smoke_check=False,
            )


# ---------------------------------------------------------------------------
# Default fitness_fn = mean of all metric scores
# ---------------------------------------------------------------------------


class TestDefaultFitness:
    @pytest.mark.asyncio
    async def test_default_is_mean_for_multi_metric(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=[_AlwaysScore("a", 0.4), _AlwaysScore("b", 1.0)],
            population_size=1, generations=1, smoke_check=False,
        )
        result = await ev.evolve(context_factory=_factory)
        # Mean of [0.4, 1.0] = 0.7
        assert pytest.approx(result.best_score, abs=1e-9) == 0.7

    @pytest.mark.asyncio
    async def test_single_metric_default_unchanged(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=_AlwaysScore("acc", 0.42),
            population_size=1, generations=1, smoke_check=False,
        )
        result = await ev.evolve(context_factory=_factory)
        assert pytest.approx(result.best_score, abs=1e-9) == 0.42


# ---------------------------------------------------------------------------
# Custom fitness_fn for true trade-off
# ---------------------------------------------------------------------------


class TestCustomFitnessFn:
    @pytest.mark.asyncio
    async def test_weighted_sum(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=[_AlwaysScore("score", 0.8), _AlwaysScore("cost", 0.2)],
            fitness_fn=lambda s: s["score"] - 0.5 * s["cost"],
            population_size=1, generations=1, smoke_check=False,
        )
        result = await ev.evolve(context_factory=_factory)
        # 0.8 - 0.5 * 0.2 = 0.7
        assert pytest.approx(result.best_score, abs=1e-9) == 0.7

    @pytest.mark.asyncio
    async def test_fitness_fn_receives_metric_names_as_keys(self) -> None:
        captured: dict[str, float] = {}

        def my_fitness(scores: dict[str, float]) -> float:
            captured.update(scores)
            return 0.5

        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=[_AlwaysScore("alpha", 0.3), _AlwaysScore("beta", 0.6)],
            fitness_fn=my_fitness,
            population_size=1, generations=1, smoke_check=False,
        )
        await ev.evolve(context_factory=_factory)
        assert set(captured.keys()) == {"alpha", "beta"}
        assert captured["alpha"] == 0.3
        assert captured["beta"] == 0.6


# ---------------------------------------------------------------------------
# IndividualMetrics.scores_by_metric populated correctly
# ---------------------------------------------------------------------------


class TestScoresByMetric:
    @pytest.mark.asyncio
    async def test_multi_metric_path_populates_scores_by_metric(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(n=2),
            metric=[_AlwaysScore("a", 0.5), _AlwaysScore("b", 0.9)],
            population_size=2, generations=1, smoke_check=False,
        )
        result = await ev.evolve(context_factory=_factory)
        # Every individual has both metrics scored
        for im in result.history[0].population_metrics:
            assert set(im.scores_by_metric.keys()) == {"a", "b"}
            assert im.scores_by_metric["a"] == 0.5
            assert im.scores_by_metric["b"] == 0.9

    @pytest.mark.asyncio
    async def test_single_metric_path_leaves_scores_by_metric_empty(self) -> None:
        """Backward compat — the single-metric path keeps using the
        legacy DatasetEvaluator-driven evaluation, which doesn't fill
        the new field."""
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=_AlwaysScore("only", 0.7),
            population_size=1, generations=1, smoke_check=False,
        )
        result = await ev.evolve(context_factory=_factory)
        assert result.history[0].population_metrics[0].scores_by_metric == {}


# ---------------------------------------------------------------------------
# Metric failure inside one case doesn't kill the run
# ---------------------------------------------------------------------------


class _RaisingMetric(MetricBase):
    @property
    def name(self) -> str:
        return "raises"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        raise RuntimeError("metric crashed")


class TestMetricFailureTolerated:
    @pytest.mark.asyncio
    async def test_other_metric_still_scored(self) -> None:
        ev = ChainEvolver(
            base_chain=_make_chain(), dataset=_make_dataset(),
            metric=[_RaisingMetric(), _AlwaysScore("ok", 0.6)],
            population_size=1, generations=1, smoke_check=False,
        )
        # Should NOT raise — the broken metric just contributes 0.0.
        result = await ev.evolve(context_factory=_factory)
        im = result.history[0].population_metrics[0]
        assert im.scores_by_metric["raises"] == 0.0
        assert im.scores_by_metric["ok"] == 0.6


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


class TestSchema:
    def test_scores_by_metric_round_trips(self) -> None:
        from mmar_carl.chain_evolution import IndividualMetrics

        im = IndividualMetrics(
            score=0.7, scores_by_metric={"acc": 0.8, "cost": 0.6},
        )
        rehydrated = IndividualMetrics.model_validate(im.model_dump())
        assert rehydrated.scores_by_metric == {"acc": 0.8, "cost": 0.6}

    def test_scores_by_metric_defaults_to_empty(self) -> None:
        from mmar_carl.chain_evolution import IndividualMetrics

        im = IndividualMetrics(score=0.5)
        assert im.scores_by_metric == {}
