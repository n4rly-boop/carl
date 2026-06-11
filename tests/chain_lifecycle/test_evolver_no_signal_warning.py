"""Tests for the no-fitness-signal warning in ``ChainEvolver``.

When every individual in a generation scores identically, selection has no
signal and subsequent generations waste tokens. The live benchmark spent
~7 minutes producing ``best=0.00 mean=0.00`` lines before the user could
diagnose the cause. The warning surfaces this loudly the first time it's
observed.
"""

from __future__ import annotations

import warnings

import pytest

from mmar_carl import (
    ChainEvolver,
    DataCase,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.chain_evolution import ChainEvolver as _ChainEvolverModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstMetric(MetricBase):
    def __init__(self, score: float = 0.0) -> None:
        self._score = score

    @property
    def name(self) -> str:
        return "const"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return self._score


def _make_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="noop",
                config=ToolStepConfig(tool_name="noop"),
            ),
        ],
    )


def _ctx_factory(case: DataCase) -> ReasoningContext:
    ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
    ctx.register_tool("noop", lambda: "ok")
    return ctx


# ---------------------------------------------------------------------------
# _scores_are_flat unit tests
# ---------------------------------------------------------------------------


class TestScoresAreFlat:
    def test_single_score_not_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.5]) is False

    def test_all_identical_zero_is_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.0, 0.0, 0.0]) is True

    def test_all_identical_nonzero_is_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.7, 0.7, 0.7]) is True

    def test_differing_scores_not_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.0, 0.5, 1.0]) is False

    def test_microscopic_diff_still_flat(self) -> None:
        """Floating-point fuzz of 1e-12 should NOT count as signal."""
        assert _ChainEvolverModule._scores_are_flat([0.5, 0.5 + 1e-12]) is True

    def test_meaningful_small_diff_not_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.5, 0.5 + 1e-6]) is False

    def test_all_minus_inf_is_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat(
            [float("-inf"), float("-inf"), float("-inf")]
        ) is True

    def test_all_nan_is_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat(
            [float("nan"), float("nan")]
        ) is True

    def test_mixed_finite_and_inf_not_flat(self) -> None:
        assert _ChainEvolverModule._scores_are_flat([0.5, float("-inf")]) is False


# ---------------------------------------------------------------------------
# evolve() integration — warning emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warning_fires_when_all_score_zero() -> None:
    dataset = SimpleDataset([DataCase(input="a"), DataCase(input="b")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _ConstMetric(score=0.0),
        population_size=3,
        generations=2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await evolver.evolve(_ctx_factory)

    flat = [w for w in caught if "no signal" in str(w.message)]
    assert len(flat) == 1, f"Expected 1 flat-signal warning, got {len(flat)}"
    message = str(flat[0].message)
    assert "generation 0" in message
    assert "0.0000" in message
    assert "case-aware metric" in message  # actionable hint


@pytest.mark.asyncio
async def test_warning_fires_only_once_across_multiple_generations() -> None:
    """Two consecutive flat generations should still emit only one warning —
    we don't want the user spammed."""
    dataset = SimpleDataset([DataCase(input="x")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _ConstMetric(score=1.0),
        population_size=2,
        generations=3,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await evolver.evolve(_ctx_factory)

    flat = [w for w in caught if "no signal" in str(w.message)]
    assert len(flat) == 1


@pytest.mark.asyncio
async def test_no_warning_when_scores_vary() -> None:
    """A well-configured metric returning varying scores should NOT warn."""

    # Use a single-case dataset and a stateful metric so each individual
    # in the population gets a different mean score.
    class _StatefulMetric(MetricBase):
        def __init__(self) -> None:
            self._calls = 0

        @property
        def name(self) -> str:
            return "stateful"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            self._calls += 1
            # First individual gets 0.1, second gets 0.9, third gets 0.5
            return [0.1, 0.9, 0.5][(self._calls - 1) % 3]

    dataset = SimpleDataset([DataCase(input="x")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _StatefulMetric(),
        population_size=3,
        generations=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await evolver.evolve(_ctx_factory)

    flat = [w for w in caught if "no signal" in str(w.message)]
    # Population should have ≥2 distinct mean scores → no warning.
    assert len(set(result.history[0].population_scores)) > 1, (
        "Test setup: population_scores must vary"
    )
    assert len(flat) == 0


@pytest.mark.asyncio
async def test_warning_message_mentions_first_observed_generation() -> None:
    """If gen 0 has signal but gen 1 collapses to flat, the warning should
    reference gen 1, not gen 0."""

    class _DegradingMetric(MetricBase):
        """Population scores vary on gen 0 (calls 1,2 → 0.1, 0.9), then all
        evaluations return 0.0 in gen 1."""

        def __init__(self) -> None:
            self._calls = 0

        @property
        def name(self) -> str:
            return "degrading"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            self._calls += 1
            # dataset=1, population=2 → 2 calls per gen.
            if self._calls <= 2:
                return [0.1, 0.9][self._calls - 1]
            return 0.0

    dataset = SimpleDataset([DataCase(input="a")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _DegradingMetric(),
        population_size=2,
        generations=2,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await evolver.evolve(_ctx_factory)

    # Gen 0 should have varied scores; gen 1 flat at 0.0.
    assert len(set(result.history[0].population_scores)) > 1
    assert all(s == 0.0 for s in result.history[1].population_scores)
    flat = [w for w in caught if "no signal" in str(w.message)]
    assert len(flat) == 1
    assert "generation 1" in str(flat[0].message)


@pytest.mark.asyncio
async def test_warning_fires_on_all_minus_inf_evaluation_failures() -> None:
    """If every individual blows up identically (each scoring -inf), warn."""

    class _AlwaysCrashMetric(MetricBase):
        @property
        def name(self) -> str:
            return "crash"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            raise RuntimeError("intentional crash")

    dataset = SimpleDataset([DataCase(input="a")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _AlwaysCrashMetric(),
        population_size=2,
        generations=1,
        smoke_check=False,  # exercise the no-signal warning path, not the smoke check
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await evolver.evolve(_ctx_factory)

    # DatasetEvaluator catches per-case exceptions and scores them 0.0,
    # so the final mean_score per individual is 0.0 (not -inf). Still flat.
    assert all(s == 0.0 for s in result.history[0].population_scores)
    flat = [w for w in caught if "no signal" in str(w.message)]
    assert len(flat) == 1


@pytest.mark.asyncio
async def test_population_size_one_does_not_warn() -> None:
    """A single-individual population trivially has 'flat' scores by
    definition. Don't spam — there's no information lost."""
    dataset = SimpleDataset([DataCase(input="a")])
    evolver = ChainEvolver(
        _make_chain(),
        dataset,
        _ConstMetric(score=0.5),
        population_size=1,
        generations=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await evolver.evolve(_ctx_factory)

    flat = [w for w in caught if "no signal" in str(w.message)]
    assert len(flat) == 0
