"""Tests for the ``ChainEvolver`` pre-flight smoke check.

The live benchmark spent ~7 minutes on a misconfigured metric run
before the no-fitness-signal warning fired at gen 0. The smoke check
catches the same failure modes (broken metric, broken chain, empty
dataset) *before* the first generation kicks off, costing exactly one
LLM call instead of ``population_size * generations * len(dataset)``.
"""

from __future__ import annotations

import asyncio

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit", config=ToolStepConfig(tool_name="emit")
            ),
        ],
    )


def _ok_ctx_factory(case: DataCase) -> ReasoningContext:
    ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
    ctx.register_tool("emit", lambda: "ok")
    return ctx


def _failing_ctx_factory(case: DataCase) -> ReasoningContext:
    # No tool registered → step fails
    return ReasoningContext(outer_context=case.input, api=None, model="default")


class _OkMetric(MetricBase):
    @property
    def name(self) -> str:
        return "ok"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 0.5


class _NaNMetric(MetricBase):
    @property
    def name(self) -> str:
        return "nan"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return float("nan")


class _InfMetric(MetricBase):
    @property
    def name(self) -> str:
        return "inf"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return float("inf")


class _RaisingMetric(MetricBase):
    @property
    def name(self) -> str:
        return "raising"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        raise RuntimeError("intentional metric bug")


# ---------------------------------------------------------------------------
# Smoke check enabled (default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_check_passes_with_legit_setup() -> None:
    """Happy path: chain runs, metric returns finite score → no raise."""
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _OkMetric(),
        population_size=1,
        generations=1,
    )
    result = await evolver.evolve(_ok_ctx_factory)
    assert result.best_score == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_smoke_check_raises_on_empty_dataset() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([]),
        _OkMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="dataset is empty"):
        await evolver.evolve(_ok_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_raises_when_chain_fails() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _OkMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="base chain returned success=False"):
        await evolver.evolve(_failing_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_raises_when_chain_raises_exception() -> None:
    """A context_factory that itself raises should surface a clear diagnostic."""

    def crashing_ctx_factory(case: DataCase) -> ReasoningContext:
        raise ValueError("context_factory bug")

    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _OkMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="base chain raised ValueError"):
        await evolver.evolve(crashing_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_raises_when_metric_returns_nan() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _NaNMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="returned nan"):
        await evolver.evolve(_ok_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_raises_when_metric_returns_inf() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _InfMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="returned inf"):
        await evolver.evolve(_ok_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_raises_when_metric_raises() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _RaisingMetric(),
        population_size=1,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="metric 'raising' raised RuntimeError"):
        await evolver.evolve(_ok_ctx_factory)


@pytest.mark.asyncio
async def test_smoke_check_allows_zero_score() -> None:
    """A constant-zero metric is valid (e.g. for testing) — the runtime
    no-fitness-signal warning catches that case after gen 0, not here."""

    class _ZeroMetric(MetricBase):
        @property
        def name(self) -> str:
            return "zero"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            return 0.0

    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ZeroMetric(),
        population_size=1,
        generations=1,
    )
    # Should NOT raise — proceeds to evolve which then warns separately.
    result = await evolver.evolve(_ok_ctx_factory)
    assert result.best_score == 0.0


# ---------------------------------------------------------------------------
# smoke_check=False bypass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_check_disabled_skips_pre_flight() -> None:
    """With ``smoke_check=False`` the broken metric reaches the main loop."""
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _NaNMetric(),
        population_size=1,
        generations=1,
        smoke_check=False,
    )
    # Should NOT raise — instead the broken metric pollutes population scores,
    # and the main loop returns whatever happened (-inf, since the metric
    # going through DatasetEvaluator returns NaN, mean is NaN, etc.).
    result = await evolver.evolve(_ok_ctx_factory)
    # Don't assert on the score — point is the run completes without
    # the pre-flight check raising.
    assert isinstance(result.best_score, float)


@pytest.mark.asyncio
async def test_smoke_check_disabled_chain_failure_still_reaches_loop() -> None:
    """With ``smoke_check=False`` a broken chain reaches the main loop;
    DatasetEvaluator catches the failure and scores it 0."""
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _OkMetric(),
        population_size=1,
        generations=1,
        smoke_check=False,
    )
    result = await evolver.evolve(_failing_ctx_factory)
    # DatasetEvaluator returns 0.0 for failed cases — population scores stay 0.
    assert result.best_score == 0.0


# ---------------------------------------------------------------------------
# Cost — one extra LLM call vs full population sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_check_cost_is_one_extra_chain_run() -> None:
    """Tally the number of chain executions: smoke check + (pop * gen * cases)."""
    call_count = {"n": 0}

    def counting_ctx_factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")

        def emit() -> str:
            call_count["n"] += 1
            return "ok"

        ctx.register_tool("emit", emit)
        return ctx

    pop = 2
    gens = 2
    cases = 3
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input=f"c{i}") for i in range(cases)]),
        _OkMetric(),
        population_size=pop,
        generations=gens,
        smoke_check=True,
    )
    await evolver.evolve(counting_ctx_factory)

    # smoke check (1) + population × generations × cases (12) = 13 chain runs
    expected = 1 + pop * gens * cases
    assert call_count["n"] == expected


# ---------------------------------------------------------------------------
# Synchronous wrapper for one path — sanity that asyncio.run works too
# ---------------------------------------------------------------------------


def test_smoke_check_via_asyncio_run() -> None:
    evolver = ChainEvolver(
        _make_chain(),
        SimpleDataset([DataCase(input="x")]),
        _OkMetric(),
        population_size=1,
        generations=1,
    )
    result = asyncio.run(evolver.evolve(_ok_ctx_factory))
    assert result.best_score == pytest.approx(0.5)
