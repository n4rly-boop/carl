"""Tests for concurrent per-individual evaluation in ``ChainEvolver``.

The live benchmark spent ~8 minutes on a 6-individual × 5-case sweep
because all evaluations ran serially. ``max_concurrent_individuals=N``
bounds a semaphore around per-individual ``_evaluate`` calls so multiple
chains can run in parallel against the LLM API. Default (1) preserves
backward compatibility.
"""

from __future__ import annotations

import time

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


class _ConstMetric(MetricBase):
    @property
    def name(self) -> str:
        return "const"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 0.5


def _slow_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="slow", config=ToolStepConfig(tool_name="slow")
            ),
        ],
    )


def _fast_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="fast", config=ToolStepConfig(tool_name="fast")
            ),
        ],
    )


# Each tool call sleeps ~120 ms — long enough that serial vs concurrent
# is observable but short enough that tests still finish fast.
SLEEP_S = 0.12


def _make_slow_ctx_factory():
    def factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
        ctx.register_tool("slow", lambda: (time.sleep(SLEEP_S), "ok")[1])
        return ctx
    return factory


def _make_fast_ctx_factory():
    def factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
        ctx.register_tool("fast", lambda: "ok")
        return ctx
    return factory


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_default_is_one_for_backward_compat(self) -> None:
        ev = ChainEvolver(
            _fast_chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=1,
            generations=1,
        )
        assert ev.max_concurrent_individuals == 1

    def test_explicit_value_accepted(self) -> None:
        ev = ChainEvolver(
            _fast_chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=1,
            max_concurrent_individuals=4,
        )
        assert ev.max_concurrent_individuals == 4

    def test_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_individuals"):
            ChainEvolver(
                _fast_chain(),
                SimpleDataset([DataCase(input="x")]),
                _ConstMetric(),
                max_concurrent_individuals=0,
            )

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_concurrent_individuals"):
            ChainEvolver(
                _fast_chain(),
                SimpleDataset([DataCase(input="x")]),
                _ConstMetric(),
                max_concurrent_individuals=-3,
            )


# ---------------------------------------------------------------------------
# Backward compatibility: sequential path unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_path_runs_individuals_sequentially() -> None:
    """With ``max_concurrent_individuals=1`` (default), the original
    serial loop is used. Total wall time ≈ N × per-individual time."""
    ev = ChainEvolver(
        _slow_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=3,
        generations=1,
        smoke_check=False,
    )
    t0 = time.perf_counter()
    result = await ev.evolve(_make_slow_ctx_factory())
    elapsed = time.perf_counter() - t0
    # 3 individuals × 1 case × 120ms = 360ms, with overhead expect ~0.3-0.6s
    assert elapsed >= 3 * SLEEP_S * 0.85
    assert len(result.history) == 1
    assert len(result.history[0].population_scores) == 3


# ---------------------------------------------------------------------------
# Concurrent path delivers speedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_evaluation_speedup_visible() -> None:
    """With ``max_concurrent_individuals=N``, total wall time should be
    closer to per-individual time, not N × per-individual time."""
    pop = 4
    dataset = SimpleDataset([DataCase(input="x")])

    # Sequential
    ev_seq = ChainEvolver(
        _slow_chain(),
        dataset,
        _ConstMetric(),
        population_size=pop,
        generations=1,
        smoke_check=False,
        max_concurrent_individuals=1,
    )
    t0 = time.perf_counter()
    await ev_seq.evolve(_make_slow_ctx_factory())
    seq_time = time.perf_counter() - t0

    # Concurrent — 4 in parallel
    ev_par = ChainEvolver(
        _slow_chain(),
        dataset,
        _ConstMetric(),
        population_size=pop,
        generations=1,
        smoke_check=False,
        max_concurrent_individuals=pop,
    )
    t0 = time.perf_counter()
    await ev_par.evolve(_make_slow_ctx_factory())
    par_time = time.perf_counter() - t0

    # Concurrent should be at least 2× faster (conservative — actual is ~4x).
    assert par_time * 2 < seq_time, (
        f"Expected >=2x speedup; got seq={seq_time:.3f}s par={par_time:.3f}s"
    )


@pytest.mark.asyncio
async def test_concurrent_results_match_sequential_results() -> None:
    """Same setup with seq vs concurrent should produce the same scores
    (just different wall time)."""
    pop = 3
    dataset = SimpleDataset([DataCase(input="x"), DataCase(input="y")])

    def build_evolver(concurrent: int) -> ChainEvolver:
        return ChainEvolver(
            _fast_chain(),
            dataset,
            _ConstMetric(),
            population_size=pop,
            generations=1,
            smoke_check=False,
            max_concurrent_individuals=concurrent,
        )

    seq_result = await build_evolver(1).evolve(_make_fast_ctx_factory())
    par_result = await build_evolver(3).evolve(_make_fast_ctx_factory())

    assert seq_result.best_score == par_result.best_score
    assert sorted(seq_result.history[0].population_scores) == sorted(
        par_result.history[0].population_scores
    )


# ---------------------------------------------------------------------------
# Semaphore correctly bounds concurrency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_bounds_concurrent_evals() -> None:
    """With ``max_concurrent_individuals=2`` and population=4, at any moment
    at most 2 evaluations should be in flight."""
    in_flight = {"current": 0, "peak": 0}

    def factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")

        def tracked_tool():
            in_flight["current"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            time.sleep(SLEEP_S)
            in_flight["current"] -= 1
            return "ok"

        ctx.register_tool("slow", tracked_tool)
        return ctx

    ev = ChainEvolver(
        _slow_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=4,
        generations=1,
        smoke_check=False,
        max_concurrent_individuals=2,
    )
    await ev.evolve(factory)
    # At most 2 should ever have been in flight simultaneously.
    assert in_flight["peak"] <= 2
    # And we should have used the parallelism (peak > 1).
    assert in_flight["peak"] >= 2


# ---------------------------------------------------------------------------
# Result ordering preserved
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoring_order_preserved_after_concurrent_eval() -> None:
    """After concurrent eval, the sorted scored list still pairs each
    individual with its correct score. We use a stateful metric that
    returns the call-count to make scores distinguishable."""

    class _DistinctMetric(MetricBase):
        def __init__(self) -> None:
            self._n = 0

        @property
        def name(self) -> str:
            return "distinct"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            self._n += 1
            return float(self._n)

    ev = ChainEvolver(
        _fast_chain(),
        SimpleDataset([DataCase(input="x")]),
        _DistinctMetric(),
        population_size=3,
        generations=1,
        smoke_check=False,
        max_concurrent_individuals=3,
    )
    result = await ev.evolve(_make_fast_ctx_factory())
    scores = result.history[0].population_scores
    # 3 calls → scores should be {1.0, 2.0, 3.0} regardless of order they arrived.
    assert set(scores) == {1.0, 2.0, 3.0}
    # And descending order
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Concurrent doesn't break smoke check or no-signal warning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smoke_check_still_runs_before_concurrent_loop() -> None:
    """Smoke check is sequential and runs once; concurrent gate only affects
    the main evolution loop."""
    smoke_calls = {"n": 0}

    def factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")

        def tool():
            smoke_calls["n"] += 1
            return "ok"

        ctx.register_tool("fast", tool)
        return ctx

    ev = ChainEvolver(
        _fast_chain(),
        SimpleDataset([DataCase(input="x")]),
        _ConstMetric(),
        population_size=2,
        generations=1,
        smoke_check=True,
        max_concurrent_individuals=2,
    )
    await ev.evolve(factory)
    # 1 smoke + (2 individuals × 1 case) = 3 tool calls
    assert smoke_calls["n"] == 3
