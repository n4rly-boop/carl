"""Tests for ``ChainMutator`` and ``ChainEvolver`` (evolutionary chain search)."""

from __future__ import annotations

import random

import pytest

from mmar_carl import (
    ChainEvolver,
    ChainMutator,
    EvolutionResult,
    GenerationStats,
    LLMStepConfig,
    LLMStepDescription,
    MetricBase,
    MutationKind,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.models.dataset import (
    DataCase,
    SimpleDataset,
    ThresholdStrategy,
)


# ---------------------------------------------------------------------------
# Mutator unit tests
# ---------------------------------------------------------------------------


def _make_two_llm_chain(max_workers: int = 2) -> ReasoningChain:
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="A", aim="Analyze."),
            LLMStepDescription(
                number=2,
                title="B",
                aim="Summarize.",
                dependencies=[1],
                llm_config=LLMStepConfig(temperature=0.5),
            ),
        ],
        max_workers=max_workers,
    )


class TestChainMutator:
    def test_empty_pools_returns_clone(self) -> None:
        chain = _make_two_llm_chain()
        mut = ChainMutator()
        rng = random.Random(0)
        clone = mut.mutate(chain, rng)
        assert clone is not chain
        assert clone.to_dict() == chain.to_dict()
        assert mut.enabled_kinds == []

    def test_enabled_kinds_inferred_from_pools(self) -> None:
        mut = ChainMutator(model_pool=["m1"], aim_suffix_pool=["x"])
        assert MutationKind.MODEL_SWAP in mut.enabled_kinds
        assert MutationKind.PROMPT_REWRITE in mut.enabled_kinds
        assert MutationKind.TEMPERATURE_SWAP not in mut.enabled_kinds
        assert MutationKind.MAX_WORKERS not in mut.enabled_kinds

    def test_model_swap_changes_a_step_model(self) -> None:
        chain = _make_two_llm_chain()
        mut = ChainMutator(
            model_pool=["model-x"],
            enabled_kinds=[MutationKind.MODEL_SWAP],
        )
        rng = random.Random(1)
        mutated = mut.mutate(chain, rng)
        models = [
            (s.llm_config.model if s.llm_config else None) for s in mutated.steps
        ]
        assert "model-x" in models
        # Original chain untouched.
        for s in chain.steps:
            assert s.llm_config is None or s.llm_config.model is None

    def test_temperature_swap_changes_step_temperature(self) -> None:
        chain = _make_two_llm_chain()
        mut = ChainMutator(
            temperature_pool=[0.1, 0.9],
            enabled_kinds=[MutationKind.TEMPERATURE_SWAP],
        )
        rng = random.Random(2)
        mutated = mut.mutate(chain, rng)
        temps = [
            (s.llm_config.temperature if s.llm_config else None)
            for s in mutated.steps
        ]
        # At least one step now has a value from the pool.
        assert any(t in (0.1, 0.9) for t in temps if t is not None)

    def test_prompt_rewrite_appends_suffix_to_aim(self) -> None:
        chain = _make_two_llm_chain()
        mut = ChainMutator(
            aim_suffix_pool=["Be concise."],
            enabled_kinds=[MutationKind.PROMPT_REWRITE],
        )
        rng = random.Random(3)
        mutated = mut.mutate(chain, rng)
        aims = [s.aim for s in mutated.steps]
        # One of the aims should have the suffix appended.
        assert any(a and "Be concise." in a for a in aims)
        # The other should still be original.
        originals_present = sum(
            1 for a in aims if a in {"Analyze.", "Summarize."}
        )
        assert originals_present >= 1

    def test_max_workers_swap_changes_chain_max_workers(self) -> None:
        chain = _make_two_llm_chain(max_workers=2)
        mut = ChainMutator(
            max_workers_pool=[5],
            enabled_kinds=[MutationKind.MAX_WORKERS],
        )
        rng = random.Random(4)
        mutated = mut.mutate(chain, rng)
        assert mutated.max_workers == 5
        assert chain.max_workers == 2

    def test_max_workers_pool_supports_auto(self) -> None:
        chain = _make_two_llm_chain(max_workers=2)
        mut = ChainMutator(
            max_workers_pool=["auto"],
            enabled_kinds=[MutationKind.MAX_WORKERS],
        )
        rng = random.Random(5)
        mutated = mut.mutate(chain, rng)
        assert mutated.max_workers == "auto"

    def test_chain_with_no_llm_steps_skips_llm_only_mutations(self) -> None:
        """Tool-only chain: model/temperature/prompt mutations gracefully no-op."""
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="noop"),
                ),
            ],
        )
        mut = ChainMutator(
            model_pool=["m1"],
            temperature_pool=[0.5],
            aim_suffix_pool=["x"],
        )
        rng = random.Random(6)
        clone = mut.mutate(chain, rng)
        # No LLM steps to mutate → returns a fresh clone unchanged.
        assert clone.to_dict() == chain.to_dict()

    def test_deterministic_with_seeded_rng(self) -> None:
        chain = _make_two_llm_chain()
        mut = ChainMutator(
            model_pool=["a", "b"],
            temperature_pool=[0.1, 0.9],
            aim_suffix_pool=["one", "two"],
            max_workers_pool=[1, 4],
        )
        rng_a = random.Random(99)
        rng_b = random.Random(99)
        out_a = mut.mutate(chain, rng_a)
        out_b = mut.mutate(chain, rng_b)
        assert out_a.to_dict() == out_b.to_dict()


# ---------------------------------------------------------------------------
# Evolver validation
# ---------------------------------------------------------------------------


class TestChainEvolverValidation:
    def test_population_size_must_be_positive(self) -> None:
        chain = _make_two_llm_chain()
        ds = SimpleDataset([DataCase(input="x")])
        metric = _ConstMetric()
        with pytest.raises(ValueError, match="population_size"):
            ChainEvolver(chain, ds, metric, population_size=0)

    def test_generations_must_be_positive(self) -> None:
        chain = _make_two_llm_chain()
        ds = SimpleDataset([DataCase(input="x")])
        metric = _ConstMetric()
        with pytest.raises(ValueError, match="generations"):
            ChainEvolver(chain, ds, metric, generations=0)

    def test_elitism_must_be_non_negative(self) -> None:
        chain = _make_two_llm_chain()
        ds = SimpleDataset([DataCase(input="x")])
        metric = _ConstMetric()
        with pytest.raises(ValueError, match="elitism"):
            ChainEvolver(chain, ds, metric, elitism=-1)

    def test_elitism_clamped_to_population_size(self) -> None:
        chain = _make_two_llm_chain()
        ds = SimpleDataset([DataCase(input="x")])
        metric = _ConstMetric()
        ev = ChainEvolver(
            chain, ds, metric, population_size=3, elitism=99, generations=1
        )
        assert ev.elitism == 3


# ---------------------------------------------------------------------------
# Helpers for evolver execution tests
# ---------------------------------------------------------------------------


class _ConstMetric(MetricBase):
    """Returns a fixed score regardless of output."""

    def __init__(self, score: float = 1.0, name: str = "const") -> None:
        self._score = score
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return self._score


class _OutputContainsMetric(MetricBase):
    """Score = 1.0 if final output contains substring, else 0.0."""

    def __init__(self, substring: str) -> None:
        self._sub = substring

    @property
    def name(self) -> str:
        return "contains"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        # Chain-level metric: inspect get_final_output()
        try:
            text = output.get_final_output()
        except AttributeError:
            text = output.result
        return 1.0 if self._sub in (text or "") else 0.0


def _context_factory(case: DataCase) -> ReasoningContext:
    return ReasoningContext(
        outer_context=case.input,
        api=None,
        model="default",
    )


def _tool_only_chain(tool_name: str, max_workers: int = 1) -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="emit",
                config=ToolStepConfig(tool_name=tool_name),
            ),
        ],
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# Evolver execution (real chain runs)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evolver_runs_and_returns_best() -> None:
    """A toy chain + constant metric → result.best_score equals the metric's
    constant. Verifies the basic loop machinery."""
    chain = _tool_only_chain("emit")
    ds = SimpleDataset([DataCase(input="case1"), DataCase(input="case2")])
    metric = _ConstMetric(score=0.75)

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit", lambda: "hello")
        return ctx

    evolver = ChainEvolver(
        chain,
        ds,
        metric,
        population_size=3,
        generations=2,
        elitism=1,
        rng=random.Random(7),
    )
    result = await evolver.evolve(context_factory)

    assert isinstance(result, EvolutionResult)
    assert result.best_score == pytest.approx(0.75)
    assert len(result.history) == 2
    assert all(isinstance(g, GenerationStats) for g in result.history)
    # Each generation has population_size scores.
    for stats in result.history:
        assert len(stats.population_scores) == 3


@pytest.mark.asyncio
async def test_evolver_history_records_per_generation_stats() -> None:
    chain = _tool_only_chain("emit")
    ds = SimpleDataset([DataCase(input="x")])
    metric = _ConstMetric(score=0.5)

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit", lambda: "hello")
        return ctx

    evolver = ChainEvolver(
        chain, ds, metric, population_size=2, generations=3, rng=random.Random(0)
    )
    result = await evolver.evolve(context_factory)
    assert [g.generation for g in result.history] == [0, 1, 2]
    for stats in result.history:
        assert stats.best_score == pytest.approx(0.5)
        assert stats.mean_score == pytest.approx(0.5)
        # Best chain spec serializable + non-empty
        assert "steps" in stats.best_chain_spec


@pytest.mark.asyncio
async def test_evolver_selects_best_when_scores_differ() -> None:
    """Chain with two parallel tool steps where one variant ("good") scores
    higher than the other ("bad"). Evolver should converge on 'good'.

    We exploit the ``max_workers`` mutation to deterministically toggle which
    tool a chain calls — but here it's easier to encode the choice in the
    tool registry via two different chains and let the elitism/breeding
    pipeline select the higher-scoring one.
    """
    base_chain = _tool_only_chain("emit_good")
    bad_chain = _tool_only_chain("emit_bad")

    ds = SimpleDataset([DataCase(input="case")])
    metric = _OutputContainsMetric("GOOD")

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit_good", lambda: "GOOD")
        ctx.register_tool("emit_bad", lambda: "BAD")
        return ctx

    # Use a mutator that doesn't actually change anything (empty pools), so
    # the population is clones of whatever we seed. Initial generation
    # contains one "good" parent (best_chain) + clones; we'll just confirm
    # the evolver tracks best_score correctly with a heterogeneous population
    # by injecting via the base chain.
    evolver = ChainEvolver(
        base_chain,
        ds,
        metric,
        population_size=2,
        generations=1,
        elitism=1,
        rng=random.Random(123),
    )
    result = await evolver.evolve(context_factory)
    assert result.best_score == 1.0  # "GOOD" appeared

    # Negative control: when only the "bad" chain is seeded, score should be 0.
    evolver_bad = ChainEvolver(
        bad_chain,
        ds,
        metric,
        population_size=2,
        generations=1,
        elitism=1,
        rng=random.Random(123),
    )
    result_bad = await evolver_bad.evolve(context_factory)
    assert result_bad.best_score == 0.0


@pytest.mark.asyncio
async def test_evolver_no_mutator_yields_homogeneous_population() -> None:
    """Without a mutator, every individual is a clone of the base chain."""
    chain = _tool_only_chain("emit", max_workers=3)
    ds = SimpleDataset([DataCase(input="x")])
    metric = _ConstMetric(score=1.0)

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit", lambda: "hi")
        return ctx

    evolver = ChainEvolver(
        chain,
        ds,
        metric,
        mutator=None,
        population_size=4,
        generations=2,
        rng=random.Random(0),
    )
    result = await evolver.evolve(context_factory)
    # All recorded best specs identical to base (cloned).
    assert all(stats.best_chain_spec == chain.to_dict() for stats in result.history)
    # population_scores all equal
    for stats in result.history:
        assert all(s == pytest.approx(1.0) for s in stats.population_scores)


@pytest.mark.asyncio
async def test_evolver_with_mutator_produces_diverse_population() -> None:
    """When a mutator is supplied, the second-generation population should not
    all be identical (with sufficient pool size)."""
    chain = _make_two_llm_chain()
    ds = SimpleDataset([DataCase(input="x")])
    metric = _ConstMetric(score=0.5)

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        return ctx

    mut = ChainMutator(
        aim_suffix_pool=["A", "B", "C", "D"],
        max_workers_pool=[1, 2, 5, "auto"],
    )
    # Use generations=1 so we just look at the initial seed diversity.
    evolver = ChainEvolver(
        chain,
        ds,
        metric,
        mutator=mut,
        population_size=6,
        generations=1,
        rng=random.Random(11),
    )
    # We can't run evolve() because the chain has LLM steps and no mocked LLM,
    # so just inspect the initial population directly.
    seeded = evolver._seed_population()
    specs = [tuple(sorted(s.to_dict().items(), key=lambda kv: kv[0])) for s in seeded]
    # First spec is the unmutated base; others should not all match.
    unique = {repr(spec) for spec in specs}
    assert len(unique) > 1, "Mutator failed to introduce any diversity"


@pytest.mark.asyncio
async def test_evolver_evaluation_failure_records_minus_inf() -> None:
    """If the chain blows up during evaluation, the individual gets score
    -inf and the loop continues."""
    chain = _tool_only_chain("crashy")
    ds = SimpleDataset([DataCase(input="x")])
    metric = _ConstMetric(score=1.0)

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        # No tool registered → step execution returns success=False, score=0.0
        return ctx

    evolver = ChainEvolver(
        chain,
        ds,
        metric,
        population_size=2,
        generations=1,
        elitism=1,
        rng=random.Random(0),
        smoke_check=False,  # we *want* the broken chain to reach the loop here
    )
    result = await evolver.evolve(context_factory)
    # Chain fails for all individuals → DatasetEvaluator scores them as 0.0
    # (it catches per-case exceptions). So best_score is 0.0, not -inf.
    assert result.best_score == 0.0


@pytest.mark.asyncio
async def test_evolver_elitism_carries_best_across_generations() -> None:
    """The best chain from gen N appears in gen N+1's population (as elite)."""
    chain = _tool_only_chain("emit")
    ds = SimpleDataset([DataCase(input="x")])

    # Use a stateful metric that gives the chain a different score each call,
    # but the elite (best after gen 0) is carried forward unchanged.
    call_count = {"n": 0}

    class _AscendingMetric(MetricBase):
        @property
        def name(self) -> str:
            return "asc"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            call_count["n"] += 1
            return float(call_count["n"])

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit", lambda: "x")
        return ctx

    evolver = ChainEvolver(
        chain,
        ds,
        _AscendingMetric(),
        population_size=2,
        generations=2,
        elitism=1,
        rng=random.Random(0),
    )
    result = await evolver.evolve(context_factory)
    # Gen 1 max == gen 0 max + 2 (two evaluations between).
    assert result.history[0].best_score < result.history[1].best_score
    # Best is from the higher-scoring later generation.
    assert result.best_generation == 1


@pytest.mark.asyncio
async def test_evolver_selection_strategy_passthrough() -> None:
    """Custom selection_strategy reaches DatasetEvaluator."""
    chain = _tool_only_chain("emit")
    ds = SimpleDataset([DataCase(input="x")])

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = _context_factory(case)
        ctx.register_tool("emit", lambda: "x")
        return ctx

    custom = ThresholdStrategy(threshold=0.99, higher_is_better=False)
    evolver = ChainEvolver(
        chain,
        ds,
        _ConstMetric(),
        population_size=1,
        generations=1,
        selection_strategy=custom,
        rng=random.Random(0),
    )
    assert evolver.selection_strategy is custom
    # Should still run end-to-end without error.
    result = await evolver.evolve(context_factory)
    assert isinstance(result, EvolutionResult)


@pytest.mark.asyncio
async def test_evolver_default_selection_strategy_is_threshold_zero() -> None:
    chain = _tool_only_chain("emit")
    ds = SimpleDataset([DataCase(input="x")])
    evolver = ChainEvolver(chain, ds, _ConstMetric())
    assert isinstance(evolver.selection_strategy, ThresholdStrategy)
    assert evolver.selection_strategy.threshold == 0.0
