"""Tests for ``EvolutionResult.format_mutation_effectiveness``.

Groups individuals by the mutation that produced
them, shows mean ``score - parent_score`` per kind. Requires the lineage
tracking I added: ``IndividualMetrics.mutation_kind`` and ``.parent_score``,
populated by ``ChainEvolver._next_generation`` / ``_seed_population`` and
attached after ``_evaluate`` runs.
"""

from __future__ import annotations

import os
import random
import tempfile

import pytest

from mmar_carl import (
    ChainEvolver,
    ChainMutator,
    DataCase,
    LLMStepDescription,
    MetricBase,
    MutationKind,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
)
from mmar_carl.chain_evolution import (
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic histories without running evolution
# ---------------------------------------------------------------------------


def _result_with_lineage(
    *deltas: tuple[str, float, float],  # (mutation_kind, child_score, parent_score)
) -> EvolutionResult:
    """Construct a 1-gen EvolutionResult where every individual has lineage."""
    population_metrics = [
        IndividualMetrics(
            score=child,
            total_tokens=100,
            mutation_kind=kind,
            parent_score=parent,
        )
        for kind, child, parent in deltas
    ]
    return EvolutionResult(
        best_chain_spec={},
        best_score=max((m.score for m in population_metrics), default=0.0),
        best_generation=0,
        history=[
            GenerationStats(
                generation=0,
                best_score=max((m.score for m in population_metrics), default=0.0),
                mean_score=sum(m.score for m in population_metrics) / len(population_metrics)
                if population_metrics else 0.0,
                population_scores=[m.score for m in population_metrics],
                population_metrics=population_metrics,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history_returns_placeholder(self) -> None:
        result = EvolutionResult(
            best_chain_spec={}, best_score=0.0, best_generation=0, history=[]
        )
        assert "no mutated individuals" in result.format_mutation_effectiveness()

    def test_only_seeds_no_mutations_placeholder(self) -> None:
        """Gen-0 seeds have None lineage → no mutation deltas to aggregate."""
        result = EvolutionResult(
            best_chain_spec={}, best_score=0.5, best_generation=0,
            history=[
                GenerationStats(
                    generation=0, best_score=0.5, mean_score=0.5,
                    population_scores=[0.5],
                    population_metrics=[IndividualMetrics(score=0.5)],
                )
            ],
        )
        assert "no mutated individuals" in result.format_mutation_effectiveness()

    def test_unknown_format_raises(self) -> None:
        result = _result_with_lineage(("prompt_rewrite", 0.6, 0.5))
        with pytest.raises(ValueError, match="Unknown format"):
            result.format_mutation_effectiveness(format="svg")

    def test_inf_scores_skipped(self) -> None:
        """Individuals with -inf score or -inf parent_score are excluded."""
        result = _result_with_lineage(
            ("prompt_rewrite", float("-inf"), 0.5),   # skip
            ("prompt_rewrite", 0.6, float("-inf")),   # skip
            ("prompt_rewrite", 0.7, 0.5),             # +0.2 → kept
        )
        out = result.format_mutation_effectiveness()
        assert "prompt_rewrite" in out
        # Sample size should be 1 (only the finite-finite pair survived)
        assert "  1  " in out


# ---------------------------------------------------------------------------
# Aggregation math
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_single_kind_mean_delta(self) -> None:
        """3 individuals from prompt_rewrite with deltas +0.1, +0.2, -0.05.
        Mean delta = +0.0833."""
        result = _result_with_lineage(
            ("prompt_rewrite", 0.6, 0.5),
            ("prompt_rewrite", 0.7, 0.5),
            ("prompt_rewrite", 0.45, 0.5),
        )
        out = result.format_mutation_effectiveness()
        # Mean: (0.1 + 0.2 - 0.05) / 3 = +0.0833
        assert "+0.0833" in out
        assert "prompt_rewrite" in out
        # Sample size 3
        assert "  3  " in out

    def test_multiple_kinds_ranked_descending(self) -> None:
        """3 kinds with different mean deltas → list sorted by mean desc."""
        result = _result_with_lineage(
            ("temperature_swap", 0.3, 0.5),   # -0.2
            ("model_swap",       0.7, 0.5),   # +0.2
            ("prompt_rewrite",   0.55, 0.5),  # +0.05
        )
        out = result.format_mutation_effectiveness()
        # model_swap first, prompt_rewrite second, temperature_swap last
        idx_model = out.index("model_swap")
        idx_prompt = out.index("prompt_rewrite")
        idx_temp = out.index("temperature_swap")
        assert idx_model < idx_prompt < idx_temp

    def test_negative_mean_renders_bar_left_of_center(self) -> None:
        """A mutation that hurts on average should render its bar on the
        LEFT of the bar column's center."""
        result = _result_with_lineage(
            ("bad_mutation", 0.2, 0.5),
            ("bad_mutation", 0.1, 0.5),
        )
        out = result.format_mutation_effectiveness(bar_width=20)
        # Find the data row
        lines = out.splitlines()
        bad_row = next(line for line in lines if "bad_mutation" in line and "█" in line)
        # The bar column is bracketed: `|<bar>|`. Find the pipes.
        pipes = [i for i, ch in enumerate(bad_row) if ch == "|"]
        assert len(pipes) == 2
        bar_start, bar_end = pipes[0] + 1, pipes[1]
        bar_chars = bad_row[bar_start:bar_end]
        # The █ blocks should appear in the FIRST half of the bar column.
        center = (bar_end - bar_start) // 2
        first_block = bar_chars.index("█")
        assert first_block < center


# ---------------------------------------------------------------------------
# Lineage propagation through real ChainEvolver
# ---------------------------------------------------------------------------


def _make_evolver_for_lineage_test() -> ChainEvolver:
    """Build a minimal evolver where every mutation kind is enabled and
    every individual fires through ``mutate_with_kind``."""

    class _ScoreLengthMetric(MetricBase):
        @property
        def name(self) -> str:
            return "len"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            return float(len(output.get_final_output() or ""))

    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="A", aim="x")],
    )

    def ctx(case: DataCase) -> ReasoningContext:
        c = ReasoningContext(outer_context=case.input, api=None, model="default")
        # No tool — chain will fail per case, but that's fine for lineage testing
        return c

    return ChainEvolver(
        chain,
        SimpleDataset([DataCase(input="x")]),
        _ScoreLengthMetric(),
        mutator=ChainMutator(
            aim_suffix_pool=["Be brief."],
            temperature_pool=[0.5],
            max_workers_pool=[2],
        ),
        population_size=3,
        generations=2,
        elitism=1,
        smoke_check=False,
        rng=random.Random(42),
    )


@pytest.mark.asyncio
async def test_lineage_attached_through_real_evolver() -> None:
    """End-to-end: after evolve() finishes, gen-1 mutants should have
    non-None mutation_kind and parent_score on their IndividualMetrics."""
    ev = _make_evolver_for_lineage_test()

    # Tool isn't registered → cases will fail; scores will be 0.0. That's
    # OK for lineage testing — we just need the metadata to flow.
    def ctx(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=None, model="default")

    result = await ev.evolve(ctx)

    # Gen-0 individuals should NOT have lineage (no measured parent yet).
    gen0_metrics = result.history[0].population_metrics
    assert all(m.parent_score is None for m in gen0_metrics)

    # Gen-1 elite (1 individual at index 0) has no mutation; bred individuals
    # at indices 1+ should have lineage.
    gen1_metrics = result.history[1].population_metrics
    # At least one bred individual should have lineage
    bred = [m for m in gen1_metrics if m.mutation_kind is not None]
    assert len(bred) > 0, "Expected at least one bred individual with mutation_kind set"
    for m in bred:
        assert m.parent_score is not None
        # The mutation_kind must be a valid enum value
        assert m.mutation_kind in {k.value for k in MutationKind}


# ---------------------------------------------------------------------------
# mutate_with_kind contract
# ---------------------------------------------------------------------------


class TestMutateWithKind:
    def test_returns_tuple_of_chain_and_kind(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
        )
        mut = ChainMutator(aim_suffix_pool=["test."])
        out, kind = mut.mutate_with_kind(chain, random.Random(0))
        assert isinstance(out, ReasoningChain)
        assert kind == MutationKind.PROMPT_REWRITE

    def test_returns_none_kind_when_no_pools_configured(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
        )
        mut = ChainMutator()  # all empty pools → enabled_kinds=[]
        out, kind = mut.mutate_with_kind(chain, random.Random(0))
        assert isinstance(out, ReasoningChain)
        assert kind is None

    def test_backward_compat_mutate_returns_chain_only(self) -> None:
        """The original `mutate()` API should still return just the chain."""
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
        )
        mut = ChainMutator(aim_suffix_pool=["test."])
        out = mut.mutate(chain, random.Random(0))
        # Just the chain, not a tuple
        assert isinstance(out, ReasoningChain)


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        result = _result_with_lineage(("prompt_rewrite", 0.6, 0.5))
        with pytest.raises(ValueError, match="png_path"):
            result.format_mutation_effectiveness(format="png")

    def test_png_raises_install_hint_when_matplotlib_missing(self) -> None:
        try:
            import matplotlib  # noqa: F401
            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        result = _result_with_lineage(
            ("prompt_rewrite", 0.6, 0.5),
            ("model_swap", 0.7, 0.5),
        )
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            result.format_mutation_effectiveness(format="png", png_path="/tmp/x.png")

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")
        result = _result_with_lineage(
            ("prompt_rewrite", 0.6, 0.5),
            ("model_swap", 0.7, 0.5),
            ("temperature_swap", 0.4, 0.5),
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "mutation.png")
            written = result.format_mutation_effectiveness(format="png", png_path=path)
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0
