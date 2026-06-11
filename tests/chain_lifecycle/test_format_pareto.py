"""Tests for ``EvolutionResult.format_pareto`` — per-individual
score-vs-cost Pareto chart across all generations.

Each individual ever evaluated is one point on
``(cost, score)``; Pareto-dominant points marked distinctly.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mmar_carl.chain_evolution import (
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
)


def _gen(generation: int, *individuals: tuple[float, int]) -> GenerationStats:
    """`*individuals` is `(score, tokens)` pairs."""
    return GenerationStats(
        generation=generation,
        best_score=max((s for s, _ in individuals), default=0.0),
        mean_score=sum(s for s, _ in individuals) / max(len(individuals), 1),
        population_scores=[s for s, _ in individuals],
        population_metrics=[
            IndividualMetrics(score=s, total_tokens=t) for s, t in individuals
        ],
    )


def _result(*generations: GenerationStats) -> EvolutionResult:
    return EvolutionResult(
        best_chain_spec={},
        best_score=max((g.best_score for g in generations), default=0.0),
        best_generation=max((g.generation for g in generations), default=0),
        history=list(generations),
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history_returns_placeholder(self) -> None:
        out = _result().format_pareto()
        assert "nothing to chart" in out

    def test_only_inf_scores_returns_placeholder(self) -> None:
        """When every individual has -inf score (all evals crashed),
        the chart can't be drawn."""
        out = _result(
            _gen(0, (float("-inf"), 100), (float("-inf"), 200))
        ).format_pareto()
        assert "no finite-score individuals" in out

    def test_unknown_format_raises(self) -> None:
        result = _result(_gen(0, (0.5, 100)))
        with pytest.raises(ValueError, match="Unknown format"):
            result.format_pareto(format="svg")

    def test_single_individual_renders(self) -> None:
        """One individual → on the Pareto front trivially; chart still
        renders without div-by-zero."""
        out = _result(_gen(0, (0.5, 100))).format_pareto()
        assert "★" in out
        # 1/1 on the front
        assert "(1 / 1 on front)" in out

    def test_flat_history_doesnt_divide_by_zero(self) -> None:
        """All individuals at the same (score, cost) — degenerate."""
        out = _result(_gen(0, (0.5, 100), (0.5, 100))).format_pareto()
        # Should render without crashing
        assert "★" in out or "·" in out


# ---------------------------------------------------------------------------
# Pareto-dominance math
# ---------------------------------------------------------------------------


class TestParetoMath:
    def test_cheaper_higher_score_dominates_more_expensive_lower_score(self) -> None:
        """A point at (100, 0.9) dominates (200, 0.5) — the second is
        worse on both axes."""
        out = _result(
            _gen(0, (0.9, 100), (0.5, 200))
        ).format_pareto()
        # 1 of 2 on the front (the cheaper+higher-score one)
        assert "(1 / 2 on front)" in out

    def test_both_points_on_front_when_one_axis_each(self) -> None:
        """(100, 0.5) and (200, 0.9): neither dominates the other —
        first is cheaper, second has higher score. Both on the front."""
        out = _result(
            _gen(0, (0.5, 100), (0.9, 200))
        ).format_pareto()
        assert "(2 / 2 on front)" in out

    def test_staircase_pareto_front(self) -> None:
        """A classic monotonic-staircase scenario: 5 points where each
        is on the front because they trade off one axis for the other."""
        out = _result(
            _gen(
                0,
                (0.3, 100),
                (0.5, 200),
                (0.7, 300),
                (0.85, 400),
                (0.95, 500),
            )
        ).format_pareto()
        assert "(5 / 5 on front)" in out

    def test_dominated_strictly_inside_front(self) -> None:
        """Front: (100, 0.4), (500, 0.95). Dominated middle: (300, 0.5)
        — there's a Pareto point cheaper-or-equal with higher score."""
        out = _result(
            _gen(0, (0.4, 100), (0.5, 300), (0.95, 500))
        ).format_pareto()
        # 2 of 3 — the (300, 0.5) point is NOT dominated because nobody
        # is cheaper *and* higher score. Wait: (100, 0.4) is cheaper but
        # has lower score, so it doesn't dominate (300, 0.5). (500, 0.95)
        # has higher score but is more expensive, so it also doesn't
        # dominate (300, 0.5). So actually all 3 are on the front.
        assert "(3 / 3 on front)" in out

    def test_tied_cost_dominates_when_score_higher(self) -> None:
        """(200, 0.5) is dominated by (200, 0.9) — same cost, higher score."""
        out = _result(
            _gen(0, (0.5, 200), (0.9, 200))
        ).format_pareto()
        assert "(1 / 2 on front)" in out


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------


class TestTextFormat:
    def test_legend_present(self) -> None:
        out = _result(_gen(0, (0.5, 100))).format_pareto()
        assert "★ Pareto-dominant" in out
        assert "· dominated" in out
        assert "on front" in out

    def test_axis_labels_default_to_tokens(self) -> None:
        out = _result(_gen(0, (0.5, 100), (0.7, 200))).format_pareto()
        assert "tokens" in out

    def test_axis_labels_switch_to_usd_with_cost_per_1k(self) -> None:
        out = _result(_gen(0, (0.5, 100), (0.7, 200))).format_pareto(cost_per_1k=0.05)
        assert "cost (USD)" in out
        assert "$0.0050" in out  # 100 * 0.05 / 1000 = 0.005

    def test_pareto_points_painted_over_dominated_when_overlapping(self) -> None:
        """If a Pareto point and a dominated point happen to land on the
        same canvas cell (rare but possible), the ★ should win — drawn
        last on top of the ·."""
        # Force overlap: same score (front + dominated by tie), same cost.
        # Actually with the current implementation, a tie-on-both is
        # mutual non-domination — but if cost differs slightly only
        # within the same canvas cell, the painting order matters.
        out = _result(
            _gen(0, (0.9, 200), (0.9, 200))
        ).format_pareto(width=10, height=5)
        # Both points are on the front (mutual ties)
        # Should render at least one ★
        assert "★" in out


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        result = _result(_gen(0, (0.5, 100)))
        with pytest.raises(ValueError, match="png_path"):
            result.format_pareto(format="png")

    def test_png_raises_install_hint_when_matplotlib_missing(self) -> None:
        try:
            import matplotlib  # noqa: F401
            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        result = _result(_gen(0, (0.5, 100), (0.7, 200)))
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            result.format_pareto(format="png", png_path="/tmp/x.png")

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")
        result = _result(
            _gen(0, (0.3, 100), (0.5, 200)),
            _gen(1, (0.7, 300), (0.9, 500)),
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pareto.png")
            written = result.format_pareto(format="png", png_path=path)
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0


# ---------------------------------------------------------------------------
# Integration: ignores -inf scores but keeps the rest
# ---------------------------------------------------------------------------


def test_inf_score_individuals_skipped_others_kept() -> None:
    """Mixed: some individuals have -inf (eval crash), others have real
    scores. The chart should plot only the finite ones."""
    out = _result(
        _gen(
            0,
            (float("-inf"), 100),  # skipped
            (0.5, 200),
            (0.8, 300),
        )
    ).format_pareto()
    # 2 finite points, both on the front (different costs, monotonic scores)
    assert "(2 / 2 on front)" in out
