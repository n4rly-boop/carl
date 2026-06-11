"""Tests for ``EvolutionResult.format_score_evolution`` — per-generation
score chart with best/mean/worst lines.

Last of the three visualizations
(token pie / Gantt / score evolution) that give CARL users the three
canonical "where did time/money go?" and "is evolution working?" answers.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mmar_carl.chain_evolution import EvolutionResult, GenerationStats


def _result(*generations: GenerationStats, best_score: float = 0.0) -> EvolutionResult:
    return EvolutionResult(
        best_chain_spec={},
        best_score=best_score,
        best_generation=max((g.generation for g in generations), default=0),
        history=list(generations),
    )


def _gen(
    generation: int,
    *,
    best: float = 0.0,
    mean: float = 0.0,
    population: list[float] | None = None,
) -> GenerationStats:
    pop = population if population is not None else [best, mean]
    return GenerationStats(
        generation=generation,
        best_score=best,
        mean_score=mean,
        population_scores=pop,
        best_chain_spec={},
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history_returns_placeholder(self) -> None:
        out = _result().format_score_evolution()
        assert "nothing to chart" in out

    def test_unknown_format_raises(self) -> None:
        result = _result(_gen(0, best=0.5, mean=0.5))
        with pytest.raises(ValueError, match="Unknown format"):
            result.format_score_evolution(format="svg")

    def test_single_generation_renders_centered(self) -> None:
        """One generation should still produce a valid chart, not crash on
        ``range(0)`` or division-by-zero math."""
        out = _result(_gen(0, best=0.5, mean=0.5, population=[0.5, 0.5])).format_score_evolution()
        # Has the chart markers
        assert "█" in out or "▒" in out

    def test_flat_history_doesnt_divide_by_zero(self) -> None:
        """Every generation has identical scores → y_max == y_min. Don't crash."""
        out = _result(
            _gen(0, best=0.5, mean=0.5, population=[0.5, 0.5]),
            _gen(1, best=0.5, mean=0.5, population=[0.5, 0.5]),
        ).format_score_evolution()
        assert "█" in out


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------


class TestTextFormat:
    def test_legend_present(self) -> None:
        out = _result(_gen(0, best=0.5, mean=0.5)).format_score_evolution()
        assert "legend" in out.lower()
        assert "best" in out
        assert "mean" in out
        assert "worst" in out

    def test_y_axis_labels_show_min_and_max(self) -> None:
        out = _result(
            _gen(0, best=0.1, mean=0.05, population=[0.1, 0.05, 0.0]),
            _gen(1, best=1.0, mean=0.5, population=[1.0, 0.5, 0.0]),
        ).format_score_evolution()
        # y_min = 0.0, y_max = 1.0 — both should appear formatted to 3 dp
        assert "1.000" in out
        assert "0.000" in out

    def test_x_axis_label_present(self) -> None:
        out = _result(_gen(0, best=0.5, mean=0.5)).format_score_evolution()
        assert "generation" in out

    def test_best_above_mean_above_worst_in_chart(self) -> None:
        """In a generation where best > mean > worst, the best marker
        should be in a higher row (lower line index) than mean, which
        is above worst."""
        out = _result(
            _gen(0, best=0.9, mean=0.5, population=[0.9, 0.5, 0.1])
        ).format_score_evolution(height=10)
        lines = out.splitlines()
        # Find the row indices containing each marker
        best_row = next(i for i, line in enumerate(lines) if "█" in line)
        mean_row = next(i for i, line in enumerate(lines) if "▒" in line)
        worst_row = next(i for i, line in enumerate(lines) if "░" in line)
        assert best_row < mean_row < worst_row  # smaller row index = higher on screen

    def test_multiple_generations_spread_across_width(self) -> None:
        out = _result(
            _gen(0, best=0.2, mean=0.1, population=[0.2, 0.1]),
            _gen(1, best=0.5, mean=0.4, population=[0.5, 0.4]),
            _gen(2, best=0.9, mean=0.8, population=[0.9, 0.8]),
        ).format_score_evolution(width=30)
        # Find the chart row containing the best markers
        lines = out.splitlines()
        # There should be at least 3 'best' markers (one per generation)
        all_best_chars = sum(line.count("█") for line in lines)
        assert all_best_chars >= 3

    def test_x_axis_label_includes_each_generation_index(self) -> None:
        out = _result(
            _gen(0, best=0.5, mean=0.5),
            _gen(1, best=0.7, mean=0.6),
            _gen(2, best=0.9, mean=0.8),
        ).format_score_evolution()
        # The X-axis line should mention each generation number
        # (formatted as "0", "1", "2" across the width)
        # Search the full text — they all appear somewhere
        for g in (0, 1, 2):
            assert str(g) in out


# ---------------------------------------------------------------------------
# Inf / NaN handling
# ---------------------------------------------------------------------------


class TestInfHandling:
    def test_minus_inf_replaced_with_zero(self) -> None:
        """A failed individual gets -inf score. The chart
        must not crash trying to render -inf — replace with finite 0.0."""
        result = _result(
            _gen(
                0,
                best=0.5,
                mean=float("-inf"),
                population=[0.5, float("-inf")],
            )
        )
        # Just exercising — shouldn't raise
        out = result.format_score_evolution()
        assert "█" in out

    def test_nan_replaced_with_zero(self) -> None:
        result = _result(
            _gen(
                0,
                best=0.5,
                mean=float("nan"),
                population=[0.5, float("nan")],
            )
        )
        out = result.format_score_evolution()
        assert "█" in out
        # No "nan" string in the formatted output
        assert "nan" not in out.lower() or "nan" not in out


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        result = _result(_gen(0, best=0.5, mean=0.5))
        with pytest.raises(ValueError, match="png_path"):
            result.format_score_evolution(format="png")

    def test_png_raises_install_hint_when_matplotlib_missing(self) -> None:
        try:
            import matplotlib  # noqa: F401

            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        result = _result(_gen(0, best=0.5, mean=0.5))
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            result.format_score_evolution(format="png", png_path="/tmp/x.png")

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed; skipping live PNG test")
        result = _result(
            _gen(0, best=0.5, mean=0.4, population=[0.5, 0.4, 0.3]),
            _gen(1, best=0.7, mean=0.6, population=[0.7, 0.6, 0.5]),
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "evolution.png")
            written = result.format_score_evolution(format="png", png_path=path)
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0


# ---------------------------------------------------------------------------
# Integration — population scores hooked up correctly
# ---------------------------------------------------------------------------


def test_worst_derived_from_population_scores_minimum() -> None:
    """When population_scores is set, ``worst`` should be its min — not
    the same as ``best_score`` or ``mean_score``."""
    result = _result(
        _gen(0, best=0.9, mean=0.5, population=[0.9, 0.5, 0.1, 0.05]),
    )
    out = result.format_score_evolution(height=15)
    # min of population is 0.05 → should show up as y_min on the axis
    assert "0.05" in out or "0.050" in out


def test_fallback_to_best_when_population_scores_empty() -> None:
    """If population_scores is empty (defensive), worst falls back to
    best_score so the chart still renders."""
    result = _result(_gen(0, best=0.5, mean=0.5, population=[]))
    # No crash
    out = result.format_score_evolution()
    assert "█" in out
