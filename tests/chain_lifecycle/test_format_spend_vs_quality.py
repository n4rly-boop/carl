"""Tests for ``EvolutionResult.format_spend_vs_quality``.

Plots cumulative evolution spend on x-axis vs
best-so-far score on y-axis. Each generation contributes one point;
the curve typically climbs steeply then plateaus — the elbow is the
"stop spending" signal.
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


def _gen(
    generation: int,
    best: float,
    *,
    tokens: int = 1000,
    n: int = 1,
) -> GenerationStats:
    """One generation with `n` individuals, each consuming `tokens // n`."""
    per_individual = tokens // max(n, 1)
    return GenerationStats(
        generation=generation,
        best_score=best,
        mean_score=best,
        population_scores=[best] * n,
        population_metrics=[
            IndividualMetrics(score=best, total_tokens=per_individual)
            for _ in range(n)
        ],
    )


def _result(*generations: GenerationStats) -> EvolutionResult:
    bests = [g.best_score for g in generations]
    return EvolutionResult(
        best_chain_spec={},
        best_score=max(bests, default=0.0),
        best_generation=max((g.generation for g in generations), default=0),
        history=list(generations),
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history_returns_placeholder(self) -> None:
        out = _result().format_spend_vs_quality()
        assert "nothing to chart" in out

    def test_unknown_format_raises(self) -> None:
        result = _result(_gen(0, 0.5))
        with pytest.raises(ValueError, match="Unknown format"):
            result.format_spend_vs_quality(format="svg")

    def test_single_generation_renders(self) -> None:
        """One gen → both x and y collapse to a single point. Should still
        render without div-by-zero."""
        out = _result(_gen(0, 0.5)).format_spend_vs_quality()
        # Latest marker present
        assert "◉" in out
        # Generation count is 1
        assert "(1 total)" in out

    def test_flat_history_doesnt_divide_by_zero(self) -> None:
        """All gens have identical score and identical tokens — degenerate
        case. Don't crash."""
        out = _result(
            _gen(0, 0.5, tokens=1000),
            _gen(1, 0.5, tokens=1000),
        ).format_spend_vs_quality()
        assert "◉" in out


# ---------------------------------------------------------------------------
# Cumulative-spend math
# ---------------------------------------------------------------------------


class TestCumulativeSpend:
    def test_tokens_accumulate_across_generations(self) -> None:
        """Cumulative tokens at end == sum of all per-generation tokens."""
        out = _result(
            _gen(0, 0.3, tokens=1000, n=2),  # gen 0: 1000 tokens
            _gen(1, 0.6, tokens=1500, n=2),  # gen 1: +1500 → 2500 total
            _gen(2, 0.8, tokens=2000, n=2),  # gen 2: +2000 → 4500 total
        ).format_spend_vs_quality()
        # X-axis max should show ~4500 (with thousands separator)
        # Actual sum: gen 0 = 1000 (500+500), gen 1 = 1500 (750+750),
        # gen 2 = 2000 (1000+1000) → cumulative 4500
        assert "4,500" in out

    def test_best_so_far_only_increases(self) -> None:
        """If gen N's best drops below gen N-1's, the displayed line
        should NOT dip (best-so-far is monotonic)."""
        out = _result(
            _gen(0, 0.5),
            _gen(1, 0.8),
            _gen(2, 0.3),  # regression — best drops
        ).format_spend_vs_quality()
        # The y-axis max should be 0.8 (gen 1's best), not 0.3
        # Best-so-far tracker means the chart shows 0.5 → 0.8 → 0.8.
        assert "0.800" in out
        # Plateau detection should fire because last marginal gain = 0
        assert "plateau detected" in out


# ---------------------------------------------------------------------------
# X-axis: tokens vs cost
# ---------------------------------------------------------------------------


class TestAxisModes:
    def test_default_x_axis_is_tokens(self) -> None:
        out = _result(_gen(0, 0.5), _gen(1, 0.7)).format_spend_vs_quality()
        assert "cumulative tokens" in out

    def test_cost_per_1k_flips_axis_to_dollars(self) -> None:
        out = _result(
            _gen(0, 0.5, tokens=1000),
            _gen(1, 0.7, tokens=2000),
        ).format_spend_vs_quality(cost_per_1k=0.05)
        assert "cumulative cost (USD)" in out
        # gen 0: 1000 tokens * 0.05/1000 = $0.0500
        # cumulative at end: (1000 + 2000) * 0.05/1000 = $0.1500
        assert "$0.1500" in out


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------


class TestPlateauDetection:
    def test_no_plateau_when_score_keeps_growing(self) -> None:
        out = _result(
            _gen(0, 0.1),
            _gen(1, 0.4),
            _gen(2, 0.9),  # large marginal gain
        ).format_spend_vs_quality()
        assert "plateau detected" not in out

    def test_plateau_detected_when_last_marginal_below_5pct(self) -> None:
        """gen 0 → 0.4, gen 1 → 0.75, gen 2 → 0.92, gen 3 → 0.93.
        Total gain = 0.53; marginal last = 0.01; ratio = 1.9% < 5% → plateau."""
        out = _result(
            _gen(0, 0.40),
            _gen(1, 0.75),
            _gen(2, 0.92),
            _gen(3, 0.93),
        ).format_spend_vs_quality()
        assert "plateau detected" in out
        assert "<5%" in out

    def test_no_plateau_label_with_only_two_generations(self) -> None:
        """Plateau heuristic requires ≥3 generations — anything less can't
        meaningfully say 'the curve is flat'."""
        out = _result(
            _gen(0, 0.5),
            _gen(1, 0.51),  # tiny gain, but only 2 gens
        ).format_spend_vs_quality()
        assert "plateau detected" not in out


# ---------------------------------------------------------------------------
# Chart structure
# ---------------------------------------------------------------------------


class TestChartStructure:
    def test_text_format_includes_axis_labels_and_legend(self) -> None:
        out = _result(_gen(0, 0.3), _gen(1, 0.7)).format_spend_vs_quality()
        assert "cumulative tokens" in out  # x-axis label
        assert "0.300" in out  # y-axis min
        assert "0.700" in out  # y-axis max
        assert "legend:" in out
        assert "◉ latest generation" in out

    def test_text_format_latest_marker_is_distinct(self) -> None:
        out = _result(_gen(0, 0.3), _gen(1, 0.5), _gen(2, 0.7)).format_spend_vs_quality()
        # Count only inside the chart canvas (rows with the │ axis char) —
        # the legend line also contains ◉/█ as part of the explanation.
        chart_lines = [line for line in out.splitlines() if "│" in line]
        chart_body = "\n".join(chart_lines)
        # ◉ should appear exactly once (the latest gen).
        assert chart_body.count("◉") == 1
        # And █ for the earlier gens (2 of them).
        assert chart_body.count("█") == 2

    def test_generation_count_in_legend(self) -> None:
        out = _result(_gen(0, 0.3), _gen(1, 0.5), _gen(2, 0.7), _gen(3, 0.9)).format_spend_vs_quality()
        assert "(4 total)" in out


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        result = _result(_gen(0, 0.5))
        with pytest.raises(ValueError, match="png_path"):
            result.format_spend_vs_quality(format="png")

    def test_png_raises_install_hint_when_matplotlib_missing(self) -> None:
        try:
            import matplotlib  # noqa: F401
            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        result = _result(_gen(0, 0.5), _gen(1, 0.7))
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            result.format_spend_vs_quality(format="png", png_path="/tmp/x.png")

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed; skipping live PNG test")
        result = _result(
            _gen(0, 0.3, tokens=500),
            _gen(1, 0.6, tokens=700),
            _gen(2, 0.8, tokens=800),
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "spend_quality.png")
            written = result.format_spend_vs_quality(format="png", png_path=path)
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0
