"""Tests for ``DatasetEvaluationReport.format_score_distribution``.

A Unicode box plot of chain-level metric scores across a dataset.
Surfaces long lower-tails / bimodal distributions that mean+min+max
hides.
"""

from __future__ import annotations

import pytest

from mmar_carl.models.dataset import (
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    ThresholdStrategy,
)


def _case(label: str, score: float) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input=label, label=label),
        score=score,
        chain_output="",
        success=True,
    )


def _report(*scores: float, metric: str = "m") -> DatasetEvaluationReport:
    cases = [_case(f"c{i}", s) for i, s in enumerate(scores)]
    return DatasetEvaluationReport(
        metric_name=metric,
        strategy=ThresholdStrategy(threshold=0.0),
        all_results=cases,
        selected_cases=[],
        mean_score=sum(scores) / len(scores) if scores else 0.0,
        min_score=min(scores, default=0.0),
        max_score=max(scores, default=0.0),
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_report_returns_placeholder(self) -> None:
        out = _report().format_score_distribution()
        assert "no cases evaluated" in out

    def test_single_case_returns_placeholder(self) -> None:
        out = _report(0.5).format_score_distribution()
        assert "only 1 case" in out
        assert "0.50" in out

    def test_all_equal_scores_returns_placeholder(self) -> None:
        out = _report(1.0, 1.0, 1.0, 1.0).format_score_distribution()
        assert "all scores equal" in out
        assert "1.00" in out

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            _report(0.1, 0.9).format_score_distribution(format="svg")


# ---------------------------------------------------------------------------
# Quantile arithmetic
# ---------------------------------------------------------------------------


class TestQuantiles:
    def test_min_and_max_taken_from_sorted_scores(self) -> None:
        out = _report(0.8, 0.1, 0.5, 0.9, 0.3).format_score_distribution()
        # min == 0.10, max == 0.90
        assert "min=0.10" in out
        assert "max=0.90" in out

    def test_median_for_odd_count_is_middle_value(self) -> None:
        out = _report(0.1, 0.5, 0.9).format_score_distribution()
        assert "med=0.50" in out

    def test_median_for_even_count_interpolates(self) -> None:
        # Four scores 0.1, 0.3, 0.7, 0.9 → median = (0.3+0.7)/2 = 0.50
        out = _report(0.1, 0.3, 0.7, 0.9).format_score_distribution()
        assert "med=0.50" in out

    def test_quartiles_for_known_dataset(self) -> None:
        # Five evenly-spaced scores; with numpy-style linear interpolation:
        # Q1 = score at index 1 = 0.25, Q3 = score at index 3 = 0.75
        out = _report(0.0, 0.25, 0.5, 0.75, 1.0).format_score_distribution()
        assert "Q1=0.25" in out
        assert "med=0.50" in out
        assert "Q3=0.75" in out


# ---------------------------------------------------------------------------
# Canvas rendering
# ---------------------------------------------------------------------------


class TestCanvas:
    def test_canvas_width_respected(self) -> None:
        out = _report(0.0, 0.25, 0.5, 0.75, 1.0).format_score_distribution(
            canvas_width=20
        )
        # The canvas line is line index 1 (line 0 is the header).
        canvas_line = out.splitlines()[1]
        assert len(canvas_line) == 20

    def test_default_canvas_width_is_40(self) -> None:
        out = _report(0.0, 0.5, 1.0).format_score_distribution()
        canvas_line = out.splitlines()[1]
        assert len(canvas_line) == 40

    def test_canvas_contains_box_and_endcap_glyphs(self) -> None:
        out = _report(0.0, 0.25, 0.5, 0.75, 1.0).format_score_distribution()
        canvas = out.splitlines()[1]
        # All five glyphs must appear
        for glyph in ("├", "─", "╞", "═", "█", "╡", "┤"):
            assert glyph in canvas, f"missing glyph {glyph!r} in {canvas!r}"

    def test_endpoints_at_canvas_boundaries(self) -> None:
        out = _report(0.0, 0.5, 1.0).format_score_distribution(canvas_width=40)
        canvas = out.splitlines()[1]
        # min endpoint at start, max endpoint at end
        assert canvas[0] == "├"
        assert canvas[-1] == "┤"

    def test_minimum_canvas_width_floor_at_5(self) -> None:
        # Width below 5 should be silently bumped to 5 so the canvas still
        # has room for ├ ╞ █ ╡ ┤.
        out = _report(0.0, 0.5, 1.0).format_score_distribution(canvas_width=2)
        canvas = out.splitlines()[1]
        assert len(canvas) >= 5

    def test_left_skew_has_more_left_whisker(self) -> None:
        """When most scores cluster near the top, the LEFT whisker (min..Q1)
        should be visibly longer than the right (Q3..max)."""
        out = _report(
            0.0, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98
        ).format_score_distribution(canvas_width=60)
        canvas = out.splitlines()[1]
        q1_idx = canvas.index("╞")
        q3_idx = canvas.index("╡")
        left_whisker = q1_idx  # 0 .. q1_idx
        right_whisker = (len(canvas) - 1) - q3_idx
        assert left_whisker > right_whisker, (
            f"expected fat left whisker, got L={left_whisker} R={right_whisker}"
        )


# ---------------------------------------------------------------------------
# Header / metric name
# ---------------------------------------------------------------------------


class TestHeader:
    def test_header_includes_metric_name(self) -> None:
        out = _report(0.1, 0.5, 0.9, metric="exact_match").format_score_distribution()
        assert "metric='exact_match'" in out

    def test_header_includes_case_count(self) -> None:
        out = _report(0.1, 0.5, 0.9, 0.7, 0.3).format_score_distribution()
        assert "5 cases" in out
