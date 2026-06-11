"""Tests for ``DatasetEvaluationReport.format_cost_trend``.

Per-run cost / token trend sparkline + regression detection. Lets
users running the same chain N times spot prompt-regressions that
inflate spend.
"""

from __future__ import annotations

import pytest

from mmar_carl.models.dataset import (
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    ThresholdStrategy,
)


def _case(
    label: str,
    *,
    total: int = 0,
    prompt_t: int = 0,
    completion_t: int = 0,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input=label, label=label),
        score=1.0, chain_output="", success=True,
        token_usage=(
            {"prompt": prompt_t, "completion": completion_t, "total": total}
            if total else {}
        ),
    )


def _report(*cases: CaseEvaluationResult) -> DatasetEvaluationReport:
    return DatasetEvaluationReport(
        metric_name="m",
        strategy=ThresholdStrategy(threshold=0.0),
        all_results=list(cases), selected_cases=[],
        mean_score=1.0, min_score=1.0, max_score=1.0,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_report_returns_placeholder(self) -> None:
        out = _report().format_cost_trend()
        assert "no runs evaluated" in out

    def test_no_token_usage_returns_placeholder(self) -> None:
        out = _report(_case("a"), _case("b")).format_cost_trend()
        assert "no per-run token usage" in out

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            _report(_case("a", total=100)).format_cost_trend(format="svg")


# ---------------------------------------------------------------------------
# Sparkline + per-run rows
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_sparkline_renders_one_block_per_run(self) -> None:
        out = _report(
            _case("r1", total=100),
            _case("r2", total=200),
            _case("r3", total=300),
        ).format_cost_trend()
        # The sparkline is the second line; it should be 3 chars long
        # (one block per run since N=3 < default sparkline_width=40).
        spark_line = out.splitlines()[1]
        assert len(spark_line) == 3

    def test_every_run_shows_at_least_min_glyph(self) -> None:
        """An outlier shouldn't blank out all other runs — even the
        lowest non-zero value must render as `▁`."""
        out = _report(
            _case("r1", total=100),
            _case("r2", total=100),
            _case("r3", total=100),
            _case("r4", total=10000),  # outlier
            _case("r5", total=100),
        ).format_cost_trend()
        spark_line = out.splitlines()[1]
        # All five runs render as a visible glyph, not space.
        assert "▁" in spark_line
        assert "█" in spark_line
        # No spaces inside the sparkline portion
        assert " " not in spark_line

    def test_per_run_row_lists_tokens(self) -> None:
        out = _report(_case("r1", total=123)).format_cost_trend()
        assert "tokens=123" in out

    def test_per_run_row_falls_back_to_em_dash_without_pricing(self) -> None:
        out = _report(_case("r1", total=100)).format_cost_trend()
        # No pricing supplied → cost column shows "—"
        assert "cost=—" in out


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_cost_computed_when_pricing_supplied(self) -> None:
        out = _report(
            _case("r1", total=2000, prompt_t=1000, completion_t=1000),
        ).format_cost_trend(
            pricing={"m1": (0.001, 0.001)},
            default_model="m1",
        )
        # cost = 1000/1000 * 0.001 + 1000/1000 * 0.001 = 0.002
        assert "cost=$0.0020" in out

    def test_default_model_missing_from_pricing_falls_back_to_tokens(self) -> None:
        out = _report(
            _case("r1", total=100, prompt_t=50, completion_t=50),
        ).format_cost_trend(
            pricing={"other-model": (0.001, 0.001)},
            default_model="not-in-pricing",
        )
        # Falls back to em-dash
        assert "cost=—" in out

    def test_header_lists_model_when_pricing_used(self) -> None:
        out = _report(
            _case("r1", total=100, prompt_t=50, completion_t=50),
        ).format_cost_trend(
            pricing={"qwen-8b": (0.001, 0.001)},
            default_model="qwen-8b",
        )
        assert "model=qwen-8b" in out.splitlines()[0]


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    def test_regression_flagged_when_run_exceeds_factor_times_median(self) -> None:
        out = _report(
            _case("r1", total=100),
            _case("r2", total=100),
            _case("r3", total=500),  # 5× median
            _case("r4", total=100),
            _case("r5", total=100),
        ).format_cost_trend(regression_factor=2.0)
        assert "regression detected at run 3" in out
        # Median × ratio should mention 5.0 (500 / 100)
        assert "× 5.0" in out

    def test_no_regression_when_all_runs_similar(self) -> None:
        out = _report(
            _case("r1", total=100),
            _case("r2", total=110),
            _case("r3", total=95),
            _case("r4", total=105),
        ).format_cost_trend(regression_factor=2.0)
        assert "regression" not in out

    def test_regression_factor_respected(self) -> None:
        """A run that's 1.5× the median should NOT trip a 2.0× threshold,
        but DOES trip a 1.2× threshold."""
        cases = (
            _case("r1", total=100),
            _case("r2", total=100),
            _case("r3", total=150),
            _case("r4", total=100),
        )
        no_trip = _report(*cases).format_cost_trend(regression_factor=2.0)
        assert "regression" not in no_trip

        trip = _report(*cases).format_cost_trend(regression_factor=1.2)
        assert "regression detected at run 3" in trip

    def test_regression_uses_cost_when_pricing_supplied(self) -> None:
        out = _report(
            _case("r1", total=200, prompt_t=100, completion_t=100),
            _case("r2", total=200, prompt_t=100, completion_t=100),
            _case("r3", total=600, prompt_t=300, completion_t=300),  # 3× cost
            _case("r4", total=200, prompt_t=100, completion_t=100),
        ).format_cost_trend(
            pricing={"m1": (0.01, 0.01)},
            default_model="m1",
            regression_factor=2.0,
        )
        assert "regression detected at run 3" in out
        # The regression line should reference cost, not tokens
        regression_line = next(
            line for line in out.splitlines() if "regression" in line
        )
        assert "cost=" in regression_line


# ---------------------------------------------------------------------------
# Median calculation
# ---------------------------------------------------------------------------


class TestMedian:
    def test_median_odd_count(self) -> None:
        out = _report(
            _case("r1", total=100),
            _case("r2", total=200),
            _case("r3", total=300),
        ).format_cost_trend()
        assert "median tokens=200" in out

    def test_median_even_count_averages(self) -> None:
        out = _report(
            _case("r1", total=100),
            _case("r2", total=200),
            _case("r3", total=300),
            _case("r4", total=400),
        ).format_cost_trend()
        # median = (200+300)/2 = 250
        assert "median tokens=250" in out


# ---------------------------------------------------------------------------
# Header / footer structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_counts_usable_runs_not_total(self) -> None:
        # 3 with usage, 2 without → header should say "3 runs"
        out = _report(
            _case("r1", total=100),
            _case("r2"),
            _case("r3", total=200),
            _case("r4"),
            _case("r5", total=300),
        ).format_cost_trend()
        assert "3 runs" in out.splitlines()[0]
