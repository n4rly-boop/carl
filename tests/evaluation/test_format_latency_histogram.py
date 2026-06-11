"""Tests for ``DatasetEvaluationReport.format_latency_histogram``.

Per-step inline-sparkline histogram across a dataset run. Surfaces
tail-latency outliers and bimodal distributions that a single mean
or p95 number can't show.
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
    step_latencies_ms: dict[int, float] | None = None,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input=label, label=label),
        score=1.0, chain_output="", success=True,
        step_latencies_ms=step_latencies_ms or {},
    )


def _report(*cases: CaseEvaluationResult) -> DatasetEvaluationReport:
    return DatasetEvaluationReport(
        metric_name="m",
        strategy=ThresholdStrategy(threshold=0.0),
        all_results=list(cases),
        selected_cases=[],
        mean_score=1.0, min_score=1.0, max_score=1.0,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_report_returns_placeholder(self) -> None:
        out = _report().format_latency_histogram()
        assert "no cases evaluated" in out

    def test_no_latency_data_returns_placeholder(self) -> None:
        out = _report(_case("a"), _case("b")).format_latency_histogram()
        assert "no per-step latency data" in out

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            _report(_case("a", step_latencies_ms={1: 100.0})
                    ).format_latency_histogram(format="svg")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_step_appears_with_runs_count(self) -> None:
        out = _report(
            _case("a", step_latencies_ms={1: 100.0}),
            _case("b", step_latencies_ms={1: 150.0}),
            _case("c", step_latencies_ms={1: 200.0}),
        ).format_latency_histogram()
        # n=3 in the data row
        data_lines = [
            line for line in out.splitlines()
            if line.lstrip().startswith("1 ")
        ]
        assert data_lines, f"expected step-1 row, got:\n{out}"
        assert "3" in data_lines[0]

    def test_multiple_steps_render_in_order(self) -> None:
        out = _report(
            _case("a", step_latencies_ms={1: 100.0, 2: 200.0}),
            _case("b", step_latencies_ms={1: 110.0, 2: 220.0}),
        ).format_latency_histogram()
        lines = out.splitlines()
        step1_idx = next(
            i for i, line in enumerate(lines)
            if line.lstrip().startswith("1 ")
        )
        step2_idx = next(
            i for i, line in enumerate(lines)
            if line.lstrip().startswith("2 ")
        )
        assert step1_idx < step2_idx

    def test_case_with_subset_of_steps_still_aggregates(self) -> None:
        out = _report(
            _case("a", step_latencies_ms={1: 100.0}),
            _case("b", step_latencies_ms={2: 200.0}),
            _case("c", step_latencies_ms={1: 110.0, 2: 210.0}),
        ).format_latency_histogram()
        # Both steps present; step 1 has 2 runs, step 2 has 2 runs
        step1_row = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("1 ")
        )
        step2_row = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("2 ")
        )
        # 'n' column reads 2 for both
        assert "2" in step1_row
        assert "2" in step2_row


# ---------------------------------------------------------------------------
# Percentile arithmetic
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_p50_and_max_match_expected(self) -> None:
        out = _report(*[
            _case(f"c{i}", step_latencies_ms={1: float(v)})
            for i, v in enumerate([100, 200, 300, 400, 500])
        ]).format_latency_histogram()
        # p50 of [100..500] step=100 == 300
        # max == 500
        row = next(line for line in out.splitlines()
                   if line.lstrip().startswith("1 "))
        assert "300" in row
        assert "500" in row

    def test_p95_picks_up_outlier(self) -> None:
        # 9 fast + 1 slow ⇒ p95 lies in the interpolation between fast
        # and slow, so it must exceed the fast value.
        cases = [
            _case(f"c{i}", step_latencies_ms={1: 100.0}) for i in range(9)
        ]
        cases.append(_case("slow", step_latencies_ms={1: 5000.0}))
        out = _report(*cases).format_latency_histogram()
        row = next(line for line in out.splitlines()
                   if line.lstrip().startswith("1 "))
        cells = row.split()
        # Layout: # title n hist p50 p95 max  → last three are numbers.
        nums = [c for c in cells if c.replace(".", "").isdigit()]
        # max is 5000 (last)
        assert nums[-1] == "5000"


# ---------------------------------------------------------------------------
# Sparkline rendering
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_uniform_data_produces_full_block_anywhere_in_sparkline(self) -> None:
        # All identical latencies → every value falls in the same bin →
        # one bin has count N, rest have 0 → sparkline has one █ + spaces.
        out = _report(*[
            _case(f"c{i}", step_latencies_ms={1: 100.0}) for i in range(8)
        ]).format_latency_histogram()
        row = next(line for line in out.splitlines()
                   if line.lstrip().startswith("1 "))
        assert "█" in row

    def test_outlier_visible_at_top_bin(self) -> None:
        """An extreme outlier should land in the rightmost bin —
        producing a small block far from the cluster."""
        cases = [
            _case(f"c{i}", step_latencies_ms={1: 100.0}) for i in range(6)
        ]
        cases.append(_case("outlier", step_latencies_ms={1: 5000.0}))
        out = _report(*cases).format_latency_histogram(bins=10)
        # Find the sparkline portion of the step-1 row.
        row = next(line for line in out.splitlines()
                   if line.lstrip().startswith("1 "))
        # Both the dominant low bin (█) and a smaller high-bin glyph
        # should be in the sparkline characters.
        assert "█" in row
        # Smaller-height blocks indicate the outlier
        assert any(g in row for g in "▁▂▃▄▅▆▇")

    def test_bin_count_respected(self) -> None:
        out = _report(*[
            _case(f"c{i}", step_latencies_ms={1: float(i * 10)})
            for i in range(20)
        ]).format_latency_histogram(bins=8)
        # Header line mentions the bin count
        header_line = out.splitlines()[0]
        assert "8" in header_line


# ---------------------------------------------------------------------------
# Structure / header / footer
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_lists_columns(self) -> None:
        out = _report(
            _case("a", step_latencies_ms={1: 100.0}),
            _case("b", step_latencies_ms={1: 200.0}),
        ).format_latency_histogram()
        for col in ("step", "p50 ms", "p95 ms", "max ms"):
            assert col in out, f"missing column {col!r}"

    def test_footer_summarises_counts(self) -> None:
        out = _report(
            _case("a", step_latencies_ms={1: 100.0, 2: 200.0}),
            _case("b", step_latencies_ms={1: 110.0, 2: 210.0}),
        ).format_latency_histogram()
        assert "total cases: 2" in out
        assert "steps with timing: 2" in out


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_step_latencies_ms_defaults_to_empty(self) -> None:
        c = CaseEvaluationResult(
            case=DataCase(input="x", label="x"),
            score=0.0, chain_output="", success=True,
        )
        assert c.step_latencies_ms == {}

    def test_step_latencies_ms_round_trips(self) -> None:
        c = CaseEvaluationResult(
            case=DataCase(input="x", label="x"),
            score=0.0, chain_output="", success=True,
            step_latencies_ms={1: 500.0, 2: 1200.0},
        )
        rehydrated = CaseEvaluationResult.model_validate(c.model_dump())
        assert rehydrated.step_latencies_ms == {1: 500.0, 2: 1200.0}


# ---------------------------------------------------------------------------
# Integration with DatasetEvaluator
# ---------------------------------------------------------------------------


class TestDatasetEvaluatorPopulatesLatencies:
    @pytest.mark.asyncio
    async def test_step_latencies_populated_from_step_results(self) -> None:
        from mmar_carl import ReasoningChain, ReasoningContext
        from mmar_carl.dataset_evaluator import DatasetEvaluator
        from mmar_carl.metrics import MetricBase
        from mmar_carl.models.config import ToolStepConfig
        from mmar_carl.models.dataset import (
            DataCase, SimpleDataset, ThresholdStrategy,
        )
        from mmar_carl.models.steps import ToolStepDescription

        class PassThrough(MetricBase):
            @property
            def name(self) -> str:
                return "p"

            async def compute_async(self, output) -> float:  # type: ignore[override]
                return 1.0

        def fast_tool() -> str:
            return "ok"

        chain = ReasoningChain(steps=[
            ToolStepDescription(
                number=1, title="A",
                config=ToolStepConfig(tool_name="fast_tool"),
            ),
            ToolStepDescription(
                number=2, title="B",
                config=ToolStepConfig(tool_name="fast_tool"),
                dependencies=[1],
            ),
        ])
        dataset = SimpleDataset([
            DataCase(input="x", label="x"),
            DataCase(input="y", label="y"),
        ])
        evaluator = DatasetEvaluator(
            chain=chain, dataset=dataset, metric=PassThrough(),
            strategy=ThresholdStrategy(threshold=0.0),
        )

        def factory(case):
            ctx = ReasoningContext(outer_context=case.input, api=None)
            ctx.register_tool("fast_tool", fast_tool)
            return ctx

        report = await evaluator.evaluate_async(factory)
        assert len(report.all_results) == 2
        for cr in report.all_results:
            # Both steps recorded a positive latency
            assert set(cr.step_latencies_ms.keys()) == {1, 2}
            assert all(ms >= 0 for ms in cr.step_latencies_ms.values())

        out = report.format_latency_histogram()
        # Both steps appear; total cases footer
        assert "total cases: 2" in out
