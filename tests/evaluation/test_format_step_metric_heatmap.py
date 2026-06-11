"""Tests for ``DatasetEvaluationReport.format_step_metric_heatmap``.

Shows per-case × per-step values of a single metric across the
dataset — useful for spotting "step 3 always scores low on long inputs"
patterns that the failure heatmap can't show (because the step still
returns ``success=True``).
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
    step_metrics: dict[int, dict[str, float]] | None = None,
    success: bool = True,
    score: float = 0.5,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input=label, label=label),
        score=score,
        chain_output="",
        success=success,
        step_metrics=step_metrics or {},
    )


def _report(*cases: CaseEvaluationResult) -> DatasetEvaluationReport:
    scores = [c.score for c in cases]
    return DatasetEvaluationReport(
        metric_name="m",
        strategy=ThresholdStrategy(threshold=0.0),
        all_results=list(cases),
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
        report = DatasetEvaluationReport(
            metric_name="m",
            strategy=ThresholdStrategy(threshold=0.0),
            all_results=[],
            selected_cases=[],
            mean_score=0.0, min_score=0.0, max_score=0.0,
        )
        out = report.format_step_metric_heatmap("anything")
        assert "no cases evaluated" in out

    def test_metric_not_recorded_returns_placeholder(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"other": 0.5}}),
        ).format_step_metric_heatmap("missing")
        assert "not recorded on any step" in out

    def test_cases_without_step_metrics_returns_placeholder(self) -> None:
        out = _report(
            _case("a"),
            _case("b"),
        ).format_step_metric_heatmap("anything")
        assert "not recorded on any step" in out


# ---------------------------------------------------------------------------
# Cell shading
# ---------------------------------------------------------------------------


class TestShading:
    def test_top_value_renders_full_block(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 1.0}, 2: {"q": 0.0}}),
        ).format_step_metric_heatmap("q")
        assert "█" in out

    def test_bottom_value_renders_dot(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 1.0}, 2: {"q": 0.0}}),
        ).format_step_metric_heatmap("q")
        assert "·" in out

    def test_missing_cell_renders_dash(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.5}}),
            _case("b", step_metrics={2: {"q": 0.7}}),
        ).format_step_metric_heatmap("q")
        # Both rows have a "-" for the step the other case had
        assert "-" in out

    def test_uniform_values_dont_crash(self) -> None:
        # All scores equal → span=0; method must avoid div-by-zero.
        out = _report(
            _case("a", step_metrics={1: {"q": 0.5}, 2: {"q": 0.5}}),
            _case("b", step_metrics={1: {"q": 0.5}, 2: {"q": 0.5}}),
        ).format_step_metric_heatmap("q")
        # All cells map to the same bucket (top, by the t<X<=1.0 rules)
        assert "█" in out


# ---------------------------------------------------------------------------
# Structure: header, mean row, legend
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_lists_step_numbers(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.1}, 3: {"q": 0.9}}),
            _case("b", step_metrics={2: {"q": 0.5}}),
        ).format_step_metric_heatmap("q")
        header_line = out.splitlines()[0]
        # Union of step numbers, sorted ascending
        assert "1" in header_line
        assert "2" in header_line
        assert "3" in header_line
        # Order: 1 before 2 before 3
        assert header_line.index("1") < header_line.index("2") < header_line.index("3")

    def test_case_labels_present_and_truncated(self) -> None:
        out = _report(
            _case("short", step_metrics={1: {"q": 0.5}}),
            _case("a-very-long-case-name-that-should-truncate",
                  step_metrics={1: {"q": 0.7}}),
        ).format_step_metric_heatmap("q", case_label_width=12)
        assert "short" in out
        # Long label truncated with …
        assert "…" in out

    def test_mean_row_present_and_correct(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.8}}),
            _case("b", step_metrics={1: {"q": 0.4}}),
        ).format_step_metric_heatmap("q")
        # mean of step 1 = 0.6
        assert "mean" in out
        assert "0.60" in out

    def test_mean_row_skips_steps_with_no_recorded_value(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.9}}),
            _case("b", step_metrics={2: {"q": 0.1}}),
        ).format_step_metric_heatmap("q")
        # Step 1 mean = 0.9; step 2 mean = 0.1; both should appear
        assert "0.90" in out
        assert "0.10" in out

    def test_legend_mentions_metric_name_and_scale(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.2}, 2: {"q": 0.8}}),
        ).format_step_metric_heatmap("q")
        assert "metric: 'q'" in out
        assert "0.20" in out  # scale low
        assert "0.80" in out  # scale high
        assert "low → high" in out

    def test_legend_counts_cases_and_steps(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.5}, 2: {"q": 0.5}}),
            _case("b", step_metrics={1: {"q": 0.5}, 2: {"q": 0.5}}),
            _case("c", step_metrics={1: {"q": 0.5}, 2: {"q": 0.5}}),
        ).format_step_metric_heatmap("q")
        assert "3 cases" in out
        assert "2 steps" in out


# ---------------------------------------------------------------------------
# Scale overrides
# ---------------------------------------------------------------------------


class TestScale:
    def test_scale_min_max_override_observed_range(self) -> None:
        # Observed min=0.5, max=0.6 (narrow range). With explicit scale
        # [0, 1] both should fall in the middle band (▒), not at extremes.
        out = _report(
            _case("a", step_metrics={1: {"q": 0.5}, 2: {"q": 0.6}}),
        ).format_step_metric_heatmap("q", scale_min=0.0, scale_max=1.0)
        # Neither value should map to the top (█) or bottom (·) cell
        chart_lines = [
            line for line in out.splitlines()
            if line.startswith("a")
        ]
        assert chart_lines, "expected a data row for case 'a'"
        row = chart_lines[0]
        assert "█" not in row
        assert "·" not in row

    def test_scale_used_in_legend(self) -> None:
        out = _report(
            _case("a", step_metrics={1: {"q": 0.3}}),
        ).format_step_metric_heatmap("q", scale_min=0.0, scale_max=1.0)
        assert "[0.00 … 1.00]" in out


# ---------------------------------------------------------------------------
# Schema additions
# ---------------------------------------------------------------------------


class TestSchema:
    def test_step_metrics_defaults_to_empty(self) -> None:
        c = CaseEvaluationResult(
            case=DataCase(input="x", label="x"),
            score=0.0, chain_output="", success=True,
        )
        assert c.step_metrics == {}

    def test_step_metrics_round_trips(self) -> None:
        c = CaseEvaluationResult(
            case=DataCase(input="x", label="x"),
            score=0.0, chain_output="", success=True,
            step_metrics={1: {"a": 0.5, "b": 0.7}, 2: {"a": 0.1}},
        )
        d = c.model_dump()
        rehydrated = CaseEvaluationResult.model_validate(d)
        assert rehydrated.step_metrics == {1: {"a": 0.5, "b": 0.7}, 2: {"a": 0.1}}


# ---------------------------------------------------------------------------
# Integration with DatasetEvaluator
# ---------------------------------------------------------------------------


class TestDatasetEvaluatorPopulatesStepMetrics:
    @pytest.mark.asyncio
    async def test_step_metrics_populated_from_step_results(self) -> None:
        """Run a real evaluator end-to-end and confirm step_metrics
        is wired into the resulting CaseEvaluationResult."""
        from mmar_carl import ReasoningChain, ReasoningContext
        from mmar_carl.dataset_evaluator import DatasetEvaluator
        from mmar_carl.metrics import MetricBase
        from mmar_carl.models.dataset import (
            DataCase, SimpleDataset, ThresholdStrategy,
        )
        from mmar_carl.models.results import (
            ReasoningResult, StepExecutionResult,
        )
        from mmar_carl.models.steps import ToolStepDescription
        from mmar_carl.models.config import ToolStepConfig

        # Step-level metric: returns 1.0 if step result is "good", else 0.0
        class GoodnessMetric(MetricBase):
            @property
            def name(self) -> str:
                return "goodness"

            async def compute_async(self, output) -> float:  # type: ignore[override]
                # Works on either StepExecutionResult or ReasoningResult.
                if isinstance(output, StepExecutionResult):
                    return 1.0 if "good" in str(output.result) else 0.0
                if isinstance(output, ReasoningResult):
                    return 1.0 if "good" in str(output.get_final_output()) else 0.0
                return 0.0

        def good_tool() -> str:
            return "good"

        def bad_tool() -> str:
            return "bad"

        # Two steps each with the GoodnessMetric attached
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1, title="A",
                    config=ToolStepConfig(tool_name="good_tool"),
                    metrics=[GoodnessMetric()],
                ),
                ToolStepDescription(
                    number=2, title="B",
                    config=ToolStepConfig(tool_name="bad_tool"),
                    dependencies=[1],
                    metrics=[GoodnessMetric()],
                ),
            ]
        )

        dataset = SimpleDataset([
            DataCase(input="case1", label="c1"),
            DataCase(input="case2", label="c2"),
        ])

        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=GoodnessMetric(),
            strategy=ThresholdStrategy(threshold=0.5),
        )

        def factory(case):
            ctx = ReasoningContext(outer_context=case.input, api=None)
            ctx.register_tool("good_tool", good_tool)
            ctx.register_tool("bad_tool", bad_tool)
            return ctx

        report = await evaluator.evaluate_async(factory)
        assert len(report.all_results) == 2
        for cr in report.all_results:
            # Step 1 should have goodness=1.0 (returns "good")
            assert cr.step_metrics.get(1, {}).get("goodness") == 1.0
            # Step 2 should have goodness=0.0 (returns "bad")
            assert cr.step_metrics.get(2, {}).get("goodness") == 0.0

        out = report.format_step_metric_heatmap("goodness")
        # Both cases should have step 1 at top, step 2 at bottom
        assert "█" in out  # high
        assert "·" in out  # low
