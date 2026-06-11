"""Tests for ``DatasetEvaluationReport.format_failure_heatmap``.

Shows per-case × per-step outcomes so users can tell whether
failures cluster on a specific step (chain bug) or specific cases (data
issue).
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
    step_outcomes: dict[int, str] | None = None,
    success: bool = True,
    score: float = 0.5,
) -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input=label, label=label),
        score=score,
        chain_output="",
        success=success,
        step_outcomes=step_outcomes or {},
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
        out = _report().format_failure_heatmap()
        assert "no cases evaluated" in out

    def test_cases_with_no_step_outcomes_returns_placeholder(self) -> None:
        out = _report(_case("a"), _case("b")).format_failure_heatmap()
        assert "no per-step outcomes" in out


# ---------------------------------------------------------------------------
# Cell symbols
# ---------------------------------------------------------------------------


class TestCellSymbols:
    def test_success_renders_as_checkmark(self) -> None:
        out = _report(_case("a", step_outcomes={1: "success"})).format_failure_heatmap()
        assert "✓" in out

    def test_failure_renders_as_cross(self) -> None:
        out = _report(_case("a", step_outcomes={1: "failure"})).format_failure_heatmap()
        assert "✗" in out

    def test_skipped_renders_as_dot(self) -> None:
        out = _report(_case("a", step_outcomes={1: "skipped"})).format_failure_heatmap()
        assert "·" in out

    def test_step_not_in_case_renders_as_dash(self) -> None:
        """Some cases didn't reach all steps (chain failed early).
        The heatmap union-merges step numbers across all cases."""
        out = _report(
            _case("a", step_outcomes={1: "success", 2: "success"}),
            _case("b", step_outcomes={1: "failure"}),  # didn't reach step 2
        ).format_failure_heatmap()
        # Case b's row for step 2 should be "-"
        lines = out.splitlines()
        b_row = next(line for line in lines if line.startswith("b "))
        assert "-" in b_row


# ---------------------------------------------------------------------------
# Failure-per-step summary row
# ---------------------------------------------------------------------------


class TestFailureSummary:
    def test_failure_count_per_step_appears_in_summary(self) -> None:
        out = _report(
            _case("a", step_outcomes={1: "success", 2: "failure"}),
            _case("b", step_outcomes={1: "failure", 2: "failure"}),
            _case("c", step_outcomes={1: "success", 2: "success"}),
        ).format_failure_heatmap()
        lines = out.splitlines()
        fails_row = next(line for line in lines if line.lstrip().startswith("fails"))
        # Step 1: 1 failure ; Step 2: 2 failures
        # Both numbers should appear in the row.
        assert " 1 " in fails_row or fails_row.rstrip().endswith(" 1")
        # Search for "2" — should be in the row
        assert " 2" in fails_row

    def test_always_fail_step_triggers_warning(self) -> None:
        """A step that fails in EVERY case → chain-bug warning."""
        out = _report(
            _case("a", step_outcomes={1: "success", 2: "failure"}),
            _case("b", step_outcomes={1: "success", 2: "failure"}),
            _case("c", step_outcomes={1: "success", 2: "failure"}),
        ).format_failure_heatmap()
        assert "step 2 failed in every case" in out
        assert "chain bug" in out

    def test_no_warning_when_no_step_fails_universally(self) -> None:
        out = _report(
            _case("a", step_outcomes={1: "success", 2: "failure"}),
            _case("b", step_outcomes={1: "failure", 2: "success"}),
        ).format_failure_heatmap()
        assert "chain bug" not in out

    def test_multiple_always_fail_steps_pluralised(self) -> None:
        out = _report(
            _case("a", step_outcomes={1: "failure", 2: "failure"}),
            _case("b", step_outcomes={1: "failure", 2: "failure"}),
        ).format_failure_heatmap()
        assert "steps 1, 2 failed in every case" in out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_lists_step_numbers(self) -> None:
        out = _report(
            _case("a", step_outcomes={1: "success", 3: "success", 5: "success"})
        ).format_failure_heatmap()
        lines = out.splitlines()
        # Header row should list 1, 3, 5
        header = lines[0]
        assert " 1" in header
        assert " 3" in header
        assert " 5" in header

    def test_case_labels_in_first_column(self) -> None:
        out = _report(
            _case("alice", step_outcomes={1: "success"}),
            _case("bob", step_outcomes={1: "success"}),
        ).format_failure_heatmap()
        assert "alice" in out
        assert "bob" in out

    def test_long_case_label_truncated_with_ellipsis(self) -> None:
        out = _report(
            _case("X" * 100, step_outcomes={1: "success"})
        ).format_failure_heatmap(case_label_width=20)
        assert "X" * 100 not in out
        assert "…" in out

    def test_legend_present(self) -> None:
        out = _report(_case("a", step_outcomes={1: "success"})).format_failure_heatmap()
        assert "✓ success" in out
        assert "✗ failure" in out
        assert "· skipped" in out
        assert "- not run" in out

    def test_legend_shows_case_and_step_count(self) -> None:
        out = _report(
            _case("a", step_outcomes={1: "success", 2: "failure"}),
            _case("b", step_outcomes={1: "success", 2: "failure"}),
            _case("c", step_outcomes={1: "success", 2: "failure"}),
        ).format_failure_heatmap()
        assert "3 cases × 2 steps" in out


# ---------------------------------------------------------------------------
# Union of step numbers across cases
# ---------------------------------------------------------------------------


def test_step_columns_union_across_cases() -> None:
    """Different cases may have run different step subsets — union them."""
    out = _report(
        _case("a", step_outcomes={1: "success", 2: "success"}),
        _case("b", step_outcomes={1: "success", 3: "failure"}),  # only 1,3
        _case("c", step_outcomes={2: "skipped", 4: "success"}),
    ).format_failure_heatmap()
    # All four step numbers appear in the header
    header = out.splitlines()[0]
    for n in (1, 2, 3, 4):
        assert f" {n}" in header


# ---------------------------------------------------------------------------
# CaseEvaluationResult schema additions
# ---------------------------------------------------------------------------


class TestCaseEvaluationResultSchema:
    def test_step_outcomes_defaults_to_empty(self) -> None:
        result = CaseEvaluationResult(
            case=DataCase(input="x"),
            score=0.5,
            chain_output="ok",
            success=True,
        )
        assert result.step_outcomes == {}

    def test_step_outcomes_accepts_explicit_dict(self) -> None:
        result = CaseEvaluationResult(
            case=DataCase(input="x"),
            score=0.5,
            chain_output="ok",
            success=True,
            step_outcomes={1: "success", 2: "failure"},
        )
        assert result.step_outcomes == {1: "success", 2: "failure"}


# ---------------------------------------------------------------------------
# DatasetEvaluator populates step_outcomes end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dataset_evaluator_populates_step_outcomes() -> None:
    """End-to-end: a real DatasetEvaluator run should populate
    step_outcomes from result.step_results."""
    from mmar_carl import (
        DatasetEvaluator,
        ReasoningChain,
        ReasoningContext,
        SimpleDataset,
        ToolStepConfig,
        ToolStepDescription,
    )
    from mmar_carl.metrics import MetricBase

    class _Const(MetricBase):
        @property
        def name(self) -> str:
            return "c"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            return 1.0

    # Two-step chain: step 1 succeeds, step 2 fails (no tool)
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="ok", config=ToolStepConfig(tool_name="ok")
            ),
            ToolStepDescription(
                number=2, title="fail", dependencies=[1],
                config=ToolStepConfig(tool_name="missing"),
            ),
        ],
    )

    def ctx_factory(case: DataCase) -> ReasoningContext:
        c = ReasoningContext(outer_context=case.input, api=None, model="default")
        c.register_tool("ok", lambda: "ok")
        # Don't register "missing" → step 2 fails
        return c

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=SimpleDataset([DataCase(input="x"), DataCase(input="y")]),
        metric=_Const(),
        strategy=ThresholdStrategy(threshold=0.5),
    )
    report = await evaluator.evaluate_async(ctx_factory)
    # Both cases should record step_outcomes for steps 1 and 2.
    for r in report.all_results:
        assert r.step_outcomes.get(1) == "success"
        assert r.step_outcomes.get(2) == "failure"
    # And the heatmap should fire the chain-bug warning.
    heatmap = report.format_failure_heatmap()
    assert "step 2 failed in every case" in heatmap
