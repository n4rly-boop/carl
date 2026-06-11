"""Tests for the Jupyter rich-display protocol.

Verifies that ``ReasoningResult``, ``DatasetEvaluationReport``,
``EvolutionResult``, ``CostEstimate``, and ``ChainVisualizer`` all
expose a ``_repr_markdown_`` method that returns valid markdown for
inline notebook rendering.
"""

from __future__ import annotations

import pytest

from mmar_carl import ChainVisualizer
from mmar_carl.chain_evolution import EvolutionResult, GenerationStats
from mmar_carl.cost import CostEstimate, StepCostEstimate
from mmar_carl.models.dataset import (
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    ThresholdStrategy,
)
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


# ---------------------------------------------------------------------------
# ReasoningResult
# ---------------------------------------------------------------------------


def _result(success: bool = True, with_tokens: bool = True) -> ReasoningResult:
    usage = {"prompt": 100, "completion": 50, "total": 150} if with_tokens else {}
    return ReasoningResult(
        success=success, history=["Step 1: hi"],
        step_results=[StepExecutionResult(
            step_number=1, step_title="Test", step_type=StepType.LLM,
            result="ok", success=success,
            error_message=None if success else "boom",
            execution_time=0.5, token_usage=usage,
        )],
        total_execution_time=0.5,
        token_usage=usage,
    )


class TestReasoningResultMarkdown:
    def test_success_banner(self) -> None:
        md = _result()._repr_markdown_()
        assert "✅ success" in md
        assert "ReasoningResult" in md

    def test_failed_banner_includes_error(self) -> None:
        md = _result(success=False)._repr_markdown_()
        assert "❌ failed" in md
        assert "boom" in md

    def test_token_count_displayed(self) -> None:
        md = _result()._repr_markdown_()
        assert "150 tokens" in md

    def test_profiling_table_in_fenced_code_block(self) -> None:
        md = _result()._repr_markdown_()
        # Markdown contains a ```text fence + the step title from the table
        assert "```text" in md
        assert "Test" in md

    def test_mermaid_block_present_when_usage_recorded(self) -> None:
        md = _result()._repr_markdown_()
        assert "```mermaid" in md
        assert "pie title" in md

    def test_no_mermaid_block_when_no_usage(self) -> None:
        md = _result(with_tokens=False)._repr_markdown_()
        assert "```mermaid" not in md


# ---------------------------------------------------------------------------
# DatasetEvaluationReport
# ---------------------------------------------------------------------------


def _make_report(n_cases: int = 3, with_outcomes: bool = True) -> DatasetEvaluationReport:
    cases = [
        CaseEvaluationResult(
            case=DataCase(input=f"c{i}", label=f"c{i}"),
            score=0.5 + i * 0.1, chain_output="", success=True,
            step_outcomes={1: "success"} if with_outcomes else {},
        )
        for i in range(n_cases)
    ]
    scores = [c.score for c in cases] if cases else [0.0]
    return DatasetEvaluationReport(
        metric_name="acc", strategy=ThresholdStrategy(threshold=0.0),
        all_results=cases, selected_cases=[],
        mean_score=sum(scores) / len(scores), min_score=min(scores), max_score=max(scores),
    )


class TestDatasetEvaluationReportMarkdown:
    def test_headline_includes_metric_name_and_counts(self) -> None:
        md = _make_report()._repr_markdown_()
        assert "metric: `acc`" in md
        assert "3 cases" in md

    def test_mean_min_max_displayed(self) -> None:
        md = _make_report()._repr_markdown_()
        assert "mean" in md.lower()
        assert "min" in md
        assert "max" in md

    def test_failure_heatmap_block_when_outcomes_recorded(self) -> None:
        md = _make_report(with_outcomes=True)._repr_markdown_()
        assert "✓" in md  # the heatmap success glyph

    def test_no_heatmap_block_when_no_outcomes(self) -> None:
        md = _make_report(with_outcomes=False)._repr_markdown_()
        assert "✓" not in md
        assert "✗" not in md

    def test_score_distribution_block_when_multiple_cases(self) -> None:
        md = _make_report(n_cases=4)._repr_markdown_()
        # Box plot legend mentions Q1/Q3
        assert "Q1=" in md or "score distribution" in md

    def test_empty_report_yields_only_headline(self) -> None:
        empty = DatasetEvaluationReport(
            metric_name="m", strategy=ThresholdStrategy(threshold=0.0),
            all_results=[], selected_cases=[],
            mean_score=0.0, min_score=0.0, max_score=0.0,
        )
        md = empty._repr_markdown_()
        # No fenced blocks because there's no data
        assert "```text" not in md
        assert "DatasetEvaluationReport" in md


# ---------------------------------------------------------------------------
# EvolutionResult
# ---------------------------------------------------------------------------


class TestEvolutionResultMarkdown:
    def test_headline_includes_best_score_and_generation(self) -> None:
        ev = EvolutionResult(
            best_chain_spec={}, best_score=0.95, best_generation=4,
            history=[GenerationStats(generation=0, best_score=0.5, mean_score=0.4)],
        )
        md = ev._repr_markdown_()
        assert "0.950" in md
        assert "generation 4" in md

    def test_score_evolution_block_when_history_present(self) -> None:
        ev = EvolutionResult(
            best_chain_spec={}, best_score=0.85, best_generation=2,
            history=[
                GenerationStats(generation=i, best_score=0.5 + i * 0.1,
                                 mean_score=0.4 + i * 0.1)
                for i in range(3)
            ],
        )
        md = ev._repr_markdown_()
        # The chart block is present
        assert "```text" in md


# ---------------------------------------------------------------------------
# CostEstimate
# ---------------------------------------------------------------------------


class TestCostEstimateMarkdown:
    def test_headline_includes_total_cost(self) -> None:
        ce = CostEstimate(
            steps=[StepCostEstimate(
                step_number=1, step_title="X", step_type=StepType.LLM,
                calls_llm=True, estimated_calls=1,
                input_tokens=100, output_tokens=50,
                input_cost_usd=0.001, output_cost_usd=0.002,
                total_cost_usd=0.003,
            )],
            total_input_tokens=100, total_output_tokens=50,
            total_tokens=150, total_cost_usd=0.003,
        )
        md = ce._repr_markdown_()
        assert "$0.0030" in md
        assert "CostEstimate" in md

    def test_table_in_fenced_code_block(self) -> None:
        ce = CostEstimate(
            steps=[StepCostEstimate(
                step_number=1, step_title="X", step_type=StepType.LLM,
                calls_llm=True, estimated_calls=1,
                input_tokens=100, output_tokens=50,
                total_cost_usd=0.003,
            )],
            total_tokens=150, total_cost_usd=0.003,
        )
        md = ce._repr_markdown_()
        assert "```text" in md
        # Step title appears inside the table
        assert "X" in md

    def test_empty_estimate_renders_placeholder(self) -> None:
        ce = CostEstimate(steps=[], total_tokens=0)
        md = ce._repr_markdown_()
        assert "(empty chain)" in md


# ---------------------------------------------------------------------------
# ChainVisualizer
# ---------------------------------------------------------------------------


class TestChainVisualizerMarkdown:
    def test_empty_visualizer_yields_placeholder(self) -> None:
        viz = ChainVisualizer(_result())
        md = viz._repr_markdown_()
        assert "no views accumulated" in md

    def test_views_become_h3_sections_with_fences(self) -> None:
        viz = ChainVisualizer(_result()).token_pie().prompt_completion()
        md = viz._repr_markdown_()
        # Each view becomes an H3 section
        assert "### Token pie (text)" in md
        assert "### Prompt vs completion" in md
        # Fenced blocks present
        assert "```text" in md

    def test_mermaid_view_renders_with_mermaid_fence(self) -> None:
        viz = ChainVisualizer(_result()).token_pie(format="mermaid")
        md = viz._repr_markdown_()
        # Pie chart should be in a ```mermaid fence, not ```text
        assert "```mermaid" in md
        # The directive should appear inside the fence
        assert "pie title" in md

    def test_text_views_use_text_fence_not_mermaid(self) -> None:
        viz = ChainVisualizer(_result()).profiling_table()
        md = viz._repr_markdown_()
        assert "```text" in md
        # The profiling table doesn't start with a Mermaid directive
        assert "```mermaid\n  #" not in md


# ---------------------------------------------------------------------------
# Method-existence smoke test
# ---------------------------------------------------------------------------


class TestMethodExists:
    @pytest.mark.parametrize("obj_factory", [
        lambda: _result(),
        lambda: _make_report(),
        lambda: EvolutionResult(
            best_chain_spec={}, best_score=0.5, best_generation=0, history=[],
        ),
        lambda: CostEstimate(steps=[], total_tokens=0),
        lambda: ChainVisualizer(_result()),
    ])
    def test_repr_markdown_returns_string(self, obj_factory) -> None:
        out = obj_factory()._repr_markdown_()
        assert isinstance(out, str)
        assert len(out) > 0
