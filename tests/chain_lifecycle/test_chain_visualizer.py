"""Tests for ``ChainVisualizer`` — chainable facade over all CARL viz methods.

Consolidates token pie / Gantt / profiling table / DAG variants /
score evolution / spend-vs-quality into one fluent API:
``ChainVisualizer(result=...).token_pie().gantt().profiling_table().print()``.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest

from mmar_carl import (
    ChainVisualizer,
    LLMStepDescription,
    ReasoningChain,
)
from mmar_carl.chain_evolution import (
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
)
from mmar_carl.execution_trace import ExecutionTrace, TraceEvent
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(*, with_trace: bool = False) -> ReasoningResult:
    trace = None
    if with_trace:
        trace = ExecutionTrace(
            chain_title="t",
            events=[
                TraceEvent(
                    step_number=1, step_title="A", step_type="llm",
                    success=True, execution_time=1.0, batch_index=0,
                ),
                TraceEvent(
                    step_number=2, step_title="B", step_type="llm",
                    success=True, execution_time=2.0, batch_index=1,
                ),
            ],
        )
    return ReasoningResult(
        success=True,
        history=[],
        total_execution_time=3.0,
        step_results=[
            StepExecutionResult(
                step_number=1, step_title="A", step_type=StepType.LLM,
                result="ok", success=True, execution_time=1.0,
                token_usage={"prompt": 100, "completion": 50, "total": 150},
            ),
            StepExecutionResult(
                step_number=2, step_title="B", step_type=StepType.LLM,
                result="ok", success=True, execution_time=2.0,
                token_usage={"prompt": 200, "completion": 100, "total": 300},
            ),
        ],
        trace=trace,
    )


def _chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="A", aim="x"),
            LLMStepDescription(number=2, title="B", aim="x", dependencies=[1]),
        ],
    )


def _evolution_result() -> EvolutionResult:
    return EvolutionResult(
        best_chain_spec={},
        best_score=0.9,
        best_generation=1,
        history=[
            GenerationStats(
                generation=0, best_score=0.5, mean_score=0.5,
                population_scores=[0.5],
                population_metrics=[IndividualMetrics(score=0.5, total_tokens=200)],
            ),
            GenerationStats(
                generation=1, best_score=0.9, mean_score=0.9,
                population_scores=[0.9],
                population_metrics=[IndividualMetrics(score=0.9, total_tokens=300)],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Constructor + buffer semantics
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_empty_render_returns_placeholder(self) -> None:
        viz = ChainVisualizer(result=_result())
        out = viz.render()
        assert "no views accumulated" in out

    def test_view_titles_starts_empty(self) -> None:
        viz = ChainVisualizer(result=_result())
        assert viz.view_titles == []


class TestBufferOps:
    def test_clear_resets_buffer(self) -> None:
        viz = ChainVisualizer(result=_result())
        viz.token_pie()
        assert len(viz.view_titles) == 1
        viz.clear()
        assert viz.view_titles == []
        assert "no views accumulated" in viz.render()

    def test_clear_returns_self_for_chaining(self) -> None:
        viz = ChainVisualizer(result=_result())
        assert viz.clear() is viz

    def test_buffer_preserves_insertion_order(self) -> None:
        viz = ChainVisualizer(result=_result())
        viz.profiling_table().token_pie().prompt_completion()
        assert viz.view_titles == [
            "Profiling table",
            "Token pie (text)",
            "Prompt vs completion",
        ]


# ---------------------------------------------------------------------------
# Builder methods that need result only
# ---------------------------------------------------------------------------


class TestResultOnlyBuilders:
    def test_token_pie_adds_view(self) -> None:
        viz = ChainVisualizer(result=_result()).token_pie()
        assert "Token pie (text)" in viz.view_titles
        # Rendered text contains the data
        assert "100.0%" in viz.render()  # 100% total

    def test_token_pie_format_arg_propagates(self) -> None:
        viz = ChainVisualizer(result=_result()).token_pie(format="mermaid")
        assert "Token pie (mermaid)" in viz.view_titles
        assert "pie title" in viz.render()

    def test_prompt_completion_adds_view(self) -> None:
        viz = ChainVisualizer(result=_result()).prompt_completion()
        assert "Prompt vs completion" in viz.view_titles
        assert "▒ prompt" in viz.render()

    def test_profiling_table_adds_view(self) -> None:
        viz = ChainVisualizer(result=_result()).profiling_table()
        assert "Profiling table" in viz.view_titles

    def test_profiling_table_pricing_propagates(self) -> None:
        viz = ChainVisualizer(result=_result()).profiling_table(
            pricing={"m": (0.001, 0.001)}, default_model="m"
        )
        # Costs should appear in the table
        assert "$" in viz.render()


# ---------------------------------------------------------------------------
# Gantt (needs trace on result)
# ---------------------------------------------------------------------------


class TestGantt:
    def test_gantt_with_trace_renders(self) -> None:
        viz = ChainVisualizer(result=_result(with_trace=True)).gantt()
        assert "Gantt (text)" in viz.view_titles
        assert "total wall time" in viz.render()

    def test_gantt_without_trace_renders_placeholder(self) -> None:
        viz = ChainVisualizer(result=_result(with_trace=False)).gantt()
        assert "Gantt" in viz.view_titles[0]
        assert "no trace attached" in viz.render()

    def test_gantt_mermaid_format(self) -> None:
        viz = ChainVisualizer(result=_result(with_trace=True)).gantt(format="mermaid")
        assert "Gantt (mermaid)" in viz.view_titles
        assert "gantt" in viz.render()


# ---------------------------------------------------------------------------
# Chain-dependent builders
# ---------------------------------------------------------------------------


class TestChainBuilders:
    def test_dag_with_chain_renders(self) -> None:
        viz = ChainVisualizer(result=_result(), chain=_chain()).dag()
        assert "DAG (mermaid)" in viz.view_titles
        assert "flowchart TD" in viz.render()

    def test_critical_path_with_chain_and_result_renders(self) -> None:
        viz = ChainVisualizer(result=_result(), chain=_chain()).critical_path()
        assert "Critical path (mermaid)" in viz.view_titles
        assert "linkStyle" in viz.render() or "%% critical path" in viz.render()

    def test_heatmap_with_metric_propagates(self) -> None:
        viz = ChainVisualizer(result=_result(), chain=_chain()).heatmap(metric="tokens")
        assert "Heatmap: tokens (mermaid)" in viz.view_titles
        assert "flowchart TD" in viz.render()


# ---------------------------------------------------------------------------
# Evolution-result-dependent builders
# ---------------------------------------------------------------------------


class TestEvolutionBuilders:
    def test_score_evolution_renders(self) -> None:
        viz = ChainVisualizer(
            result=_result(),
            evolution_result=_evolution_result(),
        ).score_evolution()
        assert "Score evolution (text)" in viz.view_titles
        assert "generation" in viz.render()

    def test_spend_vs_quality_renders(self) -> None:
        viz = ChainVisualizer(
            result=_result(),
            evolution_result=_evolution_result(),
        ).spend_vs_quality()
        assert "Spend vs quality (text)" in viz.view_titles
        assert "cumulative tokens" in viz.render()


# ---------------------------------------------------------------------------
# Validation: missing constructor args raise clear errors
# ---------------------------------------------------------------------------


class TestRequiredArgs:
    def test_dag_without_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="chain"):
            ChainVisualizer(result=_result()).dag()

    def test_critical_path_without_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="chain"):
            ChainVisualizer(result=_result()).critical_path()

    def test_heatmap_without_chain_raises(self) -> None:
        with pytest.raises(ValueError, match="chain"):
            ChainVisualizer(result=_result()).heatmap()

    def test_score_evolution_without_evo_raises(self) -> None:
        with pytest.raises(ValueError, match="evolution_result"):
            ChainVisualizer(result=_result()).score_evolution()

    def test_spend_vs_quality_without_evo_raises(self) -> None:
        with pytest.raises(ValueError, match="evolution_result"):
            ChainVisualizer(result=_result()).spend_vs_quality()

    def test_token_pie_without_result_raises(self) -> None:
        with pytest.raises(ValueError, match="result"):
            ChainVisualizer().token_pie()

    def test_error_message_shows_which_args_present(self) -> None:
        """Error should help users diagnose by listing what they did supply."""
        try:
            ChainVisualizer(result=_result()).dag()
        except ValueError as e:
            msg = str(e)
            assert "result=True" in msg
            assert "chain=False" in msg


# ---------------------------------------------------------------------------
# render() formatting
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_includes_section_headers(self) -> None:
        viz = ChainVisualizer(result=_result()).token_pie().profiling_table()
        out = viz.render()
        assert "=== Token pie (text) ===" in out
        assert "=== Profiling table ===" in out

    def test_render_separator_can_be_customised(self) -> None:
        viz = ChainVisualizer(result=_result()).token_pie().profiling_table()
        out = viz.render(separator="\n---\n")
        assert "\n---\n" in out

    def test_print_routes_to_stdout(self) -> None:
        viz = ChainVisualizer(result=_result()).token_pie()
        with patch("sys.stdout", new_callable=StringIO) as fake_out:
            ret = viz.print()
            captured = fake_out.getvalue()
        assert "=== Token pie (text) ===" in captured
        assert ret is viz  # returns self for chaining


# ---------------------------------------------------------------------------
# Chaining sanity: a long fluent call works end-to-end
# ---------------------------------------------------------------------------


def test_fluent_chain_with_all_view_types() -> None:
    viz = (
        ChainVisualizer(
            result=_result(with_trace=True),
            chain=_chain(),
            evolution_result=_evolution_result(),
        )
        .token_pie()
        .prompt_completion()
        .profiling_table()
        .gantt()
        .dag()
        .critical_path()
        .heatmap(metric="latency")
        .score_evolution()
        .spend_vs_quality()
    )
    assert len(viz.view_titles) == 9
    # Each section header appears in the bundled output
    bundle = viz.render()
    for title in viz.view_titles:
        assert f"=== {title} ===" in bundle
