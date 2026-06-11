"""Tests for ``ReasoningChain.to_mermaid_heatmap(result, metric=...)``.

Renders the chain DAG with per-node colouring
proportional to a chosen metric (latency / tokens / cost). Complements
``to_mermaid_critical_path`` — that highlights the longest *path*; this
highlights the hottest *nodes* across all metrics.
"""

from __future__ import annotations

import pytest

from mmar_carl import LLMStepDescription, ReasoningChain, ToolStepDescription
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _make_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="Cheap", aim="x"),
            LLMStepDescription(number=2, title="Expensive", aim="x", dependencies=[1]),
            ToolStepDescription(
                number=3,
                title="Save",
                config=ToolStepConfig(tool_name="save"),
                dependencies=[2],
            ),
        ],
    )


def _step(
    number: int,
    title: str,
    *,
    step_type: StepType = StepType.LLM,
    execution_time: float = 0.0,
    usage: dict[str, int] | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=step_type,
        result="",
        success=True,
        execution_time=execution_time,
        token_usage=usage or {},
    )


def _result_with_step_2_dominant() -> ReasoningResult:
    return ReasoningResult(
        success=True,
        history=[],
        step_results=[
            _step(1, "Cheap", execution_time=1.0,
                  usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, "Expensive", execution_time=10.0,
                  usage={"prompt": 500, "completion": 2000, "total": 2500}),
            _step(3, "Save", step_type=StepType.TOOL, execution_time=0.05),
        ],
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_metric_raises(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        with pytest.raises(ValueError, match="Unknown metric"):
            chain.to_mermaid_heatmap(result, metric="vibes")

    def test_default_metric_is_latency(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        # Calling without metric should not raise
        out = chain.to_mermaid_heatmap(result)
        assert "wall time per step" in out


# ---------------------------------------------------------------------------
# Latency metric
# ---------------------------------------------------------------------------


class TestLatencyMetric:
    def test_each_step_shows_its_time(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="latency")
        assert "1.00s" in out
        assert "10.00s" in out
        assert "0.05s" in out

    def test_max_step_colored_red(self) -> None:
        """The step with the highest latency must use the hot end of
        the gradient (#EF4444 = full red)."""
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="latency")
        s2_style = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("style S2")
        )
        assert "#EF4444" in s2_style

    def test_min_step_uses_green_zone(self) -> None:
        """Step with minimum value should be close to pure green."""
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="latency")
        s3_style = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("style S3")
        )
        # 0.05s vs max 10.0s → 0.5% → very close to #22C55E (full green).
        # Allow slight interpolation hex by checking the first digit's range.
        assert s3_style.upper().startswith("    STYLE S3 FILL:#2") or "#22C55E" in s3_style

    def test_annotation_includes_max_value(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="latency")
        # Annotation comment with max
        assert "Max: 10.00s" in out


# ---------------------------------------------------------------------------
# Tokens metric
# ---------------------------------------------------------------------------


class TestTokensMetric:
    def test_each_step_shows_its_tokens(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="tokens")
        assert "150 tok" in out
        assert "2,500 tok" in out  # thousands separator

    def test_non_llm_step_labeled_no_llm(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="tokens")
        # Step 3 is a tool step → "no llm" label
        assert "no llm" in out

    def test_non_llm_step_uses_lowest_colour(self) -> None:
        """A tool step (no LLM tokens) renders at value=0 → green."""
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="tokens")
        s3_style = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("style S3")
        )
        assert "#22C55E" in s3_style


# ---------------------------------------------------------------------------
# Cost metric
# ---------------------------------------------------------------------------


class TestCostMetric:
    def test_cost_without_pricing_renders_no_dollar_label(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(result, metric="cost")
        # No pricing supplied → labels show "no $"
        assert "no $" in out

    def test_cost_with_pricing_and_default_model_populates_dollars(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(
            result,
            metric="cost",
            pricing={"gpt-4o-mini": (0.00015, 0.0006)},
            default_model="gpt-4o-mini",
        )
        # Step 2: (500/1000)*0.00015 + (2000/1000)*0.0006 = 0.000075 + 0.0012 = 0.001275
        assert "$0.0013" in out or "$0.0012" in out  # allow rounding

    def test_cost_max_step_is_red(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        out = chain.to_mermaid_heatmap(
            result,
            metric="cost",
            pricing={"gpt-4o-mini": (0.00015, 0.0006)},
            default_model="gpt-4o-mini",
        )
        s2_style = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("style S2")
        )
        assert "#EF4444" in s2_style

    def test_cost_with_unpriced_model_falls_back_to_zero(self) -> None:
        chain = _make_chain()
        result = _result_with_step_2_dominant()
        # Pricing for a different model only
        out = chain.to_mermaid_heatmap(
            result,
            metric="cost",
            pricing={"some-other-model": (0.001, 0.001)},
            default_model="gpt-4o-mini",
        )
        assert "(no price)" in out


# ---------------------------------------------------------------------------
# Mermaid structure
# ---------------------------------------------------------------------------


class TestMermaidStructure:
    def test_starts_with_flowchart_directive(self) -> None:
        chain = _make_chain()
        out = chain.to_mermaid_heatmap(_result_with_step_2_dominant())
        assert out.startswith("flowchart TD")

    def test_emits_one_style_directive_per_step(self) -> None:
        chain = _make_chain()
        out = chain.to_mermaid_heatmap(_result_with_step_2_dominant())
        style_lines = [
            line for line in out.splitlines() if line.lstrip().startswith("style S")
        ]
        assert len(style_lines) == 3

    def test_edges_present(self) -> None:
        chain = _make_chain()
        out = chain.to_mermaid_heatmap(_result_with_step_2_dominant())
        assert "S1 --> S2" in out
        assert "S2 --> S3" in out

    def test_annotation_comment_present(self) -> None:
        chain = _make_chain()
        out = chain.to_mermaid_heatmap(_result_with_step_2_dominant(), metric="tokens")
        assert "    %% heatmap:" in out
        assert "green=low, red=high" in out

    def test_label_separator_uses_br_tag_not_literal_backslash_n(self) -> None:
        """Mermaid renders ``<br/>`` as a line break inside node labels
        but treats a bare ``\\n`` as a literal two-character escape, so
        the per-step label (``"1: Title\\nVALUE"``) was rendering as
        ``1: Title\\nVALUE`` in viewers. Regression guard for that bug.
        """
        chain = _make_chain()
        out = chain.to_mermaid_heatmap(_result_with_step_2_dominant(), metric="tokens")
        # The legitimate <br/> separator must appear inside each step label.
        assert "<br/>" in out
        # The buggy literal escape must NOT appear anywhere in node labels.
        # We restrict to label-bearing lines (`S<N>["..."]`) so a future
        # comment mentioning ``\n`` wouldn't trigger this guard.
        label_lines = [line for line in out.splitlines() if '"' in line and "S" in line]
        for line in label_lines:
            assert "\\n" not in line, (
                f"Found literal '\\n' in mermaid label line: {line!r}"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_zero_values_uses_neutral_grey(self) -> None:
        """When every step has value=0, the gradient denominator is 0 →
        use neutral grey rather than divide-by-zero."""
        chain = _make_chain()
        # Empty step_results (no values recorded) — values dict is empty.
        result = ReasoningResult(success=True, history=[], step_results=[])
        out = chain.to_mermaid_heatmap(result, metric="latency")
        # No crash; neutral grey for every node
        assert "#9CA3AF" in out

    def test_single_step_chain(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="Only", aim="x")]
        )
        result = ReasoningResult(
            success=True,
            history=[],
            step_results=[
                _step(1, "Only", execution_time=2.0,
                      usage={"prompt": 100, "completion": 50, "total": 150}),
            ],
        )
        out = chain.to_mermaid_heatmap(result, metric="latency")
        assert "flowchart TD" in out
        assert "S1" in out
        # Single step → max == itself → red
        s1_style = next(
            line for line in out.splitlines() if line.lstrip().startswith("style S1")
        )
        assert "#EF4444" in s1_style

    def test_implicit_linear_chain_no_deps_declared(self) -> None:
        """Mirrors to_mermaid + critical-path: synthesise linear edges
        when no deps are declared."""
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="x"),
                LLMStepDescription(number=2, title="B", aim="x"),
                LLMStepDescription(number=3, title="C", aim="x"),
            ]
        )
        result = ReasoningResult(
            success=True, history=[],
            step_results=[
                _step(1, "A", execution_time=1.0),
                _step(2, "B", execution_time=1.0),
                _step(3, "C", execution_time=1.0),
            ],
        )
        out = chain.to_mermaid_heatmap(result, metric="latency")
        assert "S1 --> S2" in out
        assert "S2 --> S3" in out
