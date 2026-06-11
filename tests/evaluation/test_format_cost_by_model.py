"""Tests for ``ReasoningResult.format_cost_by_model``.

Groups tokens + cost by model — relevant when a chain mixes
multiple LLMs (cheap for plan, expensive for synth). Steps record
``model`` via ``StepExecutionResult.model``, populated by
``LLMStepExecutor`` from the resolved client's ``model_name``.
"""

from __future__ import annotations

import pytest

from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _step(
    number: int,
    *,
    title: str = "s",
    step_type: StepType = StepType.LLM,
    success: bool = True,
    model: str | None = None,
    usage: dict[str, int] | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=step_type,
        result="",
        success=success,
        model=model,
        token_usage=usage or {},
    )


def _result(*steps: StepExecutionResult) -> ReasoningResult:
    return ReasoningResult(success=True, history=[], step_results=list(steps))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_chain_returns_placeholder(self) -> None:
        out = _result().format_cost_by_model()
        assert "no model-attributed token usage" in out

    def test_non_llm_steps_only_returns_placeholder(self) -> None:
        out = _result(
            _step(1, step_type=StepType.TOOL),
            _step(2, step_type=StepType.MEMORY),
        ).format_cost_by_model()
        assert "no model-attributed" in out

    def test_unknown_format_raises(self) -> None:
        r = _result(_step(1, model="m", usage={"prompt": 10, "completion": 5, "total": 15}))
        with pytest.raises(ValueError, match="Unknown format"):
            r.format_cost_by_model(format="svg")


# ---------------------------------------------------------------------------
# Per-model aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_single_model_aggregated_across_steps(self) -> None:
        out = _result(
            _step(1, model="gpt-4o-mini", usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, model="gpt-4o-mini", usage={"prompt": 200, "completion": 100, "total": 300}),
        ).format_cost_by_model()
        assert "gpt-4o-mini" in out
        # Total = 150 + 300 = 450
        assert "450" in out
        # 100% share
        assert "100.0%" in out

    def test_two_models_each_get_a_row(self) -> None:
        out = _result(
            _step(1, model="cheap", usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, model="expensive", usage={"prompt": 200, "completion": 100, "total": 300}),
        ).format_cost_by_model()
        assert "cheap" in out
        assert "expensive" in out

    def test_rows_sorted_by_total_desc(self) -> None:
        out = _result(
            _step(1, model="small", usage={"prompt": 50, "completion": 25, "total": 75}),
            _step(2, model="big", usage={"prompt": 500, "completion": 250, "total": 750}),
            _step(3, model="medium", usage={"prompt": 200, "completion": 100, "total": 300}),
        ).format_cost_by_model()
        # big > medium > small
        big_idx = out.index("big")
        medium_idx = out.index("medium")
        small_idx = out.index("small")
        assert big_idx < medium_idx < small_idx

    def test_step_without_model_falls_back_to_default_model(self) -> None:
        out = _result(
            _step(1, model=None, usage={"prompt": 100, "completion": 50, "total": 150}),
        ).format_cost_by_model(default_model="gpt-4o-mini")
        assert "gpt-4o-mini" in out

    def test_step_with_neither_model_nor_default_bucketed_as_unknown(self) -> None:
        out = _result(
            _step(1, model=None, usage={"prompt": 100, "completion": 50, "total": 150}),
        ).format_cost_by_model()
        assert "(unknown)" in out


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_no_pricing_leaves_cost_column_blank(self) -> None:
        out = _result(
            _step(1, model="m", usage={"prompt": 100, "completion": 50, "total": 150})
        ).format_cost_by_model()
        # Data row contains the model but no $ sign
        lines = out.splitlines()
        data_row = next(line for line in lines if line.startswith("m"))
        assert "$" not in data_row

    def test_pricing_populates_cost(self) -> None:
        out = _result(
            _step(1, model="gpt-4o-mini",
                  usage={"prompt": 1000, "completion": 500, "total": 1500})
        ).format_cost_by_model(pricing={"gpt-4o-mini": (0.00015, 0.0006)})
        # 1000/1000 * 0.00015 + 500/1000 * 0.0006 = 0.00045
        assert "$0.0004" in out or "$0.0005" in out

    def test_missing_pricing_for_model_shows_dash_and_warning(self) -> None:
        out = _result(
            _step(1, model="m-unknown",
                  usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, model="gpt-4o-mini",
                  usage={"prompt": 100, "completion": 50, "total": 150}),
        ).format_cost_by_model(pricing={"gpt-4o-mini": (0.00015, 0.0006)})
        # m-unknown row has $-
        assert "$-" in out
        # Warning footer
        assert "missing pricing for: m-unknown" in out

    def test_total_cost_sums_across_models(self) -> None:
        out = _result(
            _step(1, model="m1", usage={"prompt": 1000, "completion": 0, "total": 1000}),
            _step(2, model="m2", usage={"prompt": 0, "completion": 1000, "total": 1000}),
        ).format_cost_by_model(pricing={"m1": (0.001, 0.001), "m2": (0.002, 0.002)})
        # m1: 1000/1000 * 0.001 = 0.001
        # m2: 1000/1000 * 0.002 = 0.002
        # total = 0.003
        assert "$0.0030" in out


# ---------------------------------------------------------------------------
# Mermaid format
# ---------------------------------------------------------------------------


class TestMermaidFormat:
    def test_mermaid_starts_with_pie_directive(self) -> None:
        out = _result(
            _step(1, model="m", usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_cost_by_model(format="mermaid")
        assert out.startswith("pie title Token spend by model")

    def test_mermaid_one_line_per_model(self) -> None:
        out = _result(
            _step(1, model="A", usage={"prompt": 100, "completion": 0, "total": 100}),
            _step(2, model="B", usage={"prompt": 200, "completion": 0, "total": 200}),
        ).format_cost_by_model(format="mermaid")
        lines = out.splitlines()
        assert len(lines) == 3  # header + 2 slices

    def test_mermaid_uses_token_total_as_value(self) -> None:
        out = _result(
            _step(1, model="m", usage={"prompt": 999, "completion": 1, "total": 1000})
        ).format_cost_by_model(format="mermaid")
        # `"m" : 1000`
        assert "1000" in out

    def test_mermaid_sanitises_quotes_in_model_name(self) -> None:
        out = _result(
            _step(1, model='has "quoted" word', usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_cost_by_model(format="mermaid")
        # Double quotes inside the label would break Mermaid; should be replaced
        assert "has 'quoted' word" in out


# ---------------------------------------------------------------------------
# Structure: TOT row, header
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_columns_present(self) -> None:
        out = _result(
            _step(1, model="m", usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_cost_by_model()
        assert "model" in out
        assert "tokens" in out
        assert "bar" in out
        assert "%" in out

    def test_total_row_present(self) -> None:
        out = _result(
            _step(1, model="m1", usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, model="m2", usage={"prompt": 50, "completion": 25, "total": 75}),
        ).format_cost_by_model()
        assert "TOT" in out
        # Total = 225 (150 + 75)
        assert "225" in out

    def test_bar_proportional_to_share(self) -> None:
        """When one model is 80% of tokens, its bar should be wider than the
        other (20%)."""
        out = _result(
            _step(1, model="big", usage={"prompt": 800, "completion": 0, "total": 800}),
            _step(2, model="small", usage={"prompt": 200, "completion": 0, "total": 200}),
        ).format_cost_by_model(bar_width=20)
        lines = out.splitlines()
        big_row = next(line for line in lines if line.startswith("big"))
        small_row = next(line for line in lines if line.startswith("small"))
        big_cells = big_row.count("█")
        small_cells = small_row.count("█")
        # Roughly 4:1
        assert big_cells > small_cells
        # Both have at least one cell
        assert big_cells >= 1
        assert small_cells >= 1


# ---------------------------------------------------------------------------
# StepExecutionResult.model schema
# ---------------------------------------------------------------------------


class TestStepResultSchema:
    def test_model_field_defaults_to_none(self) -> None:
        sr = StepExecutionResult(
            step_number=1, step_title="x", step_type=StepType.LLM,
            result="", success=True,
        )
        assert sr.model is None

    def test_model_field_round_trips(self) -> None:
        sr = StepExecutionResult(
            step_number=1, step_title="x", step_type=StepType.LLM,
            result="", success=True, model="gpt-4o",
        )
        assert sr.model == "gpt-4o"
        d = sr.model_dump()
        # Pydantic includes the field even at default
        rehydrated = StepExecutionResult.model_validate(d)
        assert rehydrated.model == "gpt-4o"
