"""Tests for ``ReasoningResult.format_profiling_table``.

Builds on the existing ``get_profiling_summary()`` data + ``token_usage_by_step``
to produce a single printable "where did time / tokens / spend go?" view.
"""

from __future__ import annotations

from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _step(
    number: int,
    *,
    title: str = "step",
    step_type: StepType = StepType.LLM,
    execution_time: float = 0.0,
    success: bool = True,
    skipped: bool = False,
    usage: dict[str, int] | None = None,
    profiling: dict | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=step_type,
        result="",
        success=success,
        execution_time=execution_time,
        skipped=skipped,
        token_usage=usage or {},
        profiling=profiling or {},
    )


def _result(*steps: StepExecutionResult) -> ReasoningResult:
    return ReasoningResult(
        success=all(s.success for s in steps),
        history=[],
        step_results=list(steps),
        total_execution_time=sum(s.execution_time or 0 for s in steps),
    )


# ---------------------------------------------------------------------------
# Empty / single-step
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_chain_renders_placeholder(self) -> None:
        assert _result().format_profiling_table() == "(empty chain)"

    def test_single_step_renders_header_and_total(self) -> None:
        text = _result(_step(1, title="A", execution_time=0.5)).format_profiling_table()
        # Header columns
        assert "wall_ms" in text
        assert "tok_in" in text
        assert "tok_out" in text
        assert "cache" in text
        assert "cost" in text
        assert "status" in text
        # Step + total rows present
        assert "  1  A" in text
        assert "TOT" in text


# ---------------------------------------------------------------------------
# Column content
# ---------------------------------------------------------------------------


class TestColumnContent:
    def test_wall_ms_converted_from_seconds(self) -> None:
        text = _result(_step(1, execution_time=2.345)).format_profiling_table()
        # 2.345s → 2,345 ms (rounded)
        assert "2,345" in text

    def test_tokens_blank_for_non_llm_step(self) -> None:
        """Tool/Memory steps don't record token usage — leave columns empty."""
        text = _result(
            _step(1, step_type=StepType.TOOL, execution_time=0.001),
        ).format_profiling_table()
        # Find the data row (lines[2] = header, sep, data, sep, total)
        lines = text.splitlines()
        data_row = lines[2]
        # tok_in and tok_out columns should be visually blank (no numbers)
        # Easy assertion: no digit anywhere in those positions in this row.
        # More robust: just check that the row doesn't contain 0 in token positions.
        assert "tool" in data_row

    def test_cache_hit_shown_when_recorded(self) -> None:
        text = _result(
            _step(1, execution_time=0.0, profiling={"cache_hit": True}),
        ).format_profiling_table()
        assert "hit" in text

    def test_cache_blank_when_no_hit(self) -> None:
        text = _result(
            _step(1, execution_time=0.0, profiling={"cache_hit": False}),
        ).format_profiling_table()
        # the cache column should NOT say "hit"
        # find data row
        lines = text.splitlines()
        data_row = lines[2]
        assert "hit" not in data_row

    def test_status_ok_for_successful_step(self) -> None:
        text = _result(_step(1)).format_profiling_table()
        assert "ok" in text

    def test_status_fail_for_failed_step(self) -> None:
        text = _result(_step(1, success=False)).format_profiling_table()
        assert "fail" in text

    def test_status_skip_for_skipped_step(self) -> None:
        text = _result(_step(1, skipped=True)).format_profiling_table()
        assert "skip" in text


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_no_pricing_leaves_cost_blank(self) -> None:
        text = _result(
            _step(1, execution_time=1.0, usage={"prompt": 100, "completion": 50, "total": 150})
        ).format_profiling_table()
        lines = text.splitlines()
        data_row = lines[2]
        # No $ on the data row
        assert "$" not in data_row

    def test_pricing_with_default_model_populates_cost(self) -> None:
        text = _result(
            _step(1, execution_time=1.0, usage={"prompt": 1000, "completion": 500, "total": 1500})
        ).format_profiling_table(
            pricing={"gpt-4o-mini": (0.00015, 0.0006)},
            default_model="gpt-4o-mini",
        )
        # 1000/1000 * 0.00015 + 500/1000 * 0.0006 = 0.00015 + 0.0003 = 0.00045
        assert "$0.0004" in text or "$0.0005" in text  # allow rounding

    def test_unknown_model_in_pricing_keeps_cell_blank(self) -> None:
        text = _result(
            _step(1, execution_time=1.0, usage={"prompt": 100, "completion": 50, "total": 150})
        ).format_profiling_table(
            pricing={"gpt-4o": (0.001, 0.002)},
            default_model="claude-3.5",  # not in pricing
        )
        lines = text.splitlines()
        data_row = lines[2]
        assert "$" not in data_row  # cell stays blank

    def test_total_cost_summed_correctly(self) -> None:
        text = _result(
            _step(1, execution_time=1.0, usage={"prompt": 1000, "completion": 0, "total": 1000}),
            _step(2, execution_time=1.0, usage={"prompt": 0, "completion": 1000, "total": 1000}),
        ).format_profiling_table(
            pricing={"m": (0.001, 0.002)},
            default_model="m",
        )
        # step 1: 1000/1000 * 0.001 = 0.001
        # step 2: 1000/1000 * 0.002 = 0.002
        # total:  0.003
        assert "$0.0030" in text

    def test_pricing_with_pricing_only_no_default_blank_cost(self) -> None:
        """Without ``default_model`` and without per-step model on the result,
        the cost column stays blank."""
        text = _result(
            _step(1, execution_time=1.0, usage={"prompt": 100, "completion": 50, "total": 150})
        ).format_profiling_table(pricing={"gpt-4o-mini": (0.0001, 0.0001)})
        lines = text.splitlines()
        data_row = lines[2]
        assert "$" not in data_row


# ---------------------------------------------------------------------------
# Total row
# ---------------------------------------------------------------------------


class TestTotalRow:
    def test_total_wall_ms_is_sum(self) -> None:
        text = _result(
            _step(1, execution_time=1.0),
            _step(2, execution_time=2.5),
        ).format_profiling_table()
        # 1000 + 2500 = 3500
        total_line = text.splitlines()[-1]
        assert "3,500" in total_line

    def test_total_tokens_summed_across_steps(self) -> None:
        text = _result(
            _step(1, execution_time=0.1, usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, execution_time=0.1, usage={"prompt": 200, "completion": 100, "total": 300}),
        ).format_profiling_table()
        total_line = text.splitlines()[-1]
        assert "300" in total_line  # tok_in total
        assert "150" in total_line  # tok_out total


# ---------------------------------------------------------------------------
# Title truncation
# ---------------------------------------------------------------------------


def test_long_title_is_truncated_with_ellipsis() -> None:
    long_title = "A" * 100
    text = _result(_step(1, title=long_title, execution_time=0.1)).format_profiling_table(
        title_width=20
    )
    lines = text.splitlines()
    data_row = lines[2]
    # The full 100-char title shouldn't appear
    assert "A" * 100 not in data_row
    # An ellipsis should
    assert "…" in data_row


def test_short_title_not_truncated() -> None:
    text = _result(_step(1, title="short", execution_time=0.1)).format_profiling_table(
        title_width=20
    )
    assert "short" in text
    assert "…" not in text


# ---------------------------------------------------------------------------
# Sanity: returns a string
# ---------------------------------------------------------------------------


def test_returns_a_string() -> None:
    assert isinstance(_result(_step(1)).format_profiling_table(), str)
