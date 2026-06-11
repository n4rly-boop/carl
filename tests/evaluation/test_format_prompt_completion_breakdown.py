"""Tests for ``ReasoningResult.format_prompt_completion_breakdown`` —
per-step stacked bar showing prompt-tokens vs completion-tokens.

Answers the routine prompt-engineering question
"is my prompt bloated or my output bloated?" by visualising the
prompt/completion ratio per step.
"""

from __future__ import annotations

from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _step(
    number: int,
    *,
    title: str = "s",
    step_type: StepType = StepType.LLM,
    success: bool = True,
    usage: dict[str, int] | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=step_type,
        result="",
        success=success,
        token_usage=usage or {},
    )


def _result(*steps: StepExecutionResult) -> ReasoningResult:
    return ReasoningResult(success=True, history=[], step_results=list(steps))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_chain_returns_placeholder(self) -> None:
        out = _result().format_prompt_completion_breakdown()
        assert "no token usage recorded" in out

    def test_only_non_llm_steps_returns_placeholder(self) -> None:
        out = _result(
            _step(1, step_type=StepType.TOOL),
            _step(2, step_type=StepType.MEMORY),
        ).format_prompt_completion_breakdown()
        assert "no token usage recorded" in out

    def test_all_zero_tokens_returns_placeholder(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 0, "completion": 0, "total": 0}),
        ).format_prompt_completion_breakdown()
        assert "nothing to chart" in out


# ---------------------------------------------------------------------------
# Header / legend / total row
# ---------------------------------------------------------------------------


class TestStructure:
    def test_header_contains_required_columns(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_prompt_completion_breakdown()
        assert "step" in out
        assert "prompt" in out
        assert "compl" in out
        assert "bar" in out

    def test_legend_explains_both_markers(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_prompt_completion_breakdown()
        assert "▒ prompt" in out
        assert "█ completion" in out

    def test_total_row_sums_both_columns(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, usage={"prompt": 200, "completion": 300, "total": 500}),
        ).format_prompt_completion_breakdown()
        # Total prompt = 300, total completion = 350
        lines = out.splitlines()
        total_row = next(line for line in lines if line.startswith("TOT"))
        assert "300" in total_row
        assert "350" in total_row


# ---------------------------------------------------------------------------
# Bar character semantics
# ---------------------------------------------------------------------------


class TestBarSemantics:
    def test_prompt_only_step_has_only_prompt_chars(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 1000, "completion": 0, "total": 1000})
        ).format_prompt_completion_breakdown()
        # The bar row should have ▒ but no █
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        assert "▒" in data_row
        assert "█" not in data_row

    def test_completion_only_step_has_only_completion_chars(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 0, "completion": 1000, "total": 1000})
        ).format_prompt_completion_breakdown()
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        assert "█" in data_row
        assert "▒" not in data_row

    def test_balanced_step_has_equal_prompt_and_completion_cells(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 500, "completion": 500, "total": 1000})
        ).format_prompt_completion_breakdown(bar_width=40)
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        prompt_cells = data_row.count("▒")
        completion_cells = data_row.count("█")
        assert prompt_cells == completion_cells

    def test_prompt_then_completion_order_preserved(self) -> None:
        """The bar must read left-to-right as ▒...█... (prompt then completion)
        so the visual maps to 'first the LLM consumed N tokens, then produced
        M tokens'."""
        out = _result(
            _step(1, usage={"prompt": 100, "completion": 100, "total": 200})
        ).format_prompt_completion_breakdown(bar_width=20)
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        # Find first ▒ and first █; ▒ must come first
        prompt_idx = data_row.index("▒")
        completion_idx = data_row.index("█")
        assert prompt_idx < completion_idx

    def test_small_segment_still_shown_as_at_least_one_cell(self) -> None:
        """If completion is small relative to prompt, the completion bar
        shouldn't disappear — show at least one █ if completion > 0."""
        out = _result(
            _step(1, usage={"prompt": 10000, "completion": 1, "total": 10001})
        ).format_prompt_completion_breakdown(bar_width=20)
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        assert "█" in data_row

    def test_zero_segment_omitted_entirely(self) -> None:
        """If completion is exactly 0, no █ should appear (vs the
        "always-show-1-cell" rule for non-zero values)."""
        out = _result(
            _step(1, usage={"prompt": 1000, "completion": 0, "total": 1000})
        ).format_prompt_completion_breakdown(bar_width=20)
        data_row = next(line for line in out.splitlines() if line.startswith("  1"))
        assert "█" not in data_row


# ---------------------------------------------------------------------------
# Scaling — bars sized by the chain's heaviest step
# ---------------------------------------------------------------------------


class TestScaling:
    def test_bars_scale_proportionally_to_heaviest_step(self) -> None:
        """A step with double the tokens of another should have a bar that's
        roughly twice as wide."""
        out = _result(
            _step(1, title="big", usage={"prompt": 800, "completion": 200, "total": 1000}),
            _step(2, title="small", usage={"prompt": 400, "completion": 100, "total": 500}),
        ).format_prompt_completion_breakdown(bar_width=20)
        # Big step should fill the whole bar (1000/1000 * 20 = 20 cells)
        # Small step should fill half (500/1000 * 20 = 10 cells)
        lines = out.splitlines()
        big_row = next(line for line in lines if "big" in line)
        small_row = next(line for line in lines if "small" in line)
        big_cells = big_row.count("▒") + big_row.count("█")
        small_cells = small_row.count("▒") + small_row.count("█")
        # 2:1 ratio with a tolerance for rounding
        assert big_cells > small_cells
        assert big_cells <= 20
        # small_cells should be ~10 within tight rounding
        assert 8 <= small_cells <= 12


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_non_llm_steps_omitted_from_chart(self) -> None:
        out = _result(
            _step(1, title="LLM", usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, title="ToolStep", step_type=StepType.TOOL),
            _step(3, title="MemStep", step_type=StepType.MEMORY),
        ).format_prompt_completion_breakdown()
        assert "LLM" in out
        assert "ToolStep" not in out
        assert "MemStep" not in out

    def test_long_title_truncated_with_ellipsis(self) -> None:
        out = _result(
            _step(1, title="X" * 100, usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_prompt_completion_breakdown(title_width=20)
        assert "X" * 100 not in out
        assert "…" in out

    def test_uses_thousands_separator_for_large_values(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 12345, "completion": 6789, "total": 19134})
        ).format_prompt_completion_breakdown()
        assert "12,345" in out
        assert "6,789" in out

    def test_natural_chain_order_preserved(self) -> None:
        """Unlike format_token_pie (sorted by total descending), this view
        keeps steps in their natural chain order so adjacent steps stay
        adjacent on screen — useful for comparing neighbour ratios."""
        out = _result(
            _step(1, title="first", usage={"prompt": 10, "completion": 5, "total": 15}),
            _step(2, title="second", usage={"prompt": 1000, "completion": 500, "total": 1500}),
            _step(3, title="third", usage={"prompt": 50, "completion": 25, "total": 75}),
        ).format_prompt_completion_breakdown()
        first_idx = out.index("first")
        second_idx = out.index("second")
        third_idx = out.index("third")
        # Order preserved despite step 2 being the biggest
        assert first_idx < second_idx < third_idx
