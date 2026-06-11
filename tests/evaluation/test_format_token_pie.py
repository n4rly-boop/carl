"""Tests for ``ReasoningResult.format_token_pie`` — text / Mermaid / PNG
rendering of per-step token spend.

The highest-signal viz for the most-asked
"where did my tokens go?" question. Default text mode works without any
optional deps; Mermaid mode renders natively in GitHub; PNG requires
matplotlib and raises a clear install hint otherwise.
"""

from __future__ import annotations

import os
import tempfile

import pytest

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
# Edge cases — empty + zero-token chains
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_chain_returns_placeholder(self) -> None:
        out = _result().format_token_pie()
        assert "no token usage recorded" in out

    def test_chain_with_only_non_llm_steps_returns_placeholder(self) -> None:
        out = _result(
            _step(1, step_type=StepType.TOOL),
            _step(2, step_type=StepType.MEMORY),
        ).format_token_pie()
        assert "no token usage recorded" in out

    def test_all_zero_tokens_returns_placeholder(self) -> None:
        """All LLM steps recorded zero usage (rare but possible)."""
        out = _result(
            _step(1, usage={"prompt": 0, "completion": 0, "total": 0}),
        ).format_token_pie()
        assert "zero tokens" in out

    def test_unknown_format_raises(self) -> None:
        result = _result(_step(1, usage={"prompt": 10, "completion": 5, "total": 15}))
        with pytest.raises(ValueError, match="Unknown format"):
            result.format_token_pie(format="xml")


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------


class TestTextFormat:
    def test_text_format_header_present(self) -> None:
        out = _result(_step(1, usage={"prompt": 10, "completion": 5, "total": 15})).format_token_pie()
        assert "step" in out
        assert "tokens" in out
        assert "%" in out
        assert "bar" in out

    def test_text_format_includes_total_row(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, usage={"prompt": 200, "completion": 100, "total": 300}),
        ).format_token_pie()
        # TOT row sums to 450
        assert "TOT" in out
        assert "450" in out
        assert "100.0%" in out

    def test_text_format_rows_sorted_descending_by_tokens(self) -> None:
        out = _result(
            _step(1, title="small", usage={"prompt": 50, "completion": 0, "total": 50}),
            _step(2, title="big", usage={"prompt": 500, "completion": 0, "total": 500}),
            _step(3, title="medium", usage={"prompt": 250, "completion": 0, "total": 250}),
        ).format_token_pie()
        # "big" appears before "medium" appears before "small"
        big_idx = out.index("big")
        medium_idx = out.index("medium")
        small_idx = out.index("small")
        assert big_idx < medium_idx < small_idx

    def test_text_format_percentages_sum_to_100(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 100, "completion": 0, "total": 100}),
            _step(2, usage={"prompt": 100, "completion": 0, "total": 100}),
            _step(3, usage={"prompt": 100, "completion": 0, "total": 100}),
        ).format_token_pie()
        # Each step is 33.3% → roughly equal bars
        assert "33.3%" in out

    def test_text_format_omits_non_llm_steps(self) -> None:
        out = _result(
            _step(1, title="LLM", usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, title="ToolStep", step_type=StepType.TOOL),
            _step(3, title="MemoryStep", step_type=StepType.MEMORY),
        ).format_token_pie()
        # Tool / Memory have no token usage → omitted from the chart
        assert "ToolStep" not in out
        assert "MemoryStep" not in out
        assert "LLM" in out

    def test_text_format_truncates_long_titles_with_ellipsis(self) -> None:
        long_title = "X" * 100
        out = _result(
            _step(1, title=long_title, usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_token_pie(title_width=20)
        # Full 100-char title shouldn't appear
        assert "X" * 100 not in out
        # Ellipsis appears
        assert "…" in out

    def test_text_format_uses_thousands_separator(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 12345, "completion": 6789, "total": 19134})
        ).format_token_pie()
        # Thousands separator present (US locale: comma)
        assert "19,134" in out


# ---------------------------------------------------------------------------
# Mermaid format
# ---------------------------------------------------------------------------


class TestMermaidFormat:
    def test_mermaid_format_uses_pie_directive(self) -> None:
        out = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150})
        ).format_token_pie(format="mermaid")
        assert out.startswith("pie title")
        assert "Token spend by step" in out

    def test_mermaid_format_one_line_per_step(self) -> None:
        out = _result(
            _step(1, title="A", usage={"prompt": 100, "completion": 0, "total": 100}),
            _step(2, title="B", usage={"prompt": 200, "completion": 0, "total": 200}),
        ).format_token_pie(format="mermaid")
        lines = out.splitlines()
        # header + 2 step lines
        assert len(lines) == 3
        assert "step 1: A" in lines[1] or "step 2: B" in lines[1]

    def test_mermaid_format_token_count_present_as_value(self) -> None:
        out = _result(
            _step(1, title="Big", usage={"prompt": 999, "completion": 1, "total": 1000})
        ).format_token_pie(format="mermaid")
        # `"label" : 1000` syntax
        assert "1000" in out
        assert ":" in out

    def test_mermaid_format_escapes_double_quotes_in_title(self) -> None:
        out = _result(
            _step(1, title='Has "quoted" word', usage={"prompt": 10, "completion": 5, "total": 15})
        ).format_token_pie(format="mermaid")
        # Double quotes inside the label would break Mermaid; should be replaced
        # with single quotes.
        assert 'Has \'quoted\' word' in out
        # And no naked double quote pair inside the label
        body = out.splitlines()[1]
        # The line is structured: '    "<label>" : <number>'
        # Count of double quotes should be exactly 2 (the surrounding ones).
        assert body.count('"') == 2

    def test_mermaid_format_sorted_descending(self) -> None:
        out = _result(
            _step(1, title="small", usage={"prompt": 50, "completion": 0, "total": 50}),
            _step(2, title="big", usage={"prompt": 500, "completion": 0, "total": 500}),
        ).format_token_pie(format="mermaid")
        assert out.index("big") < out.index("small")


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        result = _result(_step(1, usage={"prompt": 10, "completion": 5, "total": 15}))
        with pytest.raises(ValueError, match="png_path"):
            result.format_token_pie(format="png")

    def test_png_returns_install_hint_when_matplotlib_missing(self) -> None:
        """If matplotlib isn't available, raise ImportError pointing at
        the install command. (Skip when matplotlib IS installed — there
        the PNG writes successfully and we cover that in the next test.)"""
        try:
            import matplotlib  # noqa: F401

            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        result = _result(_step(1, usage={"prompt": 10, "completion": 5, "total": 15}))
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            result.format_token_pie(format="png", png_path="/tmp/x.png")

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed; skipping live PNG test")
        result = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, usage={"prompt": 200, "completion": 100, "total": 300}),
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pie.png")
            written = result.format_token_pie(format="png", png_path=path)
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0
            # Return value is the absolute path
            assert written == os.path.abspath(path)
