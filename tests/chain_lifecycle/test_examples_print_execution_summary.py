"""Tests for ``examples.utils.print_execution_summary``.

The live openrouter run printed a bare "Execution: FAILED" line when the
local LLM server wasn't reachable — no error message, no traceback, no
clue what went wrong. The new helper always surfaces per-failed-step error
messages and a short traceback tail so misconfigured examples become
diagnosable without scrolling.
"""

from __future__ import annotations

import asyncio

import pytest

from examples.utils import Colors, format_status, print_execution_summary
from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.models.results import (
    ReasoningResult,
    StepExecutionResult,
)
from mmar_carl.models.enums import StepType


# ---------------------------------------------------------------------------
# Helpers — handcrafted ReasoningResult instances to drive specific shapes.
# ---------------------------------------------------------------------------


def _success_result(output: str = "all good") -> ReasoningResult:
    step = StepExecutionResult(
        step_number=1,
        step_title="step1",
        step_type=StepType.TOOL,
        result=output,
        success=True,
    )
    return ReasoningResult(
        success=True,
        history=[f"Step 1. step1 [TOOL: x]\nResult: {output}\n"],
        step_results=[step],
        total_execution_time=0.01,
    )


def _failure_result(
    *,
    error_message: str = "Connection refused",
    traceback: str | None = None,
) -> ReasoningResult:
    step = StepExecutionResult(
        step_number=1,
        step_title="local_llm_call",
        step_type=StepType.LLM,
        result="",
        success=False,
        error_message=error_message,
        error_traceback=traceback,
    )
    return ReasoningResult(
        success=False,
        history=[],
        step_results=[step],
        total_execution_time=0.01,
    )


# ---------------------------------------------------------------------------
# format_status — pre-existing, kept as regression
# ---------------------------------------------------------------------------


class TestFormatStatus:
    def test_success_is_green(self) -> None:
        out = format_status(True)
        assert "SUCCESS" in out
        assert Colors.GREEN in out
        assert Colors.RESET in out

    def test_failure_is_red(self) -> None:
        out = format_status(False)
        assert "FAILED" in out
        assert Colors.RED in out


# ---------------------------------------------------------------------------
# Success-path printing
# ---------------------------------------------------------------------------


class TestPrintSuccessPath:
    def test_prints_status_line_on_success(self, capsys: pytest.CaptureFixture) -> None:
        print_execution_summary(_success_result())
        captured = capsys.readouterr().out
        assert "Execution:" in captured
        assert "SUCCESS" in captured

    def test_no_output_preview_by_default(self, capsys: pytest.CaptureFixture) -> None:
        print_execution_summary(_success_result(output="hello world"))
        captured = capsys.readouterr().out
        assert "hello world" not in captured

    def test_output_preview_when_enabled(self, capsys: pytest.CaptureFixture) -> None:
        print_execution_summary(_success_result(output="hello world"), show_output_preview=True)
        captured = capsys.readouterr().out
        assert "hello world" in captured
        assert "Output preview" in captured

    def test_output_preview_truncated_to_max_chars(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        long = "x" * 1000
        print_execution_summary(
            _success_result(output=long),
            show_output_preview=True,
            output_preview_chars=50,
        )
        captured = capsys.readouterr().out
        # 50 chars + ellipsis marker
        assert "x" * 50 in captured
        assert "…" in captured
        # Not all 1000 x's printed
        assert "x" * 100 not in captured


# ---------------------------------------------------------------------------
# Failure-path printing — the headline fix
# ---------------------------------------------------------------------------


class TestPrintFailurePath:
    def test_status_line_shows_failed(self, capsys: pytest.CaptureFixture) -> None:
        print_execution_summary(_failure_result())
        captured = capsys.readouterr().out
        assert "FAILED" in captured

    def test_step_number_and_title_printed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        print_execution_summary(_failure_result())
        captured = capsys.readouterr().out
        assert "Step 1" in captured
        assert "local_llm_call" in captured

    def test_error_message_printed(self, capsys: pytest.CaptureFixture) -> None:
        print_execution_summary(_failure_result(error_message="Connection refused"))
        captured = capsys.readouterr().out
        assert "Connection refused" in captured

    def test_traceback_tail_printed_when_present(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tb = (
            "Traceback (most recent call last):\n"
            "  File 'a.py', line 1\n"
            "  File 'b.py', line 2\n"
            "ConnectionRefusedError: [Errno 61] Connection refused"
        )
        print_execution_summary(_failure_result(traceback=tb))
        captured = capsys.readouterr().out
        assert "ConnectionRefusedError" in captured
        assert "Connection refused" in captured

    def test_traceback_lines_limited(self, capsys: pytest.CaptureFixture) -> None:
        tb = "\n".join(f"frame {i}" for i in range(20))
        print_execution_summary(_failure_result(traceback=tb), traceback_lines=3)
        captured = capsys.readouterr().out
        # Last 3 lines present
        assert "frame 19" in captured
        assert "frame 18" in captured
        assert "frame 17" in captured
        # Earlier lines absent
        assert "frame 0" not in captured
        assert "frame 5" not in captured

    def test_traceback_suppressed_when_zero_lines(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        tb = "frame 0\nframe 1\nframe 2"
        print_execution_summary(_failure_result(traceback=tb), traceback_lines=0)
        captured = capsys.readouterr().out
        for line in ("frame 0", "frame 1", "frame 2"):
            assert line not in captured

    def test_missing_error_message_falls_back_to_placeholder(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        print_execution_summary(_failure_result(error_message=""))
        captured = capsys.readouterr().out
        assert "(no message)" in captured

    def test_no_failed_steps_emits_synthetic_note(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        """A chain that failed before any step ran (validation error etc.)."""
        result = ReasoningResult(
            success=False,
            history=[],
            step_results=[],
            total_execution_time=0.0,
        )
        print_execution_summary(result)
        captured = capsys.readouterr().out
        assert "FAILED" in captured
        assert "no failed steps recorded" in captured


# ---------------------------------------------------------------------------
# Custom label
# ---------------------------------------------------------------------------


def test_custom_label_used(capsys: pytest.CaptureFixture) -> None:
    print_execution_summary(_success_result(), label="Chain A")
    captured = capsys.readouterr().out
    assert "Chain A: " in captured
    assert "Execution:" not in captured


# ---------------------------------------------------------------------------
# Integration with a real ReasoningChain failure
# ---------------------------------------------------------------------------


def test_end_to_end_with_real_chain_failure(capsys: pytest.CaptureFixture) -> None:
    """Run a chain whose tool raises; the failure summary should pick up the
    real error message + traceback. Mirrors the openrouter-example scenario."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="boom",
                config=ToolStepConfig(tool_name="boom"),
            ),
        ],
    )
    ctx = ReasoningContext(outer_context="x", api=None, model="default")

    def boom() -> str:
        raise ConnectionRefusedError("local LLM at http://localhost:11434/v1 unreachable")

    ctx.register_tool("boom", boom)

    result = asyncio.run(chain.execute_async(ctx))
    assert result.success is False

    print_execution_summary(result)
    captured = capsys.readouterr().out
    assert "FAILED" in captured
    assert "boom" in captured  # step title
    assert "local LLM at http://localhost:11434/v1 unreachable" in captured
