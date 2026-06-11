"""Utility functions for CARL examples."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmar_carl import ReasoningResult


# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


def format_status(success: bool) -> str:
    """Format execution status with color.

    Args:
        success: Whether the execution was successful

    Returns:
        Colored status string (green SUCCESS or red FAILED)
    """
    status = "SUCCESS" if success else "FAILED"
    color = Colors.GREEN if success else Colors.RED
    return f"{color}{status}{Colors.RESET}"


def print_execution_summary(
    result: "ReasoningResult",
    *,
    label: str = "Execution",
    show_output_preview: bool = False,
    output_preview_chars: int = 240,
    traceback_lines: int = 5,
) -> None:
    """Print a one-line status header plus, on failure, a per-step
    breakdown of every failed step's error message + short traceback.

    Designed as the canonical replacement for the common example pattern::

        print(f"Execution: {format_status(result.success)}")

    which hides *why* a chain failed. With this helper the user sees the
    concrete error (e.g. "Connection refused") immediately, so a misconfigured
    example (missing API key, local LLM not running, MCP server unreachable)
    becomes diagnosable without scrolling through stack traces.

    Args:
        result: The ``ReasoningResult`` from ``chain.execute_async()``.
        label: Prefix for the status line (default "Execution").
        show_output_preview: If True and the run succeeded, print a short
            preview of ``result.get_final_output()``.
        output_preview_chars: Max characters of the preview when shown.
        traceback_lines: How many trailing traceback lines to show per
            failed step. Pass 0 to suppress tracebacks entirely.
    """
    print(f"{label}: {format_status(result.success)}")

    if result.success:
        if show_output_preview:
            output = result.get_final_output() or ""
            if output:
                preview = output.strip()
                if len(preview) > output_preview_chars:
                    preview = preview[:output_preview_chars] + "…"
                print(f"  Output preview: {preview}")
        return

    # Failure path — print each failed step's error.
    failed_steps = result.get_failed_steps()
    if not failed_steps:
        # Chain failure with no failed steps (rare — e.g. validation error
        # before execution). Surface result.metadata for context.
        print(
            f"  {Colors.RED}(no failed steps recorded — chain failed before "
            f"step execution){Colors.RESET}"
        )
        return

    for step in failed_steps:
        print(
            f"  {Colors.RED}✗ Step {step.step_number} '{step.step_title}' "
            f"({step.step_type}):{Colors.RESET} {step.error_message or '(no message)'}"
        )
        if traceback_lines > 0 and step.error_traceback:
            tb_lines = step.error_traceback.rstrip().splitlines()
            tail = tb_lines[-traceback_lines:]
            for line in tail:
                print(f"    {Colors.YELLOW}{line}{Colors.RESET}")
