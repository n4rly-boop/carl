"""Tests for ``ExecutionTrace.format_gantt`` — Gantt chart from trace.

Surfaces the parallel structure of the chain — which
steps overlap in time (same batch) vs which are serial (cross-batch).
Default text format works in terminals; Mermaid format renders natively
in GitHub READMEs and Markdown notebooks.
"""

from __future__ import annotations

import pytest

from mmar_carl.execution_trace import ExecutionTrace, TraceEvent


def _event(
    number: int,
    *,
    title: str = "s",
    type_: str = "llm",
    success: bool = True,
    skipped: bool = False,
    execution_time: float = 1.0,
    batch_index: int = 0,
) -> TraceEvent:
    return TraceEvent(
        step_number=number,
        step_title=title,
        step_type=type_,
        success=success,
        skipped=skipped,
        execution_time=execution_time,
        batch_index=batch_index,
    )


def _trace(*events: TraceEvent) -> ExecutionTrace:
    return ExecutionTrace(chain_title="t", events=list(events))


# ---------------------------------------------------------------------------
# Interval computation — the heart of the Gantt math
# ---------------------------------------------------------------------------


class TestComputeStepIntervals:
    def test_empty_trace_returns_empty_list(self) -> None:
        assert _trace()._compute_step_intervals() == []

    def test_single_step_starts_at_zero(self) -> None:
        rows = _trace(_event(1, execution_time=2.5))._compute_step_intervals()
        assert len(rows) == 1
        _, start, end = rows[0]
        assert start == 0.0
        assert end == 2.5

    def test_parallel_steps_in_same_batch_share_start(self) -> None:
        rows = _trace(
            _event(1, execution_time=1.0, batch_index=0),
            _event(2, execution_time=2.0, batch_index=0),
            _event(3, execution_time=0.5, batch_index=0),
        )._compute_step_intervals()
        starts = [s for _, s, _ in rows]
        assert all(s == 0.0 for s in starts)

    def test_serial_batches_run_after_previous_batch_max(self) -> None:
        """Batch 1 starts at max(execution_time of batch 0)."""
        rows = _trace(
            _event(1, execution_time=1.0, batch_index=0),
            _event(2, execution_time=2.5, batch_index=0),  # longest in batch 0
            _event(3, execution_time=0.5, batch_index=1),
        )._compute_step_intervals()
        # Step 3 (batch 1) starts at 2.5
        for ev, start, _ in rows:
            if ev.step_number == 3:
                assert start == 2.5

    def test_three_batches_chain_correctly(self) -> None:
        rows = _trace(
            _event(1, execution_time=1.0, batch_index=0),
            _event(2, execution_time=2.0, batch_index=1),
            _event(3, execution_time=0.5, batch_index=2),
        )._compute_step_intervals()
        starts = {ev.step_number: s for ev, s, _ in rows}
        ends = {ev.step_number: e for ev, _, e in rows}
        assert starts[1] == 0.0 and ends[1] == 1.0
        assert starts[2] == 1.0 and ends[2] == 3.0
        assert starts[3] == 3.0 and ends[3] == 3.5


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------


class TestTextFormat:
    def test_empty_trace_returns_placeholder(self) -> None:
        assert "empty trace" in _trace().format_gantt()

    def test_header_present(self) -> None:
        out = _trace(_event(1)).format_gantt()
        assert "step" in out
        assert "batch" in out
        assert "time" in out
        assert "timeline" in out

    def test_total_wall_time_line(self) -> None:
        out = _trace(
            _event(1, execution_time=1.0, batch_index=0),
            _event(2, execution_time=2.0, batch_index=1),
        ).format_gantt()
        assert "total wall time" in out
        # Two serial batches → 3.0s total
        assert "3.00s" in out

    def test_batch_index_visible_in_each_row(self) -> None:
        out = _trace(
            _event(1, batch_index=0),
            _event(2, batch_index=1),
        ).format_gantt()
        # Both 0 and 1 appear (as batch indices)
        lines = out.splitlines()
        # Find data rows (skip header + separator)
        data_lines = [
            line for line in lines
            if any(line.lstrip().startswith(str(n)) for n in (1, 2))
        ]
        assert any(" 0 " in line for line in data_lines)
        assert any(" 1 " in line for line in data_lines)

    def test_bar_starts_after_previous_batches(self) -> None:
        """Step 2 (batch 1) bar should start to the right of step 1 (batch 0)."""
        out = _trace(
            _event(1, title="A", execution_time=1.0, batch_index=0),
            _event(2, title="B", execution_time=1.0, batch_index=1),
        ).format_gantt()
        lines = out.splitlines()
        row_a = next(line for line in lines if "A" in line)
        row_b = next(line for line in lines if "B" in line and "B" != line[0])
        # Find first █ in each row
        a_start = row_a.index("█")
        b_start = row_b.index("█")
        assert b_start > a_start

    def test_parallel_steps_share_bar_start_column(self) -> None:
        out = _trace(
            _event(1, title="A", execution_time=1.0, batch_index=0),
            _event(2, title="B", execution_time=1.0, batch_index=0),
        ).format_gantt()
        lines = out.splitlines()
        row_a = next(line for line in lines if "A" in line)
        row_b = next(line for line in lines if "B" in line and "B" != line[0])
        assert row_a.index("█") == row_b.index("█")

    def test_skipped_step_renders_with_dot_character(self) -> None:
        out = _trace(
            _event(1, title="OK", execution_time=1.0),
            _event(2, title="SKIP", skipped=True, execution_time=0.0),
        ).format_gantt()
        # Dot character appears for the skipped row
        assert "·" in out

    def test_failed_step_renders_with_red_ansi(self) -> None:
        out = _trace(
            _event(1, title="OK", execution_time=1.0),
            _event(2, title="FAIL", success=False, execution_time=0.5, batch_index=1),
        ).format_gantt()
        # Red ANSI escape \033[91m present somewhere
        assert "\033[91m" in out
        assert "\033[0m" in out

    def test_long_title_truncated_with_ellipsis(self) -> None:
        out = _trace(_event(1, title="X" * 100)).format_gantt(title_width=20)
        assert "X" * 100 not in out
        assert "…" in out

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            _trace(_event(1)).format_gantt(format="svg")


# ---------------------------------------------------------------------------
# Mermaid format
# ---------------------------------------------------------------------------


class TestMermaidFormat:
    def test_empty_trace_returns_placeholder(self) -> None:
        assert "empty trace" in _trace().format_gantt(format="mermaid")

    def test_mermaid_format_starts_with_gantt(self) -> None:
        out = _trace(_event(1)).format_gantt(format="mermaid")
        assert out.startswith("gantt")

    def test_mermaid_has_axis_format_in_seconds(self) -> None:
        out = _trace(_event(1)).format_gantt(format="mermaid")
        assert "axisFormat %S s" in out
        assert "dateFormat X" in out

    def test_mermaid_groups_by_batch_section(self) -> None:
        out = _trace(
            _event(1, batch_index=0),
            _event(2, batch_index=1),
        ).format_gantt(format="mermaid")
        assert "section batch 0" in out
        assert "section batch 1" in out

    def test_mermaid_status_active_for_success(self) -> None:
        out = _trace(_event(1, success=True)).format_gantt(format="mermaid")
        assert ":active," in out

    def test_mermaid_status_crit_for_failure(self) -> None:
        out = _trace(_event(1, success=False)).format_gantt(format="mermaid")
        assert ":crit," in out

    def test_mermaid_status_done_for_skipped(self) -> None:
        out = _trace(_event(1, skipped=True)).format_gantt(format="mermaid")
        assert ":done," in out

    def test_mermaid_milliseconds_present(self) -> None:
        out = _trace(_event(1, execution_time=2.5)).format_gantt(format="mermaid")
        # 2.5s = 2500 ms
        assert "2500" in out

    def test_mermaid_sanitises_colons_in_titles(self) -> None:
        """Mermaid gantt task syntax uses colons as separators —
        colons in titles must be removed."""
        out = _trace(
            _event(1, title="Has: a colon")
        ).format_gantt(format="mermaid")
        # The literal colon-in-label is gone (replaced with space)
        assert "Has  a colon" in out
        # And the syntax colon (before status) is still present
        assert ":active," in out

    def test_mermaid_includes_step_number_in_task_label(self) -> None:
        out = _trace(
            _event(1, title="MyStep", batch_index=0)
        ).format_gantt(format="mermaid")
        assert "MyStep (step 1)" in out

    def test_mermaid_zero_duration_step_gets_minimum_width(self) -> None:
        """A skipped/zero-time step gets a 1ms bar so Mermaid doesn't render
        a malformed task line."""
        out = _trace(
            _event(1, execution_time=0.0)
        ).format_gantt(format="mermaid")
        # The task line should have an end ms > start ms
        task_line = next(line for line in out.splitlines() if "step 1" in line)
        # Parse `... :status, start, end` — end is the last comma-separated token
        end = int(task_line.rstrip().split(",")[-1].strip())
        start = int(task_line.split(",")[-2].strip())
        assert end > start
