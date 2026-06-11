"""
Structured execution trace for CARL reasoning chains.

An :class:`ExecutionTrace` is built automatically during every chain execution
and stored on the :class:`~mmar_carl.models.results.ReasoningResult`.  It gives
you a structured, serialisable record of every step — inputs, outputs, timing,
token usage, and errors — that you can persist, diff, and replay.

Usage::

    result = await chain.execute_async(ctx)
    trace = result.trace

    # Persist to disk
    Path("trace.json").write_text(trace.to_json())

    # Load back
    trace2 = ExecutionTrace.from_json(Path("trace.json").read_text())

    # Compare two runs
    diff = trace.diff(trace2)

    # Replay — re-execute the chain but feed saved LLM responses
    result2 = await chain.replay(trace, ctx)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models.enums import StepType


class TraceEvent(BaseModel):
    """
    Captures the full execution record of one reasoning step.

    Attributes
    ----------
    step_number : int
        Step number as declared in the chain.
    step_title : str
        Human-readable step title.
    step_type : StepType
        Executor type (llm, tool, memory, …).
    success : bool
        Whether the step completed without error.
    skipped : bool
        Whether the step was skipped by conditional routing.
    result : str
        Raw result text (truncated to 10 000 chars for storage).
    result_data : Any
        Structured result data (tool outputs, memory values, etc.).
    error_message : str | None
        Error message if the step failed.
    execution_time : float | None
        Wall-clock time in seconds for this step.
    token_usage : dict[str, int]
        ``{"prompt": X, "completion": Y, "total": Z}`` token counts.
    batch_index : int
        Index of the parallel batch in which this step ran (0-based).
    inputs : dict[str, Any]
        Resolved input values passed to the step executor (best-effort).
    """

    step_number: int
    step_title: str
    step_type: StepType = StepType.LLM
    success: bool
    skipped: bool = False
    result: str = ""
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    batch_index: int = 0
    inputs: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "step_number": self.step_number,
            "step_title": self.step_title,
            "step_type": str(self.step_type),
            "success": self.success,
            "skipped": self.skipped,
            "result": self.result,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "batch_index": self.batch_index,
            "inputs": self.inputs,
        }


class ExecutionTrace(BaseModel):
    """
    Full execution trace for one chain run.

    Built automatically by :class:`~mmar_carl.executor.DAGExecutor` and
    attached to :attr:`~mmar_carl.models.results.ReasoningResult.trace`.

    Attributes
    ----------
    chain_title : str
        The name of the chain that was executed.
    executed_at : str
        ISO-8601 UTC timestamp of when execution began.
    total_execution_time : float | None
        Total wall-clock time in seconds.
    success : bool
        Whether the overall execution succeeded.
    events : list[TraceEvent]
        Ordered list of step events, sorted by step_number.
    metadata : dict[str, Any]
        Chain-level metadata (batch count, token totals, etc.).
    """

    chain_title: str = ""
    executed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_execution_time: Optional[float] = None
    success: bool = False
    events: list[TraceEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_event(self, step_number: int) -> Optional[TraceEvent]:
        """Return the event for *step_number*, or None if not found."""
        for e in self.events:
            if e.step_number == step_number:
                return e
        return None

    def get_successful_events(self) -> list[TraceEvent]:
        """Return only events where ``success=True`` and ``skipped=False``."""
        return [e for e in self.events if e.success and not e.skipped]

    def get_failed_events(self) -> list[TraceEvent]:
        """Return only events where the step failed."""
        return [e for e in self.events if not e.success and not e.skipped]

    def get_skipped_events(self) -> list[TraceEvent]:
        """Return only events that were skipped by conditional routing."""
        return [e for e in self.events if e.skipped]

    def total_tokens(self) -> dict[str, int]:
        """Sum token usage across all events."""
        p = sum(e.token_usage.get("prompt", 0) for e in self.events)
        c = sum(e.token_usage.get("completion", 0) for e in self.events)
        return {"prompt": p, "completion": c, "total": p + c}

    # ------------------------------------------------------------------
    # Gantt visualization
    # ------------------------------------------------------------------

    def _compute_step_intervals(self) -> list[tuple[TraceEvent, float, float]]:
        """Derive ``(event, start_s, end_s)`` per event.

        ``TraceEvent`` carries ``batch_index`` and ``execution_time`` but not
        absolute start/end timestamps. We reconstruct them under the
        assumption every step in a batch starts when the batch begins:

        - batch 0 starts at ``t=0``
        - batch ``N``'s start = batch ``N-1``'s end =
          ``batch[N-1].start + max(execution_time of all steps in batch N-1)``

        Skipped steps appear with ``start == end`` so they don't render as
        visible bars but still show in the list ordering.
        """
        if not self.events:
            return []

        by_batch: dict[int, list[TraceEvent]] = {}
        for ev in self.events:
            by_batch.setdefault(ev.batch_index, []).append(ev)

        batch_starts: dict[int, float] = {}
        running = 0.0
        for batch_idx in sorted(by_batch):
            batch_starts[batch_idx] = running
            batch_duration = max(
                (ev.execution_time or 0.0) for ev in by_batch[batch_idx]
            )
            running += batch_duration

        rows: list[tuple[TraceEvent, float, float]] = []
        for ev in self.events:
            start = batch_starts.get(ev.batch_index, 0.0)
            end = start + (ev.execution_time or 0.0)
            rows.append((ev, start, end))
        return rows

    def format_gantt(
        self,
        *,
        format: str = "text",
        title_width: int = 24,
        bar_width: int = 50,
    ) -> str:
        """Render the per-step execution timeline as a Gantt chart.

        Surfaces the parallel structure of the chain — which steps overlap
        in time (same batch) vs which are serial (cross-batch). Two output
        formats:

        * ``format="text"`` (default): ASCII timeline with one row per step.
          The ``bar_width`` columns represent the full chain duration; each
          step's bar is positioned proportionally. Failed steps render in
          red, skipped steps in dim/grey, successful steps in default colour.
          Batch numbers are shown alongside each row so users can see which
          steps ran together.
        * ``format="mermaid"``: native Mermaid ``gantt`` block with one
          section per batch. Renders in GitHub READMEs / Markdown notebooks.
          Step durations are in seconds.

        Empty traces return a one-line placeholder. Returns the rendered
        string.
        """
        rows = self._compute_step_intervals()
        if not rows:
            return "(empty trace — no steps executed)"

        if format == "text":
            return self._format_gantt_text(rows, title_width, bar_width)
        if format == "mermaid":
            return self._format_gantt_mermaid(rows)
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'mermaid'."
        )

    @staticmethod
    def _format_gantt_text(
        rows: list[tuple["TraceEvent", float, float]],
        title_width: int,
        bar_width: int,
    ) -> str:
        # ANSI colour codes — terminal-only; harmless when piped to files.
        RED = "\033[91m"
        GREY = "\033[90m"
        RESET = "\033[0m"

        total_duration = max((end for _, _, end in rows), default=0.0)
        if total_duration <= 0:
            # All steps recorded zero time — render bars as single chars.
            total_duration = 1.0

        lines = [
            f"{'#':>3}  {'step':<{title_width}}  {'batch':>5}  "
            f"{'time':>7}  timeline"
        ]
        lines.append("-" * (3 + 2 + title_width + 2 + 5 + 2 + 7 + 2 + bar_width))

        for ev, start, end in rows:
            duration = end - start
            start_col = int((start / total_duration) * bar_width)
            bar_len = max(int((duration / total_duration) * bar_width), 1) if duration > 0 else 0

            if ev.skipped:
                bar_char = "·"
                colour = GREY
                bar = colour + (" " * start_col) + (bar_char * max(bar_len, 1)) + RESET
            elif not ev.success:
                bar_char = "█"
                colour = RED
                bar = colour + (" " * start_col) + (bar_char * max(bar_len, 1)) + RESET
            else:
                bar_char = "█"
                bar = (" " * start_col) + (bar_char * bar_len)

            short_title = (
                ev.step_title[: title_width - 1] + "…"
                if len(ev.step_title) > title_width
                else ev.step_title
            )
            time_str = f"{duration:.2f}s"
            lines.append(
                f"{ev.step_number:>3}  {short_title:<{title_width}}  "
                f"{ev.batch_index:>5}  {time_str:>7}  {bar}"
            )
        lines.append("-" * (3 + 2 + title_width + 2 + 5 + 2 + 7 + 2 + bar_width))
        lines.append(f"total wall time: {total_duration:.2f}s")
        return "\n".join(lines)

    @staticmethod
    def _format_gantt_mermaid(
        rows: list[tuple["TraceEvent", float, float]],
    ) -> str:
        # Group by batch so Mermaid can show them as sections.
        by_batch: dict[int, list[tuple["TraceEvent", float, float]]] = {}
        for ev, start, end in rows:
            by_batch.setdefault(ev.batch_index, []).append((ev, start, end))

        lines = [
            "gantt",
            "    title Chain execution timeline",
            "    dateFormat X",  # X = epoch ms; we'll use seconds*1000
            "    axisFormat %S s",
        ]
        for batch_idx in sorted(by_batch):
            lines.append(f"    section batch {batch_idx}")
            for ev, start, end in by_batch[batch_idx]:
                # Mermaid gantt syntax: `task name :status, id, start, end`
                # We use anonymous ids and milliseconds.
                start_ms = int(start * 1000)
                end_ms = max(int(end * 1000), start_ms + 1)  # avoid zero-width
                # Sanitize the title — Mermaid breaks on colons/commas in labels.
                safe_title = (
                    ev.step_title.replace(":", " ")
                    .replace(",", " ")
                    .replace("\n", " ")
                )
                if ev.skipped:
                    status = "done"
                elif not ev.success:
                    status = "crit"
                else:
                    status = "active"
                lines.append(
                    f"    {safe_title} (step {ev.step_number}) :{status}, "
                    f"{start_ms}, {end_ms}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Animated HTML export
    # ------------------------------------------------------------------

    def to_html(self, path: "str | Path | None" = None) -> str:
        """Render an animated, self-contained HTML playback of the trace.

        Produces a stand-alone HTML page with inline CSS + vanilla JS —
        no server, no external dependencies — that you can drop into a
        PR description, attach to an incident review, or open straight
        from disk. Batches light up one at a time in execution order;
        clicking any step card pins its full result/error in the right-
        side detail pane.

        Layout:

        * Top bar: chain title, total wall time, success/failure flag,
          and play / pause / step buttons.
        * Step grid: one card per :class:`TraceEvent`, grouped under a
          ``Batch N`` heading. Cards animate from idle (grey) to active
          (blue) to done (green / red / amber for success / failure /
          skipped) as playback advances.
        * Detail pane: shows the selected step's number, title, type,
          timing, token usage, full ``result`` text, ``result_data``
          (JSON-pretty), and ``error_message`` (when present).

        Args:
            path: Optional output path. When provided, the rendered
                HTML is also written to that file (parent dirs auto-
                created); regardless, the rendered string is returned.

        Returns:
            The rendered HTML as a string. When ``path`` is given, the
            same content is also persisted to disk so callers can
            ``trace.to_html("trace.html")`` in one line.
        """
        events_json = json.dumps(
            [e.to_dict() for e in self.events],
            ensure_ascii=False, indent=2, default=str,
        )
        title = self.chain_title or "(unnamed chain)"
        elapsed = (
            f"{self.total_execution_time:.2f}s"
            if self.total_execution_time is not None else "—"
        )
        status_text = "✅ success" if self.success else "❌ failed"
        status_class = "ok" if self.success else "fail"

        html = _ANIMATED_PLAYBACK_TEMPLATE.format(
            title=title.replace("</", "<\\/"),
            status_text=status_text,
            status_class=status_class,
            elapsed=elapsed,
            n_events=len(self.events),
            events_json=events_json,
        )
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
        return html

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "chain_title": self.chain_title,
            "executed_at": self.executed_at,
            "total_execution_time": self.total_execution_time,
            "success": self.success,
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the trace to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionTrace":
        """
        Deserialise an :class:`ExecutionTrace` from a JSON string produced
        by :meth:`to_json`.
        """
        data = json.loads(json_str)
        events = [
            TraceEvent(
                step_number=e["step_number"],
                step_title=e["step_title"],
                step_type=StepType(e["step_type"]),
                success=e["success"],
                skipped=e.get("skipped", False),
                result=e.get("result", ""),
                result_data=e.get("result_data"),
                error_message=e.get("error_message"),
                execution_time=e.get("execution_time"),
                token_usage=e.get("token_usage", {}),
                batch_index=e.get("batch_index", 0),
                inputs=e.get("inputs", {}),
            )
            for e in data.get("events", [])
        ]
        return cls(
            chain_title=data.get("chain_title", ""),
            executed_at=data.get("executed_at", ""),
            total_execution_time=data.get("total_execution_time"),
            success=data.get("success", False),
            events=events,
            metadata=data.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Diffing
    # ------------------------------------------------------------------

    def diff(self, other: "ExecutionTrace") -> dict[str, Any]:
        """
        Compare *self* (baseline) to *other* (new run).

        Returns a structured diff dict with:

        ``added``
            Step numbers present in *other* but not *self*.
        ``removed``
            Step numbers present in *self* but not *other*.
        ``changed``
            Steps that exist in both but whose result or success flag differ.
        ``unchanged``
            Steps that exist in both and are identical.
        ``summary``
            Human-readable one-line summary.

        Example::

            diff = trace_v1.diff(trace_v2)
            print(diff["summary"])
            # "3 unchanged, 1 changed, 0 added, 0 removed"
        """
        self_map: dict[int, TraceEvent] = {e.step_number: e for e in self.events}
        other_map: dict[int, TraceEvent] = {e.step_number: e for e in other.events}

        all_nums = set(self_map) | set(other_map)
        added: list[dict[str, Any]] = []
        removed: list[dict[str, Any]] = []
        changed: list[dict[str, Any]] = []
        unchanged: list[int] = []

        for num in sorted(all_nums):
            if num not in self_map:
                e = other_map[num]
                added.append({"step_number": num, "step_title": e.step_title})
            elif num not in other_map:
                e = self_map[num]
                removed.append({"step_number": num, "step_title": e.step_title})
            else:
                a, b = self_map[num], other_map[num]
                diffs: dict[str, Any] = {}
                if a.success != b.success:
                    diffs["success"] = {"before": a.success, "after": b.success}
                if a.result != b.result:
                    diffs["result"] = {
                        "before": a.result[:500] if a.result else "",
                        "after": b.result[:500] if b.result else "",
                    }
                if a.error_message != b.error_message:
                    diffs["error_message"] = {
                        "before": a.error_message,
                        "after": b.error_message,
                    }
                if diffs:
                    changed.append(
                        {
                            "step_number": num,
                            "step_title": a.step_title,
                            "diffs": diffs,
                        }
                    )
                else:
                    unchanged.append(num)

        summary = (
            f"{len(unchanged)} unchanged, {len(changed)} changed, "
            f"{len(added)} added, {len(removed)} removed"
        )
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "unchanged": unchanged,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Cross-trace aggregation
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    """Inclusive linear-interpolation percentile (numpy-style, no numpy dep).

    ``sorted_values`` must already be sorted ascending. ``p`` is in [0, 1].
    Returns 0.0 for an empty list.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return float(sorted_values[0])
    idx = p * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


class TraceAggregator:
    """Aggregates per-step latency + token usage across N chain runs.

    Useful when the same chain runs against a dataset (or in a load
    test) and you need "step 3 sometimes takes 8 s vs its p50 of 1 s"
    visibility for capacity planning and regression-catching that a
    single :class:`ExecutionTrace` can't give you.

    Usage::

        traces = [r.trace for r in many_results]
        agg = TraceAggregator(traces)
        print(agg.format_text())
        # or programmatic access:
        agg.latency_ms[1]  # {"p50": 1200.0, "p95": 1450.0, ...}
        agg.tokens[1]      # {"p50": 150.0, "p95": 230.0, ...}

    Traces with **different chain shapes** (different ``step_number``
    sets) are unioned; the per-step ``n_runs`` count tells you how many
    of the input traces actually included that step. Failed-step
    events still contribute their timing data so partial-failure
    cohorts don't silently disappear.
    """

    def __init__(self, traces: list["ExecutionTrace"]) -> None:
        self._traces = list(traces)
        # Per-step lists of (latency_ms, total_tokens_or_None, title).
        self._step_latencies: dict[int, list[float]] = {}
        self._step_tokens: dict[int, list[float]] = {}
        self._step_titles: dict[int, str] = {}
        for trace in self._traces:
            for ev in trace.events:
                if ev.execution_time is not None:
                    self._step_latencies.setdefault(ev.step_number, []).append(
                        float(ev.execution_time) * 1000.0
                    )
                total = (ev.token_usage or {}).get("total")
                if total:
                    self._step_tokens.setdefault(ev.step_number, []).append(
                        float(total)
                    )
                # First title wins; per-trace title drift is rare.
                self._step_titles.setdefault(ev.step_number, ev.step_title)
        # Sort once so percentile lookups are O(1).
        for vals in self._step_latencies.values():
            vals.sort()
        for vals in self._step_tokens.values():
            vals.sort()

    @property
    def n_traces(self) -> int:
        return len(self._traces)

    @property
    def step_numbers(self) -> list[int]:
        return sorted(self._step_latencies.keys() | self._step_tokens.keys())

    @property
    def latency_ms(self) -> dict[int, dict[str, float]]:
        """``{step_number: {p50, p95, p99, mean, max, n_runs}}`` in ms.

        Steps with zero recorded latency are omitted entirely.
        """
        out: dict[int, dict[str, float]] = {}
        for n, vals in self._step_latencies.items():
            if not vals:
                continue
            out[n] = {
                "p50": _percentile(vals, 0.5),
                "p95": _percentile(vals, 0.95),
                "p99": _percentile(vals, 0.99),
                "mean": sum(vals) / len(vals),
                "max": vals[-1],
                "n_runs": float(len(vals)),
            }
        return out

    @property
    def tokens(self) -> dict[int, dict[str, float]]:
        """``{step_number: {p50, p95, n_runs}}`` in total tokens.

        Steps that never recorded token usage (Tool, Memory, etc., or
        LLM steps backed by a mock client) are omitted. Use this to
        spot the step that occasionally explodes its context window.
        """
        out: dict[int, dict[str, float]] = {}
        for n, vals in self._step_tokens.items():
            if not vals:
                continue
            out[n] = {
                "p50": _percentile(vals, 0.5),
                "p95": _percentile(vals, 0.95),
                "n_runs": float(len(vals)),
            }
        return out

    def format_text(self, *, title_width: int = 22) -> str:
        """Render a one-table-per-step summary as plain ASCII.

        Columns: step number, title, n_runs, p50_ms, p95_ms, p99_ms,
        mean_ms, max_ms, p50_tok, p95_tok.
        """
        if not self._traces:
            return "(no traces — nothing to aggregate)"
        if not self._step_latencies and not self._step_tokens:
            return "(no per-step events recorded across the given traces)"

        latency = self.latency_ms
        tokens = self.tokens

        # Header
        hdr = (
            f"{'#':>3}  "
            f"{'step':<{title_width}}  "
            f"{'n':>4}  "
            f"{'p50 ms':>8}  "
            f"{'p95 ms':>8}  "
            f"{'p99 ms':>8}  "
            f"{'mean ms':>8}  "
            f"{'max ms':>8}  "
            f"{'p50 tok':>8}  "
            f"{'p95 tok':>8}"
        )
        sep = "-" * len(hdr)
        lines = [hdr, sep]

        for n in self.step_numbers:
            title = self._step_titles.get(n, f"step {n}")
            if len(title) > title_width:
                title = title[: title_width - 1] + "…"
            lat = latency.get(n)
            tok = tokens.get(n)
            n_runs = int(lat["n_runs"]) if lat else int((tok or {}).get("n_runs", 0))
            lat_cells = (
                f"{lat['p50']:>8.1f}  {lat['p95']:>8.1f}  {lat['p99']:>8.1f}  "
                f"{lat['mean']:>8.1f}  {lat['max']:>8.1f}"
                if lat
                else f"{'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}"
            )
            tok_cells = (
                f"{tok['p50']:>8.0f}  {tok['p95']:>8.0f}"
                if tok
                else f"{'-':>8}  {'-':>8}"
            )
            lines.append(
                f"{n:>3}  {title:<{title_width}}  {n_runs:>4}  "
                f"{lat_cells}  {tok_cells}"
            )

        lines.append(sep)
        lines.append(f"total traces: {self.n_traces}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML template for ``ExecutionTrace.to_html``
# ---------------------------------------------------------------------------
# Stand-alone HTML5 + inline CSS + vanilla JS. No external dependencies —
# safe to drop into a PR description, an email attachment, or a static
# file server. Curly braces are doubled because the template is consumed
# via ``str.format(events_json=...)``.

_ANIMATED_PLAYBACK_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CARL execution playback — {title}</title>
<style>
  :root {{
    --bg: #0f172a;
    --fg: #e2e8f0;
    --muted: #94a3b8;
    --card-bg: #1e293b;
    --card-border: #334155;
    --idle: #475569;
    --active: #3b82f6;
    --ok: #22c55e;
    --fail: #ef4444;
    --skip: #f59e0b;
    --hilite: #60a5fa;
  }}
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    background: var(--bg); color: var(--fg);
    display: grid; grid-template-rows: auto 1fr;
    height: 100vh;
  }}
  header {{
    padding: 14px 20px;
    background: #020617;
    border-bottom: 1px solid var(--card-border);
    display: flex; align-items: center; gap: 18px;
    flex-wrap: wrap;
  }}
  header h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
  header .meta {{ color: var(--muted); font-size: 13px; }}
  header .status.ok {{ color: var(--ok); font-weight: 600; }}
  header .status.fail {{ color: var(--fail); font-weight: 600; }}
  .controls {{ margin-left: auto; display: flex; gap: 8px; }}
  .controls button {{
    background: var(--card-bg); color: var(--fg);
    border: 1px solid var(--card-border); border-radius: 6px;
    padding: 6px 14px; font-size: 13px; cursor: pointer;
  }}
  .controls button:hover {{ background: #334155; }}
  .controls button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  main {{
    display: grid; grid-template-columns: 1fr 1fr;
    overflow: hidden;
  }}
  #steps {{
    padding: 20px; overflow: auto; border-right: 1px solid var(--card-border);
  }}
  .batch-label {{
    color: var(--muted); font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 16px 0 8px;
  }}
  .step {{
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-left: 4px solid var(--idle);
    border-radius: 6px;
    padding: 10px 14px; margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 220ms, background 220ms, transform 120ms;
  }}
  .step:hover {{ background: #2a3a52; }}
  .step.selected {{ outline: 2px solid var(--hilite); outline-offset: 1px; }}
  .step.state-active {{ border-left-color: var(--active); }}
  .step.state-active .badge {{ background: var(--active); color: white; }}
  .step.state-ok {{ border-left-color: var(--ok); }}
  .step.state-ok .badge {{ background: var(--ok); color: white; }}
  .step.state-fail {{ border-left-color: var(--fail); }}
  .step.state-fail .badge {{ background: var(--fail); color: white; }}
  .step.state-skip {{ border-left-color: var(--skip); }}
  .step.state-skip .badge {{ background: var(--skip); color: black; }}
  .step .title {{ font-weight: 600; }}
  .step .meta {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .badge {{
    display: inline-block; background: var(--idle); color: white;
    padding: 1px 8px; border-radius: 4px; font-size: 11px;
    margin-right: 8px; font-weight: 600;
  }}
  #detail {{
    padding: 20px; overflow: auto; background: #0b1426;
  }}
  #detail h2 {{ font-size: 16px; margin: 0 0 12px; }}
  #detail .kv {{ display: grid; grid-template-columns: 110px 1fr;
                  gap: 4px 12px; font-size: 13px; margin-bottom: 16px; }}
  #detail .kv dt {{ color: var(--muted); }}
  #detail pre {{
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: 4px; padding: 10px; font-size: 12px;
    overflow-x: auto; white-space: pre-wrap; word-break: break-word;
    margin: 0 0 12px;
  }}
  #detail .label {{ color: var(--muted); font-size: 11px;
                     text-transform: uppercase; letter-spacing: 0.05em;
                     margin-bottom: 4px; }}
  #detail .error {{ color: var(--fail); }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <span class="meta">{n_events} step(s) · {elapsed}</span>
  <span class="status {status_class}">{status_text}</span>
  <div class="controls">
    <button id="play">▶ Play</button>
    <button id="pause" disabled>⏸ Pause</button>
    <button id="step-btn">⏭ Step</button>
    <button id="reset">⟲ Reset</button>
  </div>
</header>
<main>
  <div id="steps"></div>
  <div id="detail">
    <h2>Select a step to inspect</h2>
    <p style="color: var(--muted); font-size: 13px;">
      Click any card on the left, or use ▶ Play to animate playback.
    </p>
  </div>
</main>
<script>
const EVENTS = {events_json};
const stepsEl = document.getElementById("steps");
const detailEl = document.getElementById("detail");
const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const stepBtn = document.getElementById("step-btn");
const resetBtn = document.getElementById("reset");

let cursor = -1;         // -1 = nothing played
let timer = null;
const cards = [];

function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => (
    {{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}}[c]
  ));
}}

function fmtTokens(t) {{
  if (!t) return "—";
  const p = t.prompt || 0, c = t.completion || 0, tot = t.total || (p + c);
  return tot + " (" + p + " in / " + c + " out)";
}}

function fmtTime(s) {{
  if (s == null) return "—";
  return s.toFixed(3) + "s";
}}

function batchOf(ev) {{ return ev.batch_index || 0; }}

// Build step cards grouped by batch
let lastBatch = -1;
EVENTS.forEach((ev, i) => {{
  if (batchOf(ev) !== lastBatch) {{
    lastBatch = batchOf(ev);
    const label = document.createElement("div");
    label.className = "batch-label";
    label.textContent = "Batch " + lastBatch;
    stepsEl.appendChild(label);
  }}
  const card = document.createElement("div");
  card.className = "step";
  card.dataset.idx = i;
  card.innerHTML = (
    '<div><span class="badge">#' + ev.step_number + '</span>' +
    '<span class="title">' + escapeHtml(ev.step_title) + '</span></div>' +
    '<div class="meta">' + escapeHtml(ev.step_type) +
    ' · ' + fmtTime(ev.execution_time) +
    ' · ' + fmtTokens(ev.token_usage) + '</div>'
  );
  card.addEventListener("click", () => selectStep(i));
  stepsEl.appendChild(card);
  cards.push(card);
}});

function setState(idx, state) {{
  const card = cards[idx];
  card.classList.remove("state-active", "state-ok", "state-fail", "state-skip");
  if (state) card.classList.add("state-" + state);
}}

function finalState(ev) {{
  if (ev.skipped) return "skip";
  return ev.success ? "ok" : "fail";
}}

function selectStep(idx) {{
  cards.forEach(c => c.classList.remove("selected"));
  if (idx < 0 || idx >= EVENTS.length) return;
  cards[idx].classList.add("selected");
  cards[idx].scrollIntoView({{block: "nearest"}});
  const ev = EVENTS[idx];
  const result_data_str = ev.result_data == null ? "—" : JSON.stringify(ev.result_data, null, 2);
  detailEl.innerHTML = (
    '<h2>#' + ev.step_number + ' — ' + escapeHtml(ev.step_title) + '</h2>' +
    '<dl class="kv">' +
      '<dt>type</dt><dd>' + escapeHtml(ev.step_type) + '</dd>' +
      '<dt>batch</dt><dd>' + batchOf(ev) + '</dd>' +
      '<dt>time</dt><dd>' + fmtTime(ev.execution_time) + '</dd>' +
      '<dt>tokens</dt><dd>' + fmtTokens(ev.token_usage) + '</dd>' +
      '<dt>status</dt><dd>' +
        (ev.skipped ? "skipped"
          : ev.success ? '<span style="color: var(--ok)">success</span>'
          : '<span class="error">failed</span>') +
      '</dd>' +
    '</dl>' +
    (ev.error_message ?
      ('<div class="label">error</div><pre class="error">' +
       escapeHtml(ev.error_message) + '</pre>') : '') +
    '<div class="label">result</div><pre>' + escapeHtml(ev.result || "(empty)") + '</pre>' +
    '<div class="label">result_data</div><pre>' + escapeHtml(result_data_str) + '</pre>' +
    '<div class="label">inputs</div><pre>' + escapeHtml(JSON.stringify(ev.inputs || {{}}, null, 2)) + '</pre>'
  );
}}

function advance() {{
  cursor += 1;
  if (cursor >= EVENTS.length) {{ stopTimer(); return; }}
  // Step within the current batch: light up as active first, then snap to
  // its final state after a short delay (parallel-batch illusion).
  setState(cursor, "active");
  const final = finalState(EVENTS[cursor]);
  setTimeout(() => setState(cursor, final), 500);
  selectStep(cursor);
}}

function startTimer() {{
  if (cursor >= EVENTS.length - 1) reset();
  playBtn.disabled = true; pauseBtn.disabled = false;
  timer = setInterval(advance, 900);
  advance();  // immediate first tick
}}

function stopTimer() {{
  if (timer) clearInterval(timer);
  timer = null;
  playBtn.disabled = false; pauseBtn.disabled = true;
}}

function reset() {{
  stopTimer();
  cursor = -1;
  cards.forEach((c, i) => setState(i, null));
}}

playBtn.addEventListener("click", startTimer);
pauseBtn.addEventListener("click", stopTimer);
stepBtn.addEventListener("click", () => {{ stopTimer(); advance(); }});
resetBtn.addEventListener("click", reset);
</script>
</body>
</html>
"""
