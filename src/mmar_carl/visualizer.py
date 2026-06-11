"""
ChainVisualizer — chainable facade over all CARL visualization methods.

With individual viz methods spread across
``ReasoningResult``, ``ExecutionTrace``, ``ReasoningChain``, and
``EvolutionResult``, users had to remember which method lives where::

    result.format_token_pie()
    result.format_profiling_table()
    result.format_prompt_completion_breakdown()
    result.trace.format_gantt()
    chain.to_mermaid_critical_path(result)
    chain.to_mermaid_heatmap(result, metric="tokens")
    evo_result.format_score_evolution()
    evo_result.format_spend_vs_quality()

``ChainVisualizer`` consolidates these into one fluent API::

    ChainVisualizer(result=result, chain=chain)
        .token_pie()
        .gantt()
        .profiling_table()
        .print()

Each builder method (a) lazily renders its view by delegating to the
underlying source class, (b) appends the result + an `Optional`
section title to an internal buffer, and (c) returns ``self`` for
chaining. ``.print()`` outputs the buffer to stdout with section
headers; ``.render() -> str`` returns the bundle as a string (the test
surface).
"""

from __future__ import annotations

from typing import Any, Optional


__all__ = ["ChainVisualizer"]


class ChainVisualizer:
    """Chainable facade over all CARL visualization methods.

    Parameters
    ----------
    result:
        Required. The ``ReasoningResult`` to visualize.
    chain:
        Optional. The originating ``ReasoningChain`` — required for
        DAG-shaped views (:meth:`dag`, :meth:`critical_path`, :meth:`heatmap`).
    evolution_result:
        Optional. An ``EvolutionResult`` for evolution-specific views
        (:meth:`score_evolution`, :meth:`spend_vs_quality`).
    """

    def __init__(
        self,
        result: Optional[Any] = None,
        *,
        chain: Optional[Any] = None,
        evolution_result: Optional[Any] = None,
    ) -> None:
        self._result = result
        self._chain = chain
        self._evolution_result = evolution_result
        # Each entry: (section_title, rendered_text)
        self._views: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Builder methods — one per visualization
    # ------------------------------------------------------------------

    def token_pie(self, *, format: str = "text", **kwargs: Any) -> "ChainVisualizer":
        """Append a per-step token spend pie.

        Forwards to ``ReasoningResult.format_token_pie``. Requires the
        visualizer to have been constructed with a ``result=``.
        """
        self._require("token_pie", "result")
        rendered = self._result.format_token_pie(format=format, **kwargs)
        self._views.append((f"Token pie ({format})", rendered))
        return self

    def prompt_completion(self, **kwargs: Any) -> "ChainVisualizer":
        """Append a per-step prompt vs completion stacked bar."""
        self._require("prompt_completion", "result")
        rendered = self._result.format_prompt_completion_breakdown(**kwargs)
        self._views.append(("Prompt vs completion", rendered))
        return self

    def profiling_table(
        self,
        *,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        default_model: Optional[str] = None,
        **kwargs: Any,
    ) -> "ChainVisualizer":
        """Append the per-step profiling table.

        Forwards to ``ReasoningResult.format_profiling_table``. Passes
        ``pricing`` + ``default_model`` through for cost columns.
        """
        self._require("profiling_table", "result")
        rendered = self._result.format_profiling_table(
            pricing=pricing, default_model=default_model, **kwargs
        )
        self._views.append(("Profiling table", rendered))
        return self

    def gantt(self, *, format: str = "text", **kwargs: Any) -> "ChainVisualizer":
        """Append a Gantt chart from the result's trace.

        Requires the result to have a ``trace`` attribute populated by
        the executor (true for all chains run through ``DAGExecutor``).
        """
        self._require("gantt", "result")
        trace = getattr(self._result, "trace", None)
        if trace is None:
            self._views.append(
                ("Gantt", "(no trace attached to result — gantt unavailable)")
            )
            return self
        rendered = trace.format_gantt(format=format, **kwargs)
        self._views.append((f"Gantt ({format})", rendered))
        return self

    def dag(self) -> "ChainVisualizer":
        """Append the chain's DAG as a Mermaid diagram (pre-existing
        ``ReasoningChain.to_mermaid``)."""
        self._require("dag", "chain")
        rendered = self._chain.to_mermaid()
        self._views.append(("DAG (mermaid)", rendered))
        return self

    def critical_path(self) -> "ChainVisualizer":
        """Append the DAG with critical-path highlighting."""
        self._require("critical_path", "chain", "result")
        rendered = self._chain.to_mermaid_critical_path(self._result)
        self._views.append(("Critical path (mermaid)", rendered))
        return self

    def heatmap(self, *, metric: str = "latency", **kwargs: Any) -> "ChainVisualizer":
        """Append the DAG with per-node heatmap colouring.

        ``metric`` is one of ``"latency"``, ``"tokens"``, ``"cost"``.
        """
        self._require("heatmap", "chain", "result")
        rendered = self._chain.to_mermaid_heatmap(self._result, metric=metric, **kwargs)
        self._views.append((f"Heatmap: {metric} (mermaid)", rendered))
        return self

    def score_evolution(self, *, format: str = "text", **kwargs: Any) -> "ChainVisualizer":
        """Append the per-generation score evolution chart."""
        self._require("score_evolution", "evolution_result")
        rendered = self._evolution_result.format_score_evolution(format=format, **kwargs)
        self._views.append((f"Score evolution ({format})", rendered))
        return self

    def spend_vs_quality(self, *, format: str = "text", **kwargs: Any) -> "ChainVisualizer":
        """Append the cumulative-spend vs best-score curve."""
        self._require("spend_vs_quality", "evolution_result")
        rendered = self._evolution_result.format_spend_vs_quality(format=format, **kwargs)
        self._views.append((f"Spend vs quality ({format})", rendered))
        return self

    # ------------------------------------------------------------------
    # Buffer access
    # ------------------------------------------------------------------

    def render(self, *, separator: str = "\n\n") -> str:
        """Return the accumulated buffer as a single string.

        Each section is prefixed with a header line
        ``=== <section title> ===`` so the output is readable even when
        captured to a log file.
        """
        if not self._views:
            return "(no views accumulated — chain a builder method before render())"
        parts: list[str] = []
        for title, body in self._views:
            parts.append(f"=== {title} ===\n{body}")
        return separator.join(parts)

    def _repr_markdown_(self) -> str:
        """Rich-display protocol for Jupyter.

        Renders each accumulated view as its own Markdown section
        (H3 title + fenced code block, with Mermaid sections detected
        by the ``mermaid`` substring so they render as diagrams rather
        than verbatim text in Jupyter / GitHub / nbviewer).
        """
        if not self._views:
            return (
                "_(no views accumulated — chain a builder method before "
                "evaluating in a notebook cell)_"
            )
        parts: list[str] = []
        for title, body in self._views:
            # Heuristic: a Mermaid block typically starts with one of these
            # directive words. We open a ```mermaid fence so notebook
            # viewers render the diagram natively.
            stripped = body.lstrip()
            mermaid_starts = (
                "flowchart", "graph", "pie", "gantt", "sequenceDiagram",
                "classDiagram", "stateDiagram", "erDiagram", "journey",
                "xychart",
            )
            fence = "mermaid" if any(
                stripped.startswith(d) for d in mermaid_starts
            ) else "text"
            parts.append(f"### {title}\n\n```{fence}\n{body}\n```")
        return "\n\n".join(parts)

    def print(self) -> "ChainVisualizer":
        """Print the accumulated buffer to stdout. Returns self for chaining."""
        print(self.render())
        return self

    def clear(self) -> "ChainVisualizer":
        """Reset the accumulated buffer. Returns self for chaining."""
        self._views = []
        return self

    @property
    def view_titles(self) -> list[str]:
        """Section titles of accumulated views, in insertion order. Read-only."""
        return [title for title, _ in self._views]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require(self, method: str, *required: str) -> None:
        """Validate that the required attributes were supplied at construction."""
        attr_map = {
            "result": ("_result", "result"),
            "chain": ("_chain", "chain"),
            "evolution_result": ("_evolution_result", "evolution_result"),
        }
        missing = []
        for key in required:
            attr_name, kwarg_name = attr_map[key]
            if getattr(self, attr_name) is None:
                missing.append(kwarg_name)
        if missing:
            raise ValueError(
                f"ChainVisualizer.{method}() requires {', '.join(missing)} "
                f"to be supplied to the constructor. Got: "
                f"result={self._result is not None}, "
                f"chain={self._chain is not None}, "
                f"evolution_result={self._evolution_result is not None}."
            )
