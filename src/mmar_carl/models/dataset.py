"""
Dataset abstractions for batch evaluation in CARL.

Provides DataCase, AbstractDataset, SimpleDataset, selection strategies,
CaseEvaluationResult, and DatasetEvaluationReport for use with DatasetEvaluator.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Any, Iterator, Literal

from pydantic import BaseModel, Field, field_validator


class DataCase(BaseModel):
    """A single evaluation case for dataset-based chain assessment."""

    input: str = Field(..., description="Input data (outer_context) for chain execution")
    label: str | None = Field(
        default=None,
        description="Human-readable identifier for this case (e.g. 'case_01')",
    )
    expected: str | None = Field(
        default=None,
        description=(
            "Expected output for reference (optional, not used in metric computation)"
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional case-specific metadata",
    )


class AbstractDataset(ABC):
    """
    Abstract base class for evaluation datasets.

    Implement :meth:`__iter__` to yield :class:`DataCase` objects.
    Override :meth:`__len__` whenever the size is cheaply known.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[DataCase]: ...

    def __len__(self) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement __len__. "
            "Override it if the size is cheaply available."
        )


class SimpleDataset(AbstractDataset):
    """Concrete dataset backed by an in-memory list of :class:`DataCase` objects."""

    def __init__(self, cases: list[DataCase]) -> None:
        self._cases = list(cases)

    def __iter__(self) -> Iterator[DataCase]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)


class DataFrameDataset(AbstractDataset):
    """
    Dataset backed by a ``pandas.DataFrame``.

    Requires pandas to be installed (``pip install pandas`` or
    ``pip install mmar-carl[pandas]``).

    Args:
        df: Source DataFrame.
        input_col: Column name whose values become :attr:`DataCase.input`.
        label_col: Optional column to use as :attr:`DataCase.label`.
        expected_col: Optional column to use as :attr:`DataCase.expected`.
        metadata_cols: Additional columns to include in :attr:`DataCase.metadata`.

    Example::

        import pandas as pd
        from mmar_carl import DataFrameDataset

        df = pd.DataFrame({
            "text": ["analyse Q1 revenue", "summarise risks"],
            "id": ["q1", "risks"],
        })
        dataset = DataFrameDataset(df, input_col="text", label_col="id")
    """

    def __init__(
        self,
        df: Any,
        *,
        input_col: str = "input",
        label_col: str | None = None,
        expected_col: str | None = None,
        metadata_cols: list[str] | None = None,
    ) -> None:
        try:
            import pandas  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrameDataset. "
                "Install it with: pip install pandas  (or: pip install mmar-carl[pandas])"
            ) from exc

        if input_col not in df.columns:
            raise ValueError(f"Column '{input_col}' not found in DataFrame. Available: {list(df.columns)}")

        self._df = df
        self._input_col = input_col
        self._label_col = label_col
        self._expected_col = expected_col
        self._metadata_cols: list[str] = metadata_cols or []

    def __iter__(self) -> Iterator[DataCase]:
        for _, row in self._df.iterrows():
            meta = {col: row[col] for col in self._metadata_cols if col in self._df.columns}
            yield DataCase(
                input=str(row[self._input_col]),
                label=str(row[self._label_col]) if self._label_col is not None else None,
                expected=str(row[self._expected_col]) if self._expected_col is not None else None,
                metadata=meta,
            )

    def __len__(self) -> int:
        return len(self._df)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


class ThresholdStrategy(BaseModel):
    """
    Select cases where the metric score crosses a fixed threshold.

    With ``higher_is_better=True`` (default), selects cases where
    ``score < threshold``.  With ``higher_is_better=False``, selects cases
    where ``score > threshold``.
    """

    mode: Literal["threshold"] = "threshold"
    threshold: float = Field(..., description="Score boundary for case selection")
    higher_is_better: bool = Field(
        default=True,
        description=(
            "Metric direction.  True → lower scores are worse (select below threshold)."
            "  False → higher scores are worse (select above threshold)."
        ),
    )

    def select(
        self, results: "list[CaseEvaluationResult]"
    ) -> "list[CaseEvaluationResult]":
        """Return cases that fail the threshold criterion."""
        if not results:
            return []
        if self.higher_is_better:
            return [r for r in results if r.score < self.threshold]
        else:
            return [r for r in results if r.score > self.threshold]


class TopKWorstStrategy(BaseModel):
    """
    Select the *k* worst-scoring cases.

    When ``include_ties=True`` (default), all cases tied at the k-th boundary
    score are included, so the actual count may exceed *k*.
    """

    mode: Literal["top_k_worst"] = "top_k_worst"
    k: int = Field(..., gt=0, description="Number of worst cases to select (must be > 0)")
    higher_is_better: bool = Field(
        default=True,
        description="Metric direction.  Determines what 'worst' means.",
    )
    include_ties: bool = Field(
        default=True,
        description=(
            "Include all cases tied at the k-th boundary score.  "
            "Actual result count may be > k."
        ),
    )

    @field_validator("k")
    @classmethod
    def _k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("k must be greater than 0")
        return v

    def select(
        self, results: "list[CaseEvaluationResult]"
    ) -> "list[CaseEvaluationResult]":
        """Return the k worst cases (ascending score for higher_is_better metrics)."""
        if not results:
            return []

        # Sort worst-first
        sorted_results = sorted(
            results,
            key=lambda r: r.score,
            reverse=not self.higher_is_better,
        )

        if self.k >= len(sorted_results):
            return list(sorted_results)

        if self.include_ties:
            boundary_score = sorted_results[self.k - 1].score
            if self.higher_is_better:
                return [r for r in sorted_results if r.score <= boundary_score]
            else:
                return [r for r in sorted_results if r.score >= boundary_score]

        return sorted_results[: self.k]


# Discriminated union — Pydantic uses the `mode` literal to pick the right model.
SelectionStrategy = Annotated[
    ThresholdStrategy | TopKWorstStrategy,
    Field(discriminator="mode"),
]


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


class CaseEvaluationResult(BaseModel):
    """Result of running a :class:`~mmar_carl.chain.ReasoningChain` on a single :class:`DataCase`."""

    case: DataCase
    score: float = Field(..., description="Metric score for this case")
    chain_output: str = Field(..., description="Final output from chain execution")
    success: bool = Field(..., description="Whether the chain executed successfully")
    execution_time: float | None = Field(
        default=None, description="Chain execution time in seconds"
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Token usage for this case's chain run: "
            "``{'prompt': X, 'completion': Y, 'total': Z}``. Empty when the chain "
            "didn't record usage (e.g. mock LLM client, or non-LLM-only chain)."
        ),
    )
    llm_calls: int = Field(
        default=0,
        description="Number of LLM steps in this case's chain run that recorded token usage.",
    )
    step_outcomes: dict[int, str] = Field(
        default_factory=dict,
        description=(
            "Per-step outcome for this case keyed by ``step_number``: one of "
            "``\"success\"`` / ``\"failure\"`` / ``\"skipped\"``. Empty when "
            "the chain failed to execute at all (e.g. context_factory raised). "
            "Populated by :class:`DatasetEvaluator` from "
            "``ReasoningResult.step_results``. Used by "
            ":meth:`DatasetEvaluationReport.format_failure_heatmap`."
        ),
    )
    step_metrics: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Per-step metric scores for this case keyed by ``step_number``, "
            "then ``{metric_name: score}``. Populated from "
            "``StepExecutionResult.metrics`` when step-level metrics are "
            "attached to the chain. Used by "
            ":meth:`DatasetEvaluationReport.format_step_metric_heatmap` to "
            "render a cases × steps matrix for a single metric across the "
            "dataset."
        ),
    )
    step_latencies_ms: dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Per-step wall-clock latency for this case keyed by "
            "``step_number``, in milliseconds. Populated by "
            ":class:`DatasetEvaluator` from "
            "``StepExecutionResult.execution_time``. Used by "
            ":meth:`DatasetEvaluationReport.format_latency_histogram` to "
            "render an inline-sparkline histogram per step across the "
            "dataset (surfaces tail-latency outliers that a single "
            ":class:`~mmar_carl.execution_trace.ExecutionTrace` can't show)."
        ),
    )


class DatasetEvaluationReport(BaseModel):
    """
    Aggregated result of evaluating a dataset against a chain.

    Pass this to :class:`~mmar_carl.chain.ReflectionOptions` via the
    ``dataset_report`` field to include a dedicated problem-cases section in the
    reflection prompt:

    .. code-block:: python

        options = ReflectionOptions(dataset_report=report)
        reflection = chain.reflect("task description", options=options)

    Alternatively, use :meth:`to_reflection_dict` for the MVP path via
    ``extra_feedback``.
    """

    metric_name: str = Field(..., description="Name of the metric used for evaluation")
    strategy: SelectionStrategy = Field(..., description="Selection strategy applied")
    all_results: list[CaseEvaluationResult] = Field(
        ..., description="Evaluation results for every case in the dataset"
    )
    selected_cases: list[CaseEvaluationResult] = Field(
        ..., description="Subset of problem cases chosen by the strategy"
    )
    mean_score: float = Field(..., description="Mean metric score across all cases")
    min_score: float = Field(..., description="Minimum metric score observed")
    max_score: float = Field(..., description="Maximum metric score observed")

    def _repr_markdown_(self) -> str:
        """Rich-display protocol for Jupyter.

        Returns a Markdown summary: metric stats + failure-heatmap +
        score-distribution box plot (when enough cases for the latter
        to be informative).
        """
        n_cases = len(self.all_results)
        n_selected = len(self.selected_cases)
        n_failed = sum(1 for r in self.all_results if not r.success)
        lines = [
            f"**DatasetEvaluationReport** — metric: `{self.metric_name}` · "
            f"{n_cases} case{'s' if n_cases != 1 else ''}"
            f"{' (' + str(n_failed) + ' failed)' if n_failed else ''}"
            f" · selected: {n_selected}",
            "",
            f"- mean: **{self.mean_score:.3f}**  ·  "
            f"min: {self.min_score:.3f}  ·  max: {self.max_score:.3f}",
            "",
        ]
        if not self.all_results:
            return "\n".join(lines)

        # Failure heatmap when any per-step outcomes are recorded.
        if any(r.step_outcomes for r in self.all_results):
            lines.append("```text")
            lines.append(self.format_failure_heatmap())
            lines.append("```")
            lines.append("")

        # Score distribution when ≥2 cases (the formatter itself guards
        # the single-case / all-equal cases with a placeholder).
        if n_cases >= 2:
            lines.append("```text")
            lines.append(self.format_score_distribution())
            lines.append("```")
        return "\n".join(lines)

    def format_failure_heatmap(
        self,
        *,
        case_label_width: int = 18,
    ) -> str:
        """Render a per-case × per-step heatmap of step outcomes.

        Rows = cases, columns = step numbers. Each cell is one of:

        * ``✓`` — step succeeded
        * ``✗`` — step failed
        * ``·`` — step was skipped (conditional routing)
        * ``-`` — step didn't run (chain failed before reaching it)

        At the bottom, a `failures per step` summary row shows the absolute
        count of failures per step across all cases — surfaces whether
        failures cluster on a particular step (chain bug) or are scattered
        (data issue).

        Empty reports return a one-line placeholder.

        Args:
            case_label_width: Max characters for each case label
                (truncated with ``…`` suffix when overflowing).

        Returns:
            Printable string.
        """
        if not self.all_results:
            return "(no cases evaluated — nothing to chart)"

        # Collect the union of step numbers across all cases.
        all_steps: set[int] = set()
        for case_result in self.all_results:
            all_steps.update(case_result.step_outcomes.keys())
        if not all_steps:
            return "(no per-step outcomes recorded — chain may not have run any steps)"

        sorted_steps = sorted(all_steps)
        # Use 2-char-wide columns so the symbols + step numbers line up.
        col_width = max(2, len(str(sorted_steps[-1])) + 1)

        # Header row: step numbers
        header_cells = [f"{n:>{col_width}}" for n in sorted_steps]
        lines = [
            f"{'case':<{case_label_width}}  " + "".join(header_cells)
        ]
        lines.append(
            "-" * (case_label_width + 2 + col_width * len(sorted_steps))
        )

        # One row per case
        failure_counts: dict[int, int] = {n: 0 for n in sorted_steps}
        for idx, case_result in enumerate(self.all_results):
            label = case_result.case.label or f"case_{idx + 1}"
            short = (
                label[: case_label_width - 1] + "…"
                if len(label) > case_label_width
                else label
            )
            cells = []
            for n in sorted_steps:
                outcome = case_result.step_outcomes.get(n)
                if outcome == "success":
                    sym = "✓"
                elif outcome == "failure":
                    sym = "✗"
                    failure_counts[n] += 1
                elif outcome == "skipped":
                    sym = "·"
                else:
                    sym = "-"
                cells.append(f"{sym:>{col_width}}")
            lines.append(f"{short:<{case_label_width}}  " + "".join(cells))

        lines.append(
            "-" * (case_label_width + 2 + col_width * len(sorted_steps))
        )
        # Summary row: failures per step
        summary_cells = [f"{failure_counts[n]:>{col_width}}" for n in sorted_steps]
        lines.append(
            f"{'fails':<{case_label_width}}  " + "".join(summary_cells)
        )
        lines.append("")
        lines.append(
            f"legend:  ✓ success   ✗ failure   · skipped   - not run "
            f"({len(self.all_results)} cases × {len(sorted_steps)} steps)"
        )
        # Highlight any "always fails" step
        always_fail = [
            n for n in sorted_steps
            if failure_counts[n] == len(self.all_results)
            and failure_counts[n] > 0
        ]
        if always_fail:
            lines.append(
                f"⚠ step{'s' if len(always_fail) > 1 else ''} "
                f"{', '.join(str(n) for n in always_fail)} "
                f"failed in every case — likely a chain bug, not a data issue."
            )
        return "\n".join(lines)

    def format_step_metric_heatmap(
        self,
        metric_name: str,
        *,
        case_label_width: int = 18,
        scale_min: float | None = None,
        scale_max: float | None = None,
    ) -> str:
        """Render a per-case × per-step heatmap of a step-level metric.

        Rows = cases, columns = step numbers, cells = the named metric's
        score for that step in that case, mapped to a 5-level shade:

        * ``·`` for the bottom 20% of the score range
        * ``░`` 20–40%
        * ``▒`` 40–60%
        * ``▓`` 60–80%
        * ``█`` top 20%
        * ``-`` (blank) when the metric wasn't recorded for that cell

        A trailing per-step `mean` row helps spot steps with consistently
        weak scores. Empty reports or reports lacking the named metric
        return a one-line placeholder.

        Args:
            metric_name: Name of the metric (matches keys in
                ``StepExecutionResult.metrics``).
            case_label_width: Max characters for each case label
                (truncated with ``…`` suffix).
            scale_min: Floor for the colour scale. When ``None`` the
                observed minimum is used (autoranging — emphasises spread).
            scale_max: Ceiling for the colour scale. When ``None`` the
                observed maximum is used.

        Returns:
            Printable string.
        """
        if not self.all_results:
            return "(no cases evaluated — nothing to chart)"

        # Build (case, step) -> score mapping; collect step universe + values.
        cell_scores: dict[tuple[int, int], float] = {}
        all_steps: set[int] = set()
        observed: list[float] = []
        for case_idx, case_result in enumerate(self.all_results):
            for step_num, metric_map in case_result.step_metrics.items():
                if metric_name in metric_map:
                    value = float(metric_map[metric_name])
                    cell_scores[(case_idx, step_num)] = value
                    all_steps.add(step_num)
                    observed.append(value)

        if not observed:
            return (
                f"(metric '{metric_name}' not recorded on any step — "
                f"attach it to step descriptions to populate the heatmap)"
            )

        lo = scale_min if scale_min is not None else min(observed)
        hi = scale_max if scale_max is not None else max(observed)
        span = hi - lo if hi > lo else 1.0  # avoid div-by-zero

        def shade(v: float) -> str:
            # Map to bucket 0..4
            t = (v - lo) / span
            t = max(0.0, min(1.0, t))
            if t < 0.2:
                return "·"
            if t < 0.4:
                return "░"
            if t < 0.6:
                return "▒"
            if t < 0.8:
                return "▓"
            return "█"

        sorted_steps = sorted(all_steps)
        # 5 chars holds "0.99 " in the mean row; bigger step numbers widen it.
        col_width = max(5, len(str(sorted_steps[-1])) + 1)

        header_cells = [f"{n:>{col_width}}" for n in sorted_steps]
        lines = [
            f"{'case':<{case_label_width}}  " + "".join(header_cells)
        ]
        lines.append(
            "-" * (case_label_width + 2 + col_width * len(sorted_steps))
        )

        # Per-step sums + counts for the trailing mean row.
        step_sums: dict[int, float] = {n: 0.0 for n in sorted_steps}
        step_counts: dict[int, int] = {n: 0 for n in sorted_steps}

        for idx, case_result in enumerate(self.all_results):
            label = case_result.case.label or f"case_{idx + 1}"
            short = (
                label[: case_label_width - 1] + "…"
                if len(label) > case_label_width
                else label
            )
            cells = []
            for n in sorted_steps:
                v = cell_scores.get((idx, n))
                if v is None:
                    sym = "-"
                else:
                    sym = shade(v)
                    step_sums[n] += v
                    step_counts[n] += 1
                cells.append(f"{sym:>{col_width}}")
            lines.append(f"{short:<{case_label_width}}  " + "".join(cells))

        lines.append(
            "-" * (case_label_width + 2 + col_width * len(sorted_steps))
        )
        # Mean row: per-step average (only over cells that recorded a value).
        mean_cells = []
        for n in sorted_steps:
            if step_counts[n]:
                mean_v = step_sums[n] / step_counts[n]
                mean_cells.append(f"{mean_v:>{col_width}.2f}")
            else:
                mean_cells.append(f"{'-':>{col_width}}")
        lines.append(f"{'mean':<{case_label_width}}  " + "".join(mean_cells))

        lines.append("")
        lines.append(
            f"metric: '{metric_name}'   scale: [{lo:.2f} … {hi:.2f}]   "
            f"shades: · ░ ▒ ▓ █  (low → high)   "
            f"({len(self.all_results)} cases × {len(sorted_steps)} steps)"
        )
        return "\n".join(lines)

    def format_score_distribution(
        self,
        *,
        canvas_width: int = 40,
        format: str = "text",
    ) -> str:
        """Render a Unicode box plot of the chain-level metric scores.

        Visualises min / Q1 / median / Q3 / max across every case in the
        dataset, so reviewers can spot a long lower-tail or a bimodal
        distribution at a glance — a single mean number can't.

        Layout (40-char default canvas)::

            score distribution: metric='accuracy', 10 cases
            ├──────────╞══════════█═══════════╡──────────┤
            min=0.10  Q1=0.45  med=0.62  Q3=0.80  max=0.95

        Glyph legend:

        * ``├`` / ``┤`` — endpoints (min / max)
        * ``─`` — whiskers (min..Q1 and Q3..max)
        * ``╞`` / ``╡`` — box edges (Q1 and Q3)
        * ``═`` — box interior (Q1..median, median..Q3)
        * ``█`` — median

        Args:
            canvas_width: Width of the glyph canvas in characters
                (default 40). Wider canvases give sub-percentile precision
                in dense datasets; narrower ones are easier to embed in a
                log line.
            format: Only ``"text"`` is supported right now. Future
                matplotlib / PNG output will follow the dual-format
                pattern once a multi-metric variant is needed.

        Returns:
            Printable string. Empty reports return a placeholder; a
            single-case report returns its score with a "need ≥2 cases"
            note since quartiles need spread to be meaningful.
        """
        if format != "text":
            raise ValueError(
                f"Unknown format {format!r}. Only 'text' is supported "
                f"(see ``format_score_distribution`` docstring)."
            )
        if not self.all_results:
            return "(no cases evaluated — nothing to chart)"

        scores = sorted(r.score for r in self.all_results)
        n = len(scores)
        if n == 1:
            return (
                f"(only 1 case scored {scores[0]:.2f} — need ≥2 cases "
                f"for a distribution)"
            )

        # Inclusive linear-interpolation quantiles (matches numpy default).
        def _q(p: float) -> float:
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return scores[lo] + (scores[hi] - scores[lo]) * frac

        s_min = scores[0]
        s_max = scores[-1]
        q1 = _q(0.25)
        med = _q(0.5)
        q3 = _q(0.75)

        # All-equal scores: collapse to a single tick rather than dividing
        # by zero. Useful when every case scores 1.0 in a smoke run.
        if s_max <= s_min:
            return (
                f"score distribution: metric='{self.metric_name}', "
                f"{n} cases\n"
                f"(all scores equal: {s_min:.2f} — no spread to plot)"
            )

        span = s_max - s_min
        width = max(5, canvas_width)

        def _pos(v: float) -> int:
            return int(round((v - s_min) / span * (width - 1)))

        canvas = [" "] * width
        p_min, p_q1, p_med, p_q3, p_max = (
            _pos(s_min), _pos(q1), _pos(med), _pos(q3), _pos(s_max),
        )
        # Whisker: min..Q1
        for i in range(p_min, p_q1 + 1):
            canvas[i] = "─"
        # Box interior: Q1..Q3 (overwrites the whisker tick at Q1)
        for i in range(p_q1, p_q3 + 1):
            canvas[i] = "═"
        # Whisker: Q3..max
        for i in range(p_q3, p_max + 1):
            canvas[i] = "─"
        # Repaint box edges + median over the spans we just drew.
        canvas[p_q3] = "╡"
        canvas[p_q1] = "╞"
        canvas[p_med] = "█"
        # Endcaps last so they win over an adjacent box edge.
        canvas[p_min] = "├"
        canvas[p_max] = "┤"

        lines = [
            f"score distribution: metric='{self.metric_name}', {n} cases",
            "".join(canvas),
            (
                f"min={s_min:.2f}  Q1={q1:.2f}  med={med:.2f}  "
                f"Q3={q3:.2f}  max={s_max:.2f}"
            ),
        ]
        return "\n".join(lines)

    def format_latency_histogram(
        self,
        *,
        bins: int = 16,
        title_width: int = 18,
        format: str = "text",
    ) -> str:
        """Render a per-step latency histogram across the dataset.

        For each step that has recorded latencies, builds a histogram of
        per-case wall-clock times and renders it as an inline Unicode
        sparkline (``▁▂▃▄▅▆▇█``). Surfaces tail-latency outliers and
        bimodal distributions that a single mean/p95 number can't show.

        Layout::

            step           runs  hist (N bins)            p50    p95    max
            ─────────────  ────  ──────────────────────  ─────  ─────  ─────
            1: Outline       5   ▂▁▁█▁▁▁▁▁▂▁▁▁▁▁▁         700   1340   1500
            2: Synth         5   ▁██▁▁▁▁▁▁▁▁▁▁▁▁▁        1300   2520   2800

        Args:
            bins: Number of histogram bins per step (default 16). More
                bins make the sparkline finer but need more runs to
                avoid noise.
            title_width: Max chars for each step label (truncated with
                ``…``).
            format: Only ``"text"`` is supported. Matplotlib output can
                land later behind the ``[viz]`` extra.

        Returns:
            Printable string. Empty / no-latency-data reports return a
            one-line placeholder.
        """
        if format != "text":
            raise ValueError(
                f"Unknown format {format!r}. Only 'text' is supported."
            )
        if not self.all_results:
            return "(no cases evaluated — nothing to chart)"

        # Build per-step lists of latencies (ms).
        per_step: dict[int, list[float]] = {}
        for cr in self.all_results:
            for step_num, ms in cr.step_latencies_ms.items():
                per_step.setdefault(step_num, []).append(float(ms))
        if not per_step:
            return (
                "(no per-step latency data recorded — chain may not have "
                "executed any steps with timing)"
            )

        # Carry titles from the first case's step_outcomes / step_metrics
        # — there's no direct title field on CaseEvaluationResult, so fall
        # back to "step N" when we don't have one.
        # (Most callers will pre-print the chain definition above this.)

        blocks = " ▁▂▃▄▅▆▇█"
        n_levels = len(blocks) - 1  # 8 height levels (0 → " ", 1..8 → blocks)

        def _percentile(sorted_vals: list[float], p: float) -> float:
            if not sorted_vals:
                return 0.0
            n = len(sorted_vals)
            if n == 1:
                return sorted_vals[0]
            idx = p * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac

        hdr = (
            f"{'#':>3}  {'step':<{title_width}}  {'n':>4}  "
            f"hist ({bins} bins){'':<{max(0, bins - 10)}}  "
            f"{'p50 ms':>7}  {'p95 ms':>7}  {'max ms':>7}"
        )
        sep = "-" * len(hdr)
        lines = [hdr, sep]

        for step_num in sorted(per_step.keys()):
            vals = per_step[step_num]
            vals_sorted = sorted(vals)
            v_min = vals_sorted[0]
            v_max = vals_sorted[-1]
            span = v_max - v_min if v_max > v_min else 1.0
            # Bin
            counts = [0] * bins
            for v in vals:
                # Last bin is inclusive of the max.
                idx = int((v - v_min) / span * bins)
                idx = min(idx, bins - 1)
                counts[idx] += 1
            max_count = max(counts) if counts else 0
            spark = "".join(
                blocks[round(c / max_count * n_levels)] if max_count else " "
                for c in counts
            )
            # Title: try to recover from the first case that recorded it.
            title = f"step {step_num}"
            for cr in self.all_results:
                if step_num in cr.step_outcomes:
                    # Best-effort: outcomes don't carry titles, so leave
                    # the generic label here. The caller usually knows.
                    break
            if len(title) > title_width:
                title = title[: title_width - 1] + "…"

            p50 = _percentile(vals_sorted, 0.5)
            p95 = _percentile(vals_sorted, 0.95)
            lines.append(
                f"{step_num:>3}  {title:<{title_width}}  {len(vals):>4}  "
                f"{spark:<{max(bins, 10)}}  "
                f"{p50:>7.0f}  {p95:>7.0f}  {v_max:>7.0f}"
            )

        lines.append(sep)
        lines.append(
            f"total cases: {len(self.all_results)}   "
            f"steps with timing: {len(per_step)}"
        )
        return "\n".join(lines)

    def format_cost_trend(
        self,
        *,
        pricing: dict[str, tuple[float, float]] | None = None,
        default_model: str | None = None,
        sparkline_width: int = 40,
        regression_factor: float = 2.0,
        format: str = "text",
    ) -> str:
        """Render a per-run cost / token trend sparkline.

        Lets users running the same chain N times (e.g. in
        :class:`DatasetEvaluator`) see how cost-per-run drifts across
        the dataset — useful for catching prompt regressions that
        sneakily inflate spend.

        Layout::

            cost-per-run trend: 8 runs, 1 model
            ▁▂▁█▂▁▂▁
            run 1: tokens=933 cost=$0.0001
            ...
            run 8: tokens=854 cost=$0.0001
            median tokens=890 cost=$0.0001
            ⚠ regression detected at run 4: cost=$0.0003 (median × 3.0)

        Args:
            pricing: Optional ``{model: (in_per_1k, out_per_1k)}`` map.
                When supplied, costs are computed via prompt+completion
                token usage and ``default_model`` (or each step's
                attributed model — but we only have chain-level usage
                here, so ``default_model`` is the primary key). When
                missing, the trend uses **total tokens** as a cost
                proxy and the legend prints `cost=—`.
            default_model: Model name used as the pricing key when
                ``pricing`` is provided. Falls back to the first
                non-empty token-usage model recorded in the report.
            sparkline_width: Max characters in the trend sparkline
                (capped to one block per run; long runs are
                downsampled by truncation, not averaging — keeps the
                outlier visible).
            regression_factor: A run is flagged as a regression when
                its cost exceeds ``factor × median``. Default 2.0.
            format: Only ``"text"`` is supported right now.

        Returns:
            Printable string. Empty / no-usage reports return a
            one-line placeholder.
        """
        if format != "text":
            raise ValueError(
                f"Unknown format {format!r}. Only 'text' is supported."
            )
        if not self.all_results:
            return "(no runs evaluated — nothing to chart)"

        # Build per-run (tokens, cost) tuples.
        per_run: list[tuple[int, int, int, float | None]] = []
        # (idx, total_tokens, prompt_tokens+completion_tokens, cost_or_None)
        for idx, cr in enumerate(self.all_results):
            usage = cr.token_usage or {}
            total = int(usage.get("total", 0))
            prompt_t = int(usage.get("prompt", 0))
            compl_t = int(usage.get("completion", 0))
            cost: float | None = None
            if pricing and default_model and default_model in pricing:
                in_price, out_price = pricing[default_model]
                cost = (prompt_t / 1000.0) * in_price + (compl_t / 1000.0) * out_price
            per_run.append((idx, total, prompt_t + compl_t, cost))

        # Strip runs with zero token usage (mock clients, hard-failure)
        # so the sparkline isn't dominated by empty bars.
        usable = [r for r in per_run if r[1] > 0]
        if not usable:
            return (
                "(no per-run token usage recorded — chain may not have "
                "called any LLM)"
            )

        # Sparkline source: cost when available, else total tokens.
        values: list[float] = [
            (r[3] if r[3] is not None else float(r[1])) for r in usable
        ]
        v_min, v_max = min(values), max(values)
        span = v_max - v_min if v_max > v_min else 1.0
        blocks = " ▁▂▃▄▅▆▇█"
        n_levels = len(blocks) - 1

        # Downsample if N > sparkline_width: keep one slot per run, drop
        # the leading runs (most recent are most useful for regression).
        rendered = usable[-sparkline_width:] if len(usable) > sparkline_width else usable
        rendered_vals = [
            (r[3] if r[3] is not None else float(r[1])) for r in rendered
        ]
        # Floor the lowest non-empty value at ▁ rather than " " so every
        # run is visible — without this floor, an outlier dominates the
        # scale and the other runs render as a blank stretch.
        spark = "".join(
            blocks[max(1, round((v - v_min) / span * n_levels))]
            for v in rendered_vals
        )

        # Median for regression-check.
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            median = sorted_vals[n // 2]
        else:
            median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

        has_cost = pricing is not None and default_model in (pricing or {})

        lines = [
            f"cost-per-run trend: {len(usable)} runs"
            + (f", model={default_model}" if has_cost else "")
        ]
        lines.append(spark)
        for idx, total, _, cost in rendered:
            label = self.all_results[idx].case.label or f"case_{idx + 1}"
            cost_cell = f"${cost:.4f}" if cost is not None else "—"
            lines.append(
                f"run {idx + 1} [{label}]: tokens={total} cost={cost_cell}"
            )
        if has_cost:
            lines.append(f"median cost=${median:.4f}")
        else:
            lines.append(f"median tokens={median:.0f}")

        # Regression flags: any run whose value exceeds factor × median.
        regressions: list[tuple[int, float]] = []
        for idx, _, _, cost in usable:
            v = cost if cost is not None else float(per_run[idx][1])
            if median > 0 and v > regression_factor * median:
                regressions.append((idx, v))
        if regressions:
            for idx, v in regressions:
                unit = "cost" if has_cost else "tokens"
                shown = f"${v:.4f}" if has_cost else f"{int(v)}"
                ratio = v / median if median else 0.0
                lines.append(
                    f"⚠ regression detected at run {idx + 1}: "
                    f"{unit}={shown} (median × {ratio:.1f})"
                )

        return "\n".join(lines)

    def to_reflection_dict(self, max_preview_chars: int = 200) -> dict[str, Any]:
        """
        Format the report as a plain dict suitable for
        ``ReflectionOptions(extra_feedback=report.to_reflection_dict())``.

        Prefer the ``dataset_report`` field on :class:`ReflectionOptions` for
        richer formatting in the prompt.
        """
        return {
            "dataset_evaluation_metric": self.metric_name,
            "dataset_evaluation_stats": (
                f"total={len(self.all_results)}, "
                f"mean={round(self.mean_score, 3)}, "
                f"min={round(self.min_score, 3)}, "
                f"max={round(self.max_score, 3)}"
            ),
            "dataset_problem_cases": "; ".join(
                f"[{r.case.label or f'case_{i + 1}'}] "
                f"score={round(r.score, 3)} "
                f"input={r.case.input[:max_preview_chars]!r} "
                f"output={r.chain_output[:max_preview_chars]!r}"
                for i, r in enumerate(self.selected_cases)
            )
            or "none",
            "dataset_overfitting_warning": (
                "Optimize for patterns visible across problem cases, "
                "not individual quirks. Mean score provides baseline context."
            ),
        }
