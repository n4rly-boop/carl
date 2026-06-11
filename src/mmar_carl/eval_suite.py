"""
``EvalSuite`` — benchmark a ``ReasoningChain`` against a dataset of expected outputs.

Builds on :class:`DatasetEvaluator` to support:

- Multiple metrics scored simultaneously per case.
- Built-in "compare to expected" metrics (``ExactMatchMetric``,
  ``ContainsMetric``, ``CaseInsensitiveMatchMetric``, ``RegexMatchMetric``)
  that score against ``DataCase.expected`` — the field
  ``DatasetEvaluator`` doesn't currently read.
- Per-metric pass rates, score distributions (min / median / p95 / max), and
  per-case detail rows.
- Cross-run regression detection: ``previous_report.diff(current_report)``
  surfaces score deltas plus regressed/improved/new/dropped cases.

Designed to coexist with the existing :class:`DatasetEvaluator` — same
``DataCase`` / dataset shapes, no breaking changes to that path.
"""

from __future__ import annotations

import re
import statistics
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel, Field

from .metrics import MetricBase
from .models.dataset import AbstractDataset, DataCase

if TYPE_CHECKING:
    from .chain import ReasoningChain
    from .models.context import ReasoningContext
    from .models.results import ReasoningResult, StepExecutionResult


ContextFactory = Callable[[DataCase], "ReasoningContext"]


# --------------------------------------------------------------------------- #
# Built-in "compare to expected" metrics
# --------------------------------------------------------------------------- #


def _extract_text(output: "ReasoningResult | StepExecutionResult") -> str:
    """Pull the comparable text out of a chain result or step result."""
    from .models.results import ReasoningResult  # local — avoid circular at import time

    if isinstance(output, ReasoningResult):
        return output.get_final_output() if output.success else ""
    return getattr(output, "result", "") or ""


class _ExpectedAwareMetric(MetricBase):
    """Base class for metrics that compare chain output against ``DataCase.expected``.

    Subclasses receive both the actual text and the expected text via
    ``_score(actual, expected)``. When the case carries no ``expected``
    value, the metric returns ``0.0`` (lenient — easy to spot in reports).
    """

    def __init__(self, *, name: Optional[str] = None) -> None:
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output: Any) -> float:
        # We don't have direct access to the case here — EvalSuite injects
        # the expected text via ``output.metadata["expected"]`` before passing
        # the ReasoningResult through (see EvalSuite._run_one).
        actual = _extract_text(output)
        expected = ""
        meta = getattr(output, "metadata", None)
        if isinstance(meta, dict):
            expected = str(meta.get("__eval_expected", "") or "")
        if not expected:
            return 0.0
        return float(self._score(actual, expected))

    def _score(self, actual: str, expected: str) -> float:
        raise NotImplementedError


class ExactMatchMetric(_ExpectedAwareMetric):
    """1.0 when the chain output matches ``case.expected`` exactly (after strip)."""

    def __init__(self) -> None:
        super().__init__(name="exact_match")

    def _score(self, actual: str, expected: str) -> float:
        return 1.0 if actual.strip() == expected.strip() else 0.0


class CaseInsensitiveMatchMetric(_ExpectedAwareMetric):
    """1.0 when stripped/lower-cased output matches ``case.expected``."""

    def __init__(self) -> None:
        super().__init__(name="case_insensitive_match")

    def _score(self, actual: str, expected: str) -> float:
        return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0


class ContainsMetric(_ExpectedAwareMetric):
    """1.0 when ``case.expected`` appears as a substring of the output (case-insensitive)."""

    def __init__(self) -> None:
        super().__init__(name="contains_expected")

    def _score(self, actual: str, expected: str) -> float:
        return 1.0 if expected.strip().lower() in actual.lower() else 0.0


class RegexMatchMetric(_ExpectedAwareMetric):
    """1.0 when ``case.expected`` is a regex that matches the output."""

    def __init__(self, flags: int = 0) -> None:
        super().__init__(name="regex_match")
        self._flags = flags

    def _score(self, actual: str, expected: str) -> float:
        try:
            pattern = re.compile(expected, self._flags)
        except re.error:
            return 0.0
        return 1.0 if pattern.search(actual) else 0.0


# --------------------------------------------------------------------------- #
# Result + report models
# --------------------------------------------------------------------------- #


class EvalCaseResult(BaseModel):
    """Per-case row in an :class:`EvalSuiteReport`."""

    case: DataCase
    chain_output: str
    success: bool
    execution_time: Optional[float] = None
    metric_scores: dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None


class MetricSummary(BaseModel):
    """Per-metric aggregate statistics across the dataset."""

    name: str
    count: int
    mean: float
    median: float
    p95: float
    min: float
    max: float
    pass_rate: float = Field(
        description=(
            "Fraction of cases whose score is >= the metric's pass_threshold "
            "(default 1.0 for binary 'compare to expected' metrics, else 0.5)."
        )
    )

    @classmethod
    def from_scores(cls, name: str, scores: list[float], pass_threshold: float) -> "MetricSummary":
        if not scores:
            return cls(
                name=name, count=0, mean=0.0, median=0.0, p95=0.0,
                min=0.0, max=0.0, pass_rate=0.0,
            )
        return cls(
            name=name,
            count=len(scores),
            mean=statistics.fmean(scores),
            median=statistics.median(scores),
            p95=_percentile(scores, 0.95),
            min=min(scores),
            max=max(scores),
            pass_rate=sum(1 for s in scores if s >= pass_threshold) / len(scores),
        )


def _percentile(values: list[float], q: float) -> float:
    """Inclusive percentile (no numpy dep)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = int(q * (len(sorted_vals) - 1))
    return sorted_vals[index]


class EvalSuiteReport(BaseModel):
    """Aggregated report from running an :class:`EvalSuite`."""

    cases: list[EvalCaseResult] = Field(default_factory=list)
    metric_summaries: dict[str, MetricSummary] = Field(default_factory=dict)
    pass_thresholds: dict[str, float] = Field(default_factory=dict)
    total_execution_time: Optional[float] = None

    # -- Convenience accessors ---------------------------------------------- #

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_success(self) -> int:
        return sum(1 for c in self.cases if c.success)

    @property
    def metric_names(self) -> list[str]:
        return list(self.metric_summaries.keys())

    def failing_cases(self, metric_name: str) -> list[EvalCaseResult]:
        """Cases whose ``metric_name`` score is below the pass threshold."""
        threshold = self.pass_thresholds.get(metric_name, 1.0)
        return [
            c for c in self.cases
            if c.metric_scores.get(metric_name, 0.0) < threshold
        ]

    # -- Pretty printing ---------------------------------------------------- #

    def format_summary(self) -> str:
        if not self.cases:
            return "(empty evaluation — no cases)"
        lines = [
            f"EvalSuite — {self.n_cases} cases, "
            f"{self.n_success} ran successfully "
            f"({self.n_success / self.n_cases * 100:.0f}%)",
            "",
            f"  {'metric':<26} {'pass':>5} {'mean':>7} {'median':>7} "
            f"{'p95':>7} {'min':>5} {'max':>5}",
            "  " + "-" * 72,
        ]
        for name, s in self.metric_summaries.items():
            lines.append(
                f"  {name[:26]:<26} {s.pass_rate * 100:>4.0f}% "
                f"{s.mean:>7.3f} {s.median:>7.3f} {s.p95:>7.3f} "
                f"{s.min:>5.2f} {s.max:>5.2f}"
            )
        return "\n".join(lines)

    def print_summary(self) -> None:
        print(self.format_summary())

    # -- Cross-run regression detection ------------------------------------- #

    def diff(self, other: "EvalSuiteReport") -> "EvalSuiteDiff":
        """Compare ``other`` (newer) to ``self`` (baseline).

        Returns a :class:`EvalSuiteDiff` with per-metric mean deltas and lists
        of regressed / improved / new / dropped case labels.
        """
        return EvalSuiteDiff.compute(baseline=self, current=other)


class EvalSuiteDiff(BaseModel):
    """Pairwise comparison between two :class:`EvalSuiteReport` runs."""

    metric_mean_deltas: dict[str, float] = Field(default_factory=dict)
    metric_pass_rate_deltas: dict[str, float] = Field(default_factory=dict)
    regressed_cases: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-metric: case labels whose score dropped from "
            "≥ pass_threshold (baseline) to < pass_threshold (current)."
        ),
    )
    improved_cases: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-metric: case labels whose score moved from "
            "< pass_threshold (baseline) to ≥ pass_threshold (current)."
        ),
    )
    new_case_labels: list[str] = Field(default_factory=list)
    dropped_case_labels: list[str] = Field(default_factory=list)

    @classmethod
    def compute(
        cls, *, baseline: "EvalSuiteReport", current: "EvalSuiteReport"
    ) -> "EvalSuiteDiff":
        diff = cls()
        # Metric-level deltas
        for name in set(baseline.metric_summaries) | set(current.metric_summaries):
            b_sum = baseline.metric_summaries.get(name)
            c_sum = current.metric_summaries.get(name)
            if b_sum and c_sum:
                diff.metric_mean_deltas[name] = c_sum.mean - b_sum.mean
                diff.metric_pass_rate_deltas[name] = c_sum.pass_rate - b_sum.pass_rate

        # Case-level regression / improvement
        baseline_by_label = {_case_key(c.case): c for c in baseline.cases}
        current_by_label = {_case_key(c.case): c for c in current.cases}
        diff.new_case_labels = sorted(set(current_by_label) - set(baseline_by_label))
        diff.dropped_case_labels = sorted(set(baseline_by_label) - set(current_by_label))

        for name in baseline.metric_summaries:
            threshold = baseline.pass_thresholds.get(name, 1.0)
            regressed: list[str] = []
            improved: list[str] = []
            for label, b_case in baseline_by_label.items():
                c_case = current_by_label.get(label)
                if c_case is None:
                    continue
                b_score = b_case.metric_scores.get(name, 0.0)
                c_score = c_case.metric_scores.get(name, 0.0)
                if b_score >= threshold and c_score < threshold:
                    regressed.append(label)
                elif b_score < threshold and c_score >= threshold:
                    improved.append(label)
            if regressed:
                diff.regressed_cases[name] = sorted(regressed)
            if improved:
                diff.improved_cases[name] = sorted(improved)
        return diff

    def has_regressions(self) -> bool:
        """True iff any metric has at least one regressed case."""
        return any(self.regressed_cases.values())


def _case_key(case: DataCase) -> str:
    """Stable identifier for a case (used for cross-run alignment)."""
    return case.label or case.input[:80]


# --------------------------------------------------------------------------- #
# EvalSuite
# --------------------------------------------------------------------------- #


class EvalSuite:
    """
    Benchmark a chain against a dataset using multiple metrics.

    Example::

        suite = EvalSuite(chain, dataset)
        suite.add_metric(ExactMatchMetric())
        suite.add_metric(ContainsMetric(), pass_threshold=1.0)
        suite.add_metric(LLMJudgeMetric(api, model="gpt-4o"), pass_threshold=0.7)
        report = await suite.run(lambda case: ReasoningContext(outer_context=case.input, api=client))
        report.print_summary()

    Args:
        chain: The :class:`ReasoningChain` under test.
        dataset: Dataset of :class:`DataCase` objects.
    """

    def __init__(
        self,
        chain: "ReasoningChain",
        dataset: AbstractDataset,
    ) -> None:
        self._chain = chain
        self._dataset = dataset
        self._metrics: list[MetricBase] = []
        self._pass_thresholds: dict[str, float] = {}

    def add_metric(self, metric: MetricBase, *, pass_threshold: float = 1.0) -> "EvalSuite":
        """Register a metric to score every case.

        Args:
            metric: A :class:`MetricBase` subclass instance.
            pass_threshold: Score below this counts as a "fail" for pass-rate
                accounting. Defaults to 1.0 — the right value for binary
                "compare to expected" metrics. Use 0.7 (or whatever) for graded
                metrics like an LLM judge.
        """
        self._metrics.append(metric)
        self._pass_thresholds[metric.name] = pass_threshold
        return self

    async def run(self, context_factory: ContextFactory) -> EvalSuiteReport:
        """Execute the chain on every case, score all metrics, aggregate.

        Cases are evaluated sequentially (same policy as
        :class:`DatasetEvaluator`) so downstream LLM rate limits aren't
        overwhelmed.
        """
        import time

        if not self._metrics:
            raise ValueError(
                "EvalSuite has no metrics — call add_metric() before run()."
            )

        cases = list(self._dataset)
        case_results: list[EvalCaseResult] = []
        per_metric_scores: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        suite_start = time.time()

        for case in cases:
            case_start = time.time()
            ctx = context_factory(case)
            try:
                result = await self._chain.execute_async(ctx)
                # Stash the case's expected output on result.metadata so the
                # built-in compare-to-expected metrics can read it. Done in a
                # private key to avoid clashing with caller metadata.
                if case.expected is not None:
                    result.metadata["__eval_expected"] = case.expected

                scores: dict[str, float] = {}
                for metric in self._metrics:
                    try:
                        score = float(await metric.compute_async(result))
                    except Exception:
                        score = 0.0
                    scores[metric.name] = score
                    per_metric_scores[metric.name].append(score)

                output_text = result.get_final_output() if result.success else ""
                case_results.append(
                    EvalCaseResult(
                        case=case,
                        chain_output=output_text,
                        success=result.success,
                        execution_time=time.time() - case_start,
                        metric_scores=scores,
                        error_message=None,
                    )
                )
            except Exception as exc:
                # Whole-case failure: every metric scores 0
                for metric in self._metrics:
                    per_metric_scores[metric.name].append(0.0)
                case_results.append(
                    EvalCaseResult(
                        case=case,
                        chain_output="",
                        success=False,
                        execution_time=time.time() - case_start,
                        metric_scores={m.name: 0.0 for m in self._metrics},
                        error_message=str(exc),
                    )
                )

        summaries = {
            m.name: MetricSummary.from_scores(
                m.name, per_metric_scores[m.name], self._pass_thresholds[m.name]
            )
            for m in self._metrics
        }
        return EvalSuiteReport(
            cases=case_results,
            metric_summaries=summaries,
            pass_thresholds=dict(self._pass_thresholds),
            total_execution_time=time.time() - suite_start,
        )
