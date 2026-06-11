"""
Tests for ``EvalSuite``.

Covers the multi-metric runner, the "compare to expected" metric family,
aggregate statistics (pass rate / median / p95), failing-case extraction,
and cross-run regression diffs.
"""

import pytest

from mmar_carl import (
    CaseInsensitiveMatchMetric,
    ContainsMetric,
    DataCase,
    EvalCaseResult,
    EvalSuite,
    EvalSuiteReport,
    ExactMatchMetric,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    MetricSummary,
    ReasoningChain,
    ReasoningContext,
    RegexMatchMetric,
    SimpleDataset,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


class _ScriptedLLM(LLMClientBase):
    """Returns a sequence of canned responses one per call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.i = 0

    async def get_response(self, prompt: str) -> str:
        if self.i >= len(self.responses):
            return self.responses[-1] if self.responses else ""
        out = self.responses[self.i]
        self.i += 1
        return out

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _basic_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[LLMStepDescription(number=1, title="x", aim="x")],
        max_workers=1,
    )


def _factory(api: LLMClientBase):
    def make(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=api)
    return make


# --------------------------------------------------------------------------- #
# Built-in metric scoring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_exact_match_metric_scores_correctly() -> None:
    api = _ScriptedLLM(responses=["paris", "London", "  rome  "])
    dataset = SimpleDataset([
        DataCase(input="france?", expected="paris", label="fr"),
        DataCase(input="uk?",     expected="london", label="uk"),  # case-mismatch
        DataCase(input="italy?",  expected="rome",   label="it"),  # whitespace forgiven
    ])
    chain = _basic_chain()
    suite = EvalSuite(chain, dataset).add_metric(ExactMatchMetric())
    report = await suite.run(_factory(api))
    scores = {c.case.label: c.metric_scores["exact_match"] for c in report.cases}
    assert scores == {"fr": 1.0, "uk": 0.0, "it": 1.0}


@pytest.mark.asyncio
async def test_case_insensitive_match() -> None:
    api = _ScriptedLLM(responses=["PARIS"])
    dataset = SimpleDataset([DataCase(input="x", expected="paris", label="a")])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(CaseInsensitiveMatchMetric())
    report = await suite.run(_factory(api))
    assert report.cases[0].metric_scores["case_insensitive_match"] == 1.0


@pytest.mark.asyncio
async def test_contains_metric() -> None:
    api = _ScriptedLLM(responses=["The capital is Paris.", "I don't know."])
    dataset = SimpleDataset([
        DataCase(input="france?", expected="paris", label="fr"),
        DataCase(input="x?",      expected="rome",  label="rm"),
    ])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(ContainsMetric())
    report = await suite.run(_factory(api))
    scores = {c.case.label: c.metric_scores["contains_expected"] for c in report.cases}
    assert scores == {"fr": 1.0, "rm": 0.0}


@pytest.mark.asyncio
async def test_regex_match_metric() -> None:
    api = _ScriptedLLM(responses=["answer is 42", "no number here"])
    dataset = SimpleDataset([
        DataCase(input="x", expected=r"\d+", label="num"),
        DataCase(input="y", expected=r"\d+", label="nonum"),
    ])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(RegexMatchMetric())
    report = await suite.run(_factory(api))
    scores = {c.case.label: c.metric_scores["regex_match"] for c in report.cases}
    assert scores == {"num": 1.0, "nonum": 0.0}


@pytest.mark.asyncio
async def test_expected_aware_metric_with_no_expected_returns_zero() -> None:
    """Cases without ``expected`` set score 0.0 — lenient by design."""
    api = _ScriptedLLM(responses=["whatever"])
    dataset = SimpleDataset([DataCase(input="x", expected=None, label="x")])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(ExactMatchMetric())
    report = await suite.run(_factory(api))
    assert report.cases[0].metric_scores["exact_match"] == 0.0


# --------------------------------------------------------------------------- #
# Custom metric integration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_custom_metric_scored_alongside_built_ins() -> None:
    """A custom metric subclass works the same as the built-ins."""

    class _LengthMetric(MetricBase):
        @property
        def name(self) -> str:
            return "len"

        async def compute_async(self, output) -> float:
            return float(len(output.get_final_output()))

    api = _ScriptedLLM(responses=["short", "much-longer-output"])
    dataset = SimpleDataset([
        DataCase(input="a", expected="short", label="a"),
        DataCase(input="b", expected="much-longer-output", label="b"),
    ])
    suite = (EvalSuite(_basic_chain(), dataset)
             .add_metric(ExactMatchMetric())
             .add_metric(_LengthMetric(), pass_threshold=10.0))
    report = await suite.run(_factory(api))
    # Length should differ
    assert report.cases[0].metric_scores["len"] == 5.0
    assert report.cases[1].metric_scores["len"] == 18.0
    # Pass rate against threshold=10
    assert report.metric_summaries["len"].pass_rate == 0.5


# --------------------------------------------------------------------------- #
# Aggregate statistics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_metric_summary_aggregates() -> None:
    api = _ScriptedLLM(responses=["a", "b", "c", "d", "e"])
    cases = [DataCase(input=f"in-{i}", expected=ch, label=f"c{i}")
             for i, ch in enumerate("abcde")]
    dataset = SimpleDataset(cases)
    suite = EvalSuite(_basic_chain(), dataset).add_metric(ExactMatchMetric())
    report = await suite.run(_factory(api))
    s = report.metric_summaries["exact_match"]
    # All 5 responses match → pass rate 100%, all 1.0
    assert s.count == 5
    assert s.mean == 1.0
    assert s.median == 1.0
    assert s.pass_rate == 1.0
    assert s.min == 1.0
    assert s.max == 1.0


@pytest.mark.asyncio
async def test_metric_summary_pass_rate_with_threshold() -> None:
    """Custom pass_threshold drives pass-rate calculation."""

    class _Grade(MetricBase):
        @property
        def name(self) -> str:
            return "grade"

        async def compute_async(self, output) -> float:
            # Output is "0.9" / "0.5" / "0.2" — parse it
            try:
                return float(output.get_final_output().strip())
            except ValueError:
                return 0.0

    # LLM echoes the input back as the answer
    api = _ScriptedLLM(responses=["0.9", "0.5", "0.2"])
    dataset = SimpleDataset([
        DataCase(input="0.9", label="hi"),
        DataCase(input="0.5", label="mid"),
        DataCase(input="0.2", label="lo"),
    ])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(_Grade(), pass_threshold=0.7)
    report = await suite.run(_factory(api))
    # Only the 0.9 case crosses pass_threshold=0.7 → 1/3
    assert report.metric_summaries["grade"].pass_rate == pytest.approx(1 / 3)
    assert report.metric_summaries["grade"].mean == pytest.approx(
        (0.9 + 0.5 + 0.2) / 3
    )


# --------------------------------------------------------------------------- #
# Failing-case extraction
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_failing_cases_returns_below_threshold() -> None:
    api = _ScriptedLLM(responses=["paris", "wrong", "rome"])
    dataset = SimpleDataset([
        DataCase(input="fr", expected="paris", label="fr"),
        DataCase(input="uk", expected="london", label="uk"),
        DataCase(input="it", expected="rome", label="it"),
    ])
    suite = EvalSuite(_basic_chain(), dataset).add_metric(ExactMatchMetric())
    report = await suite.run(_factory(api))
    failing = report.failing_cases("exact_match")
    assert [c.case.label for c in failing] == ["uk"]


# --------------------------------------------------------------------------- #
# Whole-case failure
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_whole_case_exception_scores_zero_for_all_metrics() -> None:
    class _BoomLLM(LLMClientBase):
        async def get_response(self, p):
            raise RuntimeError("boom")

        async def get_response_with_retries(self, p, retries=3):
            raise RuntimeError("boom")

    dataset = SimpleDataset([
        DataCase(input="x", expected="y", label="x"),
    ])
    suite = (EvalSuite(_basic_chain(), dataset)
             .add_metric(ExactMatchMetric())
             .add_metric(ContainsMetric()))
    report = await suite.run(_factory(_BoomLLM()))
    case = report.cases[0]
    # Chain itself catches the exception and marks the step failed — but the
    # *case* still has a record with scores=0
    assert case.metric_scores["exact_match"] == 0.0
    assert case.metric_scores["contains_expected"] == 0.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_without_metrics_raises() -> None:
    suite = EvalSuite(_basic_chain(), SimpleDataset([DataCase(input="x")]))
    with pytest.raises(ValueError, match="no metrics"):
        await suite.run(_factory(_ScriptedLLM(responses=["x"])))


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_format_summary_contains_per_metric_rows() -> None:
    api = _ScriptedLLM(responses=["paris"])
    dataset = SimpleDataset([DataCase(input="fr", expected="paris", label="fr")])
    suite = (EvalSuite(_basic_chain(), dataset)
             .add_metric(ExactMatchMetric())
             .add_metric(ContainsMetric()))
    report = await suite.run(_factory(api))
    output = report.format_summary()
    assert "exact_match" in output
    assert "contains_expected" in output
    assert "1 cases" in output


def test_format_summary_empty_report_message() -> None:
    report = EvalSuiteReport(cases=[], metric_summaries={}, pass_thresholds={})
    assert "empty" in report.format_summary().lower()


# --------------------------------------------------------------------------- #
# Regression diff
# --------------------------------------------------------------------------- #


def _make_report(scores: dict[str, float], threshold: float = 1.0) -> EvalSuiteReport:
    """Build a synthetic report from {case_label: score} for diff testing."""
    cases = [
        EvalCaseResult(
            case=DataCase(input=label, expected="x", label=label),
            chain_output="x",
            success=True,
            execution_time=0.01,
            metric_scores={"m": score},
        )
        for label, score in scores.items()
    ]
    summary = MetricSummary.from_scores("m", list(scores.values()), pass_threshold=threshold)
    return EvalSuiteReport(
        cases=cases,
        metric_summaries={"m": summary},
        pass_thresholds={"m": threshold},
    )


def test_diff_detects_regression() -> None:
    baseline = _make_report({"a": 1.0, "b": 1.0, "c": 1.0})
    current = _make_report({"a": 1.0, "b": 0.0, "c": 1.0})
    diff = baseline.diff(current)
    assert diff.has_regressions()
    assert diff.regressed_cases["m"] == ["b"]
    assert diff.metric_mean_deltas["m"] == pytest.approx(-1 / 3)


def test_diff_detects_improvement() -> None:
    baseline = _make_report({"a": 0.0, "b": 0.0, "c": 1.0})
    current = _make_report({"a": 1.0, "b": 0.0, "c": 1.0})
    diff = baseline.diff(current)
    assert diff.regressed_cases == {}
    assert diff.improved_cases["m"] == ["a"]
    assert not diff.has_regressions()


def test_diff_detects_new_and_dropped_cases() -> None:
    baseline = _make_report({"a": 1.0, "b": 1.0})
    current = _make_report({"a": 1.0, "c": 1.0})
    diff = baseline.diff(current)
    assert diff.new_case_labels == ["c"]
    assert diff.dropped_case_labels == ["b"]


def test_diff_mean_delta_zero_when_unchanged() -> None:
    baseline = _make_report({"a": 1.0, "b": 0.0})
    current = _make_report({"a": 1.0, "b": 0.0})
    diff = baseline.diff(current)
    assert diff.metric_mean_deltas["m"] == 0.0
    assert diff.regressed_cases == {}
    assert diff.improved_cases == {}


def test_diff_pass_rate_delta() -> None:
    baseline = _make_report({"a": 1.0, "b": 0.0, "c": 0.0})  # pass_rate 1/3
    current = _make_report({"a": 1.0, "b": 1.0, "c": 0.0})  # pass_rate 2/3
    diff = baseline.diff(current)
    assert diff.metric_pass_rate_deltas["m"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# MetricSummary edge cases
# --------------------------------------------------------------------------- #


def test_metric_summary_from_empty_scores() -> None:
    s = MetricSummary.from_scores("x", [], pass_threshold=1.0)
    assert s.count == 0
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.pass_rate == 0.0


def test_metric_summary_p95() -> None:
    """p95 uses inclusive percentile (no numpy dep)."""
    scores = list(range(100))  # 0..99
    s = MetricSummary.from_scores("x", [float(v) for v in scores], pass_threshold=1.0)
    assert s.p95 == pytest.approx(94.0)
    assert s.median == pytest.approx(49.5)
