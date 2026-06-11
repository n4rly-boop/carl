"""Tests for case-aware metric dispatch in ``DatasetEvaluator`` and the
public ``call_metric_async`` helper.

The live evolution benchmark surfaced a real bug: a metric needs per-case
ground truth (e.g. the expected answer for a math problem), but
``MetricBase.compute_async(output)`` only sees the ``ReasoningResult`` —
no channel to the ``DataCase``. Users worked around it by stashing the
expected value in ``context.metadata``, but that doesn't survive the
chain → ReasoningResult boundary. Result: a chain that *actually solved
every problem* was silently scored 0.0 in 6 generations.

The fix is opt-in: metrics that declare a ``case`` parameter (positional
or keyword) receive the current case. Metrics that don't are called as
before. ``DatasetEvaluator`` (and the public ``call_metric_async``)
inspect the signature and dispatch accordingly.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    DataCase,
    DatasetEvaluator,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    ThresholdStrategy,
    ToolStepConfig,
    ToolStepDescription,
    call_metric_async,
)
from mmar_carl.metrics import _metric_accepts_case


# ---------------------------------------------------------------------------
# Stub metrics covering the three signature shapes we need to support
# ---------------------------------------------------------------------------


class _LegacyMetric(MetricBase):
    """Pre-fix shape — only accepts ``output``."""

    @property
    def name(self) -> str:
        return "legacy"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 0.5


class _CaseAwareMetric(MetricBase):
    """Post-fix shape — declares ``case`` keyword."""

    @property
    def name(self) -> str:
        return "case_aware"

    async def compute_async(self, output, *, case=None) -> float:  # noqa: ANN001
        if case is None or case.expected is None:
            return 0.0
        return 1.0 if case.expected == "GOOD" else 0.0


class _KwargsMetric(MetricBase):
    """Generic shape — accepts ``**kwargs`` and pulls ``case`` out."""

    @property
    def name(self) -> str:
        return "kwargs"

    async def compute_async(self, output, **kwargs) -> float:  # noqa: ANN001
        case = kwargs.get("case")
        return 1.0 if (case is not None and case.expected == "GOOD") else 0.0


# ---------------------------------------------------------------------------
# Signature introspection helper
# ---------------------------------------------------------------------------


class TestMetricAcceptsCase:
    def test_legacy_signature_rejected(self) -> None:
        assert _metric_accepts_case(_LegacyMetric()) is False

    def test_keyword_signature_accepted(self) -> None:
        assert _metric_accepts_case(_CaseAwareMetric()) is True

    def test_var_keyword_signature_accepted(self) -> None:
        assert _metric_accepts_case(_KwargsMetric()) is True


# ---------------------------------------------------------------------------
# call_metric_async dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCallMetricAsync:
    async def test_legacy_metric_called_without_case(self) -> None:
        score = await call_metric_async(
            _LegacyMetric(), object(), case=DataCase(input="x")
        )
        assert score == 0.5  # legacy returned its constant ignoring case

    async def test_case_aware_metric_receives_case(self) -> None:
        score = await call_metric_async(
            _CaseAwareMetric(), object(), case=DataCase(input="x", expected="GOOD")
        )
        assert score == 1.0

    async def test_case_aware_metric_scores_zero_on_mismatch(self) -> None:
        score = await call_metric_async(
            _CaseAwareMetric(), object(), case=DataCase(input="x", expected="BAD")
        )
        assert score == 0.0

    async def test_case_aware_metric_handles_none_case(self) -> None:
        """Without an explicit case, the metric still works (returns 0)."""
        score = await call_metric_async(_CaseAwareMetric(), object(), case=None)
        assert score == 0.0

    async def test_kwargs_metric_receives_case_via_kwargs(self) -> None:
        score = await call_metric_async(
            _KwargsMetric(), object(), case=DataCase(input="x", expected="GOOD")
        )
        assert score == 1.0

    async def test_return_value_is_coerced_to_float(self) -> None:
        class IntMetric(MetricBase):
            @property
            def name(self) -> str:
                return "int"

            async def compute_async(self, output) -> int:  # noqa: ANN001
                return 7  # not float

        score = await call_metric_async(IntMetric(), object())
        assert score == 7.0
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# DatasetEvaluator integration — the real motivating use case
# ---------------------------------------------------------------------------


def _make_echo_chain() -> ReasoningChain:
    """Chain returning the case input verbatim — so a case-aware metric can
    score against ``case.expected``."""

    def echo(text: str) -> str:
        return text

    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="echo",
                config=ToolStepConfig(
                    tool_name="echo", input_mapping={"text": "$outer_context"}
                ),
            ),
        ],
    )


@pytest.mark.asyncio
async def test_dataset_evaluator_passes_case_to_case_aware_metric() -> None:
    """The headline bug from the live benchmark: chain output is correct,
    metric is correct, but without case access the score is 0. Confirm
    the fix wires them together end-to-end."""

    dataset = SimpleDataset(
        [
            DataCase(input="GOOD", expected="GOOD", label="g1"),
            DataCase(input="BAD", expected="GOOD", label="b1"),
            DataCase(input="GOOD", expected="GOOD", label="g2"),
        ]
    )

    chain = _make_echo_chain()

    def ctx_factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
        ctx.register_tool("echo", lambda text: text)
        return ctx

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=_CaseAwareMetric(),
        strategy=ThresholdStrategy(threshold=0.5),
    )
    report = await evaluator.evaluate_async(ctx_factory)
    # 2 cases produced "GOOD" → score 1.0, 1 produced "BAD" → 0.0
    # NB: the metric only checks expected, not the actual chain output —
    # but the dispatch is what we're testing.
    assert len(report.all_results) == 3
    assert report.all_results[0].score == 1.0
    assert report.all_results[1].score == 1.0  # case.expected is "GOOD"
    assert report.all_results[2].score == 1.0
    assert report.mean_score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_dataset_evaluator_legacy_metric_still_works() -> None:
    """Legacy metrics not declaring ``case`` continue to function unchanged."""
    dataset = SimpleDataset([DataCase(input="x"), DataCase(input="y")])
    chain = _make_echo_chain()

    def ctx_factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
        ctx.register_tool("echo", lambda text: text)
        return ctx

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=_LegacyMetric(),
        strategy=ThresholdStrategy(threshold=0.0),
    )
    report = await evaluator.evaluate_async(ctx_factory)
    assert all(r.score == 0.5 for r in report.all_results)


@pytest.mark.asyncio
async def test_dataset_evaluator_scoring_uses_actual_chain_output() -> None:
    """Real-world case: a metric checks the chain output against
    ``case.expected``. Pre-fix this required smuggling the expected via
    ``context.metadata``; post-fix it's a one-liner."""

    class ExactMatchMetric(MetricBase):
        @property
        def name(self) -> str:
            return "exact_match"

        async def compute_async(self, output, *, case=None) -> float:  # noqa: ANN001
            if case is None or case.expected is None:
                return 0.0
            got = output.get_final_output() or ""
            return 1.0 if case.expected.strip() in got.strip() else 0.0

    dataset = SimpleDataset(
        [
            DataCase(input="hello", expected="hello"),  # exact match
            DataCase(input="world", expected="planet"),  # mismatch
            DataCase(input="foo", expected="foo"),  # exact match
        ]
    )
    chain = _make_echo_chain()

    def ctx_factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=None, model="default")
        ctx.register_tool("echo", lambda text: text)
        return ctx

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=ExactMatchMetric(),
        strategy=ThresholdStrategy(threshold=0.5),
    )
    report = await evaluator.evaluate_async(ctx_factory)
    assert report.all_results[0].score == 1.0  # hello matches
    assert report.all_results[1].score == 0.0  # world ≠ planet
    assert report.all_results[2].score == 1.0  # foo matches
    assert report.mean_score == pytest.approx(2 / 3)
