"""Tests for ``TraceAggregator``.

Aggregates per-step latency and token usage across N chain runs so
users can see "step 3 sometimes takes 8 s vs its p50 of 1 s" — useful
for capacity planning, regression-catching, and dataset evaluator
post-mortems.
"""

from __future__ import annotations

import pytest

from mmar_carl import TraceAggregator
from mmar_carl.execution_trace import ExecutionTrace, TraceEvent, _percentile
from mmar_carl.models.enums import StepType


def _event(
    num: int,
    *,
    title: str = "s",
    t: float | None = 0.1,
    total_tokens: int | None = None,
    step_type: StepType = StepType.LLM,
    success: bool = True,
) -> TraceEvent:
    usage: dict[str, int] = {}
    if total_tokens is not None:
        usage = {
            "prompt": total_tokens // 2,
            "completion": total_tokens - total_tokens // 2,
            "total": total_tokens,
        }
    return TraceEvent(
        step_number=num, step_title=title, step_type=step_type,
        success=success, execution_time=t, batch_index=num - 1,
        token_usage=usage,
    )


def _trace(*events: TraceEvent) -> ExecutionTrace:
    return ExecutionTrace(
        chain_title="t", executed_at="now",
        total_execution_time=sum(e.execution_time or 0.0 for e in events),
        success=True,
        events=list(events),
    )


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------


class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_single_value_returns_that_value(self) -> None:
        assert _percentile([7.0], 0.5) == 7.0
        assert _percentile([7.0], 0.99) == 7.0

    def test_p50_is_middle_for_odd_count(self) -> None:
        assert _percentile([1.0, 5.0, 9.0], 0.5) == 5.0

    def test_p50_interpolates_for_even_count(self) -> None:
        # numpy default: (5+7)/2 = 6.0
        assert _percentile([1.0, 5.0, 7.0, 9.0], 0.5) == 6.0

    def test_p100_returns_max(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 1.0) == 3.0

    def test_p0_returns_min(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 0.0) == 1.0


# ---------------------------------------------------------------------------
# Aggregator: empty + degenerate
# ---------------------------------------------------------------------------


class TestEmptyAndDegenerate:
    def test_empty_traces_format_text(self) -> None:
        out = TraceAggregator([]).format_text()
        assert "no traces" in out

    def test_traces_with_no_timings(self) -> None:
        # TraceEvent with execution_time=None should be silently ignored.
        ev = TraceEvent(
            step_number=1, step_title="x", step_type=StepType.TOOL,
            success=True, execution_time=None, batch_index=0,
        )
        agg = TraceAggregator([_trace(ev)])
        assert agg.latency_ms == {}
        assert "no per-step events" in agg.format_text()

    def test_step_with_only_zero_tokens_is_omitted(self) -> None:
        agg = TraceAggregator([_trace(_event(1, t=0.1, total_tokens=0))])
        assert 1 not in agg.tokens
        # But latency still recorded
        assert 1 in agg.latency_ms


# ---------------------------------------------------------------------------
# Aggregator: latency percentiles
# ---------------------------------------------------------------------------


class TestLatency:
    def test_latency_recorded_in_milliseconds(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, t=0.5)),  # 500 ms
            _trace(_event(1, t=1.5)),  # 1500 ms
        ])
        # Median of [500, 1500] = 1000.0
        assert agg.latency_ms[1]["p50"] == 1000.0
        assert agg.latency_ms[1]["max"] == 1500.0
        assert agg.latency_ms[1]["mean"] == 1000.0
        assert agg.latency_ms[1]["n_runs"] == 2.0

    def test_p95_captures_tail(self) -> None:
        # 9 fast + 1 slow → p95 lands inside the interpolation between
        # the last fast sample and the slow one, so it must exceed p50.
        traces = [_trace(_event(1, t=0.5)) for _ in range(9)]
        traces.append(_trace(_event(1, t=5.0)))
        agg = TraceAggregator(traces)
        lat = agg.latency_ms[1]
        assert lat["p50"] == 500.0
        assert lat["p95"] > 500.0  # tail visible at p95
        assert lat["max"] == 5000.0

    def test_multiple_steps_aggregated_independently(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, t=0.1), _event(2, t=2.0)),
            _trace(_event(1, t=0.2), _event(2, t=2.5)),
        ])
        assert agg.latency_ms[1]["mean"] < agg.latency_ms[2]["mean"]
        assert agg.latency_ms[1]["max"] == 200.0
        assert agg.latency_ms[2]["max"] == 2500.0


# ---------------------------------------------------------------------------
# Aggregator: token percentiles
# ---------------------------------------------------------------------------


class TestTokens:
    def test_only_steps_with_tokens_appear(self) -> None:
        agg = TraceAggregator([
            _trace(
                _event(1, t=0.1, total_tokens=100),
                _event(2, t=0.05, step_type=StepType.TOOL),
            ),
        ])
        assert 1 in agg.tokens
        assert 2 not in agg.tokens
        # Step 2 still has latency
        assert 2 in agg.latency_ms

    def test_p50_p95_keys_present(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, total_tokens=n)) for n in [100, 150, 200, 250, 500]
        ])
        toks = agg.tokens[1]
        assert "p50" in toks
        assert "p95" in toks
        assert "n_runs" in toks

    def test_step_with_some_runs_lacking_usage_only_counts_recorded_ones(
        self,
    ) -> None:
        agg = TraceAggregator([
            _trace(_event(1, total_tokens=100)),
            _trace(_event(1, total_tokens=None)),   # no usage
            _trace(_event(1, total_tokens=200)),
        ])
        # Only 2 of the 3 runs contributed token usage
        assert agg.tokens[1]["n_runs"] == 2.0


# ---------------------------------------------------------------------------
# Cross-shape robustness
# ---------------------------------------------------------------------------


class TestCrossShape:
    def test_different_chains_unioned(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, t=0.1), _event(2, t=0.2)),
            _trace(_event(1, t=0.1), _event(3, t=0.3)),  # diff step
        ])
        assert set(agg.step_numbers) == {1, 2, 3}
        # Step 1 ran in both traces; 2 and 3 only in one each
        assert agg.latency_ms[1]["n_runs"] == 2.0
        assert agg.latency_ms[2]["n_runs"] == 1.0
        assert agg.latency_ms[3]["n_runs"] == 1.0

    def test_failed_step_timing_still_recorded(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, t=0.1, success=False)),
            _trace(_event(1, t=0.2, success=True)),
        ])
        # Both runs counted
        assert agg.latency_ms[1]["n_runs"] == 2.0


# ---------------------------------------------------------------------------
# format_text rendering
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_header_row_lists_metric_columns(self) -> None:
        agg = TraceAggregator([_trace(_event(1, t=0.1, total_tokens=10))])
        out = agg.format_text()
        for col in ("step", "p50 ms", "p95 ms", "p99 ms", "mean ms",
                     "max ms", "p50 tok", "p95 tok"):
            assert col in out, f"missing column {col!r}"

    def test_total_traces_footer(self) -> None:
        agg = TraceAggregator([_trace(_event(1, t=0.1)) for _ in range(7)])
        assert "total traces: 7" in agg.format_text()

    def test_step_title_truncates(self) -> None:
        long = "a-step-with-a-very-very-long-name"
        agg = TraceAggregator([_trace(_event(1, title=long, t=0.1))])
        out = agg.format_text(title_width=10)
        # Truncation marker present
        assert "…" in out

    def test_step_without_tokens_renders_dash(self) -> None:
        agg = TraceAggregator([
            _trace(_event(1, t=0.1, step_type=StepType.TOOL)),
        ])
        out = agg.format_text()
        # The tokens columns should show "-" for tool steps
        data_line = next(line for line in out.splitlines() if line.lstrip().startswith("1 "))
        assert "-" in data_line


# ---------------------------------------------------------------------------
# Integration: aggregate the trace from a real chain execution
# ---------------------------------------------------------------------------


class TestRealChainIntegration:
    @pytest.mark.asyncio
    async def test_aggregate_three_chain_runs(self) -> None:
        """Drive the aggregator with traces from three real chain runs
        — confirms that ``ReasoningResult.trace`` plugs cleanly into
        the helper without manual TraceEvent construction."""
        from mmar_carl import (
            LLMStepDescription, ReasoningChain, ReasoningContext,
        )
        from mmar_carl.models.llm_client_base import LLMClientBase

        class FakeClient(LLMClientBase):
            @property
            def model_name(self) -> str:
                return "fake"

            async def get_response(self, prompt: str) -> str:
                return "ok"

            async def get_response_with_retries(
                self, prompt: str, retries: int = 3
            ) -> str:
                return "ok"

        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="A", aim="x"),
            LLMStepDescription(number=2, title="B", aim="x", dependencies=[1]),
        ])
        traces: list[ExecutionTrace] = []
        for _ in range(3):
            ctx = ReasoningContext(outer_context="N/A", api=FakeClient())
            result = await chain.execute_async(ctx)
            assert result.success
            traces.append(result.trace)

        agg = TraceAggregator(traces)
        assert set(agg.step_numbers) == {1, 2}
        for n in (1, 2):
            assert agg.latency_ms[n]["n_runs"] == 3.0
        # Each run produced an event, so format_text returns a populated table
        out = agg.format_text()
        assert "total traces: 3" in out
