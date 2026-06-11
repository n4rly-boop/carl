"""
Tests for ExecutionTrace and TraceEvent.

Covers:
- TraceEvent model fields and to_dict()
- ExecutionTrace serialization (to_json / from_json round-trip)
- ExecutionTrace accessors (get_event, successful/failed/skipped events, total_tokens)
- ExecutionTrace.diff() between two runs
- Chain integration: result.trace is populated after execute_async()
- Trace events contain correct step data (success, type, batch_index, result)
- chain.replay(trace, context) re-runs chain using saved LLM responses
"""

import json

import pytest

from mmar_carl import (
    ExecutionTrace,
    Language,
    LLMClientBase,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    TraceEvent,
)
from mmar_carl.models.enums import StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockClient(LLMClientBase):
    def __init__(self, responses: dict[int, str] | None = None, default: str = "ok"):
        self._responses = responses or {}
        self._default = default
        self._current_step: int | None = None

    async def get_response(self, prompt: str) -> str:
        return self._responses.get(self._current_step, self._default)  # type: ignore[arg-type]

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _ctx(client: LLMClientBase | None = None, **kwargs) -> ReasoningContext:
    client = client or _MockClient()
    ctx = ReasoningContext(
        outer_context="test",
        api=client,
        model="test",
        language=Language.ENGLISH,
        **kwargs,
    )
    if isinstance(client, _MockClient):
        # Wire on_step_start so the mock knows which step is executing
        def _on_step_start(step_num: int, step_title: str) -> None:
            client._current_step = step_num
        ctx.on_step_start = _on_step_start
    return ctx


def _llm_step(number: int, deps: list[int] | None = None) -> LLMStepDescription:
    return LLMStepDescription(
        number=number,
        title=f"Step {number}",
        aim="Think about it.",
        dependencies=deps or [],
    )


# ---------------------------------------------------------------------------
# Unit: TraceEvent
# ---------------------------------------------------------------------------


class TestTraceEvent:
    def test_default_fields(self):
        event = TraceEvent(
            step_number=1,
            step_title="Step 1",
            success=True,
        )
        assert event.step_number == 1
        assert event.step_title == "Step 1"
        assert event.step_type == StepType.LLM
        assert event.success is True
        assert event.skipped is False
        assert event.result == ""
        assert event.result_data is None
        assert event.error_message is None
        assert event.execution_time is None
        assert event.token_usage == {}
        assert event.batch_index == 0
        assert event.inputs == {}

    def test_to_dict_contains_all_keys(self):
        event = TraceEvent(
            step_number=2,
            step_title="Step 2",
            step_type=StepType.TOOL,
            success=False,
            error_message="oops",
            result="partial",
            batch_index=1,
        )
        d = event.to_dict()
        assert d["step_number"] == 2
        assert d["step_title"] == "Step 2"
        assert d["step_type"] == "tool"
        assert d["success"] is False
        assert d["error_message"] == "oops"
        assert d["result"] == "partial"
        assert d["batch_index"] == 1

    def test_to_dict_step_type_is_string(self):
        event = TraceEvent(step_number=1, step_title="x", success=True, step_type=StepType.MEMORY)
        assert isinstance(event.to_dict()["step_type"], str)


# ---------------------------------------------------------------------------
# Unit: ExecutionTrace serialization
# ---------------------------------------------------------------------------


class TestExecutionTraceSerialization:
    def _make_trace(self) -> ExecutionTrace:
        trace = ExecutionTrace(chain_title="MyChain")
        trace.events = [
            TraceEvent(
                step_number=1,
                step_title="Step 1",
                success=True,
                result="result 1",
                execution_time=0.5,
                token_usage={"prompt": 10, "completion": 5, "total": 15},
                batch_index=0,
            ),
            TraceEvent(
                step_number=2,
                step_title="Step 2",
                success=False,
                error_message="boom",
                batch_index=1,
            ),
        ]
        trace.total_execution_time = 1.2
        trace.success = False
        return trace

    def test_to_json_is_valid_json(self):
        trace = self._make_trace()
        json_str = trace.to_json()
        data = json.loads(json_str)
        assert data["chain_title"] == "MyChain"
        assert len(data["events"]) == 2

    def test_from_json_round_trip(self):
        trace = self._make_trace()
        restored = ExecutionTrace.from_json(trace.to_json())
        assert restored.chain_title == trace.chain_title
        assert restored.total_execution_time == trace.total_execution_time
        assert restored.success == trace.success
        assert len(restored.events) == 2

        ev1 = restored.events[0]
        assert ev1.step_number == 1
        assert ev1.result == "result 1"
        assert ev1.token_usage == {"prompt": 10, "completion": 5, "total": 15}
        assert ev1.batch_index == 0

        ev2 = restored.events[1]
        assert ev2.step_number == 2
        assert ev2.error_message == "boom"
        assert ev2.success is False

    def test_executed_at_is_set_on_creation(self):
        trace = ExecutionTrace()
        assert trace.executed_at  # not empty
        # Should be a valid ISO timestamp
        from datetime import datetime
        dt = datetime.fromisoformat(trace.executed_at)
        assert dt.tzinfo is not None  # UTC aware


# ---------------------------------------------------------------------------
# Unit: ExecutionTrace accessors
# ---------------------------------------------------------------------------


class TestExecutionTraceAccessors:
    def _make_trace(self) -> ExecutionTrace:
        trace = ExecutionTrace(chain_title="C")
        trace.events = [
            TraceEvent(step_number=1, step_title="a", success=True, token_usage={"prompt": 10, "completion": 5, "total": 15}),
            TraceEvent(step_number=2, step_title="b", success=False, error_message="err"),
            TraceEvent(step_number=3, step_title="c", success=True, skipped=True),
            TraceEvent(step_number=4, step_title="d", success=True, token_usage={"prompt": 20, "completion": 10, "total": 30}),
        ]
        return trace

    def test_get_event_found(self):
        trace = self._make_trace()
        ev = trace.get_event(2)
        assert ev is not None
        assert ev.step_number == 2

    def test_get_event_not_found(self):
        trace = self._make_trace()
        assert trace.get_event(99) is None

    def test_get_successful_events(self):
        trace = self._make_trace()
        events = trace.get_successful_events()
        numbers = [e.step_number for e in events]
        assert 1 in numbers
        assert 4 in numbers
        assert 2 not in numbers  # failed
        assert 3 not in numbers  # skipped

    def test_get_failed_events(self):
        trace = self._make_trace()
        events = trace.get_failed_events()
        assert len(events) == 1
        assert events[0].step_number == 2

    def test_get_skipped_events(self):
        trace = self._make_trace()
        events = trace.get_skipped_events()
        assert len(events) == 1
        assert events[0].step_number == 3

    def test_total_tokens(self):
        trace = self._make_trace()
        totals = trace.total_tokens()
        assert totals["prompt"] == 30   # 10 + 20
        assert totals["completion"] == 15  # 5 + 10
        assert totals["total"] == 45


# ---------------------------------------------------------------------------
# Unit: ExecutionTrace.diff()
# ---------------------------------------------------------------------------


class TestExecutionTraceDiff:
    def _make_trace(self, results: dict[int, str]) -> ExecutionTrace:
        trace = ExecutionTrace()
        for num, result in results.items():
            trace.events.append(
                TraceEvent(step_number=num, step_title=f"Step {num}", success=True, result=result)
            )
        return trace

    def test_identical_traces_all_unchanged(self):
        t1 = self._make_trace({1: "a", 2: "b"})
        t2 = self._make_trace({1: "a", 2: "b"})
        diff = t1.diff(t2)
        assert diff["unchanged"] == [1, 2]
        assert diff["changed"] == []
        assert diff["added"] == []
        assert diff["removed"] == []

    def test_changed_result(self):
        t1 = self._make_trace({1: "old", 2: "same"})
        t2 = self._make_trace({1: "new", 2: "same"})
        diff = t1.diff(t2)
        assert len(diff["changed"]) == 1
        changed_step = diff["changed"][0]
        assert changed_step["step_number"] == 1
        assert "result" in changed_step["diffs"]
        assert diff["unchanged"] == [2]

    def test_added_step(self):
        t1 = self._make_trace({1: "a"})
        t2 = self._make_trace({1: "a", 2: "b"})
        diff = t1.diff(t2)
        assert len(diff["added"]) == 1
        assert diff["added"][0]["step_number"] == 2

    def test_removed_step(self):
        t1 = self._make_trace({1: "a", 2: "b"})
        t2 = self._make_trace({1: "a"})
        diff = t1.diff(t2)
        assert len(diff["removed"]) == 1
        assert diff["removed"][0]["step_number"] == 2

    def test_summary_format(self):
        t1 = self._make_trace({1: "x", 2: "y"})
        t2 = self._make_trace({1: "changed", 2: "y", 3: "new"})
        diff = t1.diff(t2)
        summary = diff["summary"]
        assert "unchanged" in summary
        assert "changed" in summary
        assert "added" in summary
        assert "removed" in summary


# ---------------------------------------------------------------------------
# Integration: trace attached to ReasoningResult after chain execution
# ---------------------------------------------------------------------------


class TestChainTraceIntegration:
    @pytest.mark.asyncio
    async def test_result_has_trace(self):
        client = _MockClient(default="response")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        assert result.trace is not None
        assert isinstance(result.trace, ExecutionTrace)

    @pytest.mark.asyncio
    async def test_trace_has_one_event_per_step(self):
        client = _MockClient(default="ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1), _llm_step(2, deps=[1])])
        result = await chain.execute_async(ctx)

        trace = result.trace
        assert trace is not None
        assert len(trace.events) == 2

    @pytest.mark.asyncio
    async def test_trace_event_fields_match_step_result(self):
        client = _MockClient({1: "hello there"})
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        trace = result.trace
        assert trace is not None
        event = trace.get_event(1)
        assert event is not None
        assert event.step_number == 1
        assert event.step_title == "Step 1"
        assert event.success is True
        assert "hello there" in event.result

    @pytest.mark.asyncio
    async def test_trace_success_matches_result_success(self):
        client = _MockClient(default="ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        assert result.trace is not None
        assert result.trace.success == result.success

    @pytest.mark.asyncio
    async def test_trace_batch_index_set_correctly(self):
        """Steps in different batches should have different batch_index values."""
        client = _MockClient(default="ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[
            _llm_step(1),
            _llm_step(2, deps=[1]),
        ])
        result = await chain.execute_async(ctx)

        trace = result.trace
        assert trace is not None
        ev1 = trace.get_event(1)
        ev2 = trace.get_event(2)
        assert ev1 is not None and ev2 is not None
        assert ev1.batch_index != ev2.batch_index

    @pytest.mark.asyncio
    async def test_trace_total_execution_time_set(self):
        client = _MockClient(default="ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        assert result.trace is not None
        assert result.trace.total_execution_time is not None
        assert result.trace.total_execution_time >= 0

    @pytest.mark.asyncio
    async def test_trace_json_round_trip_after_execution(self):
        client = _MockClient(default="answer")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        trace = result.trace
        assert trace is not None
        restored = ExecutionTrace.from_json(trace.to_json())
        assert restored.chain_title == trace.chain_title
        assert len(restored.events) == len(trace.events)


# ---------------------------------------------------------------------------
# Integration: chain.replay()
# ---------------------------------------------------------------------------


class TestChainReplay:
    @pytest.mark.asyncio
    async def test_replay_uses_recorded_responses(self):
        """Replaying a trace returns the same LLM responses without calling the real API."""
        # First run with a real (mock) client
        client1 = _MockClient({1: "original answer"})
        ctx1 = _ctx(client1)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result1 = await chain.execute_async(ctx1)
        trace = result1.trace
        assert trace is not None

        # Replay with a *different* context — the API should not be called
        # (replay mock intercepts all LLM calls)
        ctx2 = _ctx(_MockClient(default="THIS SHOULD NOT APPEAR"))
        result2 = await chain.replay(trace, ctx2)

        assert result2.success
        # History should contain the original response, not the new client's default
        assert any("original answer" in h for h in result2.history)

    @pytest.mark.asyncio
    async def test_replay_result_is_successful(self):
        client = _MockClient(default="response")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1), _llm_step(2, deps=[1])])
        result = await chain.execute_async(ctx)

        replay_result = await chain.replay(result.trace, _ctx())
        assert replay_result.success

    @pytest.mark.asyncio
    async def test_replay_produces_new_trace(self):
        client = _MockClient(default="ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_llm_step(1)])
        result = await chain.execute_async(ctx)

        replay_result = await chain.replay(result.trace, _ctx())
        assert replay_result.trace is not None
        assert len(replay_result.trace.events) == 1
