"""Tests for `on_step_event` sub-step callback.

`ReasoningContext.on_step_event(step_number, event_type, payload)` is
the generic intra-step progress hook. Executors that perform multiple
sub-operations inside one step (Supervisor route selection, Debate
rounds, ParallelSampling samples, AgentSkill LLM_AGENT tool calls)
fire events through `context.emit_step_event(...)`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_carl import (
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.llm_client_base import LLMClientBase


# ---------------------------------------------------------------------------
# Direct emit_step_event behaviour
# ---------------------------------------------------------------------------


class _FakeClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "ok"


class TestEmitStepEvent:
    def test_callback_invoked_with_args(self) -> None:
        seen: list = []

        def cb(n, t, p):
            seen.append((n, t, p))

        ctx = ReasoningContext(
            outer_context="N/A", api=_FakeClient(), on_step_event=cb,
        )
        ctx.emit_step_event(7, "custom.event", {"x": 1})
        assert seen == [(7, "custom.event", {"x": 1})]

    def test_no_callback_is_noop(self) -> None:
        ctx = ReasoningContext(outer_context="N/A", api=_FakeClient())
        # Should not raise
        ctx.emit_step_event(7, "custom.event", {"x": 1})

    def test_empty_payload_defaults_to_dict(self) -> None:
        seen: list = []
        ctx = ReasoningContext(
            outer_context="N/A", api=_FakeClient(),
            on_step_event=lambda n, t, p: seen.append((n, t, p)),
        )
        ctx.emit_step_event(1, "no.payload")
        assert seen[0][2] == {}

    def test_callback_exception_is_swallowed(self) -> None:
        def boom(n, t, p):
            raise RuntimeError("bad consumer")

        ctx = ReasoningContext(
            outer_context="N/A", api=_FakeClient(), on_step_event=boom,
        )
        # Must not raise — log warning is best-effort
        ctx.emit_step_event(1, "x", {})


# ---------------------------------------------------------------------------
# Supervisor: supervisor.route_selected
# ---------------------------------------------------------------------------


class TestSupervisorEvent:
    @pytest.mark.asyncio
    async def test_route_selected_event_fires(self) -> None:
        from mmar_carl import (
            SupervisorStepConfig, SupervisorStepDescription,
        )

        # Specialist sub-chain (just echoes a constant)
        specialist = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="Specialist", aim="x"),
        ])

        api = MagicMock()
        # Supervisor LLM picks "alpha"; sub-chain LLM responds normally.
        api.get_response_with_retries = AsyncMock(return_value="alpha")
        api.get_response = AsyncMock(return_value="alpha")

        events: list = []
        chain = ReasoningChain(steps=[
            SupervisorStepDescription(
                number=1, title="Route",
                config=SupervisorStepConfig(
                    task="route-task",
                    agents={"alpha": "specialist", "beta": "other"},
                    routing_prompt=(
                        "Pick an agent for: {task}. Options: {agents}. "
                        "Reply with the agent name only."
                    ),
                ),
                agents={"alpha": specialist, "beta": specialist},
            ),
        ])
        ctx = ReasoningContext(
            outer_context="N/A", api=api,
            on_step_event=lambda n, t, p: events.append((n, t, p)),
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # The route_selected event was fired
        matches = [e for e in events if e[1] == "supervisor.route_selected"]
        assert matches, f"no route_selected in {events}"
        assert matches[0][0] == 1  # step number
        assert matches[0][2] == {"agent_name": "alpha"}


# ---------------------------------------------------------------------------
# ParallelSampling: parallel_sampling.sample
# ---------------------------------------------------------------------------


class TestParallelSamplingEvent:
    @pytest.mark.asyncio
    async def test_one_event_per_sample(self) -> None:
        from mmar_carl import (
            ParallelSamplingStepConfig, ParallelSamplingStepDescription,
        )

        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="sample output")
        api.get_response = AsyncMock(return_value="sample output")

        events: list = []
        chain = ReasoningChain(steps=[
            ParallelSamplingStepDescription(
                number=1, title="Sample",
                base_step=LLMStepDescription(
                    number=1, title="Base", aim="generate",
                ),
                config=ParallelSamplingStepConfig(n_samples=3),
            ),
        ])
        ctx = ReasoningContext(
            outer_context="N/A", api=api,
            on_step_event=lambda n, t, p: events.append((n, t, p)),
        )
        result = await chain.execute_async(ctx)
        assert result.success
        sample_events = [
            e for e in events if e[1] == "parallel_sampling.sample"
        ]
        assert len(sample_events) == 3
        # Sample indices are 0, 1, 2
        idxs = sorted(e[2]["sample_idx"] for e in sample_events)
        assert idxs == [0, 1, 2]


# ---------------------------------------------------------------------------
# Debate: round_started + turn_argument
# ---------------------------------------------------------------------------


class TestDebateEvents:
    @pytest.mark.asyncio
    async def test_round_and_turn_events(self) -> None:
        from mmar_carl import DebateStepConfig, DebateStepDescription

        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="argument text")
        api.get_response = AsyncMock(return_value="judgement")

        events: list = []
        chain = ReasoningChain(steps=[
            DebateStepDescription(
                number=1, title="Debate",
                config=DebateStepConfig(
                    task="topic",
                    roles=["proponent", "skeptic"],
                    rounds=2,
                    judge_prompt=(
                        "Synthesise a verdict for {task}.\n\n{transcript}"
                    ),
                ),
            ),
        ])
        ctx = ReasoningContext(
            outer_context="N/A", api=api,
            on_step_event=lambda n, t, p: events.append((n, t, p)),
        )
        result = await chain.execute_async(ctx)
        assert result.success
        round_events = [e for e in events if e[1] == "debate.round_started"]
        turn_events = [e for e in events if e[1] == "debate.turn_argument"]
        # 2 rounds × {1 round_started + 2 role turns} = 2 + 4
        assert len(round_events) == 2
        assert len(turn_events) == 4
        # Round payload carries round number
        assert {e[2]["round"] for e in round_events} == {1, 2}
        # Turn payload carries role + argument
        assert all("role" in e[2] and "argument" in e[2] for e in turn_events)


# ---------------------------------------------------------------------------
# Snapshot context inheritance — parallel batches share the callback
# ---------------------------------------------------------------------------


class TestSnapshotPropagation:
    @pytest.mark.asyncio
    async def test_on_step_event_propagates_to_parallel_snapshots(self) -> None:
        """When DAGExecutor builds a snapshot context for a parallel
        step, the snapshot must inherit the ``on_step_event`` callback
        so sub-step events fired from within parallel steps still
        reach the original consumer."""
        from mmar_carl import (
            ParallelSamplingStepConfig, ParallelSamplingStepDescription,
        )

        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="ok")
        api.get_response = AsyncMock(return_value="ok")

        events: list = []
        chain = ReasoningChain(steps=[
            # Two parallel-sampling steps with no deps → they run in
            # the same batch, each gets a snapshot context.
            ParallelSamplingStepDescription(
                number=1, title="P1",
                base_step=LLMStepDescription(
                    number=1, title="b1", aim="x",
                ),
                config=ParallelSamplingStepConfig(n_samples=2),
            ),
            ParallelSamplingStepDescription(
                number=2, title="P2",
                base_step=LLMStepDescription(
                    number=2, title="b2", aim="x",
                ),
                config=ParallelSamplingStepConfig(n_samples=2),
            ),
        ])
        ctx = ReasoningContext(
            outer_context="N/A", api=api,
            on_step_event=lambda n, t, p: events.append((n, t, p)),
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # 2 steps × 2 samples = 4 sample events. If the callback wasn't
        # propagated to snapshots, we'd see 0 from the parallel batch.
        sample_events = [
            e for e in events if e[1] == "parallel_sampling.sample"
        ]
        assert len(sample_events) == 4
        # Events are tagged with each step's number
        assert {e[0] for e in sample_events} == {1, 2}


# ---------------------------------------------------------------------------
# Buggy consumer doesn't kill the chain
# ---------------------------------------------------------------------------


class TestConsumerSafety:
    @pytest.mark.asyncio
    async def test_buggy_callback_does_not_fail_step(self) -> None:
        from mmar_carl import (
            ParallelSamplingStepConfig, ParallelSamplingStepDescription,
        )

        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="ok")
        api.get_response = AsyncMock(return_value="ok")

        def boom(n, t, p):
            raise RuntimeError("bad consumer")

        chain = ReasoningChain(steps=[
            ParallelSamplingStepDescription(
                number=1, title="P",
                base_step=LLMStepDescription(number=1, title="b", aim="x"),
                config=ParallelSamplingStepConfig(n_samples=2),
            ),
        ])
        ctx = ReasoningContext(
            outer_context="N/A", api=api, on_step_event=boom,
        )
        result = await chain.execute_async(ctx)
        assert result.success
