"""`ReasoningChain.execute_async(resume_from=ContextSnapshot)`.

The cross-process resume primitive. A chain that ran partially (say,
got paused / cancelled mid-execution) can be picked up later — even on
a different host — by:

1. Capturing ``snap = ctx.snapshot()`` while the run is paused.
2. Persisting ``snap`` (JSON).
3. Later, on a fresh context: ``await chain.execute_async(new_ctx,
   resume_from=snap)``.

The chain restores history / memory / metadata / messages from the
snapshot AND skips every step whose number is in
``snapshot.completed_step_numbers`` — the scheduler treats those
nodes as already executed so dependents see them as done and the
chain resumes at the first un-finished step.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ContextSnapshot,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.llm_client_base import LLMClientBase


# ---------------------------------------------------------------------------
# Test client — records every call so we can verify which steps were
# skipped on resume.
# ---------------------------------------------------------------------------


class _SequencedClient(LLMClientBase):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            return "default"
        return self.responses.pop(0)

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return await self.get_response(prompt)


def _make_context(client: LLMClientBase | None = None) -> ReasoningContext:
    return ReasoningContext(
        outer_context="task", api=client or _SequencedClient([]),
    )


# ---------------------------------------------------------------------------
# Completed-step tracking on the context
# ---------------------------------------------------------------------------


class TestCompletedStepTracking:
    def test_initially_empty(self) -> None:
        ctx = _make_context()
        assert ctx.get_executed_step_numbers() == []

    def test_record_step_executed_appends(self) -> None:
        ctx = _make_context()
        ctx.record_step_executed(1)
        ctx.record_step_executed(2)
        assert ctx.get_executed_step_numbers() == [1, 2]

    def test_record_step_executed_dedups(self) -> None:
        ctx = _make_context()
        ctx.record_step_executed(1)
        ctx.record_step_executed(1)
        assert ctx.get_executed_step_numbers() == [1]

    @pytest.mark.asyncio
    async def test_executor_records_step_numbers_after_completion(self) -> None:
        """A successful chain run stamps every completed step on the
        context — `ctx.snapshot()` captures them automatically."""
        client = _SequencedClient(["a", "b"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
        ])
        ctx = _make_context(client)
        result = await chain.execute_async(ctx)
        assert result.success
        assert ctx.get_executed_step_numbers() == [1, 2]


# ---------------------------------------------------------------------------
# Snapshot ↔ resume round-trip
# ---------------------------------------------------------------------------


class TestSnapshotResume:
    @pytest.mark.asyncio
    async def test_snapshot_captures_completed_step_numbers(self) -> None:
        client = _SequencedClient(["a", "b"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
        ])
        ctx = _make_context(client)
        await chain.execute_async(ctx)
        snap = ctx.snapshot()
        assert snap.completed_step_numbers == [1, 2]

    @pytest.mark.asyncio
    async def test_resume_skips_already_completed_step(self) -> None:
        """A 3-step chain. Pretend steps 1 and 2 already finished
        (snapshot says so). On resume, ONLY step 3 should fire an LLM
        call."""
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
            LLMStepDescription(number=3, title="S3", aim="a", dependencies=[2]),
        ])

        # Pre-baked snapshot: history reflects what steps 1+2 produced;
        # completed_step_numbers tells the executor to skip them.
        snap = ContextSnapshot(
            outer_context="task",
            history=["step 1 out", "step 2 out"],
            memory={},
            metadata={},
            messages=[],
            cancelled=False,
            completed_step_numbers=[1, 2],
        )

        client = _SequencedClient(["step 3 out"])
        ctx = _make_context(client)
        result = await chain.execute_async(ctx, resume_from=snap)
        assert result.success
        # Only step 3 hit the LLM.
        assert len(client.calls) == 1
        # And the final history includes the resumed entries + step 3.
        assert "step 1 out" in ctx.history[0]
        assert "step 2 out" in ctx.history[1]
        assert "step 3 out" in ctx.history[-1]

    @pytest.mark.asyncio
    async def test_resume_with_full_completion_runs_nothing(self) -> None:
        """If every step is already in `completed_step_numbers`, the
        chain is a no-op."""
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
        ])
        snap = ContextSnapshot(
            outer_context="task",
            history=["out 1", "out 2"],
            completed_step_numbers=[1, 2],
        )
        client = _SequencedClient([])
        ctx = _make_context(client)
        result = await chain.execute_async(ctx, resume_from=snap)
        assert result.success
        # No LLM calls — every step was pre-marked done.
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_resume_restores_memory_and_metadata(self) -> None:
        """The snapshot's memory/metadata is restored before
        execution resumes — downstream steps that read them see the
        prior values."""
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
        ])
        snap = ContextSnapshot(
            outer_context="restored task",
            history=["pre-existing"],
            memory={"input": {"k": "from snapshot"}},
            metadata={"user_flag": True},
            completed_step_numbers=[1],
        )
        client = _SequencedClient(["step 2 out"])
        ctx = _make_context(client)
        await chain.execute_async(ctx, resume_from=snap)
        # Memory + metadata survived the resume.
        assert ctx.memory["input"]["k"] == "from snapshot"
        assert ctx.metadata.get("user_flag") is True
        assert ctx.outer_context == "restored task"

    @pytest.mark.asyncio
    async def test_resume_from_none_is_legacy_behaviour(self) -> None:
        """Passing ``resume_from=None`` (the default) is identical to
        not passing it — the chain executes from step 1."""
        client = _SequencedClient(["a"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
        ])
        ctx = _make_context(client)
        result = await chain.execute_async(ctx, resume_from=None)
        assert result.success
        assert len(client.calls) == 1


# ---------------------------------------------------------------------------
# JSON round-trip — the snapshot survives serialisation
# ---------------------------------------------------------------------------


class TestSnapshotJSONRoundTrip:
    @pytest.mark.asyncio
    async def test_completed_steps_survive_json_round_trip(self) -> None:
        """A snapshot saved to JSON and rebuilt later still drives the
        skip logic correctly."""
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="a"),
            LLMStepDescription(number=2, title="S2", aim="a", dependencies=[1]),
        ])

        # Process A: run partially.
        client_a = _SequencedClient(["a", "b"])
        ctx_a = _make_context(client_a)
        await chain.execute_async(ctx_a)
        snap_a = ctx_a.snapshot()
        snap_json = snap_a.model_dump_json()

        # Process B: rebuild snapshot from JSON, fresh ctx, fresh client.
        snap_b = ContextSnapshot.model_validate_json(snap_json)
        assert snap_b.completed_step_numbers == [1, 2]

        client_b = _SequencedClient([])
        ctx_b = _make_context(client_b)
        result = await chain.execute_async(ctx_b, resume_from=snap_b)
        assert result.success
        # No LLM calls on process B — both steps were pre-marked done.
        assert client_b.calls == []

    def test_default_completed_step_numbers_is_empty(self) -> None:
        snap = ContextSnapshot(outer_context="x")
        assert snap.completed_step_numbers == []
        # JSON round-trip.
        rebuilt = ContextSnapshot.model_validate_json(snap.model_dump_json())
        assert rebuilt.completed_step_numbers == []


# ---------------------------------------------------------------------------
# Restore syncs completed steps on the context too
# ---------------------------------------------------------------------------


def test_restore_writes_executed_step_numbers() -> None:
    """``ctx.restore(snap)`` populates ``_executed_step_numbers`` from
    the snapshot so a subsequent ``ctx.snapshot()`` keeps the list."""
    ctx = _make_context()
    snap = ContextSnapshot(
        outer_context="x", completed_step_numbers=[7, 11, 13],
    )
    ctx.restore(snap)
    assert ctx.get_executed_step_numbers() == [7, 11, 13]
    # A second snapshot taken right after restore must echo the list.
    snap2 = ctx.snapshot()
    assert snap2.completed_step_numbers == [7, 11, 13]
