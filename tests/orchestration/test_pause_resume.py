"""Pause / resume API + ``ContextSnapshot``.

Covers the pause flag (``wait_for_resume``) and snapshot /
restore. Executor-level ``resume_from`` is deferred to P3.

CARE TUI uses these to let the user pause a long-running chain,
inspect partial state, and resume — useful for chains with expensive
LLM calls where a midway interrupt is cheaper than starting over.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mmar_carl import (
    ContextSnapshot,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.context import _PauseToken
from mmar_carl.models.llm_client_base import ChatMessage, LLMClientBase


# ---------------------------------------------------------------------------
# Sequenced LLM client (same shape as test_cancellation.py)
# ---------------------------------------------------------------------------


class _SequencedClient(LLMClientBase):
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def get_response(self, prompt: str) -> str:
        if self._idx >= len(self._responses):
            return f"response-{self._idx}"
        r = self._responses[self._idx]
        self._idx += 1
        return r

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _make_context(client: LLMClientBase | None = None) -> ReasoningContext:
    return ReasoningContext(
        outer_context="task", api=client or _SequencedClient([]),
    )


# ---------------------------------------------------------------------------
# pause flag API
# ---------------------------------------------------------------------------


class TestPauseStateAPI:
    def test_initially_not_paused(self) -> None:
        ctx = _make_context()
        assert not ctx.is_pause_requested()

    def test_request_pause_sets_flag(self) -> None:
        ctx = _make_context()
        ctx.request_pause()
        assert ctx.is_pause_requested()

    def test_clear_pause_clears_flag(self) -> None:
        ctx = _make_context()
        ctx.request_pause()
        ctx.clear_pause()
        assert not ctx.is_pause_requested()

    @pytest.mark.asyncio
    async def test_wait_for_resume_returns_immediately_when_not_paused(self) -> None:
        ctx = _make_context()
        # Without a pause request, the resume event is set so wait_for_resume
        # returns immediately (no timeout needed).
        await asyncio.wait_for(ctx.wait_for_resume(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_for_resume_blocks_until_clear_pause(self) -> None:
        ctx = _make_context()
        ctx.request_pause()
        wait_task = asyncio.create_task(ctx.wait_for_resume())
        # Give the task a chance to start awaiting.
        await asyncio.sleep(0.01)
        assert not wait_task.done(), "wait_for_resume must block while paused"
        ctx.clear_pause()
        # After clear_pause, the wait resolves.
        await asyncio.wait_for(wait_task, timeout=0.2)


# ---------------------------------------------------------------------------
# Pause token sharing across snapshots
# ---------------------------------------------------------------------------


class TestPauseTokenSharing:
    def test_independent_contexts_have_independent_tokens(self) -> None:
        a = _make_context()
        b = _make_context()
        assert a._pause_token is not b._pause_token

    def test_shared_token_propagates_pause(self) -> None:
        parent = _make_context()
        child = _make_context()
        # Mirror what executor.py snapshot construction now does.
        child._pause_token = parent._pause_token
        parent.request_pause()
        assert child.is_pause_requested()
        parent.clear_pause()
        assert not child.is_pause_requested()


# ---------------------------------------------------------------------------
# End-to-end pause + resume through the DAG executor
# ---------------------------------------------------------------------------


class TestExecutorPause:
    @pytest.mark.asyncio
    async def test_pause_between_batches_halts_then_resumes(self) -> None:
        """Two-step linear chain. Pause after step 1; step 2 must not
        start until clear_pause fires. Both steps complete eventually.
        """
        client = _SequencedClient(["s1", "s2"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S1", aim="x"),
            LLMStepDescription(number=2, title="S2", aim="x", dependencies=[1]),
        ])
        ctx = _make_context(client)

        step_finish_order: list[int] = []

        def on_complete(result) -> None:
            step_finish_order.append(result.step_number)
            # Pause after step 1 finishes — step 2 must wait for clear.
            if result.step_number == 1:
                ctx.request_pause()
        ctx.on_step_complete = on_complete

        execution_task = asyncio.create_task(chain.execute_async(ctx))

        # Give the executor time to finish step 1 and enter the pause.
        await asyncio.sleep(0.05)
        # Step 1 done, step 2 not yet.
        assert step_finish_order == [1]
        assert not execution_task.done()
        assert ctx.is_pause_requested()

        # Resume.
        ctx.clear_pause()
        result = await asyncio.wait_for(execution_task, timeout=1.0)
        assert result.success
        assert step_finish_order == [1, 2]


# ---------------------------------------------------------------------------
# snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_snapshot_captures_basic_state(self) -> None:
        ctx = _make_context()
        ctx.outer_context = "the original task"
        ctx.history = ["step 1 output", "step 2 output"]
        ctx.memory = {"input": {"k": "v"}}
        ctx.metadata = {"user_key": "preserved"}

        snap = ctx.snapshot()
        assert isinstance(snap, ContextSnapshot)
        assert snap.outer_context == "the original task"
        assert snap.history == ["step 1 output", "step 2 output"]
        assert snap.memory == {"input": {"k": "v"}}
        assert snap.metadata == {"user_key": "preserved"}

    def test_snapshot_strips_framework_internal_metadata(self) -> None:
        ctx = _make_context()
        ctx.metadata = {
            "user_key": "kept",
            "__langfuse_trace": object(),     # framework-internal
            "__replan_feedback_by_step": {},  # framework-internal
        }
        snap = ctx.snapshot()
        assert snap.metadata == {"user_key": "kept"}

    def test_snapshot_captures_cancel_state(self) -> None:
        ctx = _make_context()
        ctx.cancel()
        snap = ctx.snapshot()
        assert snap.cancelled is True

    def test_snapshot_messages_serialisable(self) -> None:
        ctx = _make_context()
        ctx.messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ]
        snap = ctx.snapshot()
        # Messages are stored as dicts so the whole snapshot can be
        # JSON-serialised.
        as_json = snap.model_dump_json()
        loaded = json.loads(as_json)
        assert loaded["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    def test_restore_replaces_mutable_state(self) -> None:
        snap = ContextSnapshot(
            outer_context="restored task",
            history=["restored history"],
            memory={"restored_ns": {"k": "restored"}},
            metadata={"restored_meta": True},
            messages=[{"role": "system", "content": "restored"}],
            cancelled=False,
        )
        ctx = _make_context()
        ctx.outer_context = "original"
        ctx.history = ["original"]
        ctx.restore(snap)
        assert ctx.outer_context == "restored task"
        assert ctx.history == ["restored history"]
        assert ctx.memory["restored_ns"]["k"] == "restored"
        assert ctx.metadata == {"restored_meta": True}
        assert len(ctx.messages) == 1
        assert isinstance(ctx.messages[0], ChatMessage)
        assert ctx.messages[0].role == "system"
        assert ctx.messages[0].content == "restored"

    def test_restore_propagates_cancel_state(self) -> None:
        snap = ContextSnapshot(cancelled=True)
        ctx = _make_context()
        ctx.restore(snap)
        assert ctx.is_cancelled()

    def test_snapshot_then_restore_round_trip(self) -> None:
        original = _make_context()
        original.outer_context = "round-trip task"
        original.history = ["h1", "h2"]
        original.memory = {"ns1": {"a": 1, "b": 2}, "ns2": {"c": [3, 4]}}
        original.metadata = {"flag": True}

        snap = original.snapshot()

        # Hydrate a fresh context from the snapshot.
        restored = _make_context()
        restored.restore(snap)
        assert restored.outer_context == original.outer_context
        assert restored.history == original.history
        assert restored.memory == original.memory
        assert restored.metadata == original.metadata

    def test_snapshot_json_round_trip(self) -> None:
        """ContextSnapshot is a Pydantic model so JSON round-trip Just Works."""
        snap = ContextSnapshot(
            outer_context="task",
            history=["a", "b"],
            memory={"ns": {"k": "v"}},
            metadata={"x": 1},
            messages=[{"role": "user", "content": "hi"}],
            cancelled=True,
        )
        as_json = snap.model_dump_json()
        rebuilt = ContextSnapshot.model_validate_json(as_json)
        assert rebuilt == snap


# ---------------------------------------------------------------------------
# PauseToken implementation details
# ---------------------------------------------------------------------------


def test_pause_token_event_is_lazy() -> None:
    """The asyncio.Event is created lazily so a token can be
    constructed outside an event loop (e.g. at import time).
    """
    tok = _PauseToken()
    # The internal event slot is None until first access.
    assert tok._event is None

    async def _materialise() -> None:
        # Lazily materialised; default state is *set* so awaits don't block.
        event = tok.event
        assert event.is_set()

    asyncio.run(_materialise())


def test_pause_token_slots_reject_extra_attrs() -> None:
    tok = _PauseToken()
    with pytest.raises(AttributeError):
        tok.extra_field = "boom"  # type: ignore[attr-defined]
