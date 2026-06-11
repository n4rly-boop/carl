"""
Tests for the intra-chain event bus.

Steps may declare ``triggered_by=["event:name"]`` in addition to or instead of
numeric ``dependencies``. Other steps (tools, LLM-step wrappers, etc.) call
``context.emit_event(name, payload)``; the executor includes a step in a
batch only when all its declared events have been emitted *and* its numeric
dependencies are satisfied. Event payloads can be read through the new
``$event.<name>`` reference syntax.
"""

import pytest

from mmar_carl import (
    LLMClientBase,
    LLMStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #


class _Stub(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


# --------------------------------------------------------------------------- #
# ReasoningContext.emit_event / has_event / get_event_payload
# --------------------------------------------------------------------------- #


def test_emit_event_records_payload() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.emit_event("ready", {"rows": 42})
    assert ctx.has_event("ready")
    assert ctx.get_event_payload("ready") == {"rows": 42}


def test_get_event_payload_returns_default_when_unset() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    assert ctx.get_event_payload("missing") is None
    assert ctx.get_event_payload("missing", default="fallback") == "fallback"


def test_emit_event_overwrites_existing_payload() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.emit_event("e", 1)
    ctx.emit_event("e", 2)
    assert ctx.get_event_payload("e") == 2


def test_emit_event_empty_name_raises() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    with pytest.raises(ValueError, match="cannot be empty"):
        ctx.emit_event("")
    with pytest.raises(ValueError, match="cannot be empty"):
        ctx.emit_event("   ")


def test_event_names_lists_all_emissions() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.emit_event("a", 1)
    ctx.emit_event("b", 2)
    ctx.emit_event("c", None)
    assert set(ctx.event_names()) == {"a", "b", "c"}


def test_has_event_for_fire_and_forget_signal() -> None:
    """Payload-less signals still register as emitted."""
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.emit_event("ping")  # no payload
    assert ctx.has_event("ping")
    assert ctx.get_event_payload("ping") is None


# --------------------------------------------------------------------------- #
# Step gating via triggered_by
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_step_gated_by_event_runs_after_emission() -> None:
    """A step with ``triggered_by=['ready']`` runs only after ``ready`` is emitted."""
    captured_order: list[str] = []

    def step1_tool() -> str:
        captured_order.append("emitter")
        return "fired"

    def step2_tool() -> str:
        captured_order.append("consumer")
        return "consumed"

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit",
                config=ToolStepConfig(
                    tool_name="emit_tool", parameters=[], input_mapping={},
                ),
            ),
            ToolStepDescription(
                number=2, title="consume",
                triggered_by=["ready"],
                config=ToolStepConfig(
                    tool_name="consume_tool", parameters=[], input_mapping={},
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    # Wrap step1's tool so it ALSO fires the event when invoked
    def emit_wrapper() -> str:
        ctx.emit_event("ready", {"ts": "now"})
        return step1_tool()
    ctx.register_tool("emit_tool", emit_wrapper)
    ctx.register_tool("consume_tool", step2_tool)
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert captured_order == ["emitter", "consumer"]


@pytest.mark.asyncio
async def test_step_with_multiple_event_triggers_waits_for_all() -> None:
    """``triggered_by=['a', 'b']`` requires BOTH events to fire."""
    order: list[str] = []

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="fire_a",
                config=ToolStepConfig(tool_name="fa", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=2, title="fire_b",
                dependencies=[1],
                config=ToolStepConfig(tool_name="fb", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=3, title="consumer",
                triggered_by=["a", "b"],
                config=ToolStepConfig(tool_name="consume", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.register_tool("fa", lambda: (ctx.emit_event("a"), order.append("a"), "ok")[-1])
    ctx.register_tool("fb", lambda: (ctx.emit_event("b"), order.append("b"), "ok")[-1])
    ctx.register_tool("consume", lambda: order.append("consume") or "done")
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert order == ["a", "b", "consume"]


@pytest.mark.asyncio
async def test_missing_event_causes_deadlock_error() -> None:
    """A step gated on an event that never fires triggers the deadlock detector."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="noop", aim="x"),
            LLMStepDescription(
                number=2, title="waits", aim="x",
                triggered_by=["never_fires"],
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    with pytest.raises(ValueError, match="Deadlock"):
        await chain.execute_async(ctx)


@pytest.mark.asyncio
async def test_deadlock_error_lists_unmet_events() -> None:
    """The deadlock message should name the unfired events for debugging."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="noop", aim="x"),
            LLMStepDescription(
                number=2, title="needs", aim="x", triggered_by=["alpha", "beta"],
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    with pytest.raises(ValueError) as exc_info:
        await chain.execute_async(ctx)
    msg = str(exc_info.value)
    assert "Unmet events" in msg
    assert "alpha" in msg or "beta" in msg


@pytest.mark.asyncio
async def test_triggered_by_does_not_replace_numeric_dependencies() -> None:
    """When a step has BOTH dependencies and triggered_by, both must be met."""
    order: list[int] = []

    def make_tool(num: int):
        def tool() -> str:
            order.append(num)
            return f"step{num}"
        return tool

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="dep_step",
                config=ToolStepConfig(tool_name="t1", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=2, title="event_emitter",
                config=ToolStepConfig(tool_name="t2", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=3, title="needs_both",
                dependencies=[1],
                triggered_by=["go"],
                config=ToolStepConfig(tool_name="t3", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=2,  # 1 and 2 can run in parallel
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.register_tool("t1", make_tool(1))
    ctx.register_tool("t2", lambda: (ctx.emit_event("go"), order.append(2), "ok")[-1])
    ctx.register_tool("t3", make_tool(3))
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    # Step 3 must run last regardless of parallel batch ordering of {1, 2}
    assert order[-1] == 3
    assert set(order[:-1]) == {1, 2}


# --------------------------------------------------------------------------- #
# $event.<name> reference resolution
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_event_payload_readable_via_dollar_event_reference() -> None:
    captured: dict[str, object] = {}

    def emitter() -> str:
        return "emitted"

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit",
                config=ToolStepConfig(tool_name="emit", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=2, title="read",
                triggered_by=["payload_ready"],
                config=ToolStepConfig(
                    tool_name="reader",
                    parameters=[],
                    input_mapping={"value": "$event.payload_ready"},
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())

    def emit_wrapper() -> str:
        ctx.emit_event("payload_ready", {"k": "v", "n": 7})
        return emitter()

    def reader(value):
        captured["got"] = value
        return "read"

    ctx.register_tool("emit", emit_wrapper)
    ctx.register_tool("reader", reader)
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert captured["got"] == {"k": "v", "n": 7}


def test_event_reference_returns_none_for_unset_event() -> None:
    """``$event.x`` resolves to None when ``x`` was never emitted (lenient)."""
    from mmar_carl.step_executors import resolve_context_reference

    ctx = ReasoningContext(outer_context="x", api=_Stub())
    assert resolve_context_reference("$event.missing", ctx) is None


# --------------------------------------------------------------------------- #
# Parallel snapshot propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_event_emitted_in_parallel_batch_visible_to_next_batch() -> None:
    """An event emitted by one step in a parallel batch must be observable
    by a step in the next batch."""
    order: list[int] = []

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="A",
                config=ToolStepConfig(tool_name="ta", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=2, title="B",
                config=ToolStepConfig(tool_name="tb", parameters=[], input_mapping={}),
            ),
            ToolStepDescription(
                number=3, title="downstream",
                triggered_by=["both_done"],
                config=ToolStepConfig(tool_name="td", parameters=[], input_mapping={}),
            ),
        ],
        max_workers=2,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.register_tool("ta", lambda: (order.append(1), "a")[-1])

    def tb() -> str:
        order.append(2)
        # B is responsible for signaling "both done" once it completes
        ctx.emit_event("both_done", {"a": True, "b": True})
        return "b"

    ctx.register_tool("tb", tb)
    ctx.register_tool("td", lambda: (order.append(3), "d")[-1])
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert 3 in order
    assert order.index(3) > order.index(1)
    assert order.index(3) > order.index(2)


@pytest.mark.asyncio
async def test_pre_existing_event_visible_to_event_gated_step() -> None:
    """Events emitted before chain start (on the caller's context) gate steps just as if a step had emitted them."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="needs_pre_event",
                aim="x",
                triggered_by=["caller_signal"],
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    ctx.emit_event("caller_signal", "from-outside")
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success


# --------------------------------------------------------------------------- #
# Memory step writes the event payload (integration)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_step_can_pull_event_payload_into_memory() -> None:
    """End-to-end: tool emits event → memory step writes ``$event.X`` to memory."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="emit",
                config=ToolStepConfig(tool_name="emit", parameters=[], input_mapping={}),
            ),
            MemoryStepDescription(
                number=2, title="persist",
                triggered_by=["data"],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="snapshot",
                    value_source="$event.data",
                    namespace="output",
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())

    def emit_wrapper() -> str:
        ctx.emit_event("data", {"v": 99})
        return "emitted"

    ctx.register_tool("emit", emit_wrapper)
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert ctx.memory.get("output", {}).get("snapshot") == {"v": 99}


# --------------------------------------------------------------------------- #
# StepDescriptionBase config
# --------------------------------------------------------------------------- #


def test_triggered_by_default_is_empty_list() -> None:
    s = LLMStepDescription(number=1, title="x", aim="x")
    assert s.triggered_by == []


def test_triggered_by_round_trips_through_to_dict() -> None:
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="a", aim="x"),
            LLMStepDescription(
                number=2, title="b", aim="x", triggered_by=["foo", "bar"],
            ),
        ],
        max_workers=1,
    )
    data = chain.to_dict()
    step2 = next(s for s in data["steps"] if s["number"] == 2)
    assert step2["triggered_by"] == ["foo", "bar"]
