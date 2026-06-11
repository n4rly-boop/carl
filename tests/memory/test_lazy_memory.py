"""Tests for the ``LazyMemoryValue`` share-on-copy wrapper.

Covers the core wrapper invariants (deepcopy shares payloads, mutable_copy
isolates them, dunder delegation works) plus integration with the parts of
CARL that deep-copy ``context.memory``: RE-PLAN checkpoint capture/restore,
the sub-chain seed paths used by Supervisor/Debate/ParallelBranches, and
ordinary parallel-batch execution.
"""

from __future__ import annotations

import copy

import pytest

from mmar_carl import (
    LazyMemoryValue,
    ReasoningChain,
    ReasoningContext,
    unwrap_lazy,
)
from mmar_carl.executor import DAGExecutor
from mmar_carl.models import (
    ToolStepConfig,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Wrapper invariants
# ---------------------------------------------------------------------------


class TestWrapperBasics:
    def test_value_returns_underlying_payload(self) -> None:
        payload = [1, 2, 3]
        wrapper = LazyMemoryValue(payload)
        assert wrapper.value is payload
        assert wrapper.get() is payload

    def test_repr_includes_type_and_length(self) -> None:
        wrapper = LazyMemoryValue([1, 2, 3, 4, 5])
        text = repr(wrapper)
        assert "LazyMemoryValue" in text
        assert "list" in text
        assert "len=5" in text

    def test_repr_omits_length_when_not_sized(self) -> None:
        class Opaque:
            pass

        text = repr(LazyMemoryValue(Opaque()))
        assert "LazyMemoryValue" in text
        assert "len=" not in text

    def test_str_delegates_to_payload(self) -> None:
        assert str(LazyMemoryValue("hello")) == "hello"

    def test_len_delegates(self) -> None:
        assert len(LazyMemoryValue("abcd")) == 4
        assert len(LazyMemoryValue([1, 2, 3])) == 3

    def test_bool_delegates(self) -> None:
        assert bool(LazyMemoryValue("x")) is True
        assert bool(LazyMemoryValue("")) is False
        assert bool(LazyMemoryValue([])) is False

    def test_eq_with_other_wrapper(self) -> None:
        assert LazyMemoryValue([1, 2]) == LazyMemoryValue([1, 2])
        assert LazyMemoryValue([1, 2]) != LazyMemoryValue([1, 3])

    def test_eq_with_raw_payload(self) -> None:
        assert LazyMemoryValue("hello") == "hello"
        assert LazyMemoryValue(42) == 42

    def test_hash_uses_payload_when_hashable(self) -> None:
        a = LazyMemoryValue("same")
        b = LazyMemoryValue("same")
        assert hash(a) == hash(b)
        assert hash(LazyMemoryValue(42)) == hash(42)

    def test_hash_falls_back_to_identity_for_unhashable_payload(self) -> None:
        wrapper = LazyMemoryValue([1, 2])  # lists are unhashable
        # Should not raise — uses id() fallback.
        h = hash(wrapper)
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# Copy / deepcopy semantics — the whole point of the class
# ---------------------------------------------------------------------------


class TestCopySemantics:
    def test_deepcopy_shares_payload_reference(self) -> None:
        big = list(range(10_000))
        wrapper = LazyMemoryValue(big)
        snapshot = copy.deepcopy(wrapper)

        assert snapshot is not wrapper
        assert snapshot.value is big  # not a copy — shared reference

    def test_shallow_copy_shares_payload_reference(self) -> None:
        big = list(range(1000))
        wrapper = LazyMemoryValue(big)
        clone = copy.copy(wrapper)

        assert clone is not wrapper
        assert clone.value is big

    def test_deepcopy_inside_nested_dict_shares_payload(self) -> None:
        big = list(range(5000))
        memory = {"pdf": {"text": LazyMemoryValue(big)}}
        cloned = copy.deepcopy(memory)

        assert cloned["pdf"]["text"].value is big
        # Outer dicts are deep-copied normally.
        assert cloned["pdf"] is not memory["pdf"]

    def test_mutable_copy_returns_independent_payload(self) -> None:
        original_payload = [1, 2, 3]
        original = LazyMemoryValue(original_payload)
        forked = original.mutable_copy()

        assert forked is not original
        assert forked.value is not original_payload
        assert forked.value == original_payload

        # Mutating the fork's payload does NOT affect the original.
        forked.value.append(99)
        assert original.value == [1, 2, 3]
        assert forked.value == [1, 2, 3, 99]

    def test_replace_yields_new_wrapper_with_new_payload(self) -> None:
        original = LazyMemoryValue("old")
        replaced = original.replace("new")

        assert original.value == "old"
        assert replaced.value == "new"
        assert original is not replaced

    def test_deepcopy_memo_prevents_infinite_recursion(self) -> None:
        wrapper = LazyMemoryValue({"data": "value"})
        memo: dict = {}
        cloned = copy.deepcopy(wrapper, memo)
        # Memo should contain the wrapper id => copy mapping.
        assert id(wrapper) in memo
        assert memo[id(wrapper)] is cloned

    def test_deepcopy_chain_through_many_layers(self) -> None:
        """Wrapping in nested deepcopies still shares payload."""
        payload = {"large": "data" * 10_000}
        wrapper = LazyMemoryValue(payload)
        clones = [wrapper]
        for _ in range(5):
            clones.append(copy.deepcopy(clones[-1]))
        for clone in clones:
            assert clone.value is payload


# ---------------------------------------------------------------------------
# unwrap_lazy helper
# ---------------------------------------------------------------------------


class TestUnwrapLazy:
    def test_unwrap_lazy_unwraps_wrapped_value(self) -> None:
        payload = [1, 2]
        assert unwrap_lazy(LazyMemoryValue(payload)) is payload

    def test_unwrap_lazy_passes_through_plain_value(self) -> None:
        assert unwrap_lazy("plain") == "plain"
        assert unwrap_lazy(42) == 42
        d = {"a": 1}
        assert unwrap_lazy(d) is d

    def test_unwrap_lazy_handles_none(self) -> None:
        assert unwrap_lazy(None) is None


# ---------------------------------------------------------------------------
# Integration: ReasoningContext memory operations
# ---------------------------------------------------------------------------


class TestContextIntegration:
    def test_memory_write_and_read_round_trip_preserves_wrapper(self) -> None:
        ctx = ReasoningContext(outer_context="x", api=None, model="default")
        payload = [1, 2, 3]
        wrapper = LazyMemoryValue(payload)
        ctx.memory_write("text", wrapper, namespace="pdf")

        stored = ctx.memory_read("text", namespace="pdf")
        # Memory layer is type-agnostic; the wrapper is stored as-is.
        assert isinstance(stored, LazyMemoryValue)
        assert stored.value is payload

    def test_replan_snapshot_capture_does_not_copy_payload(self) -> None:
        """Capturing a checkpoint snapshot must keep the payload shared."""
        ctx = ReasoningContext(outer_context="x", api=None, model="default")
        big = list(range(20_000))
        ctx.memory_write("blob", LazyMemoryValue(big), namespace="big")

        executor = DAGExecutor()
        snapshot = executor._capture_snapshot(
            executed_nodes=set(),
            all_results=[],
            context=ctx,
        )

        wrapped_in_snapshot = snapshot.memory["big"]["blob"]
        assert isinstance(wrapped_in_snapshot, LazyMemoryValue)
        # The expensive payload list is the same object — not a 20k-element copy.
        assert wrapped_in_snapshot.value is big

    def test_replan_snapshot_restore_keeps_payload_shared(self) -> None:
        ctx = ReasoningContext(outer_context="x", api=None, model="default")
        big = list(range(20_000))
        wrapper = LazyMemoryValue(big)
        ctx.memory_write("blob", wrapper, namespace="big")

        executor = DAGExecutor()
        snapshot = executor._capture_snapshot(
            executed_nodes=set(),
            all_results=[],
            context=ctx,
        )

        # Simulate intermediate mutation that we want to roll back.
        ctx.memory_write("blob", "junk", namespace="big")
        assert ctx.memory["big"]["blob"] == "junk"

        executor._restore_snapshot(
            snapshot,
            nodes=[],
            context=ctx,
        )

        restored = ctx.memory["big"]["blob"]
        assert isinstance(restored, LazyMemoryValue)
        assert restored.value is big


# ---------------------------------------------------------------------------
# Integration: real chain execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_value_survives_parallel_batch_execution() -> None:
    """Two tool steps run in parallel; both should see the same payload id."""
    payload = list(range(5000))
    wrapper = LazyMemoryValue(payload)

    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.memory_write("blob", wrapper, namespace="big")

    observations: list[dict] = []

    def make_inspect(snapshot_ctx_holder: dict):
        def inspect() -> dict:
            # Tool runs inside a parallel snapshot. The snapshot's memory
            # is created by COWMemoryStore with the parent dict as base —
            # so reading the wrapper here returns the shared payload.
            real_ctx = snapshot_ctx_holder["ctx"]
            value = real_ctx.memory_read("blob", namespace="big")
            assert isinstance(value, LazyMemoryValue)
            observation = {"payload_id": id(value.value), "payload_len": len(value.value)}
            observations.append(observation)
            return observation
        return inspect

    holder = {"ctx": ctx}
    ctx.register_tool("inspect", make_inspect(holder))

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="inspect-a",
                config=ToolStepConfig(tool_name="inspect"),
            ),
            ToolStepDescription(
                number=2,
                title="inspect-b",
                config=ToolStepConfig(tool_name="inspect"),
            ),
        ],
        max_workers=2,
    )

    result = await chain.execute_async(ctx)
    assert all(step.success for step in result.step_results), [
        s.error_message for s in result.step_results
    ]
    assert len(observations) == 2
    for obs in observations:
        assert obs["payload_id"] == id(payload)
        assert obs["payload_len"] == 5000


@pytest.mark.asyncio
async def test_mutating_via_mutable_copy_isolates_branch() -> None:
    """A branch that mutates via mutable_copy must not affect the original."""
    payload = [1, 2, 3]
    wrapper = LazyMemoryValue(payload)

    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.memory_write("blob", wrapper, namespace="big")

    holder = {"ctx": ctx}

    def fork_mutate() -> str:
        real_ctx = holder["ctx"]
        original = real_ctx.memory_read("blob", namespace="big")
        forked = original.mutable_copy()
        forked.value.append(999)
        real_ctx.memory_write("blob_forked", forked, namespace="big")
        return "ok"

    ctx.register_tool("fork_mutate", fork_mutate)

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="mutate",
                config=ToolStepConfig(tool_name="fork_mutate"),
            ),
        ],
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success, result.step_results[0].error_message

    # Original payload untouched.
    assert payload == [1, 2, 3]
    assert ctx.memory["big"]["blob"].value is payload

    forked_wrapper = ctx.memory["big"]["blob_forked"]
    assert isinstance(forked_wrapper, LazyMemoryValue)
    assert forked_wrapper.value == [1, 2, 3, 999]
    assert forked_wrapper.value is not payload


@pytest.mark.asyncio
async def test_payload_not_duplicated_when_chain_uses_replan_checkpoints() -> None:
    """Checkpoint-enabled steps capture snapshots; payload must stay shared."""
    payload_marker = object()
    wrapper = LazyMemoryValue(payload_marker)

    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.memory_write("token", wrapper, namespace="ns")

    seen_payload_ids: list[int] = []
    holder = {"ctx": ctx}

    def record() -> str:
        real_ctx = holder["ctx"]
        v = real_ctx.memory_read("token", namespace="ns")
        assert isinstance(v, LazyMemoryValue)
        seen_payload_ids.append(id(v.value))
        return "ok"

    ctx.register_tool("record", record)

    # Mark step 1 as a checkpoint so the executor captures a snapshot,
    # which is the path that uses copy.deepcopy on memory.
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="checkpoint-step",
                checkpoint=True,
                config=ToolStepConfig(tool_name="record"),
            ),
            ToolStepDescription(
                number=2,
                title="follow",
                dependencies=[1],
                config=ToolStepConfig(tool_name="record"),
            ),
        ],
    )

    result = await chain.execute_async(ctx)
    assert all(step.success for step in result.step_results), [
        s.error_message for s in result.step_results
    ]
    # Both steps saw the same payload object — checkpoint capture did not
    # duplicate it.
    assert len(seen_payload_ids) == 2
    assert all(pid == id(payload_marker) for pid in seen_payload_ids)
