"""Tests for ``ReasoningResult.context_metadata`` propagation.

The live evolution benchmark hit this gap: a user stashed an expected
answer in ``context.metadata`` before ``chain.execute_async`` to make it
available downstream, then was surprised to find it nowhere in the
returned ``ReasoningResult``. The new ``context_metadata`` field snapshots
``context.metadata`` at the end of execution, with a
``get_context_metadata(include_internal=False)`` helper to filter out
CARL framework keys (those prefixed with ``__``).
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_simple_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="emit",
                config=ToolStepConfig(tool_name="emit"),
            ),
        ],
    )


def _make_two_step_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="A", config=ToolStepConfig(tool_name="emit")
            ),
            ToolStepDescription(
                number=2,
                title="B",
                dependencies=[1],
                config=ToolStepConfig(tool_name="emit"),
            ),
        ],
    )


def _ctx(api=None, model="default") -> ReasoningContext:
    ctx = ReasoningContext(outer_context="x", api=api, model=model)
    ctx.register_tool("emit", lambda: "ok")
    return ctx


# ---------------------------------------------------------------------------
# User metadata round-trips through execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_metadata_appears_in_context_metadata() -> None:
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["my_key"] = "my_value"
    ctx.metadata["another"] = 42

    result = await chain.execute_async(ctx)
    assert result.context_metadata["my_key"] == "my_value"
    assert result.context_metadata["another"] == 42


@pytest.mark.asyncio
async def test_expected_answer_pattern_from_benchmark() -> None:
    """The exact pattern that motivated this fix: smuggle a ground-truth
    value through context.metadata for a downstream metric to read."""
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["__expected_answer"] = 7.0  # double-underscore ⇒ internal-style

    result = await chain.execute_async(ctx)
    # Internal-style key visible in raw context_metadata...
    assert result.context_metadata["__expected_answer"] == 7.0
    # ...and via the include_internal flag
    assert result.get_context_metadata(include_internal=True)["__expected_answer"] == 7.0


@pytest.mark.asyncio
async def test_get_context_metadata_excludes_internal_by_default() -> None:
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["my_key"] = "user_value"
    ctx.metadata["__hidden"] = "secret"

    result = await chain.execute_async(ctx)
    user_metadata = result.get_context_metadata()
    assert user_metadata.get("my_key") == "user_value"
    assert "__hidden" not in user_metadata
    # CARL-set internal keys also filtered out (these are set automatically
    # by chain.execute when chain.default_llm_config / langfuse are set).
    for k in user_metadata:
        assert not k.startswith("__"), f"Internal key '{k}' leaked through"


@pytest.mark.asyncio
async def test_get_context_metadata_include_internal_returns_everything() -> None:
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["__hidden"] = "secret"

    result = await chain.execute_async(ctx)
    full = result.get_context_metadata(include_internal=True)
    assert full["__hidden"] == "secret"


@pytest.mark.asyncio
async def test_get_context_metadata_returns_a_copy() -> None:
    """Mutating the helper's return value must not affect the result."""
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["my_key"] = "v"

    result = await chain.execute_async(ctx)
    snapshot = result.get_context_metadata()
    snapshot["my_key"] = "mutated"
    assert result.context_metadata["my_key"] == "v"


# ---------------------------------------------------------------------------
# Step output keys (step_N) survive — used by $metadata.step_N references
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_output_keys_propagate() -> None:
    """Step executors write ``step_N`` keys to ``context.metadata`` so other
    steps can reference them via ``$metadata.step_N``. Those keys are
    user-facing and should appear in ``context_metadata``."""
    chain = _make_two_step_chain()
    ctx = _ctx()

    result = await chain.execute_async(ctx)
    user_md = result.get_context_metadata()
    assert "step_1" in user_md
    assert "step_2" in user_md
    assert user_md["step_1"] == "ok"
    assert user_md["step_2"] == "ok"


# ---------------------------------------------------------------------------
# Mid-execution writes inside chain are captured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metadata_written_during_chain_execution_propagates() -> None:
    """A tool that writes to context.metadata mid-execution: the written
    value should be visible in result.context_metadata."""
    chain = _make_simple_chain()
    ctx = _ctx()

    holder = {"ctx": ctx}

    def writer() -> str:
        holder["ctx"].metadata["written_by_tool"] = "during_execution"
        return "done"

    ctx.register_tool("emit", writer)

    result = await chain.execute_async(ctx)
    assert result.context_metadata.get("written_by_tool") == "during_execution"


# ---------------------------------------------------------------------------
# Pre-existing metadata field is unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_result_metadata_field_unchanged() -> None:
    """The existing ``result.metadata`` field — used for execution_stats,
    parallel_batches, etc. — must still work as before."""
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["my_key"] = "v"

    result = await chain.execute_async(ctx)
    # The execution_stats / parallel_batches keys still live in result.metadata
    assert "execution_stats" in result.metadata
    assert "parallel_batches" in result.metadata
    # And the user key is NOT polluting result.metadata
    assert "my_key" not in result.metadata


# ---------------------------------------------------------------------------
# Empty / no-step chain edge case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_metadata_present_even_when_chain_has_one_step() -> None:
    """Single-step chains still snapshot context metadata at completion."""
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["pre_exec"] = "value"

    result = await chain.execute_async(ctx)
    assert result.success is True
    assert result.context_metadata.get("pre_exec") == "value"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_metadata_field_serialises_via_model_dump() -> None:
    chain = _make_simple_chain()
    ctx = _ctx()
    ctx.metadata["my_key"] = "v"

    result = await chain.execute_async(ctx)
    dumped = result.model_dump()
    assert "context_metadata" in dumped
    assert dumped["context_metadata"]["my_key"] == "v"
