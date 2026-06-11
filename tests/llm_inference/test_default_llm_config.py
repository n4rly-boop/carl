"""Tests for chain-level ``default_llm_config`` on ``ReasoningChain``.

Covers: constructor wiring, ``ChainBuilder.with_default_llm_config``,
JSON serialization round-trip, and the merge semantics in
``ReasoningContext.get_llm_client_for_step`` — step-level config wins,
unset step-level fields fall back to chain default.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    ChainBuilder,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.llm import OpenAIClientConfig, OpenAICompatibleClient


# ---------------------------------------------------------------------------
# Construction & builder API
# ---------------------------------------------------------------------------


class TestDefaultLLMConfigConstruction:
    def test_constructor_accepts_default_llm_config(self) -> None:
        cfg = LLMStepConfig(model="m1", temperature=0.3, max_tokens=512)
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
            default_llm_config=cfg,
        )
        assert chain.default_llm_config is cfg

    def test_default_is_none_when_not_set(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
        )
        assert chain.default_llm_config is None

    def test_builder_with_default_llm_config(self) -> None:
        cfg = LLMStepConfig(model="builder-default", temperature=0.0)
        chain = (
            ChainBuilder()
            .with_default_llm_config(cfg)
            .add_step(
                number=1,
                title="A",
                aim="x",
                reasoning_questions="",
                stage_action="",
                example_reasoning="",
            )
            .build()
        )
        assert chain.default_llm_config is cfg
        assert chain.default_llm_config.model == "builder-default"

    def test_builder_returns_self_for_chaining(self) -> None:
        builder = ChainBuilder()
        returned = builder.with_default_llm_config(LLMStepConfig(model="x"))
        assert returned is builder


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestDefaultLLMConfigSerialization:
    def test_to_dict_includes_default_llm_config(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
            default_llm_config=LLMStepConfig(model="m1", temperature=0.5),
        )
        d = chain.to_dict()
        assert "default_llm_config" in d
        assert d["default_llm_config"]["model"] == "m1"
        assert d["default_llm_config"]["temperature"] == 0.5

    def test_to_dict_omits_when_none(self) -> None:
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
        )
        d = chain.to_dict()
        assert "default_llm_config" not in d

    def test_from_dict_round_trip(self) -> None:
        original = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="A", aim="x")],
            default_llm_config=LLMStepConfig(
                model="round-trip", temperature=0.7, max_tokens=2048
            ),
        )
        restored = ReasoningChain.from_dict(original.to_dict(), use_typed_steps=True)
        assert restored.default_llm_config is not None
        assert restored.default_llm_config.model == "round-trip"
        assert restored.default_llm_config.temperature == 0.7
        assert restored.default_llm_config.max_tokens == 2048


# ---------------------------------------------------------------------------
# Merge semantics — get_llm_client_for_step
# ---------------------------------------------------------------------------


def _make_openai_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        OpenAIClientConfig(api_key="dummy", model="ctx-default")
    )


class TestMergeSemantics:
    def test_chain_default_used_when_step_has_no_config(self) -> None:
        ctx = ReasoningContext(
            outer_context="x", api=_make_openai_client(), model="ctx-default"
        )
        ctx.metadata["__default_llm_config"] = LLMStepConfig(
            model="chain-default", temperature=0.2, max_tokens=100
        )
        client = ctx.get_llm_client_for_step(llm_config=None)
        assert client.config.model == "chain-default"
        assert client.config.temperature == 0.2
        assert client.config.max_tokens == 100

    def test_step_config_overrides_chain_default(self) -> None:
        ctx = ReasoningContext(
            outer_context="x", api=_make_openai_client(), model="ctx-default"
        )
        ctx.metadata["__default_llm_config"] = LLMStepConfig(
            model="chain-default", temperature=0.2, max_tokens=100
        )
        client = ctx.get_llm_client_for_step(
            llm_config=LLMStepConfig(
                model="step-model", temperature=0.9, max_tokens=4096
            )
        )
        assert client.config.model == "step-model"
        assert client.config.temperature == 0.9
        assert client.config.max_tokens == 4096

    def test_step_config_partial_override_fills_from_chain_default(self) -> None:
        """Step only sets `model`; `temperature` + `max_tokens` come from chain."""
        ctx = ReasoningContext(
            outer_context="x", api=_make_openai_client(), model="ctx-default"
        )
        ctx.metadata["__default_llm_config"] = LLMStepConfig(
            model="chain-default", temperature=0.2, max_tokens=100
        )
        client = ctx.get_llm_client_for_step(
            llm_config=LLMStepConfig(model="step-only-model")
        )
        assert client.config.model == "step-only-model"
        # Unset on step → inherited from chain default
        assert client.config.temperature == 0.2
        assert client.config.max_tokens == 100

    def test_no_chain_default_and_no_step_config_uses_context_client(self) -> None:
        ctx_client = _make_openai_client()
        ctx = ReasoningContext(outer_context="x", api=ctx_client, model="ctx-default")
        client = ctx.get_llm_client_for_step(llm_config=None)
        # Returns the default context client (no override needed).
        assert client is ctx_client

    def test_step_temperature_zero_is_respected_not_treated_as_missing(self) -> None:
        """Step explicitly sets temperature=0.0; should NOT be overwritten by
        chain default (the merge uses `is not None`, not truthiness)."""
        ctx = ReasoningContext(
            outer_context="x", api=_make_openai_client(), model="ctx-default"
        )
        ctx.metadata["__default_llm_config"] = LLMStepConfig(
            model="chain-default", temperature=0.8
        )
        client = ctx.get_llm_client_for_step(
            llm_config=LLMStepConfig(temperature=0.0)
        )
        # Step's 0.0 wins; model still inherited
        assert client.config.temperature == 0.0
        assert client.config.model == "chain-default"


# ---------------------------------------------------------------------------
# End-to-end via execute_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chain_injects_default_into_context_metadata_during_execute() -> None:
    """During `execute_async`, the chain default is put on
    `context.metadata['__default_llm_config']` so step executors see it."""
    cfg = LLMStepConfig(model="exec-default", temperature=0.1)

    from mmar_carl import ToolStepConfig, ToolStepDescription

    seen_metadata: dict = {}

    def probe_tool(ctx: ReasoningContext) -> str:
        # Snapshot metadata visible inside step execution.
        seen_metadata["snapshot"] = dict(ctx.metadata)
        return "ok"

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="probe",
                config=ToolStepConfig(
                    tool_name="noop",
                    input_mapping={"ctx": "$context"},
                ),
            ),
        ],
        default_llm_config=cfg,
    )

    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("noop", probe_tool)

    result = await chain.execute_async(ctx)
    # Even if input_mapping doesn't pass the ctx, metadata is on the *context*
    # object itself (mutated by execute_async), so we can read it directly.
    assert result.success or "__default_llm_config" in ctx.metadata
    assert "__default_llm_config" in ctx.metadata
    injected = ctx.metadata["__default_llm_config"]
    assert injected.model == "exec-default"
    assert injected.temperature == 0.1


@pytest.mark.asyncio
async def test_chain_without_default_does_not_inject_metadata_key() -> None:
    from mmar_carl import ToolStepConfig, ToolStepDescription

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="probe",
                config=ToolStepConfig(tool_name="noop"),
            ),
        ],
    )
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("noop", lambda: "ok")

    await chain.execute_async(ctx)
    assert "__default_llm_config" not in ctx.metadata
