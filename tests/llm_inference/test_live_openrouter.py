"""Opt-in live tests against OpenRouter.

These tests hit a real OpenAI-compatible API endpoint — they're
skipped by default (``addopts = -m 'not live'`` in ``pyproject.toml``)
and run only when invoked explicitly with ``pytest -m live`` or via
``make test-live``.

Each test mirrors a scenario from ``examples/llm_inference/openrouter_example.py``
but asserts on **shape** (success flag, structural fields, token-usage
> 0) rather than exact content — so the suite stays resilient to
model drift while still catching real regressions in the live-API
code paths (streaming, retries, pricing detection, per-step model
overrides, etc.).
"""

from __future__ import annotations

import os

import pytest

from mmar_carl import (
    Language,
    LLMStepConfig,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    create_openai_client,
)

# Skip the whole module when no API key is set. Lets developers run
# `pytest -m live` locally without forcing an env var on every command.
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — set it (any OpenRouter key works) to run live tests",
    ),
]

# Cheapest reliable model on OpenRouter at time of writing (~$0.05/M tokens).
LIVE_MODEL = os.environ.get("LIVE_TEST_MODEL", "qwen/qwen3-8b")
LIVE_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# A small data fixture used across scenarios.
SAMPLE_DATA = (
    "Period,Revenue,Expenses,Profit\n"
    "2024-Q1,1500000,1200000,300000\n"
    "2024-Q2,1650000,1280000,370000\n"
    "2024-Q3,1720000,1350000,370000\n"
    "2024-Q4,1850000,1400000,450000\n"
)


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    assert key, "OPENAI_API_KEY must be set for live tests"
    return key


# ---------------------------------------------------------------------------
# Scenario 1: explicit OpenAIClientConfig (mirrors example_with_openrouter)
# ---------------------------------------------------------------------------


class TestExplicitConfig:
    async def test_chain_succeeds_and_records_token_usage(self) -> None:
        config = OpenAIClientConfig(
            base_url=LIVE_BASE_URL,
            api_key=_api_key(),
            model=LIVE_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
        client = OpenAICompatibleClient(config)

        chain = ReasoningChain(steps=[
            LLMStepDescription(
                number=1, title="Revenue trend",
                aim="State whether revenue is increasing or decreasing in one sentence.",
            ),
        ])
        ctx = ReasoningContext(
            outer_context=SAMPLE_DATA, api=client, language=Language.ENGLISH,
        )
        result = await chain.execute_async(ctx)

        # Shape assertions only — content depends on the model.
        assert result.success, result.get_failed_steps()
        assert result.token_usage.get("total", 0) > 0
        assert result.token_usage.get("prompt", 0) > 0
        assert result.token_usage.get("completion", 0) > 0
        final = result.get_final_output()
        assert isinstance(final, str) and final.strip()
        # Per-step model attribution is wired through to the step result
        assert result.step_results[0].model == LIVE_MODEL


# ---------------------------------------------------------------------------
# Scenario 2: factory function (mirrors example_with_factory_function)
# ---------------------------------------------------------------------------


class TestFactoryFunction:
    async def test_create_openai_client_round_trips_kwargs(self) -> None:
        client = create_openai_client(
            api_key=_api_key(),
            model=LIVE_MODEL,
            base_url=LIVE_BASE_URL,
            temperature=0.5,
        )
        assert client.config.model == LIVE_MODEL
        assert client.config.temperature == 0.5
        assert client.model_name == LIVE_MODEL

        chain = ReasoningChain(steps=[
            LLMStepDescription(
                number=1, title="Sum",
                aim="Add 2+2 and reply with just the number.",
            ),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=client)
        result = await chain.execute_async(ctx)
        assert result.success
        assert result.token_usage.get("total", 0) > 0


# ---------------------------------------------------------------------------
# Scenario 3: multi-step chain with per-step model override
# ---------------------------------------------------------------------------


class TestMultiStepWithOverride:
    async def test_per_step_model_override_persists_in_step_result(self) -> None:
        """Per-step ``LLMStepConfig(model=...)`` should cause that step's
        ``StepExecutionResult.model`` to reflect the override, even when
        the rest of the chain uses the context default."""
        api_key = _api_key()
        default_client = create_openai_client(
            api_key=api_key, model=LIVE_MODEL, base_url=LIVE_BASE_URL,
        )

        # Same model on both steps but specified differently — proves the
        # override path is hit without depending on a second provider.
        chain = ReasoningChain(steps=[
            LLMStepDescription(
                number=1, title="Plan",
                aim="Reply with the single word OK.",
            ),
            LLMStepDescription(
                number=2, title="Synth",
                aim="Echo the previous step's output.",
                dependencies=[1],
                llm_config=LLMStepConfig(model=LIVE_MODEL),
            ),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=default_client)
        result = await chain.execute_async(ctx)

        assert result.success
        # Both steps populated their model field
        models = {sr.step_number: sr.model for sr in result.step_results}
        assert models[1] == LIVE_MODEL
        assert models[2] == LIVE_MODEL


# ---------------------------------------------------------------------------
# Scenario 4: streaming surface (mirrors create_openai_client + supports_streaming)
# ---------------------------------------------------------------------------


class TestStreamingSurface:
    async def test_supports_streaming_and_chunks_arrive(self) -> None:
        client = create_openai_client(
            api_key=_api_key(), model=LIVE_MODEL, base_url=LIVE_BASE_URL,
            temperature=0.0,
        )
        assert client.supports_streaming is True

        chunks: list[str] = []
        async for chunk in client.stream_response(
            "Reply with exactly the five digits 1 2 3 4 5 separated by spaces."
        ):
            chunks.append(chunk)

        # At least one non-empty chunk arrived
        assert chunks
        assert any(c.strip() for c in chunks)
        # Concatenation produces a non-empty string (content varies)
        joined = "".join(chunks).strip()
        assert joined
