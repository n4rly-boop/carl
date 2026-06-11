"""Tests for the new typed introspection properties on ``LLMClientBase``.

Pyright was flagging ~10 errors across tests and source where call sites
did ``isinstance(client, OpenAICompatibleClient)`` then accessed
``client.config.model`` — bypassing the base type. The base class now
exposes typed ``model_name`` / ``temperature`` / ``max_tokens`` /
``supports_streaming`` accessors so consumers can introspect any
``LLMClientBase`` instance uniformly.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from mmar_carl import LLMClientBase, OpenAIClientConfig, OpenAICompatibleClient


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------


class _BareStub(LLMClientBase):
    """Subclass that overrides nothing optional."""

    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


class _StubWithStreaming(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:  # type: ignore[override]
        for ch in "hello":
            yield ch


class _StubWithCustomIntrospection(LLMClientBase):
    @property
    def model_name(self):
        return "custom-model"

    @property
    def temperature(self):
        return 0.42

    @property
    def max_tokens(self):
        return 999

    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


# ---------------------------------------------------------------------------
# Base defaults
# ---------------------------------------------------------------------------


class TestBaseDefaults:
    def test_model_name_defaults_to_none(self) -> None:
        assert _BareStub().model_name is None

    def test_temperature_defaults_to_none(self) -> None:
        assert _BareStub().temperature is None

    def test_max_tokens_defaults_to_none(self) -> None:
        assert _BareStub().max_tokens is None

    def test_supports_streaming_defaults_to_false(self) -> None:
        assert _BareStub().supports_streaming is False


# ---------------------------------------------------------------------------
# Streaming detection
# ---------------------------------------------------------------------------


class TestStreamingDetection:
    def test_streaming_subclass_detected_as_streaming(self) -> None:
        assert _StubWithStreaming().supports_streaming is True

    def test_non_streaming_subclass_raises_on_invocation(self) -> None:
        client = _BareStub()
        with pytest.raises(NotImplementedError, match="stream_response"):
            client.stream_response("hi")

    def test_streaming_subclass_yields_chunks(self) -> None:
        client = _StubWithStreaming()
        import asyncio

        async def collect():
            chunks = []
            async for ch in client.stream_response("hi"):
                chunks.append(ch)
            return chunks

        assert asyncio.run(collect()) == list("hello")


# ---------------------------------------------------------------------------
# Custom introspection overrides
# ---------------------------------------------------------------------------


class TestCustomOverrides:
    def test_subclass_can_override_model_name(self) -> None:
        assert _StubWithCustomIntrospection().model_name == "custom-model"

    def test_subclass_can_override_temperature(self) -> None:
        assert _StubWithCustomIntrospection().temperature == 0.42

    def test_subclass_can_override_max_tokens(self) -> None:
        assert _StubWithCustomIntrospection().max_tokens == 999


# ---------------------------------------------------------------------------
# OpenAICompatibleClient — the canonical concrete client
# ---------------------------------------------------------------------------


class TestOpenAIClientWiring:
    def _make(self, model="gpt-4o", temp=0.5, max_tok=2048) -> OpenAICompatibleClient:
        cfg = OpenAIClientConfig(api_key="x", model=model, temperature=temp, max_tokens=max_tok)
        return OpenAICompatibleClient(cfg)

    def test_model_name_returns_configured_model(self) -> None:
        assert self._make(model="gpt-4o").model_name == "gpt-4o"

    def test_temperature_returns_configured_temperature(self) -> None:
        assert self._make(temp=0.7).temperature == 0.7

    def test_max_tokens_returns_configured_max(self) -> None:
        assert self._make(max_tok=4096).max_tokens == 4096

    def test_supports_streaming_is_true(self) -> None:
        assert self._make().supports_streaming is True

    def test_config_attribute_still_accessible_for_backward_compat(self) -> None:
        """Pre-existing access pattern via ``client.config.model`` still works."""
        client = self._make(model="claude-3.5", temp=0.1)
        assert client.config.model == "claude-3.5"
        assert client.config.temperature == 0.1


# ---------------------------------------------------------------------------
# Uniform introspection — the actual UX win
# ---------------------------------------------------------------------------


def test_uniform_introspection_across_clients() -> None:
    """Demonstrate that callers can probe any LLMClientBase uniformly via
    the typed properties without isinstance gymnastics."""
    bare = _BareStub()
    custom = _StubWithCustomIntrospection()
    openai = OpenAICompatibleClient(
        OpenAIClientConfig(api_key="x", model="gpt-4o", temperature=0.5, max_tokens=2048)
    )
    clients: list[LLMClientBase] = [bare, custom, openai]

    # All return optional model_name without raising
    names = [c.model_name for c in clients]
    assert names == [None, "custom-model", "gpt-4o"]

    # All declare supports_streaming
    flags = [c.supports_streaming for c in clients]
    assert flags == [False, False, True]
