"""Tests for the Anthropic native client.

Covers the contract surface :class:`AnthropicClient` exposes:

* Lazy SDK import + friendly error when ``anthropic`` is not
  installed.
* Configuration validation.
* Text / usage extraction from native Anthropic content blocks.
* Retry semantics + system-prompt routing through ``system=``.
* Multi-turn messages with system-message concatenation.
* Native tool calling — projecting ``tool_use`` blocks back to
  CARL's ``{id, name, arguments}`` shape.
* Extended thinking — gated on ``thinking_budget``.
* Vision — URL + data-URI handling.
* Streaming — async iterator yielding text chunks.
* Prompt-cache header on the system block when
  ``cache_system=True``.
* ``openai_tools_to_anthropic`` translation.

Tests inject a stub SDK client (``_StubAnthropic``) via the
``sdk_client=`` constructor kwarg so they never hit the network.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

import pytest
from pydantic import ValidationError

from mmar_carl.anthropic_client import (
    AnthropicClient,
    AnthropicClientConfig,
    AnthropicClientError,
    _extract_text,
    _extract_thinking,
    _extract_tool_calls,
    _extract_usage,
    openai_tools_to_anthropic,
)
from mmar_carl.models.llm_client_base import ChatMessage


# ---------------------------------------------------------------------------
# Stub SDK
# ---------------------------------------------------------------------------


class _StubResponse:
    """Mimics enough of ``anthropic.types.Message`` for the
    helpers: ``.content`` is a list of dict-shaped blocks and
    ``.usage`` exposes ``input_tokens`` + ``output_tokens``."""

    def __init__(
        self,
        content: list[dict[str, Any]] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.content = content or []
        if usage is not None:
            self.usage = type(
                "Usage",
                (),
                {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            )()
        else:
            self.usage = None


class _StubStream:
    """Async context manager mimicking ``client.messages.stream(...)``."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> "_StubStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self) -> AsyncIterator[str]:
        return self._text_iter()

    async def _text_iter(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class _StubMessages:
    def __init__(
        self,
        *,
        response: _StubResponse | None = None,
        stream_chunks: list[str] | None = None,
        fail_until: int = 0,
    ) -> None:
        self.response = response or _StubResponse(
            content=[{"type": "text", "text": "default"}]
        )
        self.stream_chunks = stream_chunks or []
        self.fail_until = fail_until
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.create_calls.append(kwargs)
        if len(self.create_calls) <= self.fail_until:
            raise RuntimeError("transient API failure")
        return self.response

    def stream(self, **kwargs: Any) -> _StubStream:
        self.stream_calls.append(kwargs)
        return _StubStream(self.stream_chunks)


class _StubAnthropic:
    def __init__(
        self,
        *,
        response: _StubResponse | None = None,
        stream_chunks: list[str] | None = None,
        fail_until: int = 0,
    ) -> None:
        self.messages = _StubMessages(
            response=response,
            stream_chunks=stream_chunks,
            fail_until=fail_until,
        )


def _cfg(**overrides: Any) -> AnthropicClientConfig:
    base: dict[str, Any] = {"api_key": "sk-test", "model": "claude-3-7-sonnet-latest"}
    base.update(overrides)
    return AnthropicClientConfig(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_requires_api_key(self):
        with pytest.raises(ValidationError):
            AnthropicClientConfig()  # type: ignore[call-arg]

    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            AnthropicClientConfig(api_key="x", temperature=1.5)

    def test_max_tokens_positive(self):
        with pytest.raises(ValidationError):
            AnthropicClientConfig(api_key="x", max_tokens=0)

    def test_defaults(self):
        cfg = AnthropicClientConfig(api_key="x")
        assert cfg.model == "claude-3-7-sonnet-latest"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.thinking_budget is None
        assert cfg.cache_system is False


# ---------------------------------------------------------------------------
# Lazy SDK import
# ---------------------------------------------------------------------------


def _anthropic_installed() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


class TestLazyImport:
    @pytest.mark.skipif(
        _anthropic_installed(),
        reason="needs an env where anthropic SDK is NOT installed",
    )
    def test_missing_sdk_raises_friendly_error(self):
        client = AnthropicClient(_cfg())
        with pytest.raises(AnthropicClientError) as excinfo:
            _ = client.client
        assert "anthropic SDK is not installed" in str(excinfo.value)
        assert "pip install" in str(excinfo.value)

    def test_injected_sdk_client_bypasses_import(self):
        stub = _StubAnthropic()
        client = AnthropicClient(_cfg(), sdk_client=stub)
        # No import error even if SDK is missing — the stub is used.
        assert client.client is stub


# ---------------------------------------------------------------------------
# Introspection properties
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_properties_match_config(self):
        client = AnthropicClient(
            _cfg(temperature=0.3, max_tokens=2048), sdk_client=_StubAnthropic()
        )
        assert client.model_name == "claude-3-7-sonnet-latest"
        assert client.temperature == 0.3
        assert client.max_tokens == 2048
        assert client.supports_streaming is True


# ---------------------------------------------------------------------------
# get_response — text extraction + create kwargs
# ---------------------------------------------------------------------------


class TestGetResponse:
    def test_assembles_text_from_blocks(self):
        stub = _StubAnthropic(
            response=_StubResponse(
                content=[
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ]
            )
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        result = asyncio.run(client.get_response("hi"))
        assert result == "Hello world"
        # Verify create kwargs.
        call = stub.messages.create_calls[0]
        assert call["model"] == "claude-3-7-sonnet-latest"
        assert call["max_tokens"] == 4096
        assert call["temperature"] == 0.7
        assert call["messages"] == [{"role": "user", "content": "hi"}]
        # No system, no tools, no thinking when not configured.
        assert "system" not in call
        assert "tools" not in call
        assert "thinking" not in call

    def test_ignores_non_text_blocks(self):
        stub = _StubAnthropic(
            response=_StubResponse(
                content=[
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "1", "name": "foo", "input": {}},
                ]
            )
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        assert asyncio.run(client.get_response("hi")) == "answer"


# ---------------------------------------------------------------------------
# get_response_with_retries
# ---------------------------------------------------------------------------


class TestRetries:
    def test_retries_until_success(self, monkeypatch):
        # Skip the backoff sleeps for speed.
        async def _no_sleep(_):
            return None

        monkeypatch.setattr("mmar_carl.anthropic_client.asyncio.sleep", _no_sleep)
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}]),
            fail_until=2,
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        result = asyncio.run(client.get_response_with_retries("hi", retries=3))
        assert result == "ok"
        assert len(stub.messages.create_calls) == 3

    def test_propagates_after_exhaustion(self, monkeypatch):
        async def _no_sleep(_):
            return None

        monkeypatch.setattr("mmar_carl.anthropic_client.asyncio.sleep", _no_sleep)
        stub = _StubAnthropic(fail_until=10)
        client = AnthropicClient(_cfg(), sdk_client=stub)
        with pytest.raises(RuntimeError, match="transient API failure"):
            asyncio.run(client.get_response_with_retries("hi", retries=2))


# ---------------------------------------------------------------------------
# get_response_with_system
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_routed_to_dedicated_slot(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        asyncio.run(client.get_response_with_system("be concise", "say hi"))
        call = stub.messages.create_calls[0]
        # Plain string when cache_system is off.
        assert call["system"] == "be concise"
        assert call["messages"] == [{"role": "user", "content": "say hi"}]

    def test_cache_system_wraps_in_block_with_cache_control(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}])
        )
        client = AnthropicClient(_cfg(cache_system=True), sdk_client=stub)
        asyncio.run(client.get_response_with_system("be concise", "say hi"))
        call = stub.messages.create_calls[0]
        assert isinstance(call["system"], list)
        block = call["system"][0]
        assert block["type"] == "text"
        assert block["text"] == "be concise"
        assert block["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# get_response_with_usage
# ---------------------------------------------------------------------------


class TestUsage:
    def test_extracts_usage_projection(self):
        stub = _StubAnthropic(
            response=_StubResponse(
                content=[{"type": "text", "text": "ok"}],
                usage={"input_tokens": 12, "output_tokens": 8},
            )
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        text, usage = asyncio.run(client.get_response_with_usage("hi"))
        assert text == "ok"
        assert usage == {"prompt": 12, "completion": 8, "total": 20}

    def test_empty_usage_when_sdk_omits(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        _, usage = asyncio.run(client.get_response_with_usage("hi"))
        assert usage == {}


# ---------------------------------------------------------------------------
# get_response_with_messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_system_messages_concatenated(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        msgs = [
            ChatMessage(role="system", content="rule 1"),
            ChatMessage(role="system", content="rule 2"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
            ChatMessage(role="user", content="bye"),
        ]
        text, usage = asyncio.run(client.get_response_with_messages(msgs))
        assert text == "ok"
        call = stub.messages.create_calls[0]
        assert call["system"] == "rule 1\n\nrule 2"
        assert call["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]

    def test_no_chat_messages_returns_empty(self):
        stub = _StubAnthropic()
        client = AnthropicClient(_cfg(), sdk_client=stub)
        msgs = [ChatMessage(role="system", content="only system")]
        text, usage = asyncio.run(client.get_response_with_messages(msgs))
        assert text == ""
        assert usage == {}
        assert stub.messages.create_calls == []


# ---------------------------------------------------------------------------
# get_response_with_tools
# ---------------------------------------------------------------------------


class TestTools:
    def test_native_tool_use_projected_to_carl_shape(self):
        stub = _StubAnthropic(
            response=_StubResponse(
                content=[
                    {"type": "text", "text": "calling foo"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "foo",
                        "input": {"x": 1},
                    },
                ]
            )
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        tools = [
            {
                "name": "foo",
                "description": "do foo",
                "input_schema": {"type": "object"},
            }
        ]
        text, calls = asyncio.run(
            client.get_response_with_tools("be terse", "use foo", tools)
        )
        assert text == "calling foo"
        assert calls == [{"id": "toolu_1", "name": "foo", "arguments": {"x": 1}}]
        # Verify tools were forwarded.
        call = stub.messages.create_calls[0]
        assert call["tools"] == tools

    def test_messages_arg_overrides_user_prompt(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "ok"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        history = [
            {"role": "system", "content": "from-history-system"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier-reply"},
            {"role": "user", "content": "now"},
        ]
        asyncio.run(
            client.get_response_with_tools(
                "outer-system", "ignored", tools=[], messages=history
            )
        )
        call = stub.messages.create_calls[0]
        # Both system_prompt and the system role in messages concatenated.
        assert call["system"] == "outer-system\n\nfrom-history-system"
        # User/assistant turns retained, system role stripped.
        assert call["messages"] == [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier-reply"},
            {"role": "user", "content": "now"},
        ]


# ---------------------------------------------------------------------------
# Extended thinking
# ---------------------------------------------------------------------------


class TestThinking:
    def test_requires_thinking_budget(self):
        client = AnthropicClient(_cfg(), sdk_client=_StubAnthropic())
        with pytest.raises(AnthropicClientError, match="thinking_budget"):
            asyncio.run(client.get_response_with_thinking("hi"))

    def test_returns_text_thinking_usage(self):
        stub = _StubAnthropic(
            response=_StubResponse(
                content=[
                    {"type": "thinking", "thinking": "let me think..."},
                    {"type": "text", "text": "final answer"},
                ],
                usage={"input_tokens": 5, "output_tokens": 6},
            )
        )
        client = AnthropicClient(_cfg(thinking_budget=1024), sdk_client=stub)
        out = asyncio.run(client.get_response_with_thinking("solve this"))
        assert out["text"] == "final answer"
        assert out["thinking"] == "let me think..."
        assert out["usage"] == {"prompt": 5, "completion": 6, "total": 11}
        # Verify thinking kwarg was passed.
        call = stub.messages.create_calls[0]
        assert call["thinking"] == {"type": "enabled", "budget_tokens": 1024}


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class TestVision:
    def test_url_source(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "an image"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        asyncio.run(
            client.get_response_with_image(
                "describe", "https://example.com/cat.jpg"
            )
        )
        call = stub.messages.create_calls[0]
        msg = call["messages"][0]
        assert msg["role"] == "user"
        blocks = msg["content"]
        assert blocks[0]["type"] == "image"
        assert blocks[0]["source"] == {
            "type": "url",
            "url": "https://example.com/cat.jpg",
        }
        assert blocks[1] == {"type": "text", "text": "describe"}

    def test_data_uri_source(self):
        stub = _StubAnthropic(
            response=_StubResponse(content=[{"type": "text", "text": "a graph"}])
        )
        client = AnthropicClient(_cfg(), sdk_client=stub)
        asyncio.run(
            client.get_response_with_image(
                "describe", "data:image/png;base64,AAAA"
            )
        )
        call = stub.messages.create_calls[0]
        msg = call["messages"][0]
        blocks = msg["content"]
        assert blocks[0]["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": "AAAA",
        }

    def test_malformed_data_uri_raises(self):
        client = AnthropicClient(_cfg(), sdk_client=_StubAnthropic())
        with pytest.raises(AnthropicClientError, match="malformed data URI"):
            asyncio.run(client.get_response_with_image("x", "data:image/png;base64"))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_stream_response_yields_chunks(self):
        stub = _StubAnthropic(stream_chunks=["Hel", "lo", " world"])
        client = AnthropicClient(_cfg(), sdk_client=stub)

        async def _collect() -> list[str]:
            return [chunk async for chunk in client.stream_response("hi")]

        chunks = asyncio.run(_collect())
        assert chunks == ["Hel", "lo", " world"]
        # Stream invocation was made.
        assert len(stub.messages.stream_calls) == 1
        call = stub.messages.stream_calls[0]
        assert call["model"] == "claude-3-7-sonnet-latest"
        assert call["messages"] == [{"role": "user", "content": "hi"}]

    def test_stream_skips_empty_chunks(self):
        stub = _StubAnthropic(stream_chunks=["", "x", ""])
        client = AnthropicClient(_cfg(), sdk_client=stub)

        async def _collect() -> list[str]:
            return [chunk async for chunk in client.stream_response("hi")]

        assert asyncio.run(_collect()) == ["x"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_extract_text_handles_attribute_blocks(self):
        block = type("B", (), {"type": "text", "text": "from-attr"})()
        response = type("R", (), {"content": [block]})()
        assert _extract_text(response) == "from-attr"

    def test_extract_thinking_falls_back_to_text(self):
        block = type("B", (), {"type": "thinking", "text": "thought", "thinking": ""})()
        response = type("R", (), {"content": [block]})()
        assert _extract_thinking(response) == "thought"

    def test_extract_tool_calls_handles_missing_input(self):
        response = _StubResponse(
            content=[{"type": "tool_use", "id": "t", "name": "foo"}]
        )
        assert _extract_tool_calls(response) == [
            {"id": "t", "name": "foo", "arguments": {}}
        ]

    def test_extract_usage_handles_missing(self):
        response = _StubResponse(content=[])
        assert _extract_usage(response) == {}

    def test_openai_to_anthropic_translation(self):
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ]
        result = openai_tools_to_anthropic(openai_tools)
        assert result == [
            {
                "name": "search",
                "description": "search the web",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            }
        ]

    def test_openai_to_anthropic_passes_through_anthropic_shape(self):
        # Already-Anthropic-shaped tool — no `function` nesting.
        anth = [
            {
                "name": "foo",
                "description": "do foo",
                "input_schema": {"type": "object"},
            }
        ]
        assert openai_tools_to_anthropic(anth) == anth
        # Should be a copy, not the same object.
        assert openai_tools_to_anthropic(anth) is not anth

    def test_openai_to_anthropic_empty(self):
        assert openai_tools_to_anthropic([]) == []
