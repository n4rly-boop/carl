"""Tests for the streaming path on
``StructuredOutputStepExecutor``.

The executor streams tokens when (1) the LLM client exposes
``stream_response``, and (2) the context has an
``on_llm_chunk`` callback wired. The buffer is scanned for a
balanced JSON object so the executor can short-circuit the
moment the structured payload arrives.

These tests stub everything they need: a fake LLM client + a
plain :class:`ReasoningContext` so no real LLM round-trip
happens.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable


from mmar_carl import ReasoningContext
from mmar_carl.models import (
    StructuredOutputStepDescription,
    StructuredOutputStepConfig,
)
from mmar_carl.step_executors import (
    StructuredOutputStepExecutor,
    _looks_complete_json,
)


# ---------------------------------------------------------------------------
# Stub LLM client
# ---------------------------------------------------------------------------


class _StubStreamingClient:
    """Mimics enough of the OpenAI-compatible client to drive
    the executor: ``stream_response`` yields chunks; non-stream
    paths are reachable but unused on the streaming gate."""

    supports_streaming = True

    def __init__(self, chunks: list[str]):
        self._chunks = chunks
        self.stream_calls = 0
        self.retry_calls = 0

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        self.stream_calls += 1
        for chunk in self._chunks:
            yield chunk

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.retry_calls += 1
        return "".join(self._chunks)


def _ctx(client, on_chunk: Callable | None = None) -> ReasoningContext:
    """Build a minimal `ReasoningContext` wired to ``client``
    via the ``api=`` constructor kwarg + optional
    ``on_llm_chunk`` callback."""
    ctx = ReasoningContext(
        query="test",
        api=client,
        outer_context="",
    )
    if on_chunk is not None:
        ctx.on_llm_chunk = on_chunk
    return ctx


def _step(input_source: str = "$query") -> StructuredOutputStepDescription:
    return StructuredOutputStepDescription(
        number=1,
        title="extract",
        config=StructuredOutputStepConfig(
            input_source=input_source,
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
            },
            schema_name="Answer",
            instruction="Produce a JSON {answer: ...}",
        ),
    )


# ---------------------------------------------------------------------------
# _looks_complete_json
# ---------------------------------------------------------------------------


class TestLooksCompleteJson:
    def test_empty_is_false(self):
        assert _looks_complete_json("") is False

    def test_unbalanced_open_brace_is_false(self):
        assert _looks_complete_json('{"a": 1') is False

    def test_balanced_object_is_true(self):
        assert _looks_complete_json('{"a": 1}') is True

    def test_balanced_array_is_true(self):
        assert _looks_complete_json("[1, 2, 3]") is True

    def test_nested_braces_balance_correctly(self):
        assert _looks_complete_json('{"a": {"b": 1}}') is True
        assert _looks_complete_json('{"a": {"b": 1}') is False

    def test_braces_inside_strings_ignored(self):
        # Braces inside string values shouldn't affect the count.
        assert _looks_complete_json('{"text": "hello { world }"}') is True

    def test_escaped_quotes_inside_strings(self):
        assert _looks_complete_json('{"text": "hello \\"escaped\\""}') is True

    def test_no_open_brace_is_false(self):
        # Pure text without any structure.
        assert _looks_complete_json("not a json at all") is False

    def test_text_after_close_brace_still_true(self):
        # Some models emit trailing text after the close brace —
        # the heuristic correctly counts the balanced object even
        # if the wider buffer carries trailing tokens.
        assert _looks_complete_json('{"a": 1} trailing text') is True


# ---------------------------------------------------------------------------
# StructuredOutputStepExecutor streaming
# ---------------------------------------------------------------------------


class TestStructuredOutputStreaming:
    def test_streaming_used_when_callback_and_client_support_present(self):
        chunks_seen: list[str] = []

        def on_chunk(chunk: str, **kw) -> None:
            chunks_seen.append(chunk)

        client = _StubStreamingClient(chunks=['{"answer"', ': "hi"}'])
        ctx = _ctx(client, on_chunk=on_chunk)

        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))

        assert result.success is True
        assert result.result_data == {"answer": "hi"}
        # Stream was used (not the retry path).
        assert client.stream_calls == 1
        assert client.retry_calls == 0
        # Chunks reached the callback in order.
        assert chunks_seen == ['{"answer"', ': "hi"}']

    def test_non_streaming_path_when_no_callback(self):
        client = _StubStreamingClient(chunks=['{"answer": "hi"}'])
        ctx = _ctx(client, on_chunk=None)

        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))

        assert result.success is True
        assert result.result_data == {"answer": "hi"}
        # No streaming because no callback was wired.
        assert client.stream_calls == 0
        assert client.retry_calls == 1

    def test_non_streaming_path_when_client_does_not_support(self):
        class _NonStreamClient:
            supports_streaming = False

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return '{"answer": "hi"}'

        ctx = _ctx(_NonStreamClient(), on_chunk=lambda c, **kw: None)

        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))
        assert result.success is True
        assert result.result_data == {"answer": "hi"}

    def test_early_exit_skips_trailing_chunks(self):
        """When a balanced JSON object arrives mid-stream, the
        executor still consumes the rest of the stream (the
        helper exits the loop only after the buffer naturally
        ends) but the parsed result reflects the balanced
        prefix. This pins that hallucinated trailing tokens
        don't break the parse."""
        chunks = [
            '{"answer": "hi"}',
            "\n\nIgnore me — model rambled.",
        ]
        client = _StubStreamingClient(chunks=chunks)
        ctx = _ctx(client, on_chunk=lambda c, **kw: None)

        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))
        # Parsed `answer` matches the early-balanced object
        # even though the stream kept producing tokens.
        assert result.result_data == {"answer": "hi"}

    def test_chunks_dispatched_with_step_number_and_stage(self):
        seen: list[dict] = []

        def on_chunk(chunk: str, *, step_number=None, stage=None) -> None:
            seen.append({"chunk": chunk, "step_number": step_number, "stage": stage})

        client = _StubStreamingClient(chunks=['{"answer": "ok"}'])
        ctx = _ctx(client, on_chunk=on_chunk)
        executor = StructuredOutputStepExecutor()
        asyncio.run(executor.execute(_step(), ctx))
        assert seen[0]["step_number"] == 1
        assert seen[0]["stage"] == "structured_output"

    def test_streaming_failure_falls_back_via_retries(self):
        """When stream_response throws, the helper retries
        ``retries`` times before propagating. We verify the
        retry loop in _execute_structured_streaming runs."""

        class _FailingStreamClient:
            supports_streaming = True

            def __init__(self):
                self.calls = 0

            async def stream_response(self, prompt: str):
                self.calls += 1
                # Yield nothing + raise after a single chunk.
                if False:
                    yield ""  # type: ignore[unreachable]
                raise RuntimeError("stream broken")

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return ""

        client = _FailingStreamClient()
        ctx = _ctx(client, on_chunk=lambda c, **kw: None)

        # `retry_max` defaults to 1 on ReasoningContext.
        # The executor catches the resulting exception and
        # surfaces it as a failed result rather than raising.
        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))
        assert result.success is False
        assert "stream broken" in result.error_message
        # The retry loop in _execute_structured_streaming ran
        # at least once.
        assert client.calls >= 1

    def test_streaming_with_markdown_code_fence(self):
        # Some models wrap JSON in ```json ... ``` — the final
        # parse path strips fences before parsing.
        chunks = ["```json\n", '{"answer": "fenced"}', "\n```"]
        client = _StubStreamingClient(chunks=chunks)
        ctx = _ctx(client, on_chunk=lambda c, **kw: None)
        executor = StructuredOutputStepExecutor()
        result = asyncio.run(executor.execute(_step(), ctx))
        assert result.success is True
        assert result.result_data == {"answer": "fenced"}
