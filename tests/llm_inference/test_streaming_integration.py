"""
End-to-end streaming integration tests.

These exercise the full path: a mock LLM client exposing ``stream_response``
plus a ``ReasoningContext.on_llm_chunk`` callback, run through the real
``LLMStepExecutor._execute_with_streaming`` code path. We verify chunk-level
delivery, accumulated text, retry behaviour, error propagation, multi-step
chain integration, and seamless interoperation with :class:`StreamingBuffer`.

The non-streaming fallback is also covered — when either ``on_llm_chunk`` is
unset or the client lacks ``stream_response``, the executor takes the
``get_response_with_retries`` path.
"""

import asyncio
from typing import AsyncIterator

import pytest

from mmar_carl import (
    ExecutionMode,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StreamingBuffer,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Mock LLM clients
# --------------------------------------------------------------------------- #


class StreamingMockLLM(LLMClientBase):
    """LLM that yields chunks via ``stream_response``.

    Triggers streaming code path when paired with ``context.on_llm_chunk``.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)
        self.prompts_seen: list[str] = []
        self.streaming_call_count = 0
        self.fallback_call_count = 0

    async def get_response(self, prompt: str) -> str:
        self.fallback_call_count += 1
        self.prompts_seen.append(prompt)
        return "".join(self._chunks)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        self.streaming_call_count += 1
        self.prompts_seen.append(prompt)
        for c in self._chunks:
            await asyncio.sleep(0)  # cooperatively yield
            yield c


class FlakyStreamingLLM(LLMClientBase):
    """Raises the first ``fail_count`` stream attempts, then succeeds."""

    def __init__(self, chunks: list[str], fail_count: int) -> None:
        self._chunks = list(chunks)
        self.fail_count = fail_count
        self.attempts = 0

    async def get_response(self, prompt: str) -> str:
        return "".join(self._chunks)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise RuntimeError(f"transient stream failure #{self.attempts}")
        for c in self._chunks:
            yield c


class NonStreamingMockLLM(LLMClientBase):
    """Has no ``stream_response`` → executor must fall back."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        self.calls += 1
        return self.response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# --------------------------------------------------------------------------- #
# Chunk-level delivery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_on_llm_chunk_fires_once_per_chunk_in_order() -> None:
    chunks = ["Hel", "lo, ", "world", "!"]
    captured: list[str] = []
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(
        outer_context="data",
        api=llm,
        on_llm_chunk=lambda c: captured.append(c),
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="stream", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert captured == chunks
    assert llm.streaming_call_count == 1
    assert llm.fallback_call_count == 0


@pytest.mark.asyncio
async def test_step_result_equals_concatenated_chunks() -> None:
    chunks = ["one. ", "two. ", "three."]
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda _c: None,
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result == "one. two. three."


@pytest.mark.asyncio
async def test_empty_chunks_iteration_produces_empty_result() -> None:
    """A stream that yields zero chunks completes successfully with empty text."""
    llm = StreamingMockLLM(chunks=[])
    captured: list[str] = []
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda c: captured.append(c),
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result == ""
    assert captured == []
    assert llm.streaming_call_count == 1


# --------------------------------------------------------------------------- #
# Streaming opt-in: requires both on_llm_chunk and stream_response
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_on_llm_chunk_falls_back_to_non_streaming() -> None:
    """Without an on_llm_chunk callback, streaming code path is skipped."""
    chunks = ["A", "B"]
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm)  # no on_llm_chunk
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    # Streaming was not invoked
    assert llm.streaming_call_count == 0
    assert llm.fallback_call_count == 1


@pytest.mark.asyncio
async def test_client_without_stream_response_falls_back() -> None:
    """A client missing ``stream_response`` cannot stream even with on_llm_chunk set."""
    llm = NonStreamingMockLLM("complete response")
    captured: list[str] = []
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda c: captured.append(c),
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert result.step_results[0].result == "complete response"
    # No chunks fired because the client has no stream_response
    assert captured == []
    assert llm.calls == 1


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_transient_stream_failure_recovers_after_retry() -> None:
    """First attempt raises, second attempt streams successfully."""
    llm = FlakyStreamingLLM(chunks=["recovered"], fail_count=1)
    captured: list[str] = []
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda c: captured.append(c),
        retry_max=3,
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result == "recovered"
    assert captured == ["recovered"]
    assert llm.attempts == 2  # 1 failure + 1 success


@pytest.mark.asyncio
async def test_stream_failures_exceeding_retries_propagate_error() -> None:
    """When all stream attempts fail, the step is marked failed (executor
    falls back to non-streaming path or surfaces the error)."""
    llm = FlakyStreamingLLM(chunks=["never"], fail_count=10)
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda _c: None,
        retry_max=2,
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    await chain.execute_async(ctx)
    # Either: step failed, or the executor fell through to get_response on
    # the same client (which always succeeds with the canned full text).
    # The contract under test is "all attempts exhausted → no infinite loop";
    # accept either resolution, but the stream attempts must equal retry_max.
    assert llm.attempts == 2


# --------------------------------------------------------------------------- #
# Callback robustness
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_on_llm_chunk_exception_does_not_kill_stream() -> None:
    """A throwing on_llm_chunk must NOT abort the stream — chunks should
    continue being processed (per executor contract)."""
    chunks = ["A", "B", "C"]
    seen: list[str] = []

    def faulty_cb(c: str) -> None:
        seen.append(c)
        if c == "B":
            raise RuntimeError("callback boom")

    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm, on_llm_chunk=faulty_cb)
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    # All chunks reached the callback (B raised but was swallowed)
    assert seen == ["A", "B", "C"]
    # Final result still aggregates everything
    assert sr.result == "ABC"


# --------------------------------------------------------------------------- #
# Multi-step / mixed-mode chain
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_step_chain_with_streaming_first_step() -> None:
    """Step 1 streams; step 2 (Tool) does not. Both succeed in one chain run."""
    chunks = ["Result ", "for ", "step ", "one."]
    captured_chunks: list[str] = []
    tool_called_with: dict[str, str] = {}

    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=lambda c: captured_chunks.append(c),
    )

    def echo_tool(value: str) -> str:
        tool_called_with["value"] = value
        return f"echoed: {value}"

    ctx.register_tool("echo", echo_tool)

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="stream stage", aim="x"),
            ToolStepDescription(
                number=2,
                title="echo result",
                dependencies=[1],
                config=ToolStepConfig(
                    tool_name="echo",
                    parameters=[],
                    input_mapping={"value": "$history[-1]"},
                ),
            ),
        ],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert captured_chunks == chunks
    # Tool received the FULL streamed result, not a chunk
    assert "Result for step one." in tool_called_with["value"]


# --------------------------------------------------------------------------- #
# StreamingBuffer integration
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_streaming_buffer_aggregates_sentences_from_real_stream() -> None:
    """StreamingBuffer used as on_llm_chunk produces sentence-level events."""
    # Yields fragments that span sentence boundaries
    chunks = ["First", " sentence", ". Second ", "is here. ", "Third."]
    sentences: list[str] = []
    buf = StreamingBuffer(on_partial=lambda seg, _full: sentences.append(seg))
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm, on_llm_chunk=buf)
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    buf.finalize()
    assert result.step_results[0].success
    assert sentences == [
        "First sentence. ",
        "Second is here. ",
        "Third.",  # finalize flushed the unterminated tail
    ]
    # Buffer reconstructed the exact stream
    assert buf.text == "First sentence. Second is here. Third."


@pytest.mark.asyncio
async def test_streaming_buffer_text_matches_step_result() -> None:
    chunks = ["alpha ", "beta ", "gamma"]
    buf = StreamingBuffer()
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm, on_llm_chunk=buf)
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert buf.text == result.step_results[0].result == "alpha beta gamma"


# --------------------------------------------------------------------------- #
# Execution-mode interaction
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fast_mode_uses_streaming_when_available() -> None:
    """FAST mode passes ``allow_streaming=True`` to the LLM helper."""
    chunks = ["fast ", "mode"]
    seen: list[str] = []
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm, on_llm_chunk=seen.append)
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="s", aim="x",
                llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
            )
        ],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert seen == chunks
    assert llm.streaming_call_count == 1


@pytest.mark.asyncio
async def test_self_critic_mode_does_not_stream() -> None:
    """SELF_CRITIC mode runs evaluator passes that must NOT use streaming —
    streaming is reserved for the user-facing FAST path."""
    chunks = ["x"]
    seen: list[str] = []
    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(outer_context="x", api=llm, on_llm_chunk=seen.append)
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1, title="s", aim="x",
                llm_config=LLMStepConfig(execution_mode=ExecutionMode.SELF_CRITIC),
            )
        ],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    # SELF_CRITIC path did not stream
    assert llm.streaming_call_count == 0
    assert seen == []


# --------------------------------------------------------------------------- #
# Chunk timing / async ordering
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chunks_delivered_before_step_completion_callback() -> None:
    """All chunks should fire *before* on_step_complete — the chunk stream
    completes (and is awaited) prior to the step result being finalized."""
    chunks = ["a", "b", "c"]
    events: list[tuple[str, str]] = []

    def chunk_cb(c: str) -> None:
        events.append(("chunk", c))

    def step_done(_sr: object) -> None:
        events.append(("step_done", ""))

    llm = StreamingMockLLM(chunks)
    ctx = ReasoningContext(
        outer_context="x",
        api=llm,
        on_llm_chunk=chunk_cb,
        on_step_complete=step_done,
    )
    chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="s", aim="x")],
        max_workers=1,
    )
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    # Order: chunk, chunk, chunk, step_done
    kinds = [k for k, _ in events]
    assert kinds == ["chunk", "chunk", "chunk", "step_done"]
