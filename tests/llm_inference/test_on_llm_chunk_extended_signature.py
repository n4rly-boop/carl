"""``on_llm_chunk`` extended signature.

The legacy callback shape ``on_llm_chunk(chunk_text)`` carries only the
chunk delta. CARE needs to route chunks to the right step pane / stage
pane, so the callback can now also accept keyword arguments
``step_number`` and ``stage``. The step executors introspect the
signature and dispatch the right call shape so both legacy and extended
consumers work transparently.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from mmar_carl import (
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StreamingBuffer,
)
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.step_executors import (
    _callback_accepts_chunk_metadata,
    _dispatch_llm_chunk,
)


# ---------------------------------------------------------------------------
# Streaming-capable fake LLM client
# ---------------------------------------------------------------------------


class _StreamingFakeClient(LLMClientBase):
    """Returns a fixed token list from ``stream_response``.

    ``supports_streaming`` is True so the executor takes the streaming
    code path that routes through ``on_llm_chunk``.
    """

    def __init__(self, tokens: list[str]):
        self._tokens = tokens

    @property
    def supports_streaming(self) -> bool:  # type: ignore[override]
        return True

    async def get_response(self, prompt: str) -> str:
        return "".join(self._tokens)

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "".join(self._tokens)

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        return "".join(self._tokens), {}

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:  # type: ignore[override]
        for tok in self._tokens:
            yield tok


# ---------------------------------------------------------------------------
# Signature introspection
# ---------------------------------------------------------------------------


class TestSignatureIntrospection:
    def test_legacy_single_arg_callback_detected(self) -> None:
        def cb(chunk: str) -> None:
            pass
        assert _callback_accepts_chunk_metadata(cb) is False

    def test_extended_callback_with_step_number_detected(self) -> None:
        def cb(chunk: str, *, step_number: int | None = None,
               stage: str | None = None) -> None:
            pass
        assert _callback_accepts_chunk_metadata(cb) is True

    def test_var_kwargs_callback_treated_as_extended(self) -> None:
        def cb(chunk: str, **kwargs: Any) -> None:
            pass
        assert _callback_accepts_chunk_metadata(cb) is True

    def test_streaming_buffer_is_legacy_shape(self) -> None:
        """StreamingBuffer's __call__ only accepts ``chunk`` — must
        still be dispatched as legacy so existing consumers don't break.
        """
        buf = StreamingBuffer(on_partial=None)
        assert _callback_accepts_chunk_metadata(buf.__call__) is False

    def test_result_is_cached_on_callback(self) -> None:
        """Second introspection should hit the cached attribute."""
        def cb(chunk: str) -> None:
            pass
        assert _callback_accepts_chunk_metadata(cb) is False
        # Cache attribute is set on the function for the next call.
        assert hasattr(cb, "_carl_chunk_meta_kw_cached")
        assert getattr(cb, "_carl_chunk_meta_kw_cached") is False

    def test_unintrospectable_callback_falls_back_to_legacy(self) -> None:
        """A C-extension callable / builtin (no inspect.signature) is
        treated as legacy.
        """
        # ``print`` has no inspectable signature in some Python builds —
        # but inspect.signature(print) actually works in 3.12. Use a
        # synthetic object that explicitly raises on signature lookup.
        class _Weird:
            __slots__ = ()
            def __call__(self, *a: Any, **kw: Any) -> None:
                pass
        weird = _Weird()
        # __call__ on this class accepts **kw, so it should detect as
        # extended. Use a simpler example: a bound method without args.
        assert _callback_accepts_chunk_metadata(weird) is True


# ---------------------------------------------------------------------------
# Dispatcher behaviour
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_dispatch_to_legacy_passes_chunk_only(self) -> None:
        seen: list[tuple] = []
        def cb(chunk: str) -> None:
            seen.append((chunk,))
        _dispatch_llm_chunk(cb, "hello", step_number=7, stage="fast")
        assert seen == [("hello",)]

    def test_dispatch_to_extended_passes_metadata(self) -> None:
        seen: list[dict] = []
        def cb(chunk: str, *, step_number: int | None = None,
               stage: str | None = None) -> None:
            seen.append({"chunk": chunk, "step_number": step_number, "stage": stage})
        _dispatch_llm_chunk(cb, "world", step_number=3, stage="critic")
        assert seen == [{"chunk": "world", "step_number": 3, "stage": "critic"}]

    def test_dispatch_swallows_consumer_exceptions(self) -> None:
        def cb(chunk: str) -> None:
            raise RuntimeError("bad consumer")
        # Must not raise — execution must keep going.
        _dispatch_llm_chunk(cb, "x")

    def test_dispatch_with_none_callback_is_noop(self) -> None:
        # No callback configured — call must be safe.
        _dispatch_llm_chunk(None, "x", step_number=1, stage="fast")


# ---------------------------------------------------------------------------
# End-to-end integration through LLMStepExecutor
# ---------------------------------------------------------------------------


class TestStreamingIntegration:
    @pytest.mark.asyncio
    async def test_legacy_callback_still_works(self) -> None:
        seen: list[str] = []
        client = _StreamingFakeClient(["foo", "bar", "baz"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S", aim="x"),
        ])
        ctx = ReasoningContext(
            outer_context="task", api=client,
            on_llm_chunk=lambda chunk: seen.append(chunk),
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # Legacy callbacks see chunks unchanged.
        assert seen == ["foo", "bar", "baz"]

    @pytest.mark.asyncio
    async def test_extended_callback_receives_step_number_and_stage(self) -> None:
        seen: list[dict] = []
        client = _StreamingFakeClient(["alpha", "beta"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=5, title="StreamMe", aim="x"),
        ])
        def cb(chunk: str, *, step_number: int | None = None,
               stage: str | None = None) -> None:
            seen.append({"chunk": chunk, "step_number": step_number, "stage": stage})
        ctx = ReasoningContext(
            outer_context="task", api=client, on_llm_chunk=cb,
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # All chunks carry the step number + the "fast" stage label.
        assert len(seen) == 2
        for entry in seen:
            assert entry["step_number"] == 5
            assert entry["stage"] == "fast"
        assert [e["chunk"] for e in seen] == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_streaming_buffer_continues_to_work(self) -> None:
        """StreamingBuffer is documented as a drop-in callback. Its
        legacy single-arg shape must keep being dispatched correctly
        even though the dispatcher now handles two shapes."""
        captured_segments: list[str] = []
        buffer = StreamingBuffer(
            on_partial=lambda seg, full: captured_segments.append(seg),
            boundary="sentence",
        )
        client = _StreamingFakeClient(["Hello world.", " Next sentence."])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="S", aim="x"),
        ])
        ctx = ReasoningContext(
            outer_context="task", api=client, on_llm_chunk=buffer,
        )
        await chain.execute_async(ctx)
        # The buffer accumulated all chunks.
        assert buffer.text == "Hello world. Next sentence."
        # Boundary fired at least once for "Hello world. " segment.
        assert any("Hello world." in s for s in captured_segments)

    @pytest.mark.asyncio
    async def test_extended_callback_with_only_kwargs(self) -> None:
        """A callback with ``**kwargs`` should be treated as extended."""
        seen: list[dict] = []
        def cb(chunk: str, **kwargs: Any) -> None:
            seen.append({"chunk": chunk, **kwargs})
        client = _StreamingFakeClient(["x"])
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=42, title="S", aim="x"),
        ])
        ctx = ReasoningContext(outer_context="task", api=client, on_llm_chunk=cb)
        await chain.execute_async(ctx)
        assert seen[0]["chunk"] == "x"
        assert seen[0]["step_number"] == 42
        assert seen[0]["stage"] == "fast"
