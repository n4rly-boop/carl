"""
Tests for ``StreamingBuffer``.

The buffer aggregates ``on_llm_chunk``-style chunks and fires a higher-level
``on_partial(segment, full_text)`` callback at sentence or paragraph
boundaries. It is callable so it can be passed straight to
``ReasoningContext(on_llm_chunk=buffer)``.
"""

import pytest

from mmar_carl import LLMClientBase, ReasoningContext, StreamingBuffer


# --------------------------------------------------------------------------- #
# Construction / validation
# --------------------------------------------------------------------------- #


def test_rejects_invalid_boundary() -> None:
    with pytest.raises(ValueError, match="boundary"):
        StreamingBuffer(boundary="word")  # type: ignore[arg-type]


def test_boundary_attribute_exposed() -> None:
    assert StreamingBuffer().boundary == "sentence"
    assert StreamingBuffer(boundary="paragraph").boundary == "paragraph"


# --------------------------------------------------------------------------- #
# Basic accumulation (no callback)
# --------------------------------------------------------------------------- #


def test_accumulates_chunks_without_callback() -> None:
    buf = StreamingBuffer()
    buf("Hello, ")
    buf("world!")
    assert buf.text == "Hello, world!"


def test_empty_chunk_is_noop() -> None:
    captured: list[tuple[str, str]] = []
    buf = StreamingBuffer(lambda seg, full: captured.append((seg, full)))
    buf("")
    buf("Done.")
    assert buf.text == "Done."
    # Single sentence without trailing whitespace → not yet a boundary.
    assert captured == []


# --------------------------------------------------------------------------- #
# Sentence boundaries
# --------------------------------------------------------------------------- #


def test_sentence_boundary_fires_on_period_space() -> None:
    captured: list[tuple[str, str]] = []
    buf = StreamingBuffer(lambda seg, full: captured.append((seg, full)))
    buf("First sentence. ")
    assert len(captured) == 1
    seg, full = captured[0]
    assert seg == "First sentence. "
    assert full == "First sentence. "


def test_sentence_boundary_handles_multiple_in_one_chunk() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("One. Two! Three? ")
    assert captured == ["One. ", "Two! ", "Three? "]


def test_sentence_boundary_split_across_chunks() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("Half a sen")
    buf("tence. And ")
    buf("another. ")
    assert captured == ["Half a sentence. ", "And another. "]


def test_sentence_with_terminator_stripped() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(
        lambda seg, _full: captured.append(seg),
        include_terminator=False,
    )
    buf("First. Second! ")
    assert captured == ["First", "Second"]


def test_no_boundary_no_emit() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("Just a partial")
    buf(" thought without")
    buf(" any terminator")
    assert captured == []
    assert buf.pending() == "Just a partial thought without any terminator"


# --------------------------------------------------------------------------- #
# Paragraph boundaries
# --------------------------------------------------------------------------- #


def test_paragraph_boundary_fires_on_double_newline() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(
        lambda seg, _full: captured.append(seg),
        boundary="paragraph",
    )
    buf("First paragraph. With two sentences.\n\nSecond para starts.")
    assert captured == ["First paragraph. With two sentences.\n\n"]
    assert buf.pending() == "Second para starts."


def test_paragraph_boundary_tolerates_whitespace_between_newlines() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(
        lambda seg, _full: captured.append(seg),
        boundary="paragraph",
    )
    buf("One.\n \nTwo.")
    assert len(captured) == 1
    assert captured[0].startswith("One.")


def test_paragraph_does_not_fire_on_single_newline() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(
        lambda seg, _full: captured.append(seg),
        boundary="paragraph",
    )
    buf("Line 1.\nLine 2.\nLine 3.")
    assert captured == []


# --------------------------------------------------------------------------- #
# Unicode / non-Latin terminators
# --------------------------------------------------------------------------- #


def test_unicode_sentence_terminator() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("日本語のテスト。 続く文。 ")
    assert captured == ["日本語のテスト。 ", "続く文。 "]


# --------------------------------------------------------------------------- #
# finalize()
# --------------------------------------------------------------------------- #


def test_finalize_flushes_trailing_partial_sentence() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("Complete. Tail without terminator")
    assert captured == ["Complete. "]
    tail = buf.finalize()
    assert tail == "Tail without terminator"
    assert captured == ["Complete. ", "Tail without terminator"]


def test_finalize_emits_nothing_when_buffer_clean() -> None:
    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    buf("Done. ")
    assert buf.finalize() == ""
    assert captured == ["Done. "]


def test_finalize_without_callback_still_advances() -> None:
    buf = StreamingBuffer()
    buf("Tail without callback")
    tail = buf.finalize()
    assert tail == "Tail without callback"
    assert buf.pending() == ""


# --------------------------------------------------------------------------- #
# Callback invariants
# --------------------------------------------------------------------------- #


def test_callback_receives_full_text_snapshot() -> None:
    snapshots: list[str] = []
    buf = StreamingBuffer(lambda _seg, full: snapshots.append(full))
    buf("A. B. C. ")
    # All three boundaries detected within one chunk → full_text passed to
    # each emit is the full text *at the time of the chunk*, not the prefix.
    assert snapshots == ["A. B. C. "] * 3


def test_no_callback_silent_accumulation_still_tracks_pending() -> None:
    buf = StreamingBuffer()
    buf("One. Two. Trailing")
    # Without callback, the buffer still advances _pending_start past matched
    # boundaries so pending() returns only the trailing fragment.
    assert buf.pending() == "Trailing"
    assert buf.text == "One. Two. Trailing"


# --------------------------------------------------------------------------- #
# Integration: works as ReasoningContext.on_llm_chunk
# --------------------------------------------------------------------------- #


def test_works_as_on_llm_chunk_callback() -> None:
    """StreamingBuffer is callable, so it satisfies the on_llm_chunk signature."""

    class _Client(LLMClientBase):
        async def get_response(self, prompt: str) -> str:
            return ""

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return ""

    captured: list[str] = []
    buf = StreamingBuffer(lambda seg, _full: captured.append(seg))
    ctx = ReasoningContext(outer_context="", api=_Client(), on_llm_chunk=buf)

    # Simulate the runtime calling on_llm_chunk per token
    assert ctx.on_llm_chunk is not None
    for token in ["Hel", "lo, ", "world! ", "Tail"]:
        ctx.on_llm_chunk(token)
    assert captured == ["Hello, world! "]
    assert buf.pending() == "Tail"
    buf.finalize()
    assert captured[-1] == "Tail"
