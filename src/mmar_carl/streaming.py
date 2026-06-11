"""
StreamingBuffer — aggregate LLM streaming chunks and emit boundary events.

CARL's ``on_llm_chunk`` callback fires once per token-level chunk. That's the
right granularity for low-level UIs (token-by-token render) but rarely what
higher-level code wants: most callers want to react to *complete sentences*
or *paragraphs* as soon as they arrive.

``StreamingBuffer`` is a tiny callable that:

* Accumulates chunks into a running buffer.
* Detects sentence- or paragraph-level boundaries in the new tail.
* Fires the user's ``on_partial`` callback once per boundary, passing the
  newly completed segment and the full text so far.
* Exposes ``.text`` (running accumulator) and ``finalize()`` (flush any
  trailing buffer at end-of-stream).

It is intentionally instantiated and passed directly as ``on_llm_chunk``::

    buffer = StreamingBuffer(on_partial=lambda seg, full: print("[seg]", seg))
    context = ReasoningContext(..., on_llm_chunk=buffer)
    # ... after execution:
    buffer.finalize()
"""

from __future__ import annotations

import re
from typing import Callable, Literal, Optional

Boundary = Literal["sentence", "paragraph"]

# Sentence terminator followed by whitespace (or string end). Includes ?!… and
# the standard CJK full-stop variants people occasionally feed in. We anchor on
# the *trailing* whitespace so we know the sentence has truly ended — a stray
# "Dr." mid-sentence won't fire prematurely because the next char is a space
# but the next-next is lowercase; we accept the false-positive cost there.
_SENTENCE_RE = re.compile(r"[.!?。！？…]+[\s\"')\]]*\s+", re.UNICODE)
# Paragraph break: 2+ consecutive newlines (possibly with intervening spaces).
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n+")


class StreamingBuffer:
    """
    Aggregate streaming chunks and emit a callback at sentence/paragraph boundaries.

    Args:
        on_partial: Callable invoked when a boundary is reached, receiving
            ``(segment, full_text_so_far)`` where ``segment`` is the newly
            completed text since the last boundary (including its terminator)
            and ``full_text_so_far`` is the entire accumulated text.
            If ``None``, the buffer accumulates silently and you can read
            ``.text`` at any time.
        boundary: ``"sentence"`` (default) or ``"paragraph"``.
        include_terminator: When ``True`` (default), the emitted segment
            includes the trailing terminator/whitespace. When ``False``, the
            terminator is stripped so callers receive a clean text body.

    The instance is itself callable: ``buffer(chunk)`` is equivalent to
    ``buffer.on_chunk(chunk)``. This lets it be passed straight to
    ``ReasoningContext(on_llm_chunk=buffer)``.
    """

    def __init__(
        self,
        on_partial: Optional[Callable[[str, str], None]] = None,
        *,
        boundary: Boundary = "sentence",
        include_terminator: bool = True,
    ) -> None:
        if boundary not in ("sentence", "paragraph"):
            raise ValueError(
                f"boundary must be 'sentence' or 'paragraph', got {boundary!r}"
            )
        self._on_partial = on_partial
        self._boundary = boundary
        self._include_terminator = include_terminator
        self._pattern = _SENTENCE_RE if boundary == "sentence" else _PARAGRAPH_RE
        # Running full text (everything ever seen, including unflushed tail).
        self._full: list[str] = []
        # Position within self.text where the next un-emitted segment begins.
        self._pending_start: int = 0

    # ------------------------------------------------------------------ #
    # Callable interface
    # ------------------------------------------------------------------ #

    def __call__(self, chunk: str) -> None:
        """Accept a chunk — equivalent to :py:meth:`on_chunk`."""
        self.on_chunk(chunk)

    def on_chunk(self, chunk: str) -> None:
        """Append *chunk* to the buffer and emit any newly completed segments."""
        if not chunk:
            return
        self._full.append(chunk)
        self._emit_completed_segments()

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    @property
    def text(self) -> str:
        """The full accumulated text seen so far (including any unflushed tail)."""
        return "".join(self._full)

    @property
    def boundary(self) -> Boundary:
        """The boundary mode for this buffer."""
        return self._boundary

    def pending(self) -> str:
        """Return the buffer tail that has not yet been emitted as a segment."""
        return self.text[self._pending_start:]

    # ------------------------------------------------------------------ #
    # Finalisation
    # ------------------------------------------------------------------ #

    def finalize(self) -> str:
        """
        Flush any trailing buffer as a final segment.

        Returns the segment that was emitted (which may be empty if everything
        already ended on a boundary). After calling ``finalize``, further calls
        to ``on_chunk`` start a brand new segment, but the accumulated
        ``.text`` is preserved.
        """
        full_text = self.text
        tail = full_text[self._pending_start:]
        self._pending_start = len(full_text)
        if tail and self._on_partial is not None:
            self._on_partial(tail, full_text)
        return tail

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _emit_completed_segments(self) -> None:
        """Scan the unemitted tail for boundaries and fire ``on_partial``."""
        if self._on_partial is None:
            # No callback configured — still advance _pending_start so that
            # ``pending()`` and ``finalize()`` semantics stay consistent.
            full_text = self.text
            last_end = self._pending_start
            for m in self._pattern.finditer(full_text, self._pending_start):
                last_end = m.end()
            self._pending_start = last_end
            return

        full_text = self.text
        cursor = self._pending_start
        for match in self._pattern.finditer(full_text, self._pending_start):
            end = match.end()
            segment = full_text[cursor:end]
            if not self._include_terminator:
                # Strip the trailing terminator+whitespace that matched.
                segment = full_text[cursor:match.start()]
            cursor = end
            # full_text passed here is "everything up to *now*", which is what
            # the caller wants — even segments emitted earlier in the same
            # chunk see the up-to-date full snapshot.
            self._on_partial(segment, full_text)
        self._pending_start = cursor
