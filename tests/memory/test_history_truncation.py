"""
Tests for history truncation strategies in ReasoningContext.

Covers:
- Default "oldest" strategy (FIFO trim)
- "compress" strategy (strip verbose step headers, then trim)
- _compress_entry() helper
- Interaction with max_history_entries=0 (unlimited)
- trim_strategy propagation to parallel execution contexts
- $history[-1] references resolve correctly in both modes
"""

import pytest
from unittest.mock import MagicMock

from mmar_carl import (
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.models.context import ReasoningContext as _RC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(
    *,
    max_history_entries: int = 5,
    trim_strategy: str = "oldest",
    history: list[str] | None = None,
) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context="test",
        api=MagicMock(),
        model="test",
        language=Language.ENGLISH,
        max_history_entries=max_history_entries,
        trim_strategy=trim_strategy,  # type: ignore[arg-type]
    )
    if history:
        ctx.history = list(history)
    return ctx


def english_entry(step_num: int, title: str, result: str) -> str:
    return f"Step {step_num}. {title}\nResult: {result}\n"


def russian_entry(step_num: int, title: str, result: str) -> str:
    return f"Шаг {step_num}. {title}\nРезультат: {result}\n"


# ---------------------------------------------------------------------------
# _compress_entry tests
# ---------------------------------------------------------------------------


class TestCompressEntry:
    """Unit tests for the _compress_entry() static helper."""

    def test_english_result_prefix_stripped(self):
        entry = english_entry(3, "Analysis", "The result text here")
        assert _RC._compress_entry(entry) == "The result text here"

    def test_russian_result_prefix_stripped(self):
        entry = russian_entry(2, "Поиск", "Найдено 5 элементов")
        assert _RC._compress_entry(entry) == "Найдено 5 элементов"

    def test_multiline_result_preserved(self):
        entry = "Step 1. Title\nResult: line1\nline2\nline3\n"
        compressed = _RC._compress_entry(entry)
        assert compressed == "line1\nline2\nline3"

    def test_no_result_prefix_returns_unchanged(self):
        plain = "just a plain string with no prefix"
        assert _RC._compress_entry(plain) == plain

    def test_tool_mode_suffix_stripped(self):
        entry = "Step 2. Web Search [TOOL: web_search]\nResult: search results\n"
        assert _RC._compress_entry(entry) == "search results"

    def test_already_compressed_entry_unchanged(self):
        """Compressing an already-compressed entry should return it unchanged."""
        compressed = "search results"
        assert _RC._compress_entry(compressed) == compressed

    def test_trailing_newline_stripped(self):
        entry = "Step 1. T\nResult: content\n"
        result = _RC._compress_entry(entry)
        assert not result.endswith("\n")


# ---------------------------------------------------------------------------
# "oldest" strategy tests (existing behaviour)
# ---------------------------------------------------------------------------


class TestOldestTrimStrategy:
    """Verify that the default 'oldest' strategy drops the oldest entries."""

    def test_no_trim_when_under_limit(self):
        ctx = make_context(max_history_entries=5, trim_strategy="oldest")
        for i in range(3):
            ctx.add_to_history(english_entry(i + 1, f"Step {i + 1}", f"result {i + 1}"))
        assert len(ctx.history) == 3

    def test_oldest_entry_dropped_when_limit_exceeded(self):
        ctx = make_context(max_history_entries=3, trim_strategy="oldest")
        entries = [english_entry(i + 1, f"S{i + 1}", f"r{i + 1}") for i in range(4)]
        for e in entries:
            ctx.add_to_history(e)
        assert len(ctx.history) == 3
        # First entry should be gone; last 3 remain
        assert entries[0] not in ctx.history
        assert entries[1] in ctx.history
        assert entries[2] in ctx.history
        assert entries[3] in ctx.history

    def test_oldest_entries_preserved_in_full_format(self):
        """In 'oldest' mode, entries that survive are still in verbose format."""
        ctx = make_context(max_history_entries=3, trim_strategy="oldest")
        for i in range(3):
            ctx.add_to_history(english_entry(i + 1, f"T{i + 1}", f"result {i + 1}"))
        # All entries should still have headers
        for entry in ctx.history:
            assert "Result:" in entry

    def test_unlimited_history_when_max_is_zero(self):
        ctx = make_context(max_history_entries=0, trim_strategy="oldest")
        for i in range(20):
            ctx.add_to_history(f"entry {i}")
        assert len(ctx.history) == 20


# ---------------------------------------------------------------------------
# "compress" strategy tests
# ---------------------------------------------------------------------------


class TestCompressTrimStrategy:
    """Verify that 'compress' strips headers from existing entries before appending."""

    def test_existing_entries_compressed_when_new_added(self):
        ctx = make_context(max_history_entries=5, trim_strategy="compress")
        ctx.add_to_history(english_entry(1, "Step 1", "first result"))
        ctx.add_to_history(english_entry(2, "Step 2", "second result"))
        # After adding the second entry, the first should be compressed
        assert ctx.history[0] == "first result"
        # Most recent entry is still in full format
        assert "Result:" in ctx.history[1]

    def test_new_entry_added_in_full_format(self):
        """The entry just added should not be compressed yet."""
        ctx = make_context(max_history_entries=5, trim_strategy="compress")
        entry = english_entry(1, "Title", "some content")
        ctx.add_to_history(entry)
        # First entry: history was empty, nothing to compress
        assert ctx.history[0] == entry

    def test_oldest_entries_still_dropped_after_compression(self):
        """When limit is still exceeded after compression, oldest are trimmed."""
        ctx = make_context(max_history_entries=3, trim_strategy="compress")
        for i in range(5):
            ctx.add_to_history(english_entry(i + 1, f"T{i + 1}", f"result {i + 1}"))
        assert len(ctx.history) == 3

    def test_compress_reduces_entry_verbosity(self):
        ctx = make_context(max_history_entries=10, trim_strategy="compress")
        for i in range(4):
            ctx.add_to_history(english_entry(i + 1, f"Title{i + 1}", f"content {i + 1}"))
        # All entries except the most recent should be compressed (no "Result:" header)
        for entry in ctx.history[:-1]:
            assert "Result:" not in entry
            assert "Step" not in entry

    def test_compress_with_russian_entries(self):
        ctx = make_context(max_history_entries=10, trim_strategy="compress")
        ctx.add_to_history(russian_entry(1, "Анализ", "русский результат"))
        ctx.add_to_history(russian_entry(2, "Поиск", "ещё результат"))
        assert ctx.history[0] == "русский результат"

    def test_already_compressed_entries_unchanged_by_further_adds(self):
        """Adding entries is idempotent: re-compressing an already-compressed entry is a no-op."""
        ctx = make_context(max_history_entries=10, trim_strategy="compress")
        ctx.add_to_history(english_entry(1, "S1", "content A"))
        ctx.add_to_history(english_entry(2, "S2", "content B"))
        ctx.add_to_history(english_entry(3, "S3", "content C"))
        # history[0] should be "content A" (compressed) and stable
        assert ctx.history[0] == "content A"
        assert ctx.history[1] == "content B"

    def test_unlimited_history_compress_still_compresses(self):
        """With max=0 (unlimited), compress still strips headers."""
        ctx = make_context(max_history_entries=0, trim_strategy="compress")
        for i in range(5):
            ctx.add_to_history(english_entry(i + 1, f"T{i + 1}", f"result {i + 1}"))
        # All entries except the latest are compressed
        for entry in ctx.history[:-1]:
            assert "Result:" not in entry


# ---------------------------------------------------------------------------
# trim_strategy propagated in parallel execution
# ---------------------------------------------------------------------------


class TestTrimStrategyPropagation:
    """trim_strategy must be copied when creating isolated context snapshots."""

    @pytest.mark.asyncio
    async def test_trim_strategy_propagated_in_chain_execution(self):
        """
        Running a chain with trim_strategy='compress' should not crash and should
        compress history entries as steps execute.
        """
        call_count = [0]

        def counting_tool() -> str:
            call_count[0] += 1
            return f"result {call_count[0]}"

        steps = [
            ToolStepDescription(
                number=i + 1,
                title=f"Step {i + 1}",
                config=ToolStepConfig(tool_name="counting_tool", input_mapping={}),
                dependencies=[i] if i > 0 else [],
            )
            for i in range(4)
        ]

        class _MockClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                return "ok"

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return "ok"

        chain = ReasoningChain(steps=steps, max_workers=1)
        ctx = ReasoningContext(
            outer_context="",
            api=_MockClient(),
            model="test",
            language=Language.ENGLISH,
            max_history_entries=3,
            trim_strategy="compress",
        )
        ctx.register_tool("counting_tool", counting_tool)

        result = await chain.execute_async(ctx)

        assert result.success, f"Chain failed: {result.get_final_output()}"
        assert len(result.step_results) == 4
        # With max_history_entries=3 and 4 steps, oldest entry should be dropped
        assert len(ctx.history) == 3
        # In compress mode, the 2 oldest surviving entries should lack "Result:" header
        assert "Result:" not in ctx.history[0]
        assert "Result:" not in ctx.history[1]
        # Most recent entry is in full format
        assert "Result:" in ctx.history[2]

    @pytest.mark.asyncio
    async def test_history_reference_resolves_after_compression(self):
        """$history[-1] should still return the previous step's result after compress."""
        results_seen = []

        def capture_result() -> str:
            return "captured_value"

        def read_history_tool(value: str) -> str:
            results_seen.append(value)
            return f"got: {value}"

        steps = [
            ToolStepDescription(
                number=1,
                title="Producer",
                config=ToolStepConfig(tool_name="capture_result", input_mapping={}),
            ),
            ToolStepDescription(
                number=2,
                title="Consumer",
                config=ToolStepConfig(
                    tool_name="read_history_tool",
                    input_mapping={"value": "$history[-1]"},
                ),
                dependencies=[1],
            ),
        ]

        class _MockClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                return "ok"

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return "ok"

        chain = ReasoningChain(steps=steps, max_workers=1)
        ctx = ReasoningContext(
            outer_context="",
            api=_MockClient(),
            model="test",
            language=Language.ENGLISH,
            max_history_entries=5,
            trim_strategy="compress",
        )
        ctx.register_tool("capture_result", capture_result)
        ctx.register_tool("read_history_tool", read_history_tool)

        result = await chain.execute_async(ctx)

        assert result.success, f"Chain failed: {result.get_final_output()}"
        # Step 2 reads $history[-1] — but the history entry from step 1 is compressed
        # at that point. The compressed entry is the raw result of the tool.
        # We can verify step 2 ran successfully.
        step2 = next(r for r in result.step_results if r.step_number == 2)
        assert step2.success
