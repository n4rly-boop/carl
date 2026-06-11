"""
Tests for the long-term memory (LTM) system.

Covers:
- LTMBase interface (search, clear)
- InMemoryLTM: store, retrieve, delete, keys, session scoping
- JsonFileLTM: store, retrieve, delete, keys, session scoping, persistence
- context.remember() / context.recall() / context.ltm_retrieve()
- context.remember() / context.recall() raise without long_term_memory
- $ltm.key reference syntax in resolve_context_reference
- Integration: chain step reads from LTM via input_mapping
"""

from pathlib import Path

import pytest

from mmar_carl import (
    InMemoryLTM,
    JsonFileLTM,
    Language,
    LLMClientBase,
    LTMBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepDescription,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.step_executors import resolve_context_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _ctx(ltm: LTMBase | None = None, session_id: str = "", **kwargs) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test",
        api=_MockLLM(),
        model="test",
        language=Language.ENGLISH,
        long_term_memory=ltm,
        session_id=session_id,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Unit: InMemoryLTM
# ---------------------------------------------------------------------------


class TestInMemoryLTM:
    def test_store_and_retrieve(self):
        ltm = InMemoryLTM()
        ltm.store("key1", "hello")
        assert ltm.retrieve("key1") == "hello"

    def test_retrieve_missing_key_returns_none(self):
        ltm = InMemoryLTM()
        assert ltm.retrieve("does_not_exist") is None

    def test_overwrite(self):
        ltm = InMemoryLTM()
        ltm.store("k", "v1")
        ltm.store("k", "v2")
        assert ltm.retrieve("k") == "v2"

    def test_session_isolation(self):
        ltm = InMemoryLTM()
        ltm.store("shared_key", "session-a-value", session_id="a")
        ltm.store("shared_key", "session-b-value", session_id="b")
        assert ltm.retrieve("shared_key", session_id="a") == "session-a-value"
        assert ltm.retrieve("shared_key", session_id="b") == "session-b-value"
        assert ltm.retrieve("shared_key") is None  # global scope untouched

    def test_delete_existing_key(self):
        ltm = InMemoryLTM()
        ltm.store("k", 42)
        deleted = ltm.delete("k")
        assert deleted is True
        assert ltm.retrieve("k") is None

    def test_delete_nonexistent_key(self):
        ltm = InMemoryLTM()
        assert ltm.delete("no_such_key") is False

    def test_keys_lists_all_keys_in_session(self):
        ltm = InMemoryLTM()
        ltm.store("a", 1, session_id="s1")
        ltm.store("b", 2, session_id="s1")
        ltm.store("c", 3, session_id="s2")
        assert set(ltm.keys(session_id="s1")) == {"a", "b"}
        assert set(ltm.keys(session_id="s2")) == {"c"}

    def test_store_complex_value(self):
        ltm = InMemoryLTM()
        data = {"name": "Alice", "scores": [1, 2, 3]}
        ltm.store("data", data)
        assert ltm.retrieve("data") == data

    def test_search_exact_match(self):
        ltm = InMemoryLTM()
        ltm.store("note1", "revenue grew 20% in Q3")
        ltm.store("note2", "expenses flat")
        hits = ltm.search("revenue")
        assert any(h["key"] == "note1" for h in hits)
        assert all(h["key"] != "note2" for h in hits)

    def test_search_case_insensitive(self):
        ltm = InMemoryLTM()
        ltm.store("item", "Revenue Report")
        hits = ltm.search("revenue")
        assert len(hits) == 1
        assert hits[0]["score"] < 1.0  # case-insensitive match has lower score

    def test_search_top_k(self):
        ltm = InMemoryLTM()
        for i in range(10):
            ltm.store(f"k{i}", f"match {i}")
        hits = ltm.search("match", top_k=3)
        assert len(hits) <= 3

    def test_search_no_match(self):
        ltm = InMemoryLTM()
        ltm.store("k", "unrelated value")
        hits = ltm.search("xyzzy")
        assert hits == []

    def test_clear_removes_all_in_session(self):
        ltm = InMemoryLTM()
        ltm.store("a", 1, session_id="sess")
        ltm.store("b", 2, session_id="sess")
        ltm.store("c", 3, session_id="other")
        removed = ltm.clear(session_id="sess")
        assert removed == 2
        assert ltm.keys(session_id="sess") == []
        assert ltm.keys(session_id="other") == ["c"]

    def test_repr(self):
        ltm = InMemoryLTM()
        ltm.store("x", 1, session_id="s")
        assert "InMemoryLTM" in repr(ltm)


# ---------------------------------------------------------------------------
# Unit: JsonFileLTM
# ---------------------------------------------------------------------------


class TestJsonFileLTM:
    def test_store_and_retrieve(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("greeting", "hello world")
        assert ltm.retrieve("greeting") == "hello world"

    def test_persistence_across_instances(self, tmp_path):
        ltm1 = JsonFileLTM(tmp_path)
        ltm1.store("fact", "sky is blue")
        # New instance pointing to same dir
        ltm2 = JsonFileLTM(tmp_path)
        assert ltm2.retrieve("fact") == "sky is blue"

    def test_session_isolation(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("k", "session-A", session_id="A")
        ltm.store("k", "session-B", session_id="B")
        assert ltm.retrieve("k", session_id="A") == "session-A"
        assert ltm.retrieve("k", session_id="B") == "session-B"

    def test_creates_separate_files_per_session(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("x", 1, session_id="s1")
        ltm.store("y", 2, session_id="s2")
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 2

    def test_delete(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("k", "v")
        assert ltm.delete("k") is True
        assert ltm.retrieve("k") is None

    def test_delete_nonexistent(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        assert ltm.delete("nope") is False

    def test_keys(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("a", 1)
        ltm.store("b", 2)
        assert set(ltm.keys()) == {"a", "b"}

    def test_search(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        ltm.store("n1", "revenue Q3")
        ltm.store("n2", "expenses Q3")
        hits = ltm.search("revenue")
        assert len(hits) == 1
        assert hits[0]["key"] == "n1"

    def test_stores_complex_json(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        data = {"list": [1, 2, 3], "nested": {"x": True}}
        ltm.store("complex", data)
        assert ltm.retrieve("complex") == data

    def test_expands_tilde(self):
        # Just check it doesn't raise on a tilde path
        ltm = JsonFileLTM("~/.carl_test_tmp_xyz/")
        assert isinstance(ltm, JsonFileLTM)
        # Cleanup
        import shutil
        p = Path("~/.carl_test_tmp_xyz/").expanduser()
        if p.exists():
            shutil.rmtree(p)

    def test_repr(self, tmp_path):
        ltm = JsonFileLTM(tmp_path)
        assert "JsonFileLTM" in repr(ltm)


# ---------------------------------------------------------------------------
# Unit: ReasoningContext.remember() / recall() / ltm_retrieve()
# ---------------------------------------------------------------------------


class TestContextLTMMethods:
    def test_remember_and_recall(self):
        ltm = InMemoryLTM()
        ctx = _ctx(ltm=ltm, session_id="u1")
        ctx.remember("user_name", "Alice")
        hits = ctx.recall("Alice")
        assert any(h["key"] == "user_name" for h in hits)

    def test_remember_uses_session_id(self):
        ltm = InMemoryLTM()
        ctx = _ctx(ltm=ltm, session_id="my-session")
        ctx.remember("pref", "verbose")
        # Directly check in LTM with the session scope
        assert ltm.retrieve("pref", session_id="my-session") == "verbose"
        assert ltm.retrieve("pref", session_id="other") is None

    def test_remember_raises_without_ltm(self):
        ctx = _ctx()  # no ltm
        with pytest.raises(RuntimeError, match="long_term_memory"):
            ctx.remember("k", "v")

    def test_recall_raises_without_ltm(self):
        ctx = _ctx()
        with pytest.raises(RuntimeError, match="long_term_memory"):
            ctx.recall("query")

    def test_ltm_retrieve_returns_none_without_ltm(self):
        ctx = _ctx()
        assert ctx.ltm_retrieve("any_key") is None

    def test_ltm_retrieve_correct_value(self):
        ltm = InMemoryLTM()
        ltm.store("fact", "Paris is capital of France", session_id="sess")
        ctx = _ctx(ltm=ltm, session_id="sess")
        assert ctx.ltm_retrieve("fact") == "Paris is capital of France"

    def test_ltm_retrieve_missing_key_returns_none(self):
        ltm = InMemoryLTM()
        ctx = _ctx(ltm=ltm)
        assert ctx.ltm_retrieve("nonexistent") is None

    def test_recall_top_k(self):
        ltm = InMemoryLTM()
        for i in range(10):
            ltm.store(f"k{i}", f"match {i}", session_id="s")
        ctx = _ctx(ltm=ltm, session_id="s")
        hits = ctx.recall("match", top_k=3)
        assert len(hits) <= 3

    def test_remember_session_override(self):
        ltm = InMemoryLTM()
        ctx = _ctx(ltm=ltm, session_id="default-sess")
        ctx.remember("key", "value", session_id="override-sess")
        assert ltm.retrieve("key", session_id="override-sess") == "value"
        assert ltm.retrieve("key", session_id="default-sess") is None


# ---------------------------------------------------------------------------
# Unit: resolve_context_reference with $ltm.key
# ---------------------------------------------------------------------------


class TestLTMReferenceResolution:
    def test_ltm_reference_resolves_stored_value(self):
        ltm = InMemoryLTM()
        ltm.store("city", "London", session_id="s1")
        ctx = _ctx(ltm=ltm, session_id="s1")
        result = resolve_context_reference("$ltm.city", ctx)
        assert result == "London"

    def test_ltm_reference_returns_none_for_missing_key(self):
        ltm = InMemoryLTM()
        ctx = _ctx(ltm=ltm)
        result = resolve_context_reference("$ltm.nope", ctx)
        assert result is None

    def test_ltm_reference_returns_none_without_ltm(self):
        ctx = _ctx()
        result = resolve_context_reference("$ltm.key", ctx)
        assert result is None

    def test_ltm_reference_does_not_interfere_with_memory(self):
        ltm = InMemoryLTM()
        ltm.store("key", "ltm_value")
        ctx = _ctx(ltm=ltm)
        ctx.memory_write("key", "memory_value")
        assert resolve_context_reference("$ltm.key", ctx) == "ltm_value"
        assert resolve_context_reference("$memory.default.key", ctx) == "memory_value"


# ---------------------------------------------------------------------------
# Integration: chain step reads $ltm.key via input_mapping
# ---------------------------------------------------------------------------


class TestLTMChainIntegration:
    @pytest.mark.asyncio
    async def test_tool_step_reads_from_ltm(self):
        """A ToolStep with input_mapping=$ltm.key receives the LTM value."""
        ltm = InMemoryLTM()
        ltm.store("topic", "climate change", session_id="sess")

        captured: list[str] = []

        def my_tool(query: str) -> str:
            captured.append(query)
            return f"result for {query}"

        ctx = _ctx(ltm=ltm, session_id="sess")
        ctx.register_tool("my_tool", my_tool)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Search",
                    config=ToolStepConfig(
                        tool_name="my_tool",
                        input_mapping={"query": "$ltm.topic"},
                    ),
                )
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert captured == ["climate change"]

    @pytest.mark.asyncio
    async def test_remember_in_tool_then_recall(self):
        """A tool writes to LTM; a subsequent recall can find it."""
        ltm = InMemoryLTM()

        def write_tool(key: str, value: str) -> str:
            ltm.store(key, value, session_id="sess")
            return f"stored {key}"

        ctx = _ctx(ltm=ltm, session_id="sess")
        ctx.register_tool("write_tool", write_tool)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Write to LTM",
                    config=ToolStepConfig(
                        tool_name="write_tool",
                        input_mapping={
                            "key": "'answer'",
                            "value": "'42'",
                        },
                    ),
                )
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # The tool should have stored the value in LTM
        assert ltm.retrieve("answer", session_id="sess") == "42"
        # Recall should find it
        hits = ctx.recall("42")
        assert any(h["key"] == "answer" for h in hits)
