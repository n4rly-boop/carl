"""``context.register_tools_from_path(glob)`` discovers
and registers ``@carl_tool``-decorated callables from disk.

CARE will load user tool directories (e.g. ``~/.config/care/tools/*.py``)
at startup; this test exercises every key behaviour against tmp-path
files so we don't touch the user's actual filesystem.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from mmar_carl import ReasoningContext
from mmar_carl.models.llm_client_base import LLMClientBase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "x"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "x"


@pytest.fixture
def ctx() -> ReasoningContext:
    return ReasoningContext(outer_context="x", api=_FakeClient())


def _write_tool_module(
    path: Path, *, body: str,
) -> None:
    """Write a tool-module file with the standard import header."""
    path.write_text(
        dedent(f"""\
        from mmar_carl import carl_tool

        {body}
        """),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Single-file discovery
# ---------------------------------------------------------------------------


class TestDiscoveryBasics:
    def test_single_file_single_tool(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "search.py"
        _write_tool_module(f, body="""
        @carl_tool
        def search(query: str) -> str:
            return f"results for {query}"
        """)
        registered = ctx.register_tools_from_path(str(f))
        assert registered == ["search"]
        assert ctx.has_tool("search")
        assert ctx.get_tool("search")(query="foo") == "results for foo"

    def test_undecorated_callables_are_skipped(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "mixed.py"
        _write_tool_module(f, body="""
        @carl_tool
        def included(x: int) -> int:
            return x + 1

        def excluded(x: int) -> int:
            return x - 1
        """)
        registered = ctx.register_tools_from_path(str(f))
        assert registered == ["included"]
        assert ctx.has_tool("included")
        assert not ctx.has_tool("excluded")

    def test_private_attributes_skipped(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "private.py"
        _write_tool_module(f, body="""
        @carl_tool
        def _hidden(x: int) -> int:
            return x

        @carl_tool
        def public(x: int) -> int:
            return x
        """)
        registered = ctx.register_tools_from_path(str(f))
        assert registered == ["public"]


# ---------------------------------------------------------------------------
# Glob expansion
# ---------------------------------------------------------------------------


class TestGlobExpansion:
    def test_glob_matches_multiple_files(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        _write_tool_module(tmp_path / "a.py", body="""
        @carl_tool
        def alpha() -> str: return "a"
        """)
        _write_tool_module(tmp_path / "b.py", body="""
        @carl_tool
        def beta() -> str: return "b"
        """)
        registered = ctx.register_tools_from_path(str(tmp_path / "*.py"))
        assert set(registered) == {"alpha", "beta"}
        assert ctx.has_tool("alpha")
        assert ctx.has_tool("beta")

    def test_recursive_glob_finds_nested(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        nested = tmp_path / "sub" / "deeper"
        nested.mkdir(parents=True)
        _write_tool_module(nested / "deep.py", body="""
        @carl_tool
        def deep_tool() -> str: return "d"
        """)
        registered = ctx.register_tools_from_path(str(tmp_path / "**" / "*.py"))
        assert "deep_tool" in registered

    def test_non_python_files_ignored(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        # A README in the same dir mustn't break discovery.
        (tmp_path / "README.md").write_text("hi", encoding="utf-8")
        _write_tool_module(tmp_path / "ok.py", body="""
        @carl_tool
        def ok() -> str: return "ok"
        """)
        registered = ctx.register_tools_from_path(str(tmp_path / "*"))
        assert registered == ["ok"]

    def test_empty_glob_returns_empty_list(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        registered = ctx.register_tools_from_path(str(tmp_path / "missing*.py"))
        assert registered == []


# ---------------------------------------------------------------------------
# Tag filtering
# ---------------------------------------------------------------------------


class TestTagFilter:
    def test_tag_filter_includes_only_matching(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "tools.py"
        _write_tool_module(f, body="""
        @carl_tool(tags=["read"])
        def reader() -> str: return "r"

        @carl_tool(tags=["write"])
        def writer() -> str: return "w"

        @carl_tool(tags=["read", "write"])
        def both() -> str: return "rw"
        """)
        registered = ctx.register_tools_from_path(
            str(f), tag_filter=["read"],
        )
        assert set(registered) == {"reader", "both"}
        assert not ctx.has_tool("writer")

    def test_untagged_tools_excluded_when_filter_set(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "tools.py"
        _write_tool_module(f, body="""
        @carl_tool
        def untagged() -> str: return "u"

        @carl_tool(tags=["read"])
        def tagged() -> str: return "t"
        """)
        registered = ctx.register_tools_from_path(
            str(f), tag_filter=["read"],
        )
        assert registered == ["tagged"]

    def test_none_filter_registers_everything(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "tools.py"
        _write_tool_module(f, body="""
        @carl_tool
        def untagged() -> str: return "u"

        @carl_tool(tags=["read"])
        def tagged() -> str: return "t"
        """)
        registered = ctx.register_tools_from_path(str(f), tag_filter=None)
        assert set(registered) == {"untagged", "tagged"}

    def test_tags_preserved_in_registry(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "tools.py"
        _write_tool_module(f, body="""
        @carl_tool(tags=["search", "external"])
        def web_query() -> str: return "x"
        """)
        ctx.register_tools_from_path(str(f))
        assert ctx.get_tool_tags("web_query") == {"search", "external"}


# ---------------------------------------------------------------------------
# Name prefixing & error handling
# ---------------------------------------------------------------------------


class TestPrefixAndRobustness:
    def test_name_prefix_applied(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        f = tmp_path / "tools.py"
        _write_tool_module(f, body="""
        @carl_tool
        def search() -> str: return "s"
        """)
        registered = ctx.register_tools_from_path(
            str(f), name_prefix="care_",
        )
        assert registered == ["care_search"]
        assert ctx.has_tool("care_search")
        assert not ctx.has_tool("search")

    def test_broken_file_does_not_abort_discovery(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        # File with a SyntaxError — must be skipped without crashing.
        (tmp_path / "broken.py").write_text(
            "this is not !!! valid python ;",
            encoding="utf-8",
        )
        _write_tool_module(tmp_path / "ok.py", body="""
        @carl_tool
        def survivor() -> str: return "alive"
        """)
        registered = ctx.register_tools_from_path(str(tmp_path / "*.py"))
        # The good file's tool was registered; the broken file was
        # silently skipped.
        assert "survivor" in registered

    def test_runtime_error_in_module_top_level_skipped(
        self, ctx: ReasoningContext, tmp_path: Path,
    ) -> None:
        (tmp_path / "raises.py").write_text(
            "raise RuntimeError('boom on import')",
            encoding="utf-8",
        )
        _write_tool_module(tmp_path / "ok.py", body="""
        @carl_tool
        def fine() -> str: return "fine"
        """)
        registered = ctx.register_tools_from_path(str(tmp_path / "*.py"))
        assert registered == ["fine"]
