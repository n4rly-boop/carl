"""MCP connection pooling.

:class:`MCPSessionPool` reuses one ``ClientSession`` per server across
steps in a chain run. Before pooling, every MCP call (tool dispatch /
``list_tools`` / ``list_resources``) paid the full setup-teardown cost.

The pool keys sessions by ``(server_name, transport, command-or-url)``
so two configs pointing at the same server share a session.

Tests stub the MCP SDK so they're hermetic — no real server needed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These tests stub the MCP SDK (``mcp.ClientSession`` + transport clients) via
# ``unittest.mock.patch``; without the optional ``mcp`` extra installed the
# patch targets can't be imported. Skip the module cleanly in that case.
pytest.importorskip("mcp", reason="requires the optional `mcp` extra")

from mmar_carl import (
    LLMClientBase,
    MCPServerConfig,
    MCPSessionPool,
    ReasoningContext,
)
from mmar_carl.mcp_pool import _server_key


# ---------------------------------------------------------------------------
# Helpers — mirror tests/mcp/test_list_mcp_resources.py
# ---------------------------------------------------------------------------


def _make_resource(uri: str) -> MagicMock:
    r = MagicMock()
    r.uri = uri
    return r


def _make_tool(name: str) -> MagicMock:
    t = MagicMock()
    t.name = name
    return t


def _make_session_mock(
    *,
    resources_list: list[Any] | None = None,
    tools_list: list[Any] | None = None,
) -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    if resources_list is not None:
        rr = MagicMock()
        rr.resources = resources_list
        session.list_resources = AsyncMock(return_value=rr)
    if tools_list is not None:
        tr = MagicMock()
        tr.tools = tools_list
        session.list_tools = AsyncMock(return_value=tr)

    async def fake_call(name: str, arguments: dict | None = None) -> Any:
        r = MagicMock()
        r.content = {"called": name, "args": arguments or {}}
        return r
    session.call_tool = fake_call
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_sse_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_stdio_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "ok"


def _make_context() -> ReasoningContext:
    return ReasoningContext(outer_context="x", api=_StubLLM())


# ---------------------------------------------------------------------------
# _server_key — identity by (server_name, transport, target)
# ---------------------------------------------------------------------------


class TestServerKey:
    def test_distinct_servers_distinct_keys(self) -> None:
        a = MCPServerConfig(server_name="docs", transport="sse", url="http://a")
        b = MCPServerConfig(server_name="docs", transport="sse", url="http://b")
        assert _server_key(a) != _server_key(b)

    def test_same_config_same_key(self) -> None:
        a = MCPServerConfig(server_name="docs", transport="sse", url="http://a")
        b = MCPServerConfig(server_name="docs", transport="sse", url="http://a")
        assert _server_key(a) == _server_key(b)

    def test_different_transport_different_key(self) -> None:
        a = MCPServerConfig(
            server_name="x", transport="sse", url="http://a/sse",
        )
        b = MCPServerConfig(
            server_name="x", transport="stdio", command="mcp-x",
        )
        assert _server_key(a) != _server_key(b)


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------


class TestPoolLifecycle:
    @pytest.mark.asyncio
    async def test_open_close_roundtrip(self) -> None:
        pool = MCPSessionPool()
        assert not pool.is_open
        async with pool:
            assert pool.is_open
        assert not pool.is_open

    @pytest.mark.asyncio
    async def test_acquire_outside_pool_raises(self) -> None:
        pool = MCPSessionPool()
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        with pytest.raises(RuntimeError, match="not open"):
            await pool.acquire(server)

    @pytest.mark.asyncio
    async def test_stats_reflects_state(self) -> None:
        pool = MCPSessionPool()
        assert pool.stats()["open"] is False
        async with pool:
            assert pool.stats()["open"] is True
            assert pool.stats()["active_sessions"] == 0


# ---------------------------------------------------------------------------
# Session reuse — the core pooling guarantee
# ---------------------------------------------------------------------------


class TestSessionReuse:
    @pytest.mark.asyncio
    async def test_two_acquires_same_server_reuse_session(self) -> None:
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            async with MCPSessionPool() as pool:
                s1 = await pool.acquire(server)
                s2 = await pool.acquire(server)
        assert s1 is s2
        # initialize() runs ONCE — second acquire is a cache hit.
        assert session.initialize.await_count == 1

    @pytest.mark.asyncio
    async def test_different_servers_different_sessions(self) -> None:
        s1_mock = _make_session_mock(resources_list=[])
        s2_mock = _make_session_mock(resources_list=[])
        # ClientSession is called twice with different (read, write); return
        # different sessions each time.
        sessions_iter = iter([s1_mock, s2_mock])
        def _next_session(*a, **k):
            return next(sessions_iter)
        a = MCPServerConfig(server_name="A", transport="sse", url="http://A")
        b = MCPServerConfig(server_name="B", transport="sse", url="http://B")
        with (
            patch("mcp.ClientSession", side_effect=_next_session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            async with MCPSessionPool() as pool:
                sa = await pool.acquire(a)
                sb = await pool.acquire(b)
                assert sa is not sb
                assert pool.stats()["active_sessions"] == 2


# ---------------------------------------------------------------------------
# Pool wiring through ReasoningContext.mcp_pool()
# ---------------------------------------------------------------------------


class TestContextPoolWiring:
    @pytest.mark.asyncio
    async def test_call_tool_reuses_pooled_session(self) -> None:
        """When the pool is open, repeated tool calls reuse one
        session — no per-call transport setup."""
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            async with ctx.mcp_pool():
                await ctx._mcp_call_tool(server, "search", {"q": "x"}, timeout=10)
                await ctx._mcp_call_tool(server, "fetch", {"u": "y"}, timeout=10)
        # Both calls go through ONE session: initialize() runs once,
        # session.__aenter__ is invoked exactly once via the
        # AsyncExitStack.
        assert session.initialize.await_count == 1

    @pytest.mark.asyncio
    async def test_list_resources_uses_pool(self) -> None:
        session = _make_session_mock(resources_list=[
            _make_resource("docs://x.md"),
            _make_resource("docs://y.md"),
        ])
        server = MCPServerConfig(
            server_name="docs", transport="sse", url="http://x",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            async with ctx.mcp_pool():
                r1 = await ctx.list_mcp_resources(server)
                r2 = await ctx.list_mcp_resources(server)
        assert len(r1) == 2 and len(r2) == 2
        # Pool reused — list_resources called twice, initialize once.
        assert session.list_resources.await_count == 2
        assert session.initialize.await_count == 1

    @pytest.mark.asyncio
    async def test_list_tools_uses_pool(self) -> None:
        session = _make_session_mock(
            tools_list=[_make_tool("search"), _make_tool("fetch")],
        )
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            async with ctx.mcp_pool():
                t1 = await ctx._mcp_list_tools(server, timeout=10)
                t2 = await ctx._mcp_list_tools(server, timeout=10)
        assert len(t1) == 2 and len(t2) == 2
        assert session.list_tools.await_count == 2
        assert session.initialize.await_count == 1

    @pytest.mark.asyncio
    async def test_pool_cleared_after_aexit(self) -> None:
        """Once the pool's ``async with`` block exits, the context's
        ``_mcp_pool`` is cleared — future calls fall back to per-call
        sessions."""
        ctx = _make_context()
        assert ctx._mcp_pool is None
        async with ctx.mcp_pool():
            assert ctx._mcp_pool is not None
        assert ctx._mcp_pool is None

    @pytest.mark.asyncio
    async def test_no_pool_falls_back_to_per_call_session(self) -> None:
        """When no pool is active, the legacy per-call session path runs
        unchanged — `__aenter__` is invoked once per call."""
        # Each call gets its own session mock (fresh transport CM).
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            # No `async with ctx.mcp_pool():` — legacy code path.
            await ctx.list_mcp_resources(server)
            await ctx.list_mcp_resources(server)
        # Without pooling, EACH call enters & initializes the session.
        assert session.initialize.await_count == 2


# ---------------------------------------------------------------------------
# Cleanup contract — sessions exited on pool close
# ---------------------------------------------------------------------------


class TestCleanup:
    @pytest.mark.asyncio
    async def test_sessions_cleared_on_aexit(self) -> None:
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            pool = MCPSessionPool()
            async with pool:
                await pool.acquire(server)
                assert pool.stats()["active_sessions"] == 1
            # After exit, sessions dict is cleared.
            assert pool.stats()["active_sessions"] == 0
            assert not pool.is_open

    @pytest.mark.asyncio
    async def test_exception_inside_pool_still_cleans_up(self) -> None:
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            pool = MCPSessionPool()
            with pytest.raises(RuntimeError, match="user code blew up"):
                async with pool:
                    await pool.acquire(server)
                    raise RuntimeError("user code blew up")
            assert not pool.is_open
            assert pool.stats()["active_sessions"] == 0


# ---------------------------------------------------------------------------
# Export sanity
# ---------------------------------------------------------------------------


def test_mcp_session_pool_top_level_export() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "MCPSessionPool")
    assert mmar_carl.MCPSessionPool is MCPSessionPool
