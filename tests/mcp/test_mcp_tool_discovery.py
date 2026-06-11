"""
Tests for ``ReasoningContext.register_mcp_tools(server)``.

Discovers all tools exposed by an MCP server and registers each as a CARL
tool callable via ``ToolStepConfig(tool_name="mcp:server/tool")``. Connections
are short-lived — opened per call — so no shared state lives in the context.

All tests mock the MCP SDK (``mcp.ClientSession`` + transport clients).
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
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_tool_meta(name: str) -> MagicMock:
    m = MagicMock()
    m.name = name
    return m


def _make_list_tools_result(tools: list[Any]) -> MagicMock:
    r = MagicMock()
    r.tools = tools
    return r


def _make_session_mock(
    *,
    tools_list: list[Any] | None = None,
    call_responses: dict[str, Any] | None = None,
    record_calls: list[tuple[str, dict]] | None = None,
    list_tools_exc: Exception | None = None,
) -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    if list_tools_exc is not None:
        session.list_tools = AsyncMock(side_effect=list_tools_exc)
    else:
        session.list_tools = AsyncMock(
            return_value=_make_list_tools_result(tools_list or [])
        )

    async def fake_call(name: str, arguments: dict | None = None) -> Any:
        if record_calls is not None:
            record_calls.append((name, dict(arguments or {})))
        result = MagicMock()
        result.content = (call_responses or {}).get(name, "ok")
        return result

    session.call_tool = fake_call
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_sse_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_http_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), lambda: None))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _make_context() -> ReasoningContext:
    return ReasoningContext(outer_context="x", api=_StubLLM())


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_discovers_and_registers_tools_with_default_prefix() -> None:
    session = _make_session_mock(
        tools_list=[_make_tool_meta("search"), _make_tool_meta("fetch")],
    )
    server = MCPServerConfig(server_name="docs", transport="sse", url="http://x/sse")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        registered = await ctx.register_mcp_tools(server)
    assert sorted(registered) == ["mcp:docs/fetch", "mcp:docs/search"]
    assert ctx.has_tool("mcp:docs/search")
    assert ctx.has_tool("mcp:docs/fetch")


@pytest.mark.asyncio
async def test_custom_prefix_used() -> None:
    session = _make_session_mock(tools_list=[_make_tool_meta("query")])
    server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        registered = await ctx.register_mcp_tools(server, prefix="ext-")
    assert registered == ["ext-query"]
    assert ctx.has_tool("ext-query")


@pytest.mark.asyncio
async def test_empty_prefix_uses_raw_tool_names() -> None:
    session = _make_session_mock(tools_list=[_make_tool_meta("raw_tool")])
    server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        registered = await ctx.register_mcp_tools(server, prefix="")
    assert registered == ["raw_tool"]
    assert ctx.has_tool("raw_tool")


@pytest.mark.asyncio
async def test_tools_carry_provided_tags() -> None:
    session = _make_session_mock(tools_list=[_make_tool_meta("search")])
    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        await ctx.register_mcp_tools(server, tags=["mcp", "external"])
    assert ctx.get_tool_tags("mcp:srv/search") == {"mcp", "external"}


@pytest.mark.asyncio
async def test_empty_tool_list_registers_nothing() -> None:
    session = _make_session_mock(tools_list=[])
    server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        registered = await ctx.register_mcp_tools(server)
    assert registered == []


@pytest.mark.asyncio
async def test_tool_meta_without_name_attribute_is_skipped() -> None:
    nameless = MagicMock()
    nameless.name = None  # Skipped
    valid = _make_tool_meta("valid")
    session = _make_session_mock(tools_list=[nameless, valid])
    server = MCPServerConfig(server_name="x", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        registered = await ctx.register_mcp_tools(server)
    assert registered == ["mcp:x/valid"]


# --------------------------------------------------------------------------- #
# Wrapper invocation — short-lived connection per call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_registered_tool_calls_underlying_mcp_session() -> None:
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock(
        tools_list=[_make_tool_meta("search")],
        call_responses={"search": "hits for query"},
        record_calls=recorded,
    )
    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        await ctx.register_mcp_tools(server)
        wrapper = ctx.get_tool("mcp:srv/search")
        assert wrapper is not None
        result = await wrapper(query="climate", limit=5)
    assert result == "hits for query"
    assert recorded == [("search", {"query": "climate", "limit": 5})]


@pytest.mark.asyncio
async def test_wrapper_metadata_attributes_set() -> None:
    """The registered wrapper exposes is_mcp_tool/mcp_server_name/mcp_raw_tool_name."""
    session = _make_session_mock(tools_list=[_make_tool_meta("search")])
    server = MCPServerConfig(server_name="docs", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        await ctx.register_mcp_tools(server)
    w = ctx.get_tool("mcp:docs/search")
    assert getattr(w, "is_mcp_tool", False) is True
    assert getattr(w, "mcp_server_name", None) == "docs"
    assert getattr(w, "mcp_raw_tool_name", None) == "search"


@pytest.mark.asyncio
async def test_wrappers_are_independent_per_tool() -> None:
    """A wrapper dispatches to its own underlying tool, not the last one registered."""
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock(
        tools_list=[_make_tool_meta("alpha"), _make_tool_meta("beta")],
        call_responses={"alpha": "A", "beta": "B"},
        record_calls=recorded,
    )
    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        await ctx.register_mcp_tools(server)
        a_result = await ctx.get_tool("mcp:srv/alpha")()
        b_result = await ctx.get_tool("mcp:srv/beta")()
    assert a_result == "A"
    assert b_result == "B"
    assert [name for name, _args in recorded] == ["alpha", "beta"]


# --------------------------------------------------------------------------- #
# Transport coverage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stdio_transport_used_for_stdio_server() -> None:
    session = _make_session_mock(tools_list=[_make_tool_meta("t")])
    server = MCPServerConfig(server_name="local", transport="stdio", command="my-mcp")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.stdio.stdio_client", return_value=_make_sse_cm()),
        patch("mcp.client.stdio.StdioServerParameters", MagicMock()),
    ):
        registered = await ctx.register_mcp_tools(server)
    assert registered == ["mcp:local/t"]


@pytest.mark.asyncio
async def test_http_transport_used_for_http_server() -> None:
    session = _make_session_mock(tools_list=[_make_tool_meta("t")])
    server = MCPServerConfig(server_name="h", transport="http", url="http://x/mcp")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.streamable_http.streamable_http_client", return_value=_make_http_cm()),
        patch("httpx.AsyncClient"),
    ):
        registered = await ctx.register_mcp_tools(server)
    assert registered == ["mcp:h/t"]


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_without_url_raises() -> None:
    server = MCPServerConfig(server_name="x", transport="sse")  # no url
    ctx = _make_context()
    with pytest.raises(ValueError, match="url is required"):
        await ctx.register_mcp_tools(server)


@pytest.mark.asyncio
async def test_http_without_url_raises() -> None:
    server = MCPServerConfig(server_name="x", transport="http")
    ctx = _make_context()
    with pytest.raises(ValueError, match="url is required"):
        await ctx.register_mcp_tools(server)


@pytest.mark.asyncio
async def test_malformed_list_tools_response_raises() -> None:
    bad_result = MagicMock()
    bad_result.tools = None  # not iterable
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=bad_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        with pytest.raises(RuntimeError, match="malformed"):
            await ctx.register_mcp_tools(server)


@pytest.mark.asyncio
async def test_list_tools_exception_propagates() -> None:
    session = _make_session_mock(list_tools_exc=RuntimeError("server unreachable"))
    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        with pytest.raises(RuntimeError, match="server unreachable"):
            await ctx.register_mcp_tools(server)


# --------------------------------------------------------------------------- #
# End-to-end: MCP tool runs inside a ToolStep
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_registered_mcp_tool_callable_from_toolstep() -> None:
    """Full chain integration: discover, then use the tool inside a ToolStep."""
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock(
        tools_list=[_make_tool_meta("answer")],
        call_responses={"answer": {"text": "MCP says hi"}},
        record_calls=recorded,
    )
    server = MCPServerConfig(server_name="srv", transport="sse", url="http://x")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        await ctx.register_mcp_tools(server)
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1, title="ask",
                    config=ToolStepConfig(
                        tool_name="mcp:srv/answer",
                        parameters=[],
                        input_mapping={"prompt": "$outer_context"},
                    ),
                ),
            ],
            max_workers=1,
        )
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert result.step_results[0].result_data == {"text": "MCP says hi"}
    assert recorded == [("answer", {"prompt": "x"})]


@pytest.mark.asyncio
async def test_two_servers_register_disjoint_namespaces() -> None:
    """Registering two servers gives ``mcp:A/x`` and ``mcp:B/x`` separately."""
    sess_a = _make_session_mock(tools_list=[_make_tool_meta("x")])
    sess_b = _make_session_mock(tools_list=[_make_tool_meta("x")])

    sessions = iter([sess_a, sess_b])

    def session_side_effect(*args, **kwargs):
        return next(sessions)

    server_a = MCPServerConfig(server_name="alpha", transport="sse", url="http://a/sse")
    server_b = MCPServerConfig(server_name="beta", transport="sse", url="http://b/sse")
    ctx = _make_context()
    with (
        patch("mcp.ClientSession", side_effect=session_side_effect),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        a_names = await ctx.register_mcp_tools(server_a)
        b_names = await ctx.register_mcp_tools(server_b)
    assert a_names == ["mcp:alpha/x"]
    assert b_names == ["mcp:beta/x"]
    assert ctx.has_tool("mcp:alpha/x")
    assert ctx.has_tool("mcp:beta/x")
