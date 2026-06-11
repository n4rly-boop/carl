"""``context.list_mcp_resources(server)``.

CARE's catalog screen renders the resource inventory each configured
MCP server exposes (read-only data sources: docs, files, schemas).
Distinct from ``register_mcp_tools`` which discovers *callable* tools.

Plus a stability snapshot — locks the public field set of
``MCPServerConfig`` so a future contributor renaming a field has to
update the snapshot intentionally. CARE will load this config shape
from a TOML file at startup; the field set should be stable across
minor releases.

All tests mock the MCP SDK so no real server is needed.
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
    ReasoningContext,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/mcp/test_mcp_tool_discovery.py)
# ---------------------------------------------------------------------------


def _make_resource(uri: str, name: str = "", description: str = "") -> MagicMock:
    r = MagicMock()
    r.uri = uri
    r.name = name or uri.rsplit("/", 1)[-1]
    r.description = description
    r.mimeType = "text/plain"
    return r


def _make_list_resources_result(resources: list[Any]) -> MagicMock:
    r = MagicMock()
    r.resources = resources
    return r


def _make_session_mock(
    *,
    resources_list: list[Any] | None = None,
    list_resources_exc: Exception | None = None,
) -> MagicMock:
    session = MagicMock()
    session.initialize = AsyncMock()
    if list_resources_exc is not None:
        session.list_resources = AsyncMock(side_effect=list_resources_exc)
    else:
        session.list_resources = AsyncMock(
            return_value=_make_list_resources_result(resources_list or []),
        )
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
# list_mcp_resources
# ---------------------------------------------------------------------------


class TestListMcpResources:
    @pytest.mark.asyncio
    async def test_lists_resources_via_sse(self) -> None:
        session = _make_session_mock(resources_list=[
            _make_resource("docs://api/reference.md"),
            _make_resource("docs://api/quickstart.md"),
        ])
        server = MCPServerConfig(
            server_name="docs", transport="sse", url="http://x/sse",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            resources = await ctx.list_mcp_resources(server)
        assert len(resources) == 2
        assert resources[0].uri == "docs://api/reference.md"
        assert resources[1].uri == "docs://api/quickstart.md"

    @pytest.mark.asyncio
    async def test_lists_resources_via_stdio(self) -> None:
        session = _make_session_mock(resources_list=[
            _make_resource("file:///tmp/data.csv"),
        ])
        server = MCPServerConfig(
            server_name="fs", transport="stdio", command="mcp-fs",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch(
                "mcp.client.stdio.stdio_client",
                return_value=_make_stdio_cm(),
            ),
            patch("mcp.client.stdio.StdioServerParameters"),
        ):
            resources = await ctx.list_mcp_resources(server)
        assert len(resources) == 1
        assert resources[0].uri == "file:///tmp/data.csv"

    @pytest.mark.asyncio
    async def test_empty_resource_list(self) -> None:
        session = _make_session_mock(resources_list=[])
        server = MCPServerConfig(
            server_name="x", transport="sse", url="http://x",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            resources = await ctx.list_mcp_resources(server)
        assert resources == []

    @pytest.mark.asyncio
    async def test_malformed_response_raises(self) -> None:
        """``result.resources`` missing → RuntimeError with a clear msg."""
        session = MagicMock()
        session.initialize = AsyncMock()
        bad = MagicMock(spec=[])  # no .resources attribute
        session.list_resources = AsyncMock(return_value=bad)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        server = MCPServerConfig(
            server_name="broken", transport="sse", url="http://x",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            with pytest.raises(RuntimeError, match="malformed list_resources"):
                await ctx.list_mcp_resources(server)

    @pytest.mark.asyncio
    async def test_propagates_server_exception(self) -> None:
        session = _make_session_mock(
            list_resources_exc=RuntimeError("server unreachable"),
        )
        server = MCPServerConfig(
            server_name="x", transport="sse", url="http://x",
        )
        ctx = _make_context()
        with (
            patch("mcp.ClientSession", return_value=session),
            patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
        ):
            with pytest.raises(RuntimeError, match="server unreachable"):
                await ctx.list_mcp_resources(server)

    @pytest.mark.asyncio
    async def test_sse_requires_url(self) -> None:
        server = MCPServerConfig(server_name="x", transport="sse")  # no url
        ctx = _make_context()
        with pytest.raises(ValueError, match="url is required"):
            await ctx.list_mcp_resources(server)

    @pytest.mark.asyncio
    async def test_http_requires_url(self) -> None:
        server = MCPServerConfig(server_name="x", transport="http")  # no url
        ctx = _make_context()
        with pytest.raises(ValueError, match="url is required"):
            await ctx.list_mcp_resources(server)

    @pytest.mark.asyncio
    async def test_unknown_transport_raises(self) -> None:
        # Build an MCPServerConfig-shaped duck so we can poke an invalid
        # transport past pydantic validation (which restricts the Literal).
        server = MagicMock()
        server.transport = "made-up"
        server.server_name = "x"
        server.url = "http://x"
        server.headers = {}
        server.command = None
        server.args = []
        ctx = _make_context()
        with pytest.raises(NotImplementedError, match="not supported"):
            await ctx.list_mcp_resources(server)


# ---------------------------------------------------------------------------
# MCPServerConfig field-set stability
# ---------------------------------------------------------------------------


class TestMcpServerConfigStability:
    """Locks the public field set of :class:`MCPServerConfig`.

    CARE loads this shape from ``~/.config/care/mcp_servers.toml`` at
    startup; the field set should not drift in a minor release after
    graduation. This test snapshots the current fields + types so a
    future contributor renaming a field has to update the snapshot
    intentionally (rather than silently breaking CARE's TOML loader).
    """

    EXPECTED_FIELDS: dict[str, str] = {
        "server_name": "str",
        "transport": "Literal['stdio', 'http', 'sse']",
        "command": "str | None",
        "args": "list[str]",
        "url": "str | None",
        "headers": "dict[str, str]",
    }

    def test_field_set_locked(self) -> None:
        actual = set(MCPServerConfig.model_fields)
        expected = set(self.EXPECTED_FIELDS)
        missing = expected - actual
        extra = actual - expected
        assert not missing, (
            f"MCPServerConfig dropped fields {missing!r}. "
            f"This is a breaking change for CARE's TOML loader."
        )
        assert not extra, (
            f"MCPServerConfig grew new fields {extra!r}. "
            f"Update TestMcpServerConfigStability.EXPECTED_FIELDS "
            f"intentionally before adding fields."
        )

    def test_transport_literal_values(self) -> None:
        """The supported transports MUST stay ``stdio`` / ``http`` /
        ``sse`` — CARE config files reference these strings verbatim."""
        config = MCPServerConfig(server_name="x", transport="stdio")
        assert config.transport == "stdio"
        # The other two values must validate.
        for t in ("http", "sse"):
            config = MCPServerConfig(server_name="x", transport=t, url="http://x")
            assert config.transport == t

    def test_unknown_transport_rejected(self) -> None:
        with pytest.raises(Exception):  # pydantic ValidationError
            MCPServerConfig(server_name="x", transport="grpc")

    def test_round_trip_via_model_dump(self) -> None:
        original = MCPServerConfig(
            server_name="docs",
            transport="sse",
            url="http://x/sse",
            headers={"Authorization": "Bearer xyz"},
        )
        rebuilt = MCPServerConfig.model_validate(original.model_dump())
        assert rebuilt == original

    def test_round_trip_via_json(self) -> None:
        original = MCPServerConfig(
            server_name="fs",
            transport="stdio",
            command="mcp-fs",
            args=["--root", "/tmp"],
        )
        as_json = original.model_dump_json()
        rebuilt = MCPServerConfig.model_validate_json(as_json)
        assert rebuilt == original

    def test_defaults(self) -> None:
        config = MCPServerConfig(server_name="x")
        # stdio default; everything else falsy / empty.
        assert config.transport == "stdio"
        assert config.command is None
        assert config.args == []
        assert config.url is None
        assert config.headers == {}

    def test_minimal_required_field(self) -> None:
        """Only ``server_name`` is required. Defaults cover the rest."""
        config = MCPServerConfig(server_name="x")
        assert config.server_name == "x"
