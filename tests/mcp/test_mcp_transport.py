"""
Tests for MCP transport implementations (stdio, http, sse).

All tests use mocked MCP SDK internals — no real MCP server required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These tests stub the MCP SDK (``mcp.ClientSession`` + transport clients) via
# ``unittest.mock.patch``; without the optional ``mcp`` extra installed the
# patch targets can't be imported. Skip the module cleanly in that case.
pytest.importorskip("mcp", reason="requires the optional `mcp` extra")

from mmar_carl.models.config import MCPServerConfig, MCPStepConfig
from mmar_carl.models.steps import MCPStepDescription
from mmar_carl.step_executors import MCPStepExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(
    tool_name: str,
    server_cfg: MCPServerConfig,
    arguments: dict | None = None,
    argument_mapping: dict | None = None,
    timeout: float = 10.0,
) -> MCPStepDescription:
    return MCPStepDescription(
        number=1,
        title="Test MCP step",
        config=MCPStepConfig(
            server=server_cfg,
            tool_name=tool_name,
            arguments=arguments or {},
            argument_mapping=argument_mapping or {},
            timeout=timeout,
        ),
    )


def _make_session_mock(return_content: Any) -> MagicMock:
    """Return a mock MCP ClientSession whose call_tool returns the given content."""
    result_obj = MagicMock()
    result_obj.content = return_content

    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=result_obj)
    # Make session usable as async context manager
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_client_cm(read_obj=None, write_obj=None):
    """Return an async context manager that yields (read, write)."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(read_obj or MagicMock(), write_obj or MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _make_http_cm(read_obj=None, write_obj=None, session_id_fn=None):
    """Return an async context manager that yields (read, write, get_session_id)."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(
        return_value=(
            read_obj or MagicMock(),
            write_obj or MagicMock(),
            session_id_fn or (lambda: None),
        )
    )
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


# ---------------------------------------------------------------------------
# MCPServerConfig validation
# ---------------------------------------------------------------------------


class TestMCPServerConfig:
    def test_transport_stdio_default(self):
        cfg = MCPServerConfig(server_name="srv", command="my-server")
        assert cfg.transport == "stdio"

    def test_transport_sse(self):
        cfg = MCPServerConfig(server_name="srv", transport="sse", url="http://localhost:8080/sse")
        assert cfg.transport == "sse"

    def test_transport_http(self):
        cfg = MCPServerConfig(server_name="srv", transport="http", url="http://localhost:8080/mcp")
        assert cfg.transport == "http"

    def test_headers_accepted(self):
        cfg = MCPServerConfig(
            server_name="srv",
            transport="http",
            url="http://localhost/mcp",
            headers={"Authorization": "Bearer tok123"},
        )
        assert cfg.headers == {"Authorization": "Bearer tok123"}

    def test_invalid_transport_rejected(self):
        with pytest.raises(Exception):
            MCPServerConfig(server_name="srv", transport="websocket")  # no longer valid


# ---------------------------------------------------------------------------
# stdio transport (regression guard)
# ---------------------------------------------------------------------------


class TestStdioTransport:
    @pytest.mark.asyncio
    async def test_stdio_success(self):
        session_mock = _make_session_mock(return_content="tool output")
        client_cm = _make_client_cm()

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.stdio.stdio_client", return_value=client_cm),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="echo",
                server_cfg=MCPServerConfig(
                    server_name="local",
                    transport="stdio",
                    command="echo-server",
                ),
            )
            result = await executor._execute_mcp_call(step.config, {"text": "hello"})

        assert result == "tool output"

    @pytest.mark.asyncio
    async def test_stdio_passes_args(self):
        """Verify StdioServerParameters receives the configured command and args."""
        captured: list[Any] = []

        def fake_stdio_client(server_params):
            captured.append(server_params)
            return _make_client_cm()

        session_mock = _make_session_mock(return_content="ok")
        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.stdio.stdio_client", side_effect=fake_stdio_client),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="run",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="stdio",
                    command="/usr/bin/my-mcp",
                    args=["--port", "9000"],
                ),
            )
            await executor._execute_mcp_call(step.config, {})

        assert len(captured) == 1
        params = captured[0]
        assert params.command == "/usr/bin/my-mcp"
        assert params.args == ["--port", "9000"]


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------


class TestSSETransport:
    @pytest.mark.asyncio
    async def test_sse_success(self):
        session_mock = _make_session_mock(return_content={"answer": 42})
        client_cm = _make_client_cm()

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.sse.sse_client", return_value=client_cm),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="compute",
                server_cfg=MCPServerConfig(
                    server_name="remote-sse",
                    transport="sse",
                    url="http://mcp.example.com/events",
                ),
            )
            result = await executor._execute_mcp_call(step.config, {"x": 7})

        assert result == {"answer": 42}
        session_mock.call_tool.assert_awaited_once_with("compute", arguments={"x": 7})

    @pytest.mark.asyncio
    async def test_sse_passes_url_and_headers(self):
        captured: list[dict] = []

        def fake_sse_client(url, headers=None, timeout=30.0, **kwargs):
            captured.append({"url": url, "headers": headers, "timeout": timeout})
            return _make_client_cm()

        session_mock = _make_session_mock(return_content="ok")
        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.sse.sse_client", side_effect=fake_sse_client),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="ping",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="sse",
                    url="https://api.example.com/sse",
                    headers={"Authorization": "Bearer secret"},
                ),
                timeout=15.0,
            )
            await executor._execute_mcp_call(step.config, {})

        assert len(captured) == 1
        assert captured[0]["url"] == "https://api.example.com/sse"
        assert captured[0]["headers"] == {"Authorization": "Bearer secret"}
        assert captured[0]["timeout"] == 15.0

    @pytest.mark.asyncio
    async def test_sse_requires_url(self):
        executor = MCPStepExecutor()
        step = _make_step(
            tool_name="ping",
            server_cfg=MCPServerConfig(server_name="srv", transport="sse"),  # no url
        )
        with pytest.raises(ValueError, match="url is required"):
            await executor._execute_mcp_call(step.config, {})

    @pytest.mark.asyncio
    async def test_sse_empty_headers_passes_none(self):
        """When no headers are configured, None is passed (not an empty dict)."""
        captured: list[dict] = []

        def fake_sse_client(url, headers=None, **kwargs):
            captured.append({"headers": headers})
            return _make_client_cm()

        session_mock = _make_session_mock(return_content="ok")
        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.sse.sse_client", side_effect=fake_sse_client),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="ping",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="sse",
                    url="http://localhost/sse",
                ),
            )
            await executor._execute_mcp_call(step.config, {})

        assert captured[0]["headers"] is None


# ---------------------------------------------------------------------------
# HTTP (streamable) transport
# ---------------------------------------------------------------------------


class TestHTTPTransport:
    @pytest.mark.asyncio
    async def test_http_success(self):
        session_mock = _make_session_mock(return_content="streamed result")
        client_cm = _make_http_cm()

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.streamable_http.streamable_http_client", return_value=client_cm),
            patch("httpx.AsyncClient"),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="search",
                server_cfg=MCPServerConfig(
                    server_name="http-srv",
                    transport="http",
                    url="http://mcp.example.com/mcp",
                ),
            )
            result = await executor._execute_mcp_call(step.config, {"q": "cats"})

        assert result == "streamed result"
        session_mock.call_tool.assert_awaited_once_with("search", arguments={"q": "cats"})

    @pytest.mark.asyncio
    async def test_http_creates_async_client_with_headers(self):
        """Verify httpx.AsyncClient is created with the configured headers."""
        captured_clients: list[Any] = []

        def fake_streamable_http(url, *, http_client=None, **kwargs):
            captured_clients.append(http_client)
            return _make_http_cm()

        session_mock = _make_session_mock(return_content="ok")
        fake_http_client = MagicMock(spec=["__class__"])

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch(
                "mcp.client.streamable_http.streamable_http_client",
                side_effect=fake_streamable_http,
            ),
            patch("httpx.AsyncClient", return_value=fake_http_client) as mock_async_client,
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="run",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="http",
                    url="http://localhost/mcp",
                    headers={"X-API-Key": "mykey"},
                ),
                timeout=20.0,
            )
            await executor._execute_mcp_call(step.config, {})

        # AsyncClient should have been constructed with the auth headers and timeout
        mock_async_client.assert_called_once_with(
            headers={"X-API-Key": "mykey"},
            timeout=20.0,
        )
        # The constructed client should have been passed to streamable_http_client
        assert captured_clients[0] is fake_http_client

    @pytest.mark.asyncio
    async def test_http_requires_url(self):
        executor = MCPStepExecutor()
        step = _make_step(
            tool_name="run",
            server_cfg=MCPServerConfig(server_name="srv", transport="http"),  # no url
        )
        with pytest.raises(ValueError, match="url is required"):
            await executor._execute_mcp_call(step.config, {})

    @pytest.mark.asyncio
    async def test_http_passes_url_to_client(self):
        captured: list[str] = []

        def fake_streamable_http(url, *, http_client=None, **kwargs):
            captured.append(url)
            return _make_http_cm()

        session_mock = _make_session_mock(return_content="ok")
        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch(
                "mcp.client.streamable_http.streamable_http_client",
                side_effect=fake_streamable_http,
            ),
            patch("httpx.AsyncClient"),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="run",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="http",
                    url="https://api.example.com/v1/mcp",
                ),
            )
            await executor._execute_mcp_call(step.config, {})

        assert captured == ["https://api.example.com/v1/mcp"]


# ---------------------------------------------------------------------------
# MCPStepExecutor.execute() — end-to-end
# ---------------------------------------------------------------------------


class TestMCPStepExecutorEndToEnd:
    @pytest.mark.asyncio
    async def test_execute_sse_success(self):
        """MCPStepExecutor.execute() with SSE transport returns success result."""
        from unittest.mock import MagicMock

        from mmar_carl.models.context import ReasoningContext

        api = MagicMock()
        api.get_response = AsyncMock(return_value="ok")
        ctx = ReasoningContext(outer_context="test", api=api)

        session_mock = _make_session_mock(return_content="search result")
        client_cm = _make_client_cm()

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.sse.sse_client", return_value=client_cm),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="search",
                server_cfg=MCPServerConfig(
                    server_name="my-sse",
                    transport="sse",
                    url="http://localhost:3000/sse",
                ),
            )
            result = await executor.execute(step, ctx)

        assert result.success
        assert result.result == "search result"  # string content passed through as-is
        assert result.step_type.value == "mcp"

    @pytest.mark.asyncio
    async def test_execute_http_success(self):
        """MCPStepExecutor.execute() with HTTP transport returns success result."""
        from unittest.mock import MagicMock

        from mmar_carl.models.context import ReasoningContext

        api = MagicMock()
        api.get_response = AsyncMock(return_value="ok")
        ctx = ReasoningContext(outer_context="test", api=api)

        session_mock = _make_session_mock(return_content={"data": "result"})
        client_cm = _make_http_cm()

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.streamable_http.streamable_http_client", return_value=client_cm),
            patch("httpx.AsyncClient"),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="fetch",
                server_cfg=MCPServerConfig(
                    server_name="my-http",
                    transport="http",
                    url="http://localhost:3000/mcp",
                ),
            )
            result = await executor.execute(step, ctx)

        assert result.success
        assert result.result_data == {"data": "result"}

    @pytest.mark.asyncio
    async def test_execute_missing_url_returns_failure(self):
        """MCPStepExecutor.execute() wraps ValueError from missing url into failure."""
        from unittest.mock import MagicMock

        from mmar_carl.models.context import ReasoningContext

        api = MagicMock()
        api.get_response = AsyncMock(return_value="ok")
        ctx = ReasoningContext(outer_context="test", api=api)

        executor = MCPStepExecutor()
        step = _make_step(
            tool_name="search",
            server_cfg=MCPServerConfig(server_name="srv", transport="sse"),  # no url
        )
        result = await executor.execute(step, ctx)

        assert not result.success
        assert "url is required" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_argument_mapping_resolved(self):
        """argument_mapping resolves values from context before passing to tool."""
        from unittest.mock import MagicMock

        from mmar_carl.models.context import ReasoningContext

        api = MagicMock()
        api.get_response = AsyncMock(return_value="ok")
        ctx = ReasoningContext(outer_context="test", api=api)
        ctx.memory["input"] = {"query": "climate change"}

        captured_args: list[dict] = []

        def fake_sse_client(url, **kwargs):
            return _make_client_cm()

        result_obj = MagicMock()
        result_obj.content = "results"
        session_mock = MagicMock()
        session_mock.initialize = AsyncMock()
        session_mock.call_tool = AsyncMock(side_effect=lambda name, arguments: captured_args.append(arguments) or result_obj)
        session_mock.__aenter__ = AsyncMock(return_value=session_mock)
        session_mock.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("mcp.ClientSession", return_value=session_mock),
            patch("mcp.client.sse.sse_client", side_effect=fake_sse_client),
        ):
            executor = MCPStepExecutor()
            step = _make_step(
                tool_name="search",
                server_cfg=MCPServerConfig(
                    server_name="srv",
                    transport="sse",
                    url="http://localhost/sse",
                ),
                argument_mapping={"q": "$memory.input.query"},
            )
            await executor.execute(step, ctx)

        assert captured_args == [{"q": "climate change"}]
