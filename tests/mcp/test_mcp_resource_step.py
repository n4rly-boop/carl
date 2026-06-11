"""
Tests for ``MCPResourceStepDescription``.

The resource-reading counterpart to ``MCPStepDescription`` — fetches a named
read-only resource from an MCP server (e.g. files, docs, schemas) and stores
its content in memory + history. Uses the same transport infrastructure as
the tool-calling step.

All tests mock the MCP SDK (``mcp.ClientSession`` + transport clients) so
no real MCP server is required.
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
    MCPResourceStepConfig,
    MCPResourceStepDescription,
    MCPServerConfig,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_resource_result(
    *,
    text: str | None = None,
    blob: bytes | None = None,
    items: list[Any] | None = None,
) -> MagicMock:
    """Build a fake ``ReadResourceResult`` with the given content shape."""
    result = MagicMock()
    if items is not None:
        result.contents = items
    elif text is not None or blob is not None:
        item = MagicMock()
        item.text = text
        item.blob = blob
        result.contents = [item]
    else:
        result.contents = []
    return result


def _make_session_mock(read_resource_result: Any, *, record_calls: list[str] | None = None) -> MagicMock:
    """Mock ``ClientSession`` whose ``read_resource`` returns *read_resource_result*."""

    async def fake_read(uri: str) -> Any:
        if record_calls is not None:
            record_calls.append(uri)
        return read_resource_result

    session = MagicMock()
    session.initialize = AsyncMock()
    session.read_resource = fake_read
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
    cm.__aenter__ = AsyncMock(
        return_value=(MagicMock(), MagicMock(), lambda: None)
    )
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


# --------------------------------------------------------------------------- #
# Config + step description
# --------------------------------------------------------------------------- #


def test_config_requires_resource_uri() -> None:
    with pytest.raises(Exception):
        MCPResourceStepConfig(
            server=MCPServerConfig(server_name="x", transport="sse", url="http://x"),
            resource_uri="",
        )


def test_config_default_namespace_is_mcp_resource() -> None:
    cfg = MCPResourceStepConfig(
        server=MCPServerConfig(server_name="x", transport="sse", url="http://x"),
        resource_uri="docs://x",
    )
    assert cfg.output_namespace == "mcp_resource"
    assert cfg.output_memory_key == ""
    assert cfg.timeout == 30.0


def test_step_description_step_type() -> None:
    step = MCPResourceStepDescription(
        number=1, title="x",
        config=MCPResourceStepConfig(
            server=MCPServerConfig(server_name="x", transport="sse", url="http://x"),
            resource_uri="docs://x",
        ),
    )
    assert step.step_type.value == "mcp_resource"


# --------------------------------------------------------------------------- #
# Happy path — single text content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reads_single_text_resource_into_memory() -> None:
    """Single-content resource → text returned and stored in memory."""
    recorded: list[str] = []
    session = _make_session_mock(
        _make_resource_result(text="API REFERENCE BODY"),
        record_calls=recorded,
    )
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="docs",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="srv", transport="sse",
                                               url="http://srv/sse"),
                        resource_uri="docs://api/ref.md",
                        output_memory_key="api_docs",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result == "API REFERENCE BODY"
    assert sr.result_data == "API REFERENCE BODY"
    assert ctx.memory["mcp_resource"]["api_docs"] == "API REFERENCE BODY"
    assert recorded == ["docs://api/ref.md"]


@pytest.mark.asyncio
async def test_default_namespace_used_when_omitted() -> None:
    session = _make_session_mock(_make_resource_result(text="x"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                        output_memory_key="k",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        await chain.execute_async(ctx)
    # Default namespace
    assert "mcp_resource" in ctx.memory
    assert ctx.memory["mcp_resource"]["k"] == "x"


@pytest.mark.asyncio
async def test_custom_namespace_used_when_set() -> None:
    session = _make_session_mock(_make_resource_result(text="x"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                        output_memory_key="k",
                        output_namespace="docs",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        await chain.execute_async(ctx)
    assert ctx.memory["docs"]["k"] == "x"
    assert "mcp_resource" not in ctx.memory


@pytest.mark.asyncio
async def test_empty_output_memory_key_does_not_write_memory() -> None:
    """When ``output_memory_key`` is empty, the resource is only in history/result."""
    session = _make_session_mock(_make_resource_result(text="x"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert "mcp_resource" not in ctx.memory


# --------------------------------------------------------------------------- #
# Multi-content / binary resources
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_blob_content_returned_when_no_text() -> None:
    """A content item with ``blob`` and ``text=None`` returns the blob bytes."""
    session = _make_session_mock(_make_resource_result(blob=b"binary-data"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="bin",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="bin://x",
                        output_memory_key="bin",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert ctx.memory["mcp_resource"]["bin"] == b"binary-data"


@pytest.mark.asyncio
async def test_multi_content_returns_list_as_is() -> None:
    """A multi-item ``contents`` list is returned intact (caller inspects mime types etc.)."""
    item1 = MagicMock()
    item1.text = "part 1"
    item1.blob = None
    item2 = MagicMock()
    item2.text = "part 2"
    item2.blob = None
    session = _make_session_mock(_make_resource_result(items=[item1, item2]))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="multi",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://multi",
                        output_memory_key="parts",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    parts = ctx.memory["mcp_resource"]["parts"]
    assert isinstance(parts, list)
    assert len(parts) == 2


@pytest.mark.asyncio
async def test_empty_contents_returns_empty_string() -> None:
    session = _make_session_mock(_make_resource_result())  # contents=[]
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="empty",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://empty",
                        output_memory_key="e",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert ctx.memory["mcp_resource"]["e"] == ""


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stdio_transport_uses_stdio_client() -> None:
    """The executor honours stdio transport by going through ``stdio_client``."""
    session = _make_session_mock(_make_resource_result(text="stdio-content"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.stdio.stdio_client", return_value=_make_sse_cm()),
        patch("mcp.client.stdio.StdioServerParameters", MagicMock()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(
                            server_name="local", transport="stdio", command="my-mcp",
                        ),
                        resource_uri="r://x",
                        output_memory_key="k",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert ctx.memory["mcp_resource"]["k"] == "stdio-content"


@pytest.mark.asyncio
async def test_http_transport_uses_streamable_http_client() -> None:
    session = _make_session_mock(_make_resource_result(text="http-content"))
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.streamable_http.streamable_http_client", return_value=_make_http_cm()),
        patch("httpx.AsyncClient"),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(
                            server_name="srv", transport="http", url="http://x/mcp",
                        ),
                        resource_uri="r://x",
                        output_memory_key="k",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert ctx.memory["mcp_resource"]["k"] == "http-content"


@pytest.mark.asyncio
async def test_sse_transport_without_url_fails() -> None:
    chain = ReasoningChain(
        steps=[
            MCPResourceStepDescription(
                number=1, title="r",
                config=MCPResourceStepConfig(
                    server=MCPServerConfig(server_name="s", transport="sse"),  # no url
                    resource_uri="r://x",
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_StubLLM())
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "url is required" in sr.error_message


@pytest.mark.asyncio
async def test_http_transport_without_url_fails() -> None:
    chain = ReasoningChain(
        steps=[
            MCPResourceStepDescription(
                number=1, title="r",
                config=MCPResourceStepConfig(
                    server=MCPServerConfig(server_name="s", transport="http"),
                    resource_uri="r://x",
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_StubLLM())
    result = await chain.execute_async(ctx)
    assert not result.step_results[0].success


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_read_resource_exception_marks_step_failed() -> None:
    async def boom(uri: str) -> Any:
        raise RuntimeError("resource not found")

    session = MagicMock()
    session.initialize = AsyncMock()
    session.read_resource = boom
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="r",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "resource not found" in sr.error_message


# --------------------------------------------------------------------------- #
# Integration with downstream steps
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_resource_content_flows_via_memory_to_tool_step() -> None:
    """End-to-end: MCP resource → memory → Tool reads it via ``$memory.ns.key``."""
    session = _make_session_mock(_make_resource_result(text="DOC TEXT"))
    captured: dict[str, str] = {}

    def capture(text: str) -> str:
        captured["got"] = text
        return "captured"

    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="load",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                        output_memory_key="doc",
                        output_namespace="docs",
                    ),
                ),
                ToolStepDescription(
                    number=2, title="use", dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="cap", parameters=[],
                        input_mapping={"text": "$memory.docs.doc"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.register_tool("cap", capture)
        result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert captured["got"] == "DOC TEXT"


@pytest.mark.asyncio
async def test_resource_content_flows_via_history_to_next_step() -> None:
    """Without ``output_memory_key`` the resource is still visible via ``$history``."""
    session = _make_session_mock(_make_resource_result(text="HISTORY-RESOURCE"))
    captured: dict[str, str] = {}

    def capture(text: str) -> str:
        captured["got"] = text
        return "ok"

    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_cm()),
    ):
        chain = ReasoningChain(
            steps=[
                MCPResourceStepDescription(
                    number=1, title="load",
                    config=MCPResourceStepConfig(
                        server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                        resource_uri="r://x",
                    ),
                ),
                ToolStepDescription(
                    number=2, title="use", dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="cap", parameters=[],
                        input_mapping={"text": "$history[-1]"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.register_tool("cap", capture)
        await chain.execute_async(ctx)
    assert "HISTORY-RESOURCE" in captured["got"]


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #


def test_chain_to_dict_serialises_mcp_resource_step() -> None:
    chain = ReasoningChain(
        steps=[
            MCPResourceStepDescription(
                number=1, title="r",
                config=MCPResourceStepConfig(
                    server=MCPServerConfig(server_name="s", transport="sse", url="http://x"),
                    resource_uri="docs://api",
                    output_memory_key="api",
                    output_namespace="docs",
                ),
            ),
        ],
        max_workers=1,
    )
    data = chain.to_dict()
    step_data = data["steps"][0]
    assert step_data["step_type"] == "mcp_resource"
    cfg = step_data["step_config"]
    assert cfg["resource_uri"] == "docs://api"
    assert cfg["output_memory_key"] == "api"
    assert cfg["output_namespace"] == "docs"
    assert cfg["server"]["server_name"] == "s"
