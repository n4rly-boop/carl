"""live MCP integration tests.

Promotes the MCP stack from EXPERIMENTAL to STABLE by exercising every
CARL→MCP path end-to-end against a **real** MCP server (no SDK mocks).
The server is a tiny `FastMCP` instance spawned as a subprocess over the
stdio transport — same shape as the official ``mcp-server-filesystem``
/ ``mcp-server-memory`` reference servers, but lives inside this test
file so CI doesn't need to install external binaries.

Marker
------
Every test is decorated with ``@pytest.mark.mcp_live``. The default
``pytest`` run deselects them (``addopts = -m 'not live and not
mcp_live'``) so the hermetic-mock test suite stays fast. Run the live
suite with::

    pytest -m mcp_live tests/mcp/

The marker fires a fresh subprocess per test — slower than the mock
tests but verifies actual wire-protocol behaviour: ``initialize``,
``list_tools``, ``call_tool``, ``list_resources``, plus the
:class:`MCPSessionPool` reuse contract.

Why stdio only
--------------
SSE / HTTP transports need the server to bind to a port and an
asyncio-ready handshake — testable but flakier in CI. The stdio path
covers the full ``ClientSession`` lifecycle (initialize / list /
call / shutdown) plus the pool's ``AsyncExitStack`` integration, which
is the bulk of what CARE depends on. SSE / HTTP integration can land
in a follow-up loop alongside the connection pool's
keep-alive behaviour.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from mmar_carl import (
    LLMClientBase,
    MCPServerConfig,
    MCPSessionPool,
    ReasoningContext,
)


# ---------------------------------------------------------------------------
# Fixtures — write a tiny FastMCP server to a tmp file, point a
# ``MCPServerConfig(transport="stdio", command=python, args=[script])``
# at it.
# ---------------------------------------------------------------------------


_SERVER_SCRIPT = textwrap.dedent("""\
    \"\"\"Minimal FastMCP server for CARL live integration tests.

    Exposes one tool (``echo``), one tool with structured args (``add``),
    and one resource (``test://greeting``).
    \"\"\"
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("carl-live-test")


    @mcp.tool()
    def echo(text: str) -> str:
        \"\"\"Return the text unchanged.\"\"\"
        return f"echo:{text}"


    @mcp.tool()
    def add(a: int, b: int) -> int:
        \"\"\"Return a + b.\"\"\"
        return a + b


    @mcp.resource("test://greeting")
    def greeting() -> str:
        \"\"\"A canned greeting resource.\"\"\"
        return "hello from FastMCP"


    if __name__ == "__main__":
        mcp.run("stdio")
    """)


@pytest.fixture
def server_script(tmp_path: Path) -> Path:
    """Write the FastMCP test server to a tmp file and return the path."""
    script = tmp_path / "mcp_test_server.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    return script


@pytest.fixture
def server_config(server_script: Path) -> MCPServerConfig:
    """A stdio MCPServerConfig that spawns the test server."""
    return MCPServerConfig(
        server_name="carl-live-test",
        transport="stdio",
        command=sys.executable,
        args=[str(server_script)],
    )


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(
        self, prompt: str, retries: int = 3,
    ) -> str:
        return "ok"


@pytest.fixture
def ctx() -> ReasoningContext:
    return ReasoningContext(outer_context="x", api=_StubLLM())


# ---------------------------------------------------------------------------
# Tool discovery + invocation over real stdio
# ---------------------------------------------------------------------------


@pytest.mark.mcp_live
@pytest.mark.asyncio
async def test_register_and_invoke_tools(
    ctx: ReasoningContext, server_config: MCPServerConfig,
) -> None:
    """End-to-end: spawn server → list_tools → register wrappers →
    invoke a wrapped tool → assert the right value came back over
    the wire."""
    registered = await ctx.register_mcp_tools(server_config)
    # Server exposes echo + add — both should show up under the default
    # `mcp:<server_name>/` prefix.
    assert sorted(registered) == [
        "mcp:carl-live-test/add",
        "mcp:carl-live-test/echo",
    ]

    # Invoke the echo tool through the registered wrapper.
    echo_tool = ctx.get_tool("mcp:carl-live-test/echo")
    assert echo_tool is not None
    result = await echo_tool(text="hello")
    # FastMCP returns a list of content blocks per the spec; the wrapper
    # surfaces them as `result.content`. We just assert the echoed
    # string appears.
    assert "echo:hello" in str(result)

    # Invoke `add` with structured args.
    add_tool = ctx.get_tool("mcp:carl-live-test/add")
    assert add_tool is not None
    result = await add_tool(a=2, b=3)
    assert "5" in str(result)


# ---------------------------------------------------------------------------
# Resource discovery over real stdio
# ---------------------------------------------------------------------------


@pytest.mark.mcp_live
@pytest.mark.asyncio
async def test_list_resources(
    ctx: ReasoningContext, server_config: MCPServerConfig,
) -> None:
    resources = await ctx.list_mcp_resources(server_config)
    uris = [getattr(r, "uri", None) for r in resources]
    # `test://greeting` from the @mcp.resource decorator. Pydantic v2
    # AnyUrl normalises this — match by string substring.
    assert any("greeting" in str(u) for u in uris if u is not None)


# ---------------------------------------------------------------------------
# MCPSessionPool — verify the pool reuses a real session across calls
# ---------------------------------------------------------------------------


@pytest.mark.mcp_live
@pytest.mark.asyncio
async def test_pool_reuses_session_across_calls(
    ctx: ReasoningContext, server_config: MCPServerConfig,
) -> None:
    """Inside an active pool, two consecutive MCP calls share a single
    ``ClientSession`` — the second `acquire` returns the cached one.

    We can't easily count subprocess spawns from the test, but we can
    verify the pool's stats report one active session after two
    `list_resources` calls.
    """
    async with ctx.mcp_pool() as pool:
        await ctx.list_mcp_resources(server_config)
        stats_after_first = pool.stats()
        await ctx.list_mcp_resources(server_config)
        stats_after_second = pool.stats()

    assert stats_after_first["active_sessions"] == 1
    assert stats_after_second["active_sessions"] == 1
    # Pool closed; sessions cleared on exit.
    assert not pool.is_open


@pytest.mark.mcp_live
@pytest.mark.asyncio
async def test_pool_explicit_acquire(server_config: MCPServerConfig) -> None:
    """Direct ``MCPSessionPool`` usage (no ReasoningContext wiring)."""
    async with MCPSessionPool() as pool:
        session = await pool.acquire(server_config)
        # The session is already `initialize()`'d; calling list_tools
        # against it should work.
        result = await session.list_tools()
        tool_names = sorted(getattr(t, "name", "") for t in result.tools)
        assert tool_names == ["add", "echo"]

        # Second acquire returns the same session (cache hit).
        session2 = await pool.acquire(server_config)
        assert session2 is session


# ---------------------------------------------------------------------------
# Sanity — confirm the marker is wired correctly
# ---------------------------------------------------------------------------


def test_mcp_live_marker_present() -> None:
    """The mcp_live marker must be declared in pyproject.toml.

    This test is *not* marked mcp_live — it runs in the default suite
    so a missing marker declaration surfaces immediately rather than
    only when someone tries the opt-in suite.
    """
    pyproject = (
        Path(__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "mcp_live:" in pyproject, (
        "The `mcp_live` marker must be declared in [tool.pytest.ini_options] "
        "markers in pyproject.toml."
    )
    assert "not mcp_live" in pyproject, (
        "The default pytest addopts must deselect `mcp_live` so the "
        "hermetic test suite stays fast."
    )


def test_marker_skips_by_default() -> None:
    """When opted out (the default), an mcp_live test must NOT run.

    This is a sanity check: if a contributor accidentally drops the
    marker from a live test, it'd start running in the hermetic
    suite — and likely fail or spawn a real subprocess. We exercise
    this by checking the marker actually applies to our tests.
    """
    # Read this file and assert every test that hits the real server
    # is decorated with the marker.
    text = Path(__file__).read_text(encoding="utf-8")
    # Each test that uses `server_config` fixture must be marked.
    test_funcs = [
        "test_register_and_invoke_tools",
        "test_list_resources",
        "test_pool_reuses_session_across_calls",
        "test_pool_explicit_acquire",
    ]
    for fn in test_funcs:
        # Find the function declaration and walk a few lines up; the
        # @pytest.mark.mcp_live decorator must appear before it.
        idx = text.index(f"async def {fn}(")
        # Look back ~200 chars; we expect the marker to be there.
        prefix = text[max(0, idx - 200):idx]
        assert "@pytest.mark.mcp_live" in prefix, (
            f"{fn} hits the live server but lacks the @pytest.mark.mcp_live "
            f"decorator — it would run unexpectedly in the hermetic suite."
        )


# ---------------------------------------------------------------------------
# Diagnostic: skip the live suite gracefully when the script isn't
# runnable (e.g. an older mcp SDK without FastMCP).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _skip_if_fastmcp_missing(request: pytest.FixtureRequest) -> None:
    """If FastMCP isn't importable on the host running the live suite,
    skip every mcp_live test with a clear reason."""
    if "mcp_live" not in request.node.keywords:
        return
    try:
        import mcp.server.fastmcp  # noqa: F401
    except ImportError:
        pytest.skip(
            "FastMCP unavailable — `pip install mcp` provides it. "
            "Live MCP integration tests are opt-in via `pytest -m mcp_live`."
        )


# ---------------------------------------------------------------------------
# Diagnostic — sanity that the suite is reachable
# ---------------------------------------------------------------------------


def test_live_suite_discovery_works() -> None:
    """A no-op test in the default suite that ensures the module
    imports cleanly. If the imports above fail, every live test
    silently drops — this canary surfaces that immediately.
    """
    # The mere act of this module being imported by pytest validates
    # that the test fixtures + scripts above don't have a syntax error.
    assert os.path.isfile(__file__)
