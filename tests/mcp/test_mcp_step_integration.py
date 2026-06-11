"""
End-to-end integration tests for ``MCPStepDescription`` inside full chains.

The existing ``tests/test_mcp_transport.py`` exercises the transport layer
in isolation. This file drives MCP steps through ``chain.execute_async()``
in realistic multi-step DAGs — testing how the MCP step interacts with
other step types (Tool, Memory, LLM, Conditional), context references
(``$history`` / ``$memory`` / ``$metadata.step_N``), parallel execution,
and chain-level concerns (timeout, failure propagation, retry).

All tests mock the MCP SDK (``mcp.ClientSession`` + transport clients) so
no real MCP server is required.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These tests stub the MCP SDK (``mcp.ClientSession`` + transport clients) via
# ``unittest.mock.patch``; without the optional ``mcp`` extra installed the
# patch targets can't be imported. Skip the module cleanly in that case.
pytest.importorskip("mcp", reason="requires the optional `mcp` extra")

from mmar_carl import (
    LLMClientBase,
    LLMStepDescription,
    MCPServerConfig,
    MCPStepConfig,
    MCPStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Mocking helpers (mirrors the patterns in tests/test_mcp_transport.py)
# --------------------------------------------------------------------------- #


def _make_session_mock(
    tool_responses: dict[str, Any] | Callable[[str, dict], Any],
    *,
    record_calls: list[tuple[str, dict]] | None = None,
) -> MagicMock:
    """Create a mock ``mcp.ClientSession`` whose ``call_tool`` dispatches per name.

    Args:
        tool_responses: Either a dict ``{tool_name: payload}`` or a callable
            ``(name, args) -> payload`` for dynamic responses.
        record_calls: Optional list to append ``(tool_name, arguments)`` for assertions.
    """
    async def fake_call_tool(name: str, arguments: dict) -> Any:
        if record_calls is not None:
            record_calls.append((name, dict(arguments)))
        result = MagicMock()
        if callable(tool_responses):
            result.content = tool_responses(name, arguments)
        else:
            result.content = tool_responses.get(name, "no-such-tool")
        return result

    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = fake_call_tool
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session


def _make_sse_client_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "llm-ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "llm-ok"


def _patched_sse(session_mock: MagicMock):
    """Convenience: return the patch context managers for SSE transport."""
    return [
        patch("mcp.ClientSession", return_value=session_mock),
        patch("mcp.client.sse.sse_client", return_value=_make_sse_client_cm()),
    ]


def _enter_all(patches: list) -> list:
    return [p.start() for p in patches]


def _exit_all(patches: list) -> None:
    for p in reversed(patches):
        p.stop()


# --------------------------------------------------------------------------- #
# Single-MCP-step chain
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_step_runs_inside_chain_via_execute_async() -> None:
    session = _make_session_mock({"search": "MCP result for the query"})
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1,
                    title="search MCP",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="test", transport="sse", url="http://x/sse"),
                        tool_name="search",
                        arguments={"q": "climate"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="topic", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert result.success
    assert result.step_results[0].result == "MCP result for the query"
    assert result.step_results[0].step_type.value == "mcp"


# --------------------------------------------------------------------------- #
# Argument resolution from chain context
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_argument_mapping_pulls_from_memory_namespace() -> None:
    """``argument_mapping`` should resolve ``$memory.ns.key`` against context memory."""
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock({"lookup": "found"}, record_calls=recorded)
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1,
                    title="mcp lookup",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="lookup",
                        argument_mapping={"user_id": "$memory.input.user"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.memory_write("user", "alice", namespace="input")
        await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert recorded == [("lookup", {"user_id": "alice"})]


@pytest.mark.asyncio
async def test_mcp_argument_mapping_reads_outer_context() -> None:
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock({"echo": "ok"}, record_calls=recorded)
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="echo",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="echo",
                        argument_mapping={"prompt": "$outer_context"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="hello world", api=_StubLLM())
        await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert recorded[0][1] == {"prompt": "hello world"}


@pytest.mark.asyncio
async def test_mcp_step_static_arguments_passed_through() -> None:
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock({"do_thing": "ok"}, record_calls=recorded)
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="static-args",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="do_thing",
                        arguments={"mode": "fast", "depth": 3},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert recorded == [("do_thing", {"mode": "fast", "depth": 3})]


@pytest.mark.asyncio
async def test_mcp_step_argument_mapping_overrides_static_argument() -> None:
    """When the same key appears in both ``arguments`` and ``argument_mapping``,
    the mapped (dynamic) value wins."""
    recorded: list[tuple[str, dict]] = []
    session = _make_session_mock({"go": "ok"}, record_calls=recorded)
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="override",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="go",
                        arguments={"mode": "static"},
                        argument_mapping={"mode": "$memory.cfg.mode"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.memory_write("mode", "dynamic", namespace="cfg")
        await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert recorded[0][1] == {"mode": "dynamic"}


# --------------------------------------------------------------------------- #
# Downstream visibility — MCP output flows to following steps
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_result_flows_through_history_to_next_tool() -> None:
    """A Tool step reading ``$history[-1]`` should see the MCP result text."""
    session = _make_session_mock({"fetch": "fresh data from mcp"})
    patches = _patched_sse(session)
    _enter_all(patches)
    captured: dict[str, str] = {}

    def capture(value: str) -> str:
        captured["got"] = value
        return value

    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="mcp",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="fetch",
                    ),
                ),
                ToolStepDescription(
                    number=2, title="capture",
                    dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="capture",
                        parameters=[],
                        input_mapping={"value": "$history[-1]"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.register_tool("capture", capture)
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    assert "fresh data from mcp" in captured["got"]


@pytest.mark.asyncio
async def test_mcp_result_data_flows_through_metadata_step_n() -> None:
    """``result_data`` (structured payload) flows via ``$metadata.step_1``."""
    session = _make_session_mock({"query": {"hits": 7, "items": ["a", "b"]}})
    patches = _patched_sse(session)
    _enter_all(patches)
    captured: dict[str, Any] = {}

    def capture(payload: Any) -> str:
        captured["payload"] = payload
        return "ok"

    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="query",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="query",
                    ),
                ),
                ToolStepDescription(
                    number=2, title="downstream",
                    dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="cap", parameters=[],
                        input_mapping={"payload": "$metadata.step_1"},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.register_tool("cap", capture)
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    assert captured["payload"] == {"hits": 7, "items": ["a", "b"]}


@pytest.mark.asyncio
async def test_mcp_step_result_persisted_via_memory_step() -> None:
    """An MCP step result can be persisted to memory by a downstream MemoryStep."""
    session = _make_session_mock({"do": "MCP output"})
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="mcp",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="do",
                    ),
                ),
                MemoryStepDescription(
                    number=2, title="store",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="mcp_result",
                        value_source="$metadata.step_1",
                        namespace="output",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    assert ctx.memory["output"]["mcp_result"] == "MCP output"


# --------------------------------------------------------------------------- #
# Parallel siblings — two MCP calls run concurrently
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_two_parallel_mcp_steps_both_execute() -> None:
    recorded: list[tuple[str, dict]] = []

    def dispatch(name: str, args: dict) -> str:
        return f"{name}-result"

    session = _make_session_mock(dispatch, record_calls=recorded)
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="mcp-a",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="tool_a",
                    ),
                ),
                MCPStepDescription(
                    number=2, title="mcp-b",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="tool_b",
                    ),
                ),
            ],
            max_workers=2,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    called_tools = sorted(name for name, _args in recorded)
    assert called_tools == ["tool_a", "tool_b"]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_session_call_failure_marks_step_failed() -> None:
    """A raised exception inside the MCP session call becomes ``step.error_message``."""
    async def boom(name: str, arguments: dict) -> Any:
        raise RuntimeError("mcp server error 500")

    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = boom
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="boom",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="explode",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    sr = result.step_results[0]
    assert not sr.success
    assert "mcp server error" in sr.error_message.lower()


@pytest.mark.asyncio
async def test_mcp_step_with_missing_url_fails_within_chain() -> None:
    """Validation errors raised by the transport propagate to the step result."""
    chain = ReasoningChain(
        steps=[
            MCPStepDescription(
                number=1, title="no-url",
                config=MCPStepConfig(
                    server=MCPServerConfig(server_name="t", transport="sse"),  # url=None
                    tool_name="x",
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
async def test_mcp_failure_does_not_halt_downstream_dependent_steps() -> None:
    """CARL's DAG considers a step "executed" even on failure, so downstream
    steps still run (and may surface secondary failures). This test documents
    that semantic — downstream code that needs to short-circuit on upstream
    failure should check its inputs explicitly."""
    async def boom(name: str, arguments: dict) -> Any:
        raise RuntimeError("network down")

    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = boom
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    patches = _patched_sse(session)
    _enter_all(patches)

    downstream_called = {"hit": False}

    def downstream() -> str:
        downstream_called["hit"] = True
        return "ran anyway"

    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="mcp",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="t",
                    ),
                ),
                ToolStepDescription(
                    number=2, title="downstream",
                    dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="d", parameters=[], input_mapping={},
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        ctx.register_tool("d", downstream)
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    # Chain-level success is False because step 1 failed
    assert not result.success
    # Step 1 marked failed, step 2 still ran (DAG considers it executed)
    sr1, sr2 = result.step_results
    assert not sr1.success
    assert sr2.success
    assert downstream_called["hit"] is True


# --------------------------------------------------------------------------- #
# MCP step that returns a list payload
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_list_payload_serialised_to_json_string_in_history() -> None:
    """Non-string MCP payloads are JSON-serialised for ``history``/``result``."""
    session = _make_session_mock({"list_things": [{"id": 1}, {"id": 2}]})
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="list",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="list_things",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    sr = result.step_results[0]
    # result string is JSON-encoded; result_data preserves the native list
    assert sr.result_data == [{"id": 1}, {"id": 2}]
    assert '"id": 1' in sr.result or '"id":1' in sr.result


# --------------------------------------------------------------------------- #
# Multi-step DAG with MCP middle
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chain_with_llm_then_mcp_then_memory_executes_end_to_end() -> None:
    """LLM → MCP → MemoryWrite with dependency chain end-to-end."""
    session = _make_session_mock({"enrich": "enriched-output"})
    patches = _patched_sse(session)
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="prompt", aim="prep"),
                MCPStepDescription(
                    number=2, title="enrich", dependencies=[1],
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                        tool_name="enrich",
                        argument_mapping={"prior": "$history[-1]"},
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="store", dependencies=[2],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="final",
                        value_source="$history[-1]",
                        namespace="output",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="topic", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    final_text = ctx.memory["output"]["final"]
    assert "enriched-output" in final_text


# --------------------------------------------------------------------------- #
# Chain serialisation includes MCP step config
# --------------------------------------------------------------------------- #


def test_chain_to_dict_includes_mcp_step_config() -> None:
    """Constructing a chain with an MCP step serialises faithfully."""
    chain = ReasoningChain(
        steps=[
            MCPStepDescription(
                number=1, title="mcp",
                config=MCPStepConfig(
                    server=MCPServerConfig(server_name="t", transport="sse", url="http://x/sse"),
                    tool_name="search",
                    arguments={"q": "x"},
                    argument_mapping={"y": "$outer_context"},
                ),
            ),
        ],
        max_workers=1,
    )
    data = chain.to_dict()
    step_data = data["steps"][0]
    assert step_data["step_type"] == "mcp"
    assert step_data["step_config"]["server"]["server_name"] == "t"
    assert step_data["step_config"]["tool_name"] == "search"
    assert step_data["step_config"]["arguments"] == {"q": "x"}
    assert step_data["step_config"]["argument_mapping"] == {"y": "$outer_context"}


# --------------------------------------------------------------------------- #
# Multiple MCP servers in one chain
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mcp_steps_against_two_different_servers_record_urls() -> None:
    """Two MCP steps pointed at different SSE URLs — both transport calls fire."""
    sse_urls_called: list[str] = []

    def fake_sse(url, headers=None, timeout=30.0, **kwargs):
        sse_urls_called.append(url)
        return _make_sse_client_cm()

    session = _make_session_mock(lambda n, a: f"reply-from-{n}")
    patches = [
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.sse.sse_client", side_effect=fake_sse),
    ]
    _enter_all(patches)
    try:
        chain = ReasoningChain(
            steps=[
                MCPStepDescription(
                    number=1, title="a",
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="alpha", transport="sse",
                                               url="http://alpha/sse"),
                        tool_name="t",
                    ),
                ),
                MCPStepDescription(
                    number=2, title="b", dependencies=[1],
                    config=MCPStepConfig(
                        server=MCPServerConfig(server_name="beta", transport="sse",
                                               url="http://beta/sse"),
                        tool_name="t",
                    ),
                ),
            ],
            max_workers=1,
        )
        ctx = ReasoningContext(outer_context="x", api=_StubLLM())
        result = await chain.execute_async(ctx)
    finally:
        _exit_all(patches)
    assert all(sr.success for sr in result.step_results)
    assert sse_urls_called == ["http://alpha/sse", "http://beta/sse"]
