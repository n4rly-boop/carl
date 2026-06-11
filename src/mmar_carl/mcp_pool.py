"""MCP session pool — one ``ClientSession`` per server reused across steps.

Before this change every MCP call (tool dispatch,
``list_tools``, ``list_resources``) opened a fresh transport + session
via ``async with`` blocks, then tore them down — per-call overhead that
adds up when a chain hits the same MCP server N times.

:class:`MCPSessionPool` keeps an :class:`AsyncExitStack` that owns the
transport + session context managers for the lifetime of the pool. The
stack guarantees cleanup happens in the same task that opened the
sessions (asyncio's hard constraint on async generators / async
context managers). Sessions are keyed by ``(server_name, transport,
command-or-url)`` so two configs pointing at the same server share a
session.

Usage
-----

Two ways to opt in:

1. **Explicit context manager** (most predictable lifecycle)::

       async with MCPSessionPool() as pool:
           session = await pool.acquire(server)
           result = await session.call_tool("search", {"q": "x"})

2. **Auto-pooling on the context** (CARE's path)::

       async with context.mcp_pool() as pool:
           # `register_mcp_tools`, `_mcp_call_tool`, `_mcp_list_tools`,
           # and `list_mcp_resources` all reuse pooled sessions for the
           # duration of this `async with` block.
           chain_result = await chain.execute_async(context)

When no pool is active the legacy per-call session path runs unchanged
— pooling is purely opt-in so existing callers don't see a behaviour
change.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Optional

_log = logging.getLogger(__name__)


def _server_key(server: Any) -> tuple[str, str, str]:
    """Identity key for a server config.

    Two ``MCPServerConfig``s that hit the same server (same transport,
    same command-or-url) share a session even if they're different
    Python objects.
    """
    transport = getattr(server, "transport", "")
    target = (
        getattr(server, "command", None)
        or getattr(server, "url", None)
        or ""
    )
    name = getattr(server, "server_name", "")
    return (name, transport, str(target))


class MCPSessionPool:
    """Pool of long-lived MCP ``ClientSession`` instances.

    Lifecycle is managed via :class:`contextlib.AsyncExitStack`. Enter
    the pool with ``async with`` (or call :meth:`__aenter__` and
    :meth:`__aexit__` manually); every session opened during the pool's
    lifetime is exited at ``__aexit__`` time, in the same task that
    opened it.
    """

    def __init__(self) -> None:
        self._stack: Optional[AsyncExitStack] = None
        self._sessions: dict[tuple[str, str, str], Any] = {}
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "MCPSessionPool":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        stack = self._stack
        self._stack = None
        self._sessions.clear()
        if stack is not None:
            await stack.__aexit__(exc_type, exc, tb)

    @property
    def is_open(self) -> bool:
        """Whether the pool is between ``__aenter__`` and ``__aexit__``."""
        return self._stack is not None

    async def acquire(self, server: Any, *, timeout: float = 30.0) -> Any:
        """Get (or open) a pooled session for ``server``.

        Args:
            server: An ``MCPServerConfig`` describing the server.
            timeout: Used both as the transport-level timeout and as
                an upper bound on ``session.initialize()``.

        Returns:
            A :class:`mcp.ClientSession` that has already been
            ``initialize()``'d. The session lives until the pool's
            ``__aexit__`` runs.

        Raises:
            RuntimeError: If the pool isn't open (call inside an
                ``async with`` block or after ``__aenter__``).
        """
        if self._stack is None:
            raise RuntimeError(
                "MCPSessionPool is not open. Use `async with pool:` or "
                "call `__aenter__()` before `acquire()`."
            )

        key = _server_key(server)
        async with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                return cached
            session = await self._open_session(server, timeout=timeout)
            self._sessions[key] = session
            return session

    def stats(self) -> dict[str, Any]:
        """Diagnostic snapshot — useful in tests / CARE's TUI footer."""
        return {
            "open": self.is_open,
            "active_sessions": len(self._sessions),
            "server_keys": sorted(
                f"{n}|{t}|{u}" for (n, t, u) in self._sessions
            ),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _open_session(self, server: Any, *, timeout: float) -> Any:
        """Open the transport CM + ClientSession CM, both entered into
        ``self._stack`` so cleanup is deterministic.
        """
        assert self._stack is not None  # narrowed by `acquire`
        from mcp import ClientSession

        transport = getattr(server, "transport", "stdio")

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=getattr(server, "command", None) or "",
                args=list(getattr(server, "args", []) or []),
            )
            read, write = await self._stack.enter_async_context(
                stdio_client(params),
            )
        elif transport == "sse":
            from mcp.client.sse import sse_client

            url = getattr(server, "url", None)
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='sse'."
                )
            headers = dict(getattr(server, "headers", None) or {})
            read, write = await self._stack.enter_async_context(
                sse_client(
                    url=url,
                    headers=headers or None,
                    timeout=timeout,
                ),
            )
        elif transport == "http":
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            url = getattr(server, "url", None)
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='http'."
                )
            headers = dict(getattr(server, "headers", None) or {})
            http_client = httpx.AsyncClient(headers=headers, timeout=timeout)
            # Note: streamable_http_client yields 3 values; we keep only the
            # read/write halves. The session-id getter is unused by the
            # current pool — we'd capture it if CARE needs MCP resumption.
            read, write, _get_session_id = (
                await self._stack.enter_async_context(
                    streamable_http_client(url=url, http_client=http_client),
                )
            )
        else:
            raise NotImplementedError(
                f"MCP transport {transport!r} is not supported. "
                "Use one of: 'stdio', 'http', 'sse'."
            )

        session = await self._stack.enter_async_context(
            ClientSession(read, write),
        )
        await asyncio.wait_for(session.initialize(), timeout=timeout)
        return session


__all__ = ["MCPSessionPool"]
