"""Tests for the improved diagnostic when ``ReasoningChain.execute()`` is
called from inside a running event loop.

Before this fix the warning said only "may cause issues". The new
diagnostic:
1. Includes the exact replacement snippet (``await chain.execute_async``).
2. Explains the concrete failure mode (nested loop + httpx deadlocks).
3. Adds a Jupyter-specific hint when IPython is importable.
4. Supports ``strict_async=True`` to raise ``RuntimeError`` instead of warn.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="noop", config=ToolStepConfig(tool_name="noop")
            ),
        ],
    )


def _ctx() -> ReasoningContext:
    ctx = ReasoningContext(outer_context="x", api=None, model="default")
    ctx.register_tool("noop", lambda: "ok")
    return ctx


# ---------------------------------------------------------------------------
# Diagnostic message content
# ---------------------------------------------------------------------------


class TestDiagnosticMessage:
    def test_message_includes_exact_replacement_snippet(self) -> None:
        msg = ReasoningChain._build_async_context_diagnostic()
        assert "await chain.execute_async(context)" in msg

    def test_message_explains_thread_pool_danger(self) -> None:
        msg = ReasoningChain._build_async_context_diagnostic()
        assert "thread-pool" in msg
        # "nested" + "event loop" appear close together (with * emphasis chars)
        assert "nested" in msg
        assert "event loop" in msg
        assert "deadlock" in msg

    def test_message_mentions_strict_async_opt_in(self) -> None:
        msg = ReasoningChain._build_async_context_diagnostic()
        assert "strict_async" in msg

    def test_message_mentions_jupyter_when_ipython_imported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a stub IPython into sys.modules
        monkeypatch.setitem(sys.modules, "IPython", object())
        msg = ReasoningChain._build_async_context_diagnostic()
        assert "Jupyter" in msg or "IPython" in msg
        assert "autoawait" in msg

    def test_message_omits_jupyter_hint_when_ipython_not_imported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ensure IPython is not imported
        monkeypatch.delitem(sys.modules, "IPython", raising=False)
        msg = ReasoningChain._build_async_context_diagnostic()
        assert "autoawait" not in msg


# ---------------------------------------------------------------------------
# Default behaviour (warning + thread-pool fallback) — backward compat
# ---------------------------------------------------------------------------


class TestDefaultWarningBehaviour:
    def test_sync_call_from_sync_context_no_warning(self) -> None:
        chain = _make_chain()
        ctx = _ctx()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = chain.execute(ctx)
        assert result.success is True
        async_warnings = [w for w in caught if "execute_async" in str(w.message)]
        assert async_warnings == []

    @pytest.mark.asyncio
    async def test_sync_call_from_async_context_warns_with_new_diagnostic(self) -> None:
        chain = _make_chain()
        ctx = _ctx()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = chain.execute(ctx)
        # Chain still runs to completion (backward compat).
        assert result.success is True
        async_warnings = [
            w
            for w in caught
            if "running event loop" in str(w.message)
        ]
        assert len(async_warnings) == 1
        msg = str(async_warnings[0].message)
        assert "await chain.execute_async(context)" in msg
        assert "thread-pool" in msg
        assert "strict_async" in msg


# ---------------------------------------------------------------------------
# strict_async=True path
# ---------------------------------------------------------------------------


class TestStrictAsyncMode:
    def test_strict_async_no_op_when_no_running_loop(self) -> None:
        chain = _make_chain()
        ctx = _ctx()
        # No running loop → strict_async should not raise.
        result = chain.execute(ctx, strict_async=True)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_strict_async_raises_in_async_context(self) -> None:
        chain = _make_chain()
        ctx = _ctx()
        with pytest.raises(RuntimeError, match="execute_async"):
            chain.execute(ctx, strict_async=True)

    @pytest.mark.asyncio
    async def test_strict_async_error_message_is_actionable(self) -> None:
        chain = _make_chain()
        ctx = _ctx()
        try:
            chain.execute(ctx, strict_async=True)
        except RuntimeError as e:
            msg = str(e)
            assert "await chain.execute_async(context)" in msg
            assert "running event loop" in msg
        else:
            pytest.fail("Expected RuntimeError")


# ---------------------------------------------------------------------------
# Regression — async API itself is unaffected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_async_works_normally_when_called_from_async() -> None:
    """Sanity: the recommended replacement just works."""
    chain = _make_chain()
    ctx = _ctx()
    result = await chain.execute_async(ctx)
    assert result.success is True
