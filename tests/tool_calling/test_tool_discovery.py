"""
Tests for runtime tool discovery protocol.

Covers:
- carl_tool decorator
- DictToolSource registration
- CallableToolSource factory
- ModuleToolSource scanning with prefix/tag filters
- output_memory_key write
- Downstream ToolStep using discovered tools
- Chain integration
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_carl import (
    ReasoningChain,
    ReasoningContext,
    carl_tool,
)
from mmar_carl.models import (
    CallableToolSource,
    DictToolSource,
    ModuleToolSource,
    ToolDiscoveryStepConfig,
    ToolDiscoveryStepDescription,
    ToolStepDescription,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.step_executors import ToolDiscoveryStepExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ReasoningContext:
    api = MagicMock()
    api.get_response = AsyncMock(return_value="ok")
    return ReasoningContext(outer_context="", api=api, **kwargs)


def _make_tool_step(step_number: int, tool_name: str, input_args: dict | None = None) -> ToolStepDescription:
    return ToolStepDescription(
        number=step_number,
        title=f"run {tool_name}",
        config=ToolStepConfig(
            tool_name=tool_name,
            input_mapping=input_args or {},
        ),
    )


# ---------------------------------------------------------------------------
# carl_tool decorator
# ---------------------------------------------------------------------------

class TestCarlToolDecorator:
    def test_marks_function(self):
        @carl_tool
        def my_fn():
            pass

        assert getattr(my_fn, "__carl_tool__", False) is True

    def test_no_tags_by_default(self):
        @carl_tool
        def my_fn():
            pass

        assert not hasattr(my_fn, "__carl_tool_tags__")

    def test_tags_attached(self):
        @carl_tool(tags=["search", "external"])
        def my_fn():
            pass

        assert my_fn.__carl_tool_tags__ == ["search", "external"]

    def test_decorator_preserves_callable(self):
        @carl_tool
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_factory_form(self):
        tagged = carl_tool(tags=["info"])

        def raw():
            pass

        result = tagged(raw)
        assert result.__carl_tool__ is True
        assert result.__carl_tool_tags__ == ["info"]


# ---------------------------------------------------------------------------
# DictToolSource
# ---------------------------------------------------------------------------

class TestDictToolSource:
    @pytest.mark.asyncio
    async def test_registers_static_dict(self):
        ctx = _make_context()

        def my_search(query: str) -> str:
            return f"results for {query}"

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=DictToolSource(tools={"my_search": my_search}),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success
        assert "my_search" in result.result_data["tool_names"]
        assert ctx.get_tool("my_search") is my_search

    @pytest.mark.asyncio
    async def test_multiple_tools_registered(self):
        ctx = _make_context()

        def fn_a():
            return "a"

        def fn_b():
            return "b"

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=DictToolSource(tools={"fn_a": fn_a, "fn_b": fn_b}),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success
        assert set(result.result_data["tool_names"]) >= {"fn_a", "fn_b"}


# ---------------------------------------------------------------------------
# CallableToolSource
# ---------------------------------------------------------------------------

class TestCallableToolSource:
    @pytest.mark.asyncio
    async def test_factory_called_once(self):
        ctx = _make_context()

        call_count = [0]

        def factory() -> dict:
            call_count[0] += 1

            def greet(name: str) -> str:
                return f"Hello, {name}"

            return {"greet": greet}

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=CallableToolSource(factory=factory),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success
        assert call_count[0] == 1
        assert "greet" in result.result_data["tool_names"]

    @pytest.mark.asyncio
    async def test_factory_tools_callable(self):
        ctx = _make_context()

        def factory() -> dict:
            def double(x):
                return x * 2

            return {"double": double}

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=CallableToolSource(factory=factory),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        await executor.execute(step, ctx)

        tool = ctx.get_tool("double")
        assert tool is not None
        assert tool(5) == 10


# ---------------------------------------------------------------------------
# ModuleToolSource
# ---------------------------------------------------------------------------

def _make_test_module() -> types.ModuleType:
    """Build a synthetic module with a mix of functions."""
    mod = types.ModuleType("fake_tools")

    @carl_tool
    def search(query: str) -> str:  # marked
        return f"search:{query}"

    @carl_tool(tags=["information"])
    def fetch(url: str) -> str:  # marked + tagged
        return f"fetch:{url}"

    def _private():  # underscore-prefixed → skipped
        return "private"

    def plain():  # not marked
        return "plain"

    mod.search = search
    mod.fetch = fetch
    mod._private = _private
    mod.plain = plain
    return mod


class TestModuleToolSource:
    @pytest.mark.asyncio
    async def test_discovers_public_callables(self, monkeypatch):
        """ModuleToolSource registers all public callables; private (underscore) names are skipped."""
        mod = _make_test_module()
        monkeypatch.setattr("importlib.import_module", lambda name: mod)

        ctx = _make_context()
        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="fake_tools"),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        assert result.success
        names = set(result.result_data["tool_names"])
        # All public callables (including unmarked 'plain') are registered
        assert "search" in names
        assert "fetch" in names
        assert "plain" in names
        # Underscore-prefixed names are always skipped
        assert "_private" not in names

    @pytest.mark.asyncio
    async def test_name_prefix_filter(self, monkeypatch):
        mod = types.ModuleType("mod")

        @carl_tool
        def search_web(q):
            return q

        @carl_tool
        def search_docs(q):
            return q

        @carl_tool
        def calc(x):
            return x

        mod.search_web = search_web
        mod.search_docs = search_docs
        mod.calc = calc
        monkeypatch.setattr("importlib.import_module", lambda name: mod)

        ctx = _make_context()
        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="mod", name_prefix="search_"),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        names = set(result.result_data["tool_names"])
        assert "search_web" in names
        assert "search_docs" in names
        assert "calc" not in names

    @pytest.mark.asyncio
    async def test_tag_filter(self, monkeypatch):
        mod = types.ModuleType("mod")

        @carl_tool(tags=["information"])
        def fetch(url):
            return url

        @carl_tool(tags=["compute"])
        def calc(x):
            return x

        @carl_tool
        def search(q):  # no tags
            return q

        mod.fetch = fetch
        mod.calc = calc
        mod.search = search
        monkeypatch.setattr("importlib.import_module", lambda name: mod)

        ctx = _make_context()
        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="mod", tag="information"),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        result = await executor.execute(step, ctx)

        names = set(result.result_data["tool_names"])
        assert "fetch" in names
        assert "calc" not in names
        assert "search" not in names


# ---------------------------------------------------------------------------
# output_memory_key
# ---------------------------------------------------------------------------

class TestOutputMemoryKey:
    @pytest.mark.asyncio
    async def test_writes_tool_names_to_memory(self):
        ctx = _make_context()

        def my_fn():
            return "x"

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=DictToolSource(tools={"my_fn": my_fn}),
                output_memory_key="discovered",
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        await executor.execute(step, ctx)

        mem = ctx.memory.get("tools", {})
        assert "discovered" in mem
        assert "my_fn" in mem["discovered"]

    @pytest.mark.asyncio
    async def test_no_memory_write_without_key(self):
        ctx = _make_context()

        def my_fn():
            return "x"

        step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover",
            config=ToolDiscoveryStepConfig(
                source=DictToolSource(tools={"my_fn": my_fn}),
            ),
        )

        executor = ToolDiscoveryStepExecutor()
        await executor.execute(step, ctx)

        # No tools key should be written
        assert "tools" not in ctx.memory or "discovered" not in ctx.memory.get("tools", {})


# ---------------------------------------------------------------------------
# Chain integration: ToolDiscovery → ToolStep
# ---------------------------------------------------------------------------

class TestChainIntegration:
    @pytest.mark.asyncio
    async def test_downstream_tool_step_uses_discovered_tool(self):
        """A ToolDiscoveryStep registers a tool; a ToolStep uses it next."""
        captured = []

        def greet(name: str) -> str:
            captured.append(name)
            return f"Hello, {name}!"

        # Build chain
        discovery_step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover greet tool",
            config=ToolDiscoveryStepConfig(
                source=DictToolSource(tools={"greet": greet}),
            ),
        )
        tool_step = ToolStepDescription(
            number=2,
            title="Call greet",
            dependencies=[1],
            config=ToolStepConfig(
                tool_name="greet",
                input_mapping={"name": '"World"'},
            ),
        )

        chain = ReasoningChain(steps=[discovery_step, tool_step])
        ctx = _make_context()
        result = await chain.execute_async(ctx)

        assert result.success, result.get_failed_steps()
        assert "World" in captured
        # The tool step result should contain the greet output
        step_result = result.get_step_result(2)
        assert step_result is not None
        assert "Hello" in step_result.result

    @pytest.mark.asyncio
    async def test_chain_with_callable_factory(self):
        """CallableToolSource factory is invoked at runtime during the chain."""

        def factory() -> dict:
            def uppercase(text: str) -> str:
                return text.upper()

            return {"uppercase": uppercase}

        discovery_step = ToolDiscoveryStepDescription(
            number=1,
            title="Discover uppercase",
            config=ToolDiscoveryStepConfig(
                source=CallableToolSource(factory=factory),
            ),
        )
        tool_step = ToolStepDescription(
            number=2,
            title="Call uppercase",
            dependencies=[1],
            config=ToolStepConfig(
                tool_name="uppercase",
                input_mapping={"text": '"hello"'},
            ),
        )

        chain = ReasoningChain(steps=[discovery_step, tool_step])
        ctx = _make_context()
        result = await chain.execute_async(ctx)

        assert result.success, result.get_failed_steps()
        step_result = result.get_step_result(2)
        assert step_result is not None
        assert step_result.result == "HELLO"
