"""
Tests for ``ReasoningContext.register_tool_pipeline``.

A tool pipeline is a single registered tool whose body sequentially invokes
other registered tools. Each step in the pipeline pulls its arguments either
from the pipeline call's kwargs (``$input`` / ``$input.<path>``) or from the
previous step's output (``$prev_output`` / ``$prev_output.<path>``). The final
step's return value is the pipeline's return value.
"""

import asyncio
from typing import Any

import pytest

from mmar_carl import (
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)


class _NoopLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return ""

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return ""


def _ctx() -> ReasoningContext:
    return ReasoningContext(outer_context="", api=_NoopLLM())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_register_tool_pipeline_rejects_empty_steps() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="at least one step"):
        ctx.register_tool_pipeline("p", steps=[])


def test_register_tool_pipeline_rejects_malformed_step() -> None:
    ctx = _ctx()
    with pytest.raises(ValueError, match="must be a"):
        ctx.register_tool_pipeline("p", steps=[("just_a_name",)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be a"):
        ctx.register_tool_pipeline("p", steps=[(123, {})])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be a"):
        ctx.register_tool_pipeline("p", steps=[("ok", "not-a-dict")])  # type: ignore[list-item]


def test_pipeline_raises_when_referenced_tool_missing() -> None:
    ctx = _ctx()
    ctx.register_tool_pipeline("p", steps=[("missing", {})])
    runner = ctx.get_tool("p")
    assert runner is not None
    with pytest.raises(ValueError, match="references unregistered tool 'missing'"):
        asyncio.run(runner())


# --------------------------------------------------------------------------- #
# Reference resolution
# --------------------------------------------------------------------------- #


def test_pipeline_passes_input_kwargs_to_first_step() -> None:
    ctx = _ctx()
    captured: dict[str, Any] = {}

    def upper(text: str) -> str:
        captured["got"] = text
        return text.upper()

    ctx.register_tool("upper", upper)
    ctx.register_tool_pipeline("pipe", steps=[("upper", {"text": "$input.text"})])

    runner = ctx.get_tool("pipe")
    assert runner is not None
    out = asyncio.run(runner(text="hello"))
    assert out == "HELLO"
    assert captured["got"] == "hello"


def test_pipeline_threads_prev_output_between_steps() -> None:
    ctx = _ctx()

    def search(query: str) -> dict[str, Any]:
        return {"results": [f"hit-for:{query}"], "count": 1}

    def summarize(text: str) -> str:
        return f"summary({text})"

    ctx.register_tool("search", search)
    ctx.register_tool("summarize", summarize)
    ctx.register_tool_pipeline(
        "search_and_summarize",
        steps=[
            ("search", {"query": "$input.q"}),
            ("summarize", {"text": "$prev_output.results"}),
        ],
    )

    out = asyncio.run(ctx.get_tool("search_and_summarize")(q="cats"))
    assert out == "summary(['hit-for:cats'])"


def test_pipeline_dollar_input_passes_full_kwargs_dict() -> None:
    ctx = _ctx()

    def echo(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    ctx.register_tool("echo", echo)
    ctx.register_tool_pipeline("p", steps=[("echo", {"payload": "$input"})])
    out = asyncio.run(ctx.get_tool("p")(a=1, b=2))
    assert out == {"a": 1, "b": 2}


def test_pipeline_dollar_prev_output_no_path_returns_raw_value() -> None:
    ctx = _ctx()
    ctx.register_tool("a", lambda: 42)
    ctx.register_tool("b", lambda value: value * 2)
    ctx.register_tool_pipeline(
        "p", steps=[("a", {}), ("b", {"value": "$prev_output"})]
    )
    assert asyncio.run(ctx.get_tool("p")()) == 84


def test_pipeline_literal_values_pass_through() -> None:
    ctx = _ctx()

    def adder(a: int, b: int) -> int:
        return a + b

    ctx.register_tool("add", adder)
    ctx.register_tool_pipeline(
        "p",
        steps=[("add", {"a": "$input.x", "b": 100})],  # 100 is a literal, not a ref
    )
    assert asyncio.run(ctx.get_tool("p")(x=5)) == 105


def test_pipeline_missing_dotted_path_yields_none() -> None:
    ctx = _ctx()

    def show(value: Any) -> Any:
        return value

    ctx.register_tool("show", show)
    ctx.register_tool_pipeline("p", steps=[("show", {"value": "$input.a.b.c"})])
    # 'a' isn't present in kwargs → resolves to None (lenient)
    assert asyncio.run(ctx.get_tool("p")()) is None


def test_pipeline_first_step_prev_output_is_none() -> None:
    ctx = _ctx()
    captured: dict[str, Any] = {}

    def show(value: Any) -> Any:
        captured["v"] = value
        return value

    ctx.register_tool("show", show)
    ctx.register_tool_pipeline("p", steps=[("show", {"value": "$prev_output"})])
    assert asyncio.run(ctx.get_tool("p")()) is None
    assert captured["v"] is None


# --------------------------------------------------------------------------- #
# Async tool composition
# --------------------------------------------------------------------------- #


def test_pipeline_calls_async_tool() -> None:
    ctx = _ctx()

    async def async_double(n: int) -> int:
        await asyncio.sleep(0)
        return n * 2

    ctx.register_tool("dbl", async_double)
    ctx.register_tool_pipeline("p", steps=[("dbl", {"n": "$input.n"})])
    assert asyncio.run(ctx.get_tool("p")(n=7)) == 14


def test_pipeline_mixes_sync_and_async_steps() -> None:
    ctx = _ctx()

    def sync_inc(n: int) -> int:
        return n + 1

    async def async_negate(n: int) -> int:
        await asyncio.sleep(0)
        return -n

    ctx.register_tool("inc", sync_inc)
    ctx.register_tool("neg", async_negate)
    ctx.register_tool_pipeline(
        "p",
        steps=[("inc", {"n": "$input.n"}), ("neg", {"n": "$prev_output"})],
    )
    assert asyncio.run(ctx.get_tool("p")(n=5)) == -6


# --------------------------------------------------------------------------- #
# Behavioural / housekeeping
# --------------------------------------------------------------------------- #


def test_pipeline_is_marked_with_metadata() -> None:
    ctx = _ctx()
    ctx.register_tool("t", lambda: 1)
    ctx.register_tool_pipeline("p", steps=[("t", {})])
    runner = ctx.get_tool("p")
    assert runner is not None
    assert getattr(runner, "is_pipeline", False) is True
    assert getattr(runner, "pipeline_steps", None) == (("t", {}),)


def test_pipeline_spec_is_snapshot_not_live_reference() -> None:
    """Mutating the input ``steps`` after registration must not change the pipeline."""
    ctx = _ctx()
    ctx.register_tool("t", lambda x: x)
    steps: list[tuple[str, dict[str, Any]]] = [("t", {"x": "$input.v"})]
    ctx.register_tool_pipeline("p", steps=steps)
    steps.append(("t", {"x": 999}))  # mutate after registration
    steps[0][1]["x"] = "$input.evil"  # mutate inner dict
    out = asyncio.run(ctx.get_tool("p")(v="ok"))
    assert out == "ok"


def test_pipeline_accepts_tags() -> None:
    ctx = _ctx()
    ctx.register_tool("base", lambda: 1)
    ctx.register_tool_pipeline("p", steps=[("base", {})], tags=["composite"])
    assert ctx.get_tool_tags("p") == {"composite"}
    assert ctx.list_tools(tags=["composite"]) == ["p"]


def test_pipeline_propagates_underlying_tool_exception() -> None:
    ctx = _ctx()

    def boom(**_: Any) -> Any:
        raise RuntimeError("kaboom")

    ctx.register_tool("boom", boom)
    ctx.register_tool_pipeline("p", steps=[("boom", {})])
    with pytest.raises(RuntimeError, match="kaboom"):
        asyncio.run(ctx.get_tool("p")())


# --------------------------------------------------------------------------- #
# Integration: pipeline callable from a ToolStep
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pipeline_works_from_a_toolstep() -> None:
    ctx = _ctx()

    def fetch(query: str) -> dict[str, Any]:
        return {"text": f"raw-text-about:{query}", "score": 0.9}

    def summarize(text: str) -> str:
        return f"SUMMARY[{text}]"

    ctx.register_tool("fetch", fetch, tags=["information"])
    ctx.register_tool("summarize", summarize, tags=["information"])
    ctx.register_tool_pipeline(
        "fetch_and_summarize",
        steps=[
            ("fetch", {"query": "$input.query"}),
            ("summarize", {"text": "$prev_output.text"}),
        ],
        tags=["information"],
    )

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="run pipeline",
                config=ToolStepConfig(
                    tool_name="fetch_and_summarize",
                    parameters=[],
                    input_mapping={"query": "'\"climate\"'"},
                    timeout=5.0,
                    allowed_tool_tags=["information"],
                ),
            )
        ],
        max_workers=1,
    )

    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data == 'SUMMARY[raw-text-about:"climate"]'
