"""
Tests for ``ChainBuilder.from_description(task, llm_client, ...)``.

A meta-agent that asks an LLM to plan a chain in JSON form and parses that
plan through :py:meth:`ReasoningChain.from_dict`, inheriting all of the
existing validation (cycles, dependency references, reference syntax
warnings) automatically.
"""

import json

import pytest

from mmar_carl import (
    ChainBuilder,
    LLMClientBase,
    LLMStepDescription,
    ReasoningChain,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _ScriptedLLM(LLMClientBase):
    """Returns a canned text reply; records every prompt seen."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class _GetResponseOnlyLLM:
    """Bare async object exposing only ``get_response`` — used to confirm
    the meta-agent works against clients without ``get_response_with_retries``."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        self.calls += 1
        return self.reply


def _valid_plan(n: int = 2) -> str:
    steps = []
    for i in range(1, n + 1):
        step = {
            "number": i,
            "title": f"step {i}",
            "step_type": "llm",
            "aim": f"aim {i}",
        }
        if i > 1:
            step["dependencies"] = [i - 1]
        steps.append(step)
    return json.dumps({"steps": steps})


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_minimal_plan_produces_valid_chain() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(2))
    chain = await ChainBuilder.from_description("Analyze a doc", llm)
    assert isinstance(chain, ReasoningChain)
    assert len(chain.steps) == 2
    assert chain.steps[0].number == 1
    assert chain.steps[1].number == 2
    assert chain.steps[1].dependencies == [1]


@pytest.mark.asyncio
async def test_tool_step_in_plan() -> None:
    plan = json.dumps({
        "steps": [
            {"number": 1, "title": "search", "step_type": "tool",
             "step_config": {"tool_name": "web_search",
                             "input_mapping": {"query": "$outer_context"}}},
        ]
    })
    llm = _ScriptedLLM(reply=plan)
    chain = await ChainBuilder.from_description("x", llm, available_tools=["web_search"])
    assert len(chain.steps) == 1
    assert isinstance(chain.steps[0], ToolStepDescription)
    assert chain.steps[0].config.tool_name == "web_search"


@pytest.mark.asyncio
async def test_mixed_step_types_in_plan() -> None:
    plan = json.dumps({
        "steps": [
            {"number": 1, "title": "extract", "step_type": "llm", "aim": "extract"},
            {"number": 2, "title": "store",
             "step_type": "memory", "dependencies": [1],
             "step_config": {"operation": "write", "memory_key": "extracted",
                             "value_source": "$history[-1]", "namespace": "default"}},
        ]
    })
    llm = _ScriptedLLM(reply=plan)
    chain = await ChainBuilder.from_description("x", llm)
    assert len(chain.steps) == 2
    assert isinstance(chain.steps[0], LLMStepDescription)


# --------------------------------------------------------------------------- #
# Code-fence stripping
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handles_code_fenced_reply() -> None:
    llm = _ScriptedLLM(reply=f"```json\n{_valid_plan(1)}\n```")
    chain = await ChainBuilder.from_description("x", llm)
    assert len(chain.steps) == 1


@pytest.mark.asyncio
async def test_handles_code_fence_without_language_tag() -> None:
    llm = _ScriptedLLM(reply=f"```\n{_valid_plan(1)}\n```")
    chain = await ChainBuilder.from_description("x", llm)
    assert len(chain.steps) == 1


@pytest.mark.asyncio
async def test_handles_whitespace_around_reply() -> None:
    llm = _ScriptedLLM(reply=f"   \n\n{_valid_plan(1)}\n\n   ")
    chain = await ChainBuilder.from_description("x", llm)
    assert len(chain.steps) == 1


# --------------------------------------------------------------------------- #
# Planning prompt content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_prompt_includes_task_text() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(1))
    task = "Extract every email address from the input"
    await ChainBuilder.from_description(task, llm)
    assert task in llm.prompts[0]


@pytest.mark.asyncio
async def test_prompt_includes_available_tools() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(1))
    await ChainBuilder.from_description(
        "x", llm, available_tools=["search", "calculate"]
    )
    prompt = llm.prompts[0]
    assert "search" in prompt
    assert "calculate" in prompt


@pytest.mark.asyncio
async def test_prompt_includes_available_skills() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(1))
    await ChainBuilder.from_description(
        "x", llm, available_skills=["pdf", "web-search"]
    )
    prompt = llm.prompts[0]
    assert "pdf" in prompt
    assert "web-search" in prompt


@pytest.mark.asyncio
async def test_prompt_includes_extra_instructions() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(1))
    await ChainBuilder.from_description(
        "x", llm, extra_instructions="Output must be in JSON only."
    )
    assert "JSON only" in llm.prompts[0]


@pytest.mark.asyncio
async def test_prompt_mentions_max_steps() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(1))
    await ChainBuilder.from_description("x", llm, max_steps=7)
    assert "at most 7" in llm.prompts[0]


# --------------------------------------------------------------------------- #
# Client interface flexibility
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_works_with_bare_get_response_client() -> None:
    """The meta-agent falls back to ``get_response`` if ``get_response_with_retries`` is absent."""
    llm = _GetResponseOnlyLLM(reply=_valid_plan(1))
    chain = await ChainBuilder.from_description("x", llm)
    assert len(chain.steps) == 1
    assert llm.calls == 1


# --------------------------------------------------------------------------- #
# Validation errors
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_invalid_json_reply_raises() -> None:
    llm = _ScriptedLLM(reply="Sure, here's the plan: not actually JSON")
    with pytest.raises(ValueError, match="not valid JSON"):
        await ChainBuilder.from_description("x", llm)


@pytest.mark.asyncio
async def test_missing_steps_key_raises() -> None:
    llm = _ScriptedLLM(reply='{"plan": [{"number": 1}]}')  # wrong top-level key
    with pytest.raises(ValueError, match="missing top-level 'steps' key"):
        await ChainBuilder.from_description("x", llm)


@pytest.mark.asyncio
async def test_empty_steps_list_raises() -> None:
    llm = _ScriptedLLM(reply='{"steps": []}')
    with pytest.raises(ValueError, match="non-empty list"):
        await ChainBuilder.from_description("x", llm)


@pytest.mark.asyncio
async def test_non_list_steps_raises() -> None:
    llm = _ScriptedLLM(reply='{"steps": "not a list"}')
    with pytest.raises(ValueError, match="non-empty list"):
        await ChainBuilder.from_description("x", llm)


@pytest.mark.asyncio
async def test_too_many_steps_raises() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(8))
    with pytest.raises(ValueError, match="exceeding max_steps=3"):
        await ChainBuilder.from_description("x", llm, max_steps=3)


@pytest.mark.asyncio
async def test_invalid_dependency_propagates_from_chain_validation() -> None:
    """A plan with a dangling dependency fails downstream chain validation."""
    plan = json.dumps({
        "steps": [
            {"number": 1, "title": "x", "step_type": "llm", "aim": "x",
             "dependencies": [99]},  # 99 doesn't exist
        ]
    })
    llm = _ScriptedLLM(reply=plan)
    with pytest.raises(ValueError, match="non-existent step 99"):
        await ChainBuilder.from_description("x", llm)


@pytest.mark.asyncio
async def test_cycle_in_plan_propagates_from_chain_validation() -> None:
    plan = json.dumps({
        "steps": [
            {"number": 1, "title": "a", "step_type": "llm", "aim": "a", "dependencies": [2]},
            {"number": 2, "title": "b", "step_type": "llm", "aim": "b", "dependencies": [1]},
        ]
    })
    llm = _ScriptedLLM(reply=plan)
    with pytest.raises(ValueError, match=r"[Cc]ycle"):
        await ChainBuilder.from_description("x", llm)


# --------------------------------------------------------------------------- #
# max_workers + metadata pass-through
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_max_workers_passes_through() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(2))
    chain = await ChainBuilder.from_description("x", llm, max_workers=5)
    assert chain.max_workers == 5


@pytest.mark.asyncio
async def test_default_max_workers_is_auto() -> None:
    llm = _ScriptedLLM(reply=_valid_plan(2))
    chain = await ChainBuilder.from_description("x", llm)
    assert chain.max_workers == "auto"


@pytest.mark.asyncio
async def test_generated_chain_records_origin_in_metadata() -> None:
    """The original task description is stashed in chain.metadata for provenance."""
    llm = _ScriptedLLM(reply=_valid_plan(1))
    task = "Build a fact-checking pipeline"
    chain = await ChainBuilder.from_description(task, llm)
    assert chain.metadata.get("generated_from_description") == task


@pytest.mark.asyncio
async def test_long_task_description_truncated_in_metadata() -> None:
    """Task descriptions longer than ~200 chars are truncated for metadata storage."""
    llm = _ScriptedLLM(reply=_valid_plan(1))
    long_task = "x" * 500
    chain = await ChainBuilder.from_description(long_task, llm)
    stored = chain.metadata.get("generated_from_description", "")
    assert len(stored) <= 200


# --------------------------------------------------------------------------- #
# End-to-end: generated chain actually executes
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_generated_chain_executes_end_to_end() -> None:
    """After generation, the chain runs successfully against an LLM context."""
    from mmar_carl import ReasoningContext

    class _RunnerLLM(LLMClientBase):
        def __init__(self, plan: str) -> None:
            self.plan = plan
            self.is_planner = True
            self.step_responses: list[str] = []

        async def get_response(self, prompt: str) -> str:
            if self.is_planner:
                self.is_planner = False
                return self.plan
            return f"step-output-{len(self.step_responses)}"

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return await self.get_response(prompt)

    plan = json.dumps({
        "steps": [
            {"number": 1, "title": "a", "step_type": "llm", "aim": "a"},
            {"number": 2, "title": "b", "step_type": "llm", "aim": "b", "dependencies": [1]},
        ]
    })
    llm = _RunnerLLM(plan)
    chain = await ChainBuilder.from_description("Do a thing", llm)
    ctx = ReasoningContext(outer_context="data", api=llm)
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert len(result.step_results) == 2
