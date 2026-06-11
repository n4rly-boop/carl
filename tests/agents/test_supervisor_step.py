"""
Tests for ``SupervisorStepDescription``.

A supervisor step asks an LLM to pick one of N registered ``agents`` (named
``ReasoningChain`` instances) and runs the chosen sub-chain with an isolated
context derived from the parent. Coverage matrix:

- Routing reply matching (exact / whole-word / substring) and case-insensitivity.
- Fallback agent behaviour when the reply doesn't match any registered agent.
- Failure when no agent matches and no fallback is configured.
- Empty-agents validation.
- Input mapping seeded into sub-chain memory + automatic ``input.task`` /
  ``input.supervisor_agent`` keys.
- Output memory key written back to parent on success only.
- Sub-chain failure propagation respecting ``propagate_failure``.
- Tool registry inheritance (with tags).
- ``timeout`` enforcement for long-running sub-chains.
- Full end-to-end chain execution with multiple specialist chains.
"""

import asyncio

import pytest

from mmar_carl import (
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    SupervisorStepConfig,
    SupervisorStepDescription,
    ToolStepConfig,
    ToolStepDescription,
)


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #


class _CannedReplyLLM(LLMClientBase):
    """Returns a fixed reply for every routing call; records prompts seen."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class _SlowLLM(LLMClientBase):
    """Sleeps before returning — used by the timeout test."""

    def __init__(self, reply: str, delay: float) -> None:
        self.reply = reply
        self.delay = delay

    async def get_response(self, prompt: str) -> str:
        await asyncio.sleep(self.delay)
        return self.reply

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _stub_chain(label: str) -> ReasoningChain:
    """A trivial single-step chain whose final output equals *label*."""
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title=f"{label}-step", aim=label),
        ],
        max_workers=1,
    )


def _build_supervisor(
    *,
    agents: dict[str, ReasoningChain],
    reply: str,
    routing_prompt: str = "Pick one of: {agents}. Task: {task}",
    fallback_agent: str | None = None,
    output_memory_key: str = "specialist_out",
    propagate_failure: bool = True,
    input_mapping: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[ReasoningChain, ReasoningContext]:
    sup = SupervisorStepDescription(
        number=1,
        title="Route to specialist",
        agents=agents,
        config=SupervisorStepConfig(
            routing_prompt=routing_prompt,
            fallback_agent=fallback_agent,
            output_memory_key=output_memory_key,
            propagate_failure=propagate_failure,
            input_mapping=input_mapping or {},
            timeout=timeout,
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="task data", api=_CannedReplyLLM(reply=reply))
    return chain, ctx


# --------------------------------------------------------------------------- #
# Routing reply matching
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_exact_reply_routes_correctly() -> None:
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(agents=agents, reply="pdf")
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "pdf"


@pytest.mark.asyncio
async def test_reply_with_whitespace_and_case_is_normalised() -> None:
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(agents=agents, reply="  SEARCH \n")
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "search"


@pytest.mark.asyncio
async def test_reply_with_whole_word_in_sentence_matches() -> None:
    """LLM gave a verbose reply containing the agent name as a word."""
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(
        agents=agents,
        reply="I think we should use the search specialist for this.",
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "search"


@pytest.mark.asyncio
async def test_reply_substring_fallback_match() -> None:
    """No exact / whole-word match → substring fallback chooses agent.

    Reply ``pdfreader`` doesn't tokenise to ``pdf`` (it's a single word), so the
    matcher must fall through to the substring scan to find ``pdf``.
    """
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(agents=agents, reply="pdfreader")
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "pdf"


@pytest.mark.asyncio
async def test_reply_with_no_match_uses_fallback_agent() -> None:
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(
        agents=agents,
        reply="UNKNOWN gibberish",
        fallback_agent="search",
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "search"


@pytest.mark.asyncio
async def test_no_match_no_fallback_fails_step_with_clear_message() -> None:
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(agents=agents, reply="UNKNOWN gibberish")
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "did not match any agent" in sr.error_message
    assert "no fallback_agent" in sr.error_message


@pytest.mark.asyncio
async def test_no_match_unknown_fallback_treated_as_no_fallback() -> None:
    """A ``fallback_agent`` pointing to a non-existent agent still fails."""
    agents = {"pdf": _stub_chain("pdf")}
    chain, ctx = _build_supervisor(
        agents=agents,
        reply="UNKNOWN",
        fallback_agent="ghost-agent",
    )
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    # Because the fallback_agent name isn't in agents, the executor reports
    # the original "did not match" error, with no fallback being applied.
    assert "did not match any agent" in sr.error_message


# --------------------------------------------------------------------------- #
# Routing prompt templating
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_routing_prompt_receives_task_and_agents_placeholders() -> None:
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(
        agents=agents,
        reply="pdf",
        routing_prompt="AGENTS={agents}|TASK={task}",
    )
    await chain.execute_async(ctx)
    prompt = ctx.api.prompts[0]
    assert "AGENTS=pdf, search" in prompt
    assert "TASK=task data" in prompt


@pytest.mark.asyncio
async def test_task_source_can_pull_from_memory() -> None:
    agents = {"pdf": _stub_chain("pdf")}
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents=agents,
        config=SupervisorStepConfig(
            routing_prompt="Task: {task}",
            task_source="$memory.input.user_query",
            fallback_agent="pdf",
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="ignored", api=_CannedReplyLLM(reply="pdf"))
    ctx.memory_write("user_query", "How many pages in the PDF?", namespace="input")
    await chain.execute_async(ctx)
    prompt = ctx.api.prompts[0]
    assert "Task: How many pages in the PDF?" in prompt


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_empty_agents_fails_step() -> None:
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents={},
        config=SupervisorStepConfig(routing_prompt="x"),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="x", api=_CannedReplyLLM(reply="anything"))
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "agents is empty" in sr.error_message


def test_routing_prompt_min_length_enforced_by_pydantic() -> None:
    with pytest.raises(Exception):
        SupervisorStepConfig(routing_prompt="")


# --------------------------------------------------------------------------- #
# Sub-context wiring
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sub_chain_sees_routed_task_in_input_namespace() -> None:
    """The sub-chain's memory should contain ``input.task`` and ``input.supervisor_agent``."""

    captured: dict[str, str] = {}

    def capture_tool(task: str, agent: str) -> str:
        captured["task"] = task
        captured["agent"] = agent
        return "ok"

    pdf_chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="read inputs",
                config=ToolStepConfig(
                    tool_name="capture",
                    parameters=[],
                    input_mapping={
                        "task": "$memory.input.task",
                        "agent": "$memory.input.supervisor_agent",
                    },
                ),
            )
        ],
        max_workers=1,
    )
    chain, ctx = _build_supervisor(agents={"pdf": pdf_chain}, reply="pdf")
    ctx.register_tool("capture", capture_tool)
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert captured["task"] == "task data"
    assert captured["agent"] == "pdf"


@pytest.mark.asyncio
async def test_input_mapping_seeds_sub_chain_memory() -> None:
    captured: dict[str, str] = {}

    def capture(value: str) -> str:
        captured["got"] = value
        return value

    pdf_chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="cap",
                config=ToolStepConfig(
                    tool_name="capture",
                    parameters=[],
                    input_mapping={"value": "$memory.config.lang"},
                ),
            ),
        ],
        max_workers=1,
    )
    chain, ctx = _build_supervisor(
        agents={"pdf": pdf_chain},
        reply="pdf",
        input_mapping={"config.lang": "'english'"},
    )
    ctx.register_tool("capture", capture)
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert captured["got"] == "english"


# --------------------------------------------------------------------------- #
# Output handling
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_success_writes_output_memory_key() -> None:
    agents = {"pdf": _stub_chain("pdf")}
    chain, ctx = _build_supervisor(agents=agents, reply="pdf")
    await chain.execute_async(ctx)
    written = ctx.memory.get("supervisor", {}).get("specialist_out")
    assert written is not None
    assert written  # non-empty string


@pytest.mark.asyncio
async def test_failure_does_not_write_output_memory() -> None:
    agents = {"pdf": _stub_chain("pdf")}
    chain, ctx = _build_supervisor(agents=agents, reply="UNKNOWN")
    await chain.execute_async(ctx)
    # No write because the step failed before sub-chain ran
    assert "specialist_out" not in ctx.memory.get("supervisor", {})


@pytest.mark.asyncio
async def test_history_entry_names_selected_agent() -> None:
    from mmar_carl import Language
    agents = {"pdf": _stub_chain("pdf"), "search": _stub_chain("search")}
    chain, ctx = _build_supervisor(agents=agents, reply="search")
    ctx.language = Language.ENGLISH
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    final_entry = sr.updated_history[-1]
    assert "SUPERVISOR" in final_entry
    assert "search" in final_entry


# --------------------------------------------------------------------------- #
# Failure propagation
# --------------------------------------------------------------------------- #


class _FailingLLM(LLMClientBase):
    """First call (supervisor routing) succeeds; subsequent calls raise."""

    def __init__(self, route_reply: str) -> None:
        self.route_reply = route_reply
        self.calls = 0

    async def get_response(self, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return self.route_reply
        raise RuntimeError("sub-chain LLM exploded")

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


@pytest.mark.asyncio
async def test_propagate_failure_true_marks_step_failed() -> None:
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents={"pdf": _stub_chain("pdf")},
        config=SupervisorStepConfig(
            routing_prompt="Pick: {agents}",
            propagate_failure=True,
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="x", api=_FailingLLM(route_reply="pdf"), retry_max=1)
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "Sub-chain failed" in sr.error_message


@pytest.mark.asyncio
async def test_propagate_failure_false_step_still_succeeds() -> None:
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents={"pdf": _stub_chain("pdf")},
        config=SupervisorStepConfig(
            routing_prompt="Pick: {agents}",
            propagate_failure=False,
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="x", api=_FailingLLM(route_reply="pdf"), retry_max=1)
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert sr.success
    assert sr.result_data["agent_selected"] == "pdf"
    assert sr.result_data["sub_chain_success"] is False


# --------------------------------------------------------------------------- #
# Tool inheritance
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tools_and_tags_inherited_into_sub_chain() -> None:
    captured: dict[str, str] = {}

    def parent_tool() -> str:
        captured["called"] = "yes"
        return "tool result"

    pdf_chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="use parent tool",
                config=ToolStepConfig(
                    tool_name="parent_tool",
                    parameters=[],
                    input_mapping={},
                    allowed_tool_tags=["information"],
                ),
            )
        ],
        max_workers=1,
    )
    chain, ctx = _build_supervisor(agents={"pdf": pdf_chain}, reply="pdf")
    ctx.register_tool("parent_tool", parent_tool, tags=["information"])
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
    assert captured["called"] == "yes"


# --------------------------------------------------------------------------- #
# Timeout
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sub_chain_timeout_marks_step_failed() -> None:
    slow_chain = ReasoningChain(
        steps=[LLMStepDescription(number=1, title="slow", aim="x")],
        max_workers=1,
    )
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents={"slow": slow_chain},
        config=SupervisorStepConfig(
            routing_prompt="Pick: {agents}",
            timeout=0.05,  # very short
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    # Slow LLM: routing reply comes back instantly, sub-chain LLM sleeps
    class _Mixed(LLMClientBase):
        def __init__(self) -> None:
            self.calls = 0

        async def get_response(self, prompt: str) -> str:
            self.calls += 1
            if self.calls == 1:
                return "slow"
            await asyncio.sleep(1.0)
            return "late"

        async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
            return await self.get_response(prompt)

    ctx = ReasoningContext(outer_context="x", api=_Mixed(), retry_max=1)
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "timed out" in sr.error_message.lower()


# --------------------------------------------------------------------------- #
# Step config overrides
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_supervisor_llm_config_override_does_not_crash() -> None:
    """``config.llm_config`` is honoured when resolving the routing client.

    The mock client doesn't actually use the config but we exercise the path
    to make sure no ``AttributeError`` slips in for atypical clients.
    """
    sup = SupervisorStepDescription(
        number=1,
        title="route",
        agents={"pdf": _stub_chain("pdf")},
        config=SupervisorStepConfig(
            routing_prompt="Pick: {agents}",
            llm_config=LLMStepConfig(temperature=0.0),
        ),
    )
    chain = ReasoningChain(steps=[sup], max_workers=1)
    ctx = ReasoningContext(outer_context="x", api=_CannedReplyLLM(reply="pdf"))
    result = await chain.execute_async(ctx)
    assert result.step_results[0].success
