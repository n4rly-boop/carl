#!/usr/bin/env python3
"""
Example: Supervisor / hierarchical routing in CARL.

Demonstrates ``SupervisorStepDescription`` — a step where a supervisor LLM
picks one of N named specialist sub-chains and runs it. The chosen
sub-chain's result is merged back into the parent's history / memory.

Covered patterns:
  1. Basic routing — supervisor picks the right specialist for each task.
  2. Custom routing prompt and per-task LLM behaviour.
  3. Fallback agent — when the supervisor's reply doesn't match any agent.
  4. Failure isolation — `propagate_failure=False` keeps the parent step
     successful even when the sub-chain fails.
  5. Multi-task batch dispatch — calling the same supervisor across a list of
     inputs.

No LLM or API key required.  Every example uses a tiny heuristic mock client
that picks an agent by counting agent-name occurrences in the task text.

Usage:
    python examples/supervisor_routing_example.py
    PYTHONPATH=$(pwd) python examples/supervisor_routing_example.py
"""

import asyncio
from typing import Callable

from mmar_carl import (
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    SupervisorStepConfig,
    SupervisorStepDescription,
    ToolStepConfig,
    ToolStepDescription,
)
from examples.utils import format_status


# ---------------------------------------------------------------------------
# Mock LLM client — no API key needed
# ---------------------------------------------------------------------------


class HeuristicRoutingLLM(LLMClientBase):
    """Picks the agent whose name appears most often in the task portion of the prompt.

    Returns ``"UNKNOWN"`` if no agent name appears — this exercises the
    ``fallback_agent`` path. The matcher only looks at text after the literal
    ``"Task:"`` marker so the agent-name *list* in the prompt template
    doesn't bias the count.
    """

    def __init__(self, agent_names: list[str]) -> None:
        self.agent_names = agent_names

    async def get_response(self, prompt: str) -> str:
        body = prompt.lower().split("task:", 1)[-1]
        best: str | None = None
        best_n = 0
        for name in self.agent_names:
            n = body.count(name.lower())
            if n > best_n:
                best, best_n = name, n
        return best or "UNKNOWN"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ---------------------------------------------------------------------------
# Specialist sub-chains
# ---------------------------------------------------------------------------


def make_specialist(label: str) -> tuple[ReasoningChain, Callable]:
    """Build a single-step specialist sub-chain.

    Each specialist exposes a tool registered under ``<label>_tool`` that
    echoes its label plus the routed task (read from ``$memory.input.task``,
    which the SupervisorStepExecutor seeds automatically).
    """

    def specialist_tool(task: str) -> str:
        return f"[{label}] handled: {task[:60]}"

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title=f"{label} specialist",
                config=ToolStepConfig(
                    tool_name=f"{label}_tool",
                    parameters=[],
                    input_mapping={"task": "$memory.input.task"},
                ),
            ),
        ],
        max_workers=1,
    )
    return chain, specialist_tool


def make_supervisor_chain(
    agents: dict[str, ReasoningChain],
    *,
    fallback_agent: str | None = None,
    routing_prompt: str | None = None,
    propagate_failure: bool = True,
    output_memory_key: str = "result",
) -> ReasoningChain:
    return ReasoningChain(
        steps=[
            SupervisorStepDescription(
                number=1,
                title="Route to specialist",
                agents=agents,
                config=SupervisorStepConfig(
                    routing_prompt=routing_prompt
                    or (
                        "Choose ONE specialist from [{agents}] for the task below.\n"
                        "Reply with just the specialist name.\n\nTask: {task}"
                    ),
                    fallback_agent=fallback_agent,
                    output_memory_key=output_memory_key,
                    propagate_failure=propagate_failure,
                ),
            ),
        ],
        max_workers=1,
    )


def make_context(
    task: str,
    agents: dict[str, ReasoningChain],
    specialist_tools: dict[str, Callable],
    *,
    llm: LLMClientBase | None = None,
) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context=task,
        api=llm or HeuristicRoutingLLM(list(agents.keys())),
        language=Language.ENGLISH,
    )
    for label, tool in specialist_tools.items():
        ctx.register_tool(f"{label}_tool", tool)
    return ctx


# ---------------------------------------------------------------------------
# Example 1 — Basic routing across multiple tasks
# ---------------------------------------------------------------------------


async def example_basic_routing() -> None:
    print("\n" + "=" * 70)
    print("Example 1: Basic routing across multiple tasks")
    print("=" * 70)

    agents: dict[str, ReasoningChain] = {}
    tools: dict[str, Callable] = {}
    for label in ("pdf", "search", "code"):
        chain, tool = make_specialist(label)
        agents[label] = chain
        tools[label] = tool

    sup_chain = make_supervisor_chain(agents)
    tasks = [
        "Extract text from the attached pdf file.",
        "Search the web for recent climate research.",
        "Refactor this code to use a list comprehension.",
    ]

    for task in tasks:
        ctx = make_context(task, agents, tools)
        result = await sup_chain.execute_async(ctx)
        sr = result.step_results[0]
        chosen = sr.result_data["agent_selected"]
        out = ctx.memory.get("supervisor", {}).get("result", "<no output>")
        print(f"  {format_status(sr.success)}  task={task[:38]!r:40}  → {chosen:<7}  "
              f"output={out!r}")


# ---------------------------------------------------------------------------
# Example 2 — Custom routing prompt
# ---------------------------------------------------------------------------


async def example_custom_routing_prompt() -> None:
    print("\n" + "=" * 70)
    print("Example 2: Custom routing prompt template")
    print("=" * 70)

    agents: dict[str, ReasoningChain] = {}
    tools: dict[str, Callable] = {}
    for label in ("billing", "support"):
        chain, tool = make_specialist(label)
        agents[label] = chain
        tools[label] = tool

    sup_chain = make_supervisor_chain(
        agents,
        routing_prompt=(
            "You are a customer-service triage bot. The user query is:\n"
            "    Task: {task}\n"
            "Pick exactly one team: {agents}. Output the team name only."
        ),
    )

    queries = [
        "My billing statement looks wrong.",
        "I can't log in — please help support me.",
    ]
    for q in queries:
        ctx = make_context(q, agents, tools)
        result = await sup_chain.execute_async(ctx)
        sr = result.step_results[0]
        print(f"  {format_status(sr.success)}  query={q[:38]!r:40}  → "
              f"{sr.result_data['agent_selected']}")


# ---------------------------------------------------------------------------
# Example 3 — Fallback agent
# ---------------------------------------------------------------------------


async def example_fallback_agent() -> None:
    print("\n" + "=" * 70)
    print("Example 3: Fallback agent for unrecognized tasks")
    print("=" * 70)

    agents: dict[str, ReasoningChain] = {}
    tools: dict[str, Callable] = {}
    for label in ("pdf", "search", "code"):
        chain, tool = make_specialist(label)
        agents[label] = chain
        tools[label] = tool

    sup_chain = make_supervisor_chain(agents, fallback_agent="search")
    unrecognized_task = "Random task with no clear specialist."
    ctx = make_context(unrecognized_task, agents, tools)
    result = await sup_chain.execute_async(ctx)
    sr = result.step_results[0]
    print(f"  {format_status(sr.success)}  task={unrecognized_task!r}")
    print(f"    routing_reply: {sr.result_data['routing_reply']!r}")
    print(f"    fell back to: {sr.result_data['agent_selected']!r}")
    assert sr.result_data["agent_selected"] == "search"


# ---------------------------------------------------------------------------
# Example 4 — Failure isolation (propagate_failure=False)
# ---------------------------------------------------------------------------


async def example_failure_isolation() -> None:
    print("\n" + "=" * 70)
    print("Example 4: Failure isolation with propagate_failure=False")
    print("=" * 70)

    # A specialist whose tool always raises — simulates a downstream outage.
    def broken_specialist_tool(task: str) -> str:
        raise RuntimeError(f"simulated specialist outage while handling: {task[:40]}")

    broken_chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="broken specialist",
                config=ToolStepConfig(
                    tool_name="broken_tool",
                    parameters=[],
                    input_mapping={"task": "$memory.input.task"},
                    timeout=2.0,
                ),
            ),
        ],
        max_workers=1,
    )

    agents: dict[str, ReasoningChain] = {"pdf": broken_chain}
    sup_chain_strict = make_supervisor_chain(agents, propagate_failure=True)
    sup_chain_lenient = make_supervisor_chain(agents, propagate_failure=False)

    print("\n  -- propagate_failure=True (default) --")
    ctx_strict = ReasoningContext(
        outer_context="Read the pdf file.",
        api=HeuristicRoutingLLM(["pdf"]),
        language=Language.ENGLISH,
    )
    ctx_strict.register_tool("broken_tool", broken_specialist_tool)
    result_strict = await sup_chain_strict.execute_async(ctx_strict)
    sr_strict = result_strict.step_results[0]
    print(f"     supervisor success: {sr_strict.success}  (sub-chain failure propagates)")
    print(f"     error: {sr_strict.error_message!r}")
    assert sr_strict.success is False

    print("\n  -- propagate_failure=False --")
    ctx_lenient = ReasoningContext(
        outer_context="Read the pdf file.",
        api=HeuristicRoutingLLM(["pdf"]),
        language=Language.ENGLISH,
    )
    ctx_lenient.register_tool("broken_tool", broken_specialist_tool)
    result_lenient = await sup_chain_lenient.execute_async(ctx_lenient)
    sr_lenient = result_lenient.step_results[0]
    print(f"     supervisor success: {sr_lenient.success}  (failure swallowed)")
    print(f"     agent_selected:     {sr_lenient.result_data['agent_selected']!r}")
    print(f"     sub_chain_success:  {sr_lenient.result_data['sub_chain_success']}")
    assert sr_lenient.success is True
    assert sr_lenient.result_data["sub_chain_success"] is False


# ---------------------------------------------------------------------------
# Example 5 — Multi-task batch dispatch
# ---------------------------------------------------------------------------


async def example_batch_dispatch() -> None:
    print("\n" + "=" * 70)
    print("Example 5: Batch — re-use one supervisor across many tasks")
    print("=" * 70)

    agents: dict[str, ReasoningChain] = {}
    tools: dict[str, Callable] = {}
    for label in ("pdf", "search", "code", "math"):
        chain, tool = make_specialist(label)
        agents[label] = chain
        tools[label] = tool

    sup_chain = make_supervisor_chain(agents, fallback_agent="search")
    batch = [
        "Solve the math problem 17 * 23.",
        "Search for the latest CARL release notes.",
        "Extract tables from this pdf.",
        "Improve the readability of this code.",
        "Pure gibberish, doesn't fit.",
    ]
    # Process in parallel — each invocation gets its own context so memory
    # writes from siblings don't collide.
    coros = []
    for task in batch:
        ctx = make_context(task, agents, tools)
        coros.append(sup_chain.execute_async(ctx))
    results = await asyncio.gather(*coros)

    for task, result in zip(batch, results):
        sr = result.step_results[0]
        chosen = sr.result_data["agent_selected"]
        print(f"  {format_status(sr.success)}  task={task[:38]!r:40}  → {chosen}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await example_basic_routing()
    await example_custom_routing_prompt()
    await example_fallback_agent()
    await example_failure_isolation()
    await example_batch_dispatch()
    print("\n" + "=" * 70)
    print("All supervisor-routing examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
