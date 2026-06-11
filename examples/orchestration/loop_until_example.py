#!/usr/bin/env python3
"""
Example: Loop-Until Pattern in CARL.

Demonstrates iterative reasoning chains that repeat until a stopping condition
is satisfied — without hard-coding a fixed number of steps.

Covered patterns:
  1. Research loop — search → evaluate → search again until quality is high enough
  2. Retry-until-success loop — attempt an operation, back off, retry until it works
  3. Manual LoopConfig API — low-level control over loop-back and condition

No LLM or API key required.  All examples use ToolSteps + a minimal mock client.

Usage:
    python examples/loop_until_example.py
    PYTHONPATH=$(pwd) python examples/loop_until_example.py
"""

import asyncio

from mmar_carl import (
    ChainBuilder,
    Language,
    LLMClientBase,
    LoopConfig,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription
from examples.utils import print_execution_summary


# ---------------------------------------------------------------------------
# Minimal mock LLM client (no API key needed)
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    """No-op client — all work is done by tool steps."""

    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ============================================================================
# Example 1: Research loop
# ============================================================================
# Goal: keep searching until the result contains at least 3 facts.
# Each iteration either finds more facts or reports a quality score.
# The loop exits once $memory.research.done is truthy.
# ============================================================================


def make_research_state():
    """Return shared mutable state for the research tool functions."""
    return {"iteration": 0, "facts": []}


def build_research_tools(state: dict):
    """Build the tool callables for Example 1."""

    FACT_BATCHES = [
        ["Penguins cannot fly."],
        ["Penguins live in the Southern Hemisphere.", "Some species reach 1.1 m tall."],
        ["Emperor penguins are the largest species.", "They breed during Antarctic winter."],
    ]

    def search_facts() -> str:
        """Search for facts — returns more facts each call (simulating real retrieval)."""
        batch_index = min(state["iteration"], len(FACT_BATCHES) - 1)
        new_facts = FACT_BATCHES[batch_index]
        state["facts"].extend(new_facts)
        state["iteration"] += 1
        return f"Found {len(new_facts)} fact(s) this round. Total: {len(state['facts'])}"

    def evaluate_research() -> str:
        """Evaluate research quality; writes 'done' flag to memory when threshold met."""
        count = len(state["facts"])
        quality = "sufficient" if count >= 3 else "insufficient"
        return f"Quality: {quality} ({count}/3 facts required)"

    def check_done() -> bool:
        """Return True once we have 3 or more facts — this stops the loop."""
        return len(state["facts"]) >= 3

    return search_facts, evaluate_research, check_done


async def example_research_loop():
    """
    Example 1: Loop until research quality threshold is met.

    Flow per iteration:
        search_facts  ──►  evaluate_research  ──►  check_done
             ▲                                           │
             └───────────────────────────────── (loop while not done)
    """
    print("\n" + "=" * 60)
    print("Example 1: Research Loop (loop until quality threshold)")
    print("=" * 60)

    state = make_research_state()
    search_facts, evaluate_research, check_done = build_research_tools(state)

    # Build the loop body as a list of ToolStepDescriptions.
    # add_until_loop() renumbers them and attaches the LoopConfig automatically.
    body = [
        ToolStepDescription(
            number=0,  # will be renumbered
            title="Search Facts",
            config=ToolStepConfig(tool_name="search_facts", input_mapping={}),
        ),
        ToolStepDescription(
            number=0,  # will be renumbered
            title="Evaluate Research",
            config=ToolStepConfig(tool_name="evaluate_research", input_mapping={}),
        ),
        ToolStepDescription(
            number=0,  # will be renumbered
            title="Check Done",
            config=ToolStepConfig(tool_name="check_done", input_mapping={}),
        ),
    ]

    # add_until_loop: repeats body until check_done result is truthy
    chain = (
        ChainBuilder()
        .add_until_loop(
            body_steps=body,
            condition_key="$metadata.step_3",  # step 3 = check_done result
            max_iterations=10,
        )
        .build()
    )

    print(f"\nChain steps: {[s.title for s in chain.steps]}")
    print("Loop exits when: check_done() returns True (≥ 3 facts gathered)")

    ctx = ReasoningContext(
        outer_context="Research topic: penguins",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("search_facts", search_facts)
    ctx.register_tool("evaluate_research", evaluate_research)
    ctx.register_tool("check_done", check_done)

    result = await chain.execute_async(ctx)

    print_execution_summary(result, label="\nExecution")
    loop_history = ctx.metadata.get("loop_iteration_history", {})
    iterations = loop_history.get(3, 0) if loop_history else 0
    print(f"Loop iterations: {iterations}")
    print(f"Facts gathered: {state['facts']}")
    print("\nHistory:")
    for entry in result.history:
        print(f"  {entry.strip()}")


# ============================================================================
# Example 2: Retry-until-success loop
# ============================================================================
# Goal: attempt a flaky network call; retry until it succeeds.
# Each attempt has an increasing chance of success.
# ============================================================================


def build_retry_tools():
    """Return tools for Example 2 with shared attempt counter."""
    state = {"attempt": 0}

    def try_network_call() -> str:
        """Simulate a flaky call that succeeds on the 3rd attempt."""
        state["attempt"] += 1
        attempt = state["attempt"]
        if attempt < 3:
            return f"FAILED: connection timeout (attempt {attempt})"
        return f"SUCCESS: data received (attempt {attempt})"

    def is_success() -> bool:
        """Return True if the last network call succeeded."""
        # reads from the parent step's metadata via tool
        # For simplicity, we track state directly
        return state["attempt"] >= 3

    return try_network_call, is_success, state


async def example_retry_loop():
    """
    Example 2: Retry-until-success loop.

    Flow per iteration:
        try_network_call ──► is_success
               ▲                 │
               └──────── (loop while not success)
    """
    print("\n" + "=" * 60)
    print("Example 2: Retry-Until-Success Loop")
    print("=" * 60)

    try_network_call, is_success, state = build_retry_tools()

    body = [
        ToolStepDescription(
            number=0,
            title="Try Network Call",
            config=ToolStepConfig(tool_name="try_network_call", input_mapping={}),
        ),
        ToolStepDescription(
            number=0,
            title="Check Success",
            config=ToolStepConfig(tool_name="is_success", input_mapping={}),
        ),
    ]

    chain = (
        ChainBuilder()
        .add_until_loop(
            body_steps=body,
            condition_key="$metadata.step_2",  # step 2 = is_success result
            max_iterations=5,
        )
        .build()
    )

    print(f"\nChain steps: {[s.title for s in chain.steps]}")
    print("Max retries: 5 (budget guard)")
    print("Success on: attempt 3")

    ctx = ReasoningContext(
        outer_context="",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("try_network_call", try_network_call)
    ctx.register_tool("is_success", is_success)

    result = await chain.execute_async(ctx)

    print_execution_summary(result, label="\nExecution")
    print(f"Total attempts: {state['attempt']}")
    print("\nHistory:")
    for entry in result.history:
        print(f"  {entry.strip()}")


# ============================================================================
# Example 3: Manual LoopConfig API (fine-grained control)
# ============================================================================
# Shows the lower-level API for when ChainBuilder helpers aren't flexible enough.
# A counter increments on each iteration; the loop exits when it reaches 3.
# ============================================================================


async def example_manual_loop_config():
    """
    Example 3: Low-level LoopConfig API.

    Uses loop_back_to and loop_config fields directly on step descriptions.
    Counter increments from 0; loop exits at count == 3.
    """
    print("\n" + "=" * 60)
    print("Example 3: Manual LoopConfig API")
    print("=" * 60)

    counter = {"value": 0}

    def increment() -> int:
        counter["value"] += 1
        return counter["value"]

    def not_at_limit() -> bool:
        """Return True while counter < 3 (loop should continue)."""
        return counter["value"] < 3

    # Build steps manually:
    # Step 1: increment counter
    # Step 2: check if we should continue (loop_back_to=1, loop_config)
    steps = [
        ToolStepDescription(
            number=1,
            title="Increment Counter",
            config=ToolStepConfig(tool_name="increment", input_mapping={}),
        ),
        ToolStepDescription(
            number=2,
            title="Check Limit",
            config=ToolStepConfig(tool_name="not_at_limit", input_mapping={}),
            dependencies=[1],
            loop_back_to=1,  # jump back to step 1 when condition holds
            loop_config=LoopConfig(
                condition_key="$metadata.step_2",  # step 2 = not_at_limit()
                max_iterations=10,
            ),
        ),
    ]

    chain = ReasoningChain(steps=steps)

    print(f"\nSteps: {[(s.number, s.title) for s in chain.steps]}")
    print(f"loop_back_to: step {steps[1].loop_back_to}")
    print(f"condition_key: {steps[1].loop_config.condition_key}")
    print("Exits when: not_at_limit() returns False (counter == 3)")

    ctx = ReasoningContext(
        outer_context="",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("increment", increment)
    ctx.register_tool("not_at_limit", not_at_limit)

    result = await chain.execute_async(ctx)

    print_execution_summary(result, label="\nExecution")
    print(f"Final counter value: {counter['value']}")
    loop_history = ctx.metadata.get("loop_iteration_history", {})
    iterations = loop_history.get(2, 0)
    print(f"Loop iterations: {iterations}")
    print("\nHistory:")
    for entry in result.history:
        print(f"  {entry.strip()}")


# ============================================================================
# Example 4: While-loop (complement of until-loop)
# ============================================================================
# add_while_loop continues AS LONG AS the condition is truthy (inverse of until).
# Useful when you have a "should_continue" flag rather than a "done" flag.
# ============================================================================


async def example_while_loop():
    """
    Example 4: add_while_loop (continue while condition is truthy).

    A queue drainer: pop items from a queue while items remain.
    Loop exits when the queue is empty.
    """
    print("\n" + "=" * 60)
    print("Example 4: While-Loop (drain a queue)")
    print("=" * 60)

    queue = ["task-A", "task-B", "task-C"]
    processed = []

    def process_next() -> str:
        """Pop and process the next item from the queue."""
        if queue:
            item = queue.pop(0)
            processed.append(item)
            return f"Processed: {item}"
        return "Queue empty"

    def has_more() -> bool:
        """Return True while items remain in the queue."""
        return len(queue) > 0

    body = [
        ToolStepDescription(
            number=0,
            title="Process Item",
            config=ToolStepConfig(tool_name="process_next", input_mapping={}),
        ),
        ToolStepDescription(
            number=0,
            title="Has More Items",
            config=ToolStepConfig(tool_name="has_more", input_mapping={}),
        ),
    ]

    chain = (
        ChainBuilder()
        .add_while_loop(
            body_steps=body,
            condition_key="$metadata.step_2",  # step 2 = has_more result
            max_iterations=10,
        )
        .build()
    )

    print(f"\nInitial queue: {['task-A', 'task-B', 'task-C']}")
    print(f"Chain steps: {[s.title for s in chain.steps]}")

    ctx = ReasoningContext(
        outer_context="",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("process_next", process_next)
    ctx.register_tool("has_more", has_more)

    result = await chain.execute_async(ctx)

    print_execution_summary(result, label="\nExecution")
    print(f"Processed items: {processed}")
    print("\nHistory:")
    for entry in result.history:
        print(f"  {entry.strip()}")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all loop-until examples."""
    print("CARL Loop-Until Pattern Examples")
    print("=" * 60)
    print("\nThese examples demonstrate iterative reasoning chains.")
    print("No LLM or API key required.\n")

    await example_research_loop()
    await example_retry_loop()
    await example_manual_loop_config()
    await example_while_loop()

    print("\n" + "=" * 60)
    print("Loop-until examples completed!")
    print("=" * 60)
    print("\nKey APIs:")
    print("  ChainBuilder.add_until_loop(body, condition_key, max_iterations)")
    print("  ChainBuilder.add_while_loop(body, condition_key, max_iterations)")
    print("  StepDescriptionBase(loop_back_to=N, loop_config=LoopConfig(...))")
    print("\nCondition keys:")
    print("  $metadata.step_N  — result of step N (from tool/transform steps)")
    print("  $memory.ns.key    — a memory value written by a prior step")
    print("  $history[-1]      — last history entry (truthy when non-empty)")
    print("\nBudget guard:")
    print("  max_iterations prevents infinite loops (default: 10)")
    print("  loop_iteration_history tracked in context.metadata")
    print("\nNext steps:")
    print("  See conditions_example.py for conditional branching")
    print("  See tool_steps_example.py for memory and transform steps")


if __name__ == "__main__":
    asyncio.run(main())
