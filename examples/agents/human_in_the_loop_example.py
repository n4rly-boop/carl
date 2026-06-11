#!/usr/bin/env python3
"""
Example: Human-in-the-Loop Approval Gate.

Demonstrates how to pause chain execution to collect human input — approval,
feedback, or decisions — before continuing.

Covered patterns:
  1. Approval gate — tool drafts content; human approves or rejects before publish
  2. Feedback loop — human provides feedback; tool revises based on input
  3. Timeout fallback — if no human responds within N seconds, auto-continue
  4. No-callback mode — useful in batch/test contexts (fallback used immediately)

No LLM or API key required.  All examples use ToolSteps + a minimal mock
client so you can run them immediately.

Data-flow note:
  - ToolStep results are written to history (access via ``$history[N]``).
  - HumanInputStep results are written to memory (``$memory.human_input.<key>``).
  - Pre-populated input data lives in ``$memory.input.*``.

Usage:
    python examples/human_in_the_loop_example.py
    PYTHONPATH=$(pwd) python examples/human_in_the_loop_example.py
"""

import asyncio
import time

from mmar_carl import (
    HumanInputStepDescription,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import HumanInputStepConfig, ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription
from examples.utils import print_execution_summary


# ---------------------------------------------------------------------------
# Minimal mock LLM client (no API key needed)
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    """No-op client — all work is done by tool steps."""

    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return await self.get_response(prompt)


# ============================================================================
# Example 1: Simple Approval Gate
# ============================================================================
# Chain structure:
#   Step 1 (Tool):       draft_post(topic) → result in history[0]
#   Step 2 (HumanInput): prompt reviewer → decision in memory.human_input.decision
#   Step 3 (Tool):       route_decision(draft=history[0], decision=memory.human_input.decision)
#
# A simulated approver coroutine runs concurrently, providing input after a
# short delay to show the async pause/resume mechanism.
# ============================================================================


def draft_post(topic: str) -> str:
    """Simulate drafting a blog post."""
    return (
        f"Draft post about '{topic}':\n"
        "  • Key insight: AI is transforming software development.\n"
        "  • Main argument: developers who embrace AI tools will outperform those who don't.\n"
        "  • Call to action: Start with small automation tasks.\n"
        "(word count: 47)"
    )


def route_decision(draft: str, decision: str) -> str:
    """Publish or discard the draft based on the human decision."""
    decision_lower = decision.lower().strip()
    if "approve" in decision_lower or decision_lower in ("yes", "y", "ok", "go"):
        return f"PUBLISHED: {draft[:60]}..."
    else:
        return f"DISCARDED. Reason: {decision}"


async def example1_approval_gate() -> None:
    """Simple approval gate: human approves before publishing."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Approval Gate")
    print("=" * 60)

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Draft Blog Post",
                config=ToolStepConfig(
                    tool_name="draft_post",
                    input_mapping={"topic": "$memory.input.topic"},
                ),
            ),
            HumanInputStepDescription(
                number=2,
                title="Human Approval",
                dependencies=[1],
                config=HumanInputStepConfig(
                    prompt="Please review the draft and type 'approve' to publish, or explain why to reject:",
                    timeout=10.0,           # auto-continue after 10 s
                    fallback_value="approve",  # auto-approve if no response
                    output_memory_key="decision",
                ),
            ),
            ToolStepDescription(
                number=3,
                title="Publish or Discard",
                dependencies=[2],
                config=ToolStepConfig(
                    tool_name="route_decision",
                    input_mapping={
                        # Tool step results land in history; step 1 is history[0]
                        "draft": "$history[0]",
                        # HumanInput step writes to memory.human_input.<key>
                        "decision": "$memory.human_input.decision",
                    },
                ),
            ),
        ],
    )

    context = ReasoningContext(
        outer_context="Blog post workflow",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    context.memory_write("topic", "AI in software development", namespace="input")
    context.register_tool("draft_post", draft_post)
    context.register_tool("route_decision", route_decision)

    async def human_approver(prompt: str, future: asyncio.Future) -> None:
        """Simulate a human reviewing after 0.5 seconds."""
        print(f"  [Human sees prompt]: {prompt}")
        await asyncio.sleep(0.5)
        answer = "approve"
        print(f"  [Human responds]: {answer!r}")
        if not future.done():
            future.set_result(answer)

    context.on_human_input_requested = human_approver

    result = await chain.execute_async(context)
    print_execution_summary(result, label="\nExecution")

    if result.success:
        # Step 3 result is last history entry (after stripping the "Step N." header)
        last_history = result.history[-1] if result.history else ""
        outcome_line = [line for line in last_history.split("\n") if line.startswith("Result:")]
        outcome = outcome_line[0].replace("Result: ", "") if outcome_line else last_history
        print(f"  Outcome: {outcome}")
    else:
        for step in result.get_failed_steps():
            print(f"  Failed step {step.step_number}: {step.error_message}")

    assert result.success, f"Expected success, got: {result.get_failed_steps()}"
    assert any("PUBLISHED" in h for h in result.history)
    print("Example 1 passed.")


# ============================================================================
# Example 2: Feedback Loop with Revision
# ============================================================================
# Chain structure:
#   Step 1 (Tool):       generate_code(task) → code in state dict (closure)
#   Step 2 (HumanInput): prompt reviewer → feedback in memory.human_input.review
#   Step 3 (Tool):       revise_code(feedback=memory.human_input.review) → revised code
#
# The tool functions share mutable state via a closure so the code from step 1
# is available to step 3 without routing through history indices.
# ============================================================================


def make_code_state():
    """Mutable state shared between the code-generation tools in this example."""
    return {"code": ""}


def build_code_tools(state: dict):
    """Build tool callables with access to shared state."""

    def generate_code(task: str) -> str:
        snippet = (
            f"# Generated for: {task}\n"
            "def process(data):\n"
            "    return [item * 2 for item in data]"
        )
        state["code"] = snippet
        return snippet

    def revise_code(feedback: str) -> str:
        feedback_lower = feedback.lower()
        if "lgtm" in feedback_lower or "approve" in feedback_lower or "looks good" in feedback_lower:
            return f"FINALIZED:\n{state['code']}"
        elif "type hint" in feedback_lower or "typing" in feedback_lower:
            revised = state["code"].replace(
                "def process(data):",
                "def process(data: list[int]) -> list[int]:",
            )
            state["code"] = revised
            return f"REVISED (type hints added):\n{revised}"
        else:
            return f"REVISION NEEDED: {feedback}\n---\n{state['code']}"

    return generate_code, revise_code


async def example2_feedback_loop() -> None:
    """Feedback loop: human provides review feedback before code is finalized."""
    print("\n" + "=" * 60)
    print("Example 2: Code Review Feedback Loop")
    print("=" * 60)

    state = make_code_state()
    generate_code, revise_code = build_code_tools(state)

    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1,
                title="Generate Code",
                config=ToolStepConfig(
                    tool_name="generate_code",
                    input_mapping={"task": "$memory.input.task"},
                ),
            ),
            HumanInputStepDescription(
                number=2,
                title="Code Review",
                dependencies=[1],
                config=HumanInputStepConfig(
                    prompt="Please review the generated code. Type 'LGTM' to approve or describe changes needed:",
                    timeout=15.0,
                    fallback_value="LGTM",
                    output_memory_key="review",
                ),
            ),
            ToolStepDescription(
                number=3,
                title="Revise or Finalize",
                dependencies=[2],
                config=ToolStepConfig(
                    tool_name="revise_code",
                    input_mapping={
                        "feedback": "$memory.human_input.review",
                    },
                ),
            ),
        ],
    )

    context = ReasoningContext(
        outer_context="Code generation workflow",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    context.memory_write("task", "double all items in a list", namespace="input")
    context.register_tool("generate_code", generate_code)
    context.register_tool("revise_code", revise_code)

    async def code_reviewer(prompt: str, future: asyncio.Future) -> None:
        print(f"  [Reviewer sees]: {prompt}")
        await asyncio.sleep(0.3)
        feedback = "please add type hints"
        print(f"  [Reviewer responds]: {feedback!r}")
        if not future.done():
            future.set_result(feedback)

    context.on_human_input_requested = code_reviewer

    result = await chain.execute_async(context)
    print_execution_summary(result, label="\nExecution")

    if result.success:
        last_history = result.history[-1] if result.history else ""
        lines = [ln for ln in last_history.split("\n") if ln.startswith("Result:")]
        final = lines[0].replace("Result: ", "") if lines else last_history
        print(f"  Final: {final[:80]}...")
    else:
        for step in result.get_failed_steps():
            print(f"  Failed step {step.step_number}: {step.error_message}")

    assert result.success, f"Expected success: {result.get_failed_steps()}"
    assert any("REVISED" in h and "list[int]" in h for h in result.history)
    print("Example 2 passed.")


# ============================================================================
# Example 3: Timeout Fallback (no human responds)
# ============================================================================
# Demonstrates the auto-continue behaviour: if no human responds within the
# configured timeout, the step uses fallback_value and marks timed_out=True
# in result_data.  The chain continues without interruption.
# ============================================================================


def log_decision(decision: str) -> str:
    """Record the deployment decision."""
    return f"Deployment decision recorded: {decision!r}"


async def example3_timeout_fallback() -> None:
    """Timeout fallback: no human responds, chain auto-continues."""
    print("\n" + "=" * 60)
    print("Example 3: Timeout Fallback (no human responds)")
    print("=" * 60)

    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Awaiting Approval",
                config=HumanInputStepConfig(
                    prompt="Approve deployment? (auto-approves in 0.2 seconds)",
                    timeout=0.2,           # very short timeout for demo purposes
                    fallback_value="auto-approved",
                    output_memory_key="deploy_decision",
                ),
            ),
            ToolStepDescription(
                number=2,
                title="Log Decision",
                dependencies=[1],
                config=ToolStepConfig(
                    tool_name="log_decision",
                    input_mapping={
                        "decision": "$memory.human_input.deploy_decision",
                    },
                ),
            ),
        ],
    )

    context = ReasoningContext(
        outer_context="Deployment workflow",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    context.register_tool("log_decision", log_decision)

    # Register a callback that intentionally never responds (simulates absent human)
    async def absent_human(prompt: str, future: asyncio.Future) -> None:  # noqa: ARG001
        print(f"  [No human responds to]: {prompt}")
        # Deliberately do not resolve the future — timeout will trigger fallback

    context.on_human_input_requested = absent_human

    result = await chain.execute_async(context)
    print_execution_summary(result, label="\nExecution")

    if result.success:
        # Inspect timed_out from the HumanInput step's result_data
        human_step = next(
            (s for s in result.step_results if s.step_type and s.step_type.value == "human_input"),
            None,
        )
        timed_out = human_step.result_data.get("timed_out") if human_step else False
        log = next((h for h in result.history if "Deployment decision" in h), "")
        print(f"  Timed out: {timed_out}")
        print(f"  Log: {log.strip()[:80]}")
    else:
        for step in result.get_failed_steps():
            print(f"  Failed step {step.step_number}: {step.error_message}")

    assert result.success, f"Expected success: {result.get_failed_steps()}"
    assert any("auto-approved" in h for h in result.history)
    human_step = next(
        (s for s in result.step_results if s.step_type and s.step_type.value == "human_input"),
        None,
    )
    assert human_step is not None
    assert human_step.result_data.get("timed_out") is True
    print("Example 3 passed.")


# ============================================================================
# Example 4: No Callback (batch / test mode)
# ============================================================================
# When no on_human_input_requested callback is registered, the executor uses
# fallback_value immediately — no pause, no timeout wait.  This makes chains
# with HumanInputStep fully testable in automated environments.
# ============================================================================


async def example4_no_callback_batch_mode() -> None:
    """No callback mode: fallback used immediately for automated/test contexts."""
    print("\n" + "=" * 60)
    print("Example 4: No Callback — Batch / Test Mode")
    print("=" * 60)

    chain = ReasoningChain(
        steps=[
            HumanInputStepDescription(
                number=1,
                title="Human Review",
                config=HumanInputStepConfig(
                    prompt="Approve? (will use fallback in batch mode)",
                    timeout=30.0,           # long timeout — but fallback fires immediately
                    fallback_value="batch-approved",
                    output_memory_key="approval",
                ),
            ),
        ],
    )

    context = ReasoningContext(
        outer_context="Batch processing",
        api=MockClient(),
        # No on_human_input_requested registered
    )

    start = time.monotonic()
    result = await chain.execute_async(context)
    elapsed = time.monotonic() - start

    print_execution_summary(result, label="\nExecution")
    decision = context.memory_read("approval", namespace="human_input")
    print(f"  Decision: {decision!r}  (elapsed: {elapsed:.3f}s)")

    assert result.success
    assert decision == "batch-approved"
    assert elapsed < 1.0, "Should complete immediately without waiting for timeout"
    print("Example 4 passed — no timeout wait in batch mode.")


# ============================================================================
# Main runner
# ============================================================================


async def main() -> None:
    print("Human-in-the-Loop Approval Gate Examples")
    print("=" * 60)

    await example1_approval_gate()
    await example2_feedback_loop()
    await example3_timeout_fallback()
    await example4_no_callback_batch_mode()

    print("\n" + "=" * 60)
    print("All examples completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
