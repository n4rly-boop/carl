#!/usr/bin/env python3
"""
Example: generate a CARL chain from a natural-language task description.

Uses ``ChainBuilder.from_description(task, llm_client, ...)`` — a meta-agent
that asks an LLM to plan a chain in JSON form, then parses the plan back
into a fully validated :class:`ReasoningChain`.

This example uses a scripted mock planner LLM that returns canned JSON
plans for several tasks, so it runs without an API key. In production,
swap the mock for any client implementing
``LLMClientBase.get_response_with_retries`` (e.g. ``OpenAICompatibleClient``).

Run:
    python examples/chain_from_description_example.py
    PYTHONPATH=$(pwd) python examples/chain_from_description_example.py
"""

import asyncio
import json

from mmar_carl import (
    ChainBuilder,
    LLMClientBase,
    ReasoningContext,
)
from examples.utils import format_status


class ScriptedPlannerLLM(LLMClientBase):
    """Planner mock — returns a different JSON plan for each task keyword.

    Real-world replacement is any provider client: in production this would
    be an ``OpenAICompatibleClient`` (or any ``LLMClientBase`` subclass)
    pointed at gpt-4o / claude-3-5-sonnet / etc.
    """

    def __init__(self, plans: dict[str, str]) -> None:
        self.plans = plans
        self.planner_calls = 0

    async def get_response(self, prompt: str) -> str:
        # The first call is always the planning request. Subsequent calls
        # (during chain execution) get short synthesized answers.
        if self.planner_calls == 0 and "JSON plan" in prompt:
            self.planner_calls += 1
            for keyword, plan in self.plans.items():
                if keyword.lower() in prompt.lower():
                    return plan
            # Fallback if no keyword matches
            return self.plans.get("default", '{"steps": []}')
        return f"step-answer-{self.planner_calls}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ---------------------------------------------------------------------------
# Canned plans (what a real planner LLM would generate)
# ---------------------------------------------------------------------------

PDF_PLAN = json.dumps({
    "steps": [
        {"number": 1, "title": "Extract text", "step_type": "llm",
         "aim": "Pull out all text from the PDF.", "dependencies": []},
        {"number": 2, "title": "Identify claims", "step_type": "llm",
         "aim": "List every factual claim made in the document.",
         "dependencies": [1]},
        {"number": 3, "title": "Persist findings", "step_type": "memory",
         "dependencies": [2],
         "step_config": {"operation": "write", "memory_key": "claims",
                         "value_source": "$history[-1]", "namespace": "output"}},
    ]
})

FACTCHECK_PLAN = json.dumps({
    "steps": [
        {"number": 1, "title": "Decompose claim", "step_type": "llm",
         "aim": "Break the claim into atomic verifiable statements."},
        {"number": 2, "title": "Find evidence", "step_type": "tool",
         "dependencies": [1],
         "step_config": {"tool_name": "web_search",
                         "input_mapping": {"query": "$history[-1]"}}},
        {"number": 3, "title": "Assess", "step_type": "llm",
         "aim": "Rate each claim true/false/unclear based on the evidence.",
         "dependencies": [2]},
        {"number": 4, "title": "Final verdict", "step_type": "llm",
         "aim": "Synthesize the per-claim ratings into one verdict.",
         "dependencies": [3]},
    ]
})


# ---------------------------------------------------------------------------
# Example 1 — Generate, inspect, then execute
# ---------------------------------------------------------------------------


async def example_pdf_pipeline() -> None:
    print("=" * 72)
    print("Example 1: Generate a PDF-analysis chain from a description")
    print("=" * 72)

    llm = ScriptedPlannerLLM({"pdf": PDF_PLAN})
    chain = await ChainBuilder.from_description(
        "Analyze a PDF: extract its text, list factual claims, and persist them",
        llm,
        available_skills=["pdf"],
        available_tools=["web_search"],
        max_steps=5,
    )
    print(f"\nGenerated chain with {len(chain.steps)} steps:")
    for s in chain.steps:
        deps = f" deps={s.dependencies}" if s.dependencies else ""
        print(f"  step {s.number}. {s.title}  ({s.step_type}){deps}")

    print(f"\nMetadata origin: "
          f"{chain.metadata.get('generated_from_description', '')[:60]!r}")

    # Execute the generated chain end-to-end
    ctx = ReasoningContext(outer_context="document.pdf", api=llm)
    result = await chain.execute_async(ctx)
    print(f"\nExecution: {format_status(result.success)} "
          f"({len(result.step_results)}/{len(chain.steps)} steps succeeded)")
    print(f"Output memory: {ctx.memory.get('output', {})}")


# ---------------------------------------------------------------------------
# Example 2 — More elaborate plan with tool steps
# ---------------------------------------------------------------------------


async def example_factcheck_pipeline() -> None:
    print("\n" + "=" * 72)
    print("Example 2: Fact-checking pipeline with mixed step types")
    print("=" * 72)

    llm = ScriptedPlannerLLM({"fact": FACTCHECK_PLAN})

    def fake_web_search(query: str) -> str:
        return f"[3 results for {query[:40]!r}]"

    chain = await ChainBuilder.from_description(
        "Build a fact-checking pipeline for the given claim.",
        llm,
        available_tools=["web_search"],
        max_steps=6,
        extra_instructions="Prefer tool steps over LLM steps for evidence gathering.",
    )
    print(f"\nGenerated chain with {len(chain.steps)} steps:")
    for s in chain.steps:
        deps = f" deps={s.dependencies}" if s.dependencies else ""
        print(f"  step {s.number}. {s.title}  ({s.step_type}){deps}")

    ctx = ReasoningContext(
        outer_context="The Earth's circumference is 40,075 km.",
        api=llm,
    )
    ctx.register_tool("web_search", fake_web_search)
    result = await chain.execute_async(ctx)
    print(f"\nExecution: {format_status(result.success)} "
          f"({len(result.step_results)}/{len(chain.steps)} steps succeeded)")


# ---------------------------------------------------------------------------
# Example 3 — Error handling for malformed plans
# ---------------------------------------------------------------------------


async def example_invalid_plan_diagnostics() -> None:
    print("\n" + "=" * 72)
    print("Example 3: Diagnostics for malformed planner output")
    print("=" * 72)

    # A planner that returns prose instead of JSON
    llm = ScriptedPlannerLLM({"default": "Sorry, I can't help with that today."})
    try:
        await ChainBuilder.from_description("any task", llm)
    except ValueError as exc:
        print(f"\n  caught (invalid JSON): {exc}")

    # A planner that returns JSON missing 'steps'
    llm = ScriptedPlannerLLM({"default": '{"plan": "but no steps key"}'})
    try:
        await ChainBuilder.from_description("any task", llm)
    except ValueError as exc:
        print(f"  caught (missing key):  {exc}")

    # A planner that produces too many steps
    plan = json.dumps({
        "steps": [
            {"number": i, "title": f"s{i}", "step_type": "llm", "aim": "x"}
            for i in range(1, 11)
        ]
    })
    llm = ScriptedPlannerLLM({"default": plan})
    try:
        await ChainBuilder.from_description("any task", llm, max_steps=3)
    except ValueError as exc:
        print(f"  caught (too many):     {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await example_pdf_pipeline()
    await example_factcheck_pipeline()
    await example_invalid_plan_diagnostics()
    print("\n" + "=" * 72)
    print("All chain-from-description examples completed.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
