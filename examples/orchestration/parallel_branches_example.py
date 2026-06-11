#!/usr/bin/env python3
"""
Example: True Parallel Skill Branches with output_memory_key.

Demonstrates a chain where 3 AgentHandoffSteps run in parallel — each
representing a specialist "agent" (document extractor, web searcher, and
metadata analyzer).  Each writes its result to a named memory key via
``output_memory_key``, and a final synthesis step reads from all three via
``$memory.handoff.*`` references.

Covered patterns:
  1. Fan-out: three parallel AgentHandoffStep branches with ``output_memory_key``
  2. Fan-in:  synthesis ToolStep reads ``$memory.handoff.<key>`` from all branches
  3. Mixed:   sub-chains that themselves use ToolSteps (no LLM required)
  4. Failure isolation: one branch can fail without blocking the others

No LLM or API key required.  All work is done by tool steps and a mock client.

Usage:
    python examples/parallel_branches_example.py
    PYTHONPATH=$(pwd) python examples/parallel_branches_example.py
"""

import asyncio
import time

from mmar_carl import (
    AgentHandoffStepDescription,
    Language,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import AgentHandoffStepConfig, ToolStepConfig
from mmar_carl.models.steps import ToolStepDescription
from examples.utils import format_status, print_execution_summary


# ---------------------------------------------------------------------------
# Minimal mock LLM client (no API key needed)
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    """No-op client — all work is done by tool steps."""

    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return "ok"


# ---------------------------------------------------------------------------
# Specialist tool functions (simulating real processing work)
# ---------------------------------------------------------------------------


def extract_document_text(topic: str) -> str:
    """Simulate PDF/document text extraction for the given topic."""
    time.sleep(0.01)  # simulate I/O
    return (
        f"[Document Extract] Topic: {topic}\n"
        "Key findings: Revenue grew 23% YoY. Operating margins improved to 18.5%. "
        "Three new product lines launched in Q3. Headcount increased by 420 FTEs."
    )


def search_web(topic: str) -> str:
    """Simulate a web search for the given topic."""
    time.sleep(0.01)  # simulate network round-trip
    return (
        f"[Web Search] Query: '{topic}'\n"
        "Top result: Industry analysts project continued growth through 2026. "
        "Competitor X reported 15% growth; Company outperforms sector average by 8pp. "
        "Three analyst upgrades issued in past 30 days."
    )


def extract_metadata(topic: str) -> str:
    """Simulate metadata / structured attribute extraction."""
    time.sleep(0.01)
    return (
        f"[Metadata] Topic: {topic}\n"
        "fiscal_year: 2024  |  currency: USD  |  reporting_standard: GAAP  |  "
        "auditor: PwC  |  pages: 142  |  tables: 38"
    )


def synthesize_results(doc_result: str, web_result: str, meta_result: str) -> str:
    """Combine parallel branch outputs into a final research summary."""
    sections = [
        "=== RESEARCH SYNTHESIS ===",
        "",
        "--- Document Findings ---",
        doc_result,
        "",
        "--- Web Intelligence ---",
        web_result,
        "",
        "--- Metadata Attributes ---",
        meta_result,
        "",
        "--- Summary ---",
        "All three specialist agents completed successfully.",
        "Document extraction, web search, and metadata analysis are consistent.",
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Helper: build a single-step sub-chain for one specialist
# ---------------------------------------------------------------------------


def _build_specialist_chain(tool_name: str, title: str) -> ReasoningChain:
    """
    Build a single-step ReasoningChain that calls ``tool_name`` with the
    topic from sub-chain memory and writes the result to history.
    """
    step = ToolStepDescription(
        number=1,
        title=title,
        config=ToolStepConfig(
            tool_name=tool_name,
            input_mapping={"topic": "$memory.input.topic"},
        ),
    )
    return ReasoningChain(steps=[step])


# ============================================================================
# Example 1: Three parallel branches — fan-out / fan-in
# ============================================================================


async def example_parallel_branches():
    """
    Fan-out: steps 1, 2, 3 run concurrently (no dependencies on each other).
    Fan-in:  step 4 depends on [1, 2, 3] and reads from $memory.handoff.*.
    """
    print("\n" + "=" * 60)
    print("Example 1: Three Parallel Branches (fan-out / fan-in)")
    print("=" * 60)

    topic = "Acme Corp 2024 Annual Report"

    # --- Build specialist sub-chains ---
    doc_chain = _build_specialist_chain("extract_document_text", "Extract Document")
    web_chain = _build_specialist_chain("search_web", "Web Search")
    meta_chain = _build_specialist_chain("extract_metadata", "Extract Metadata")

    # --- Assemble parent chain ---
    # Steps 1, 2, 3 share no dependencies → DAGExecutor runs them in one batch.
    # Step 4 depends on all three → runs in a second batch after all complete.
    chain = ReasoningChain(
        steps=[
            AgentHandoffStepDescription(
                number=1,
                title="Document Extractor",
                sub_chain=doc_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="doc_findings",  # → memory.handoff.doc_findings
                ),
            ),
            AgentHandoffStepDescription(
                number=2,
                title="Web Searcher",
                sub_chain=web_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="web_findings",  # → memory.handoff.web_findings
                ),
            ),
            AgentHandoffStepDescription(
                number=3,
                title="Metadata Analyzer",
                sub_chain=meta_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="meta_findings",  # → memory.handoff.meta_findings
                ),
            ),
            ToolStepDescription(
                number=4,
                title="Synthesize Research",
                dependencies=[1, 2, 3],  # wait for all three branches
                config=ToolStepConfig(
                    tool_name="synthesize_results",
                    input_mapping={
                        "doc_result": "$memory.handoff.doc_findings",
                        "web_result": "$memory.handoff.web_findings",
                        "meta_result": "$memory.handoff.meta_findings",
                    },
                ),
            ),
        ]
    )

    print(f"\nTopic: {topic!r}")
    print(f"Chain steps: {[s.title for s in chain.steps]}")
    print("Steps 1-3 run in parallel; step 4 waits for all three.")

    # Register tools in parent context — sub-chain contexts inherit them
    ctx = ReasoningContext(
        outer_context=f"Research topic: {topic}",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("extract_document_text", extract_document_text)
    ctx.register_tool("search_web", search_web)
    ctx.register_tool("extract_metadata", extract_metadata)
    ctx.register_tool("synthesize_results", synthesize_results)

    start = time.perf_counter()
    result = await chain.execute_async(ctx)
    elapsed = time.perf_counter() - start

    print(f"\nExecution: {format_status(result.success)}  ({elapsed:.2f}s)")

    # Show what each branch wrote to memory
    handoff_ns = ctx.memory.get("handoff", {})
    print("\nMemory written by parallel branches:")
    for key, val in handoff_ns.items():
        snippet = val[:80].replace("\n", " ") if isinstance(val, str) else repr(val)
        print(f"  $memory.handoff.{key}: {snippet!r}...")

    print("\nFinal synthesis (step 4 output):")
    if result.success:
        print(result.history[-1][:400])

    return result


# ============================================================================
# Example 2: Full synthesis using explicit memory reads
# ============================================================================


async def example_full_synthesis():
    """
    More complete example: sub-chains write findings to named keys;
    synthesis ToolStep reads each key and produces a structured report.
    """
    print("\n" + "=" * 60)
    print("Example 2: Full Synthesis with Explicit Memory Reads")
    print("=" * 60)

    topic = "TechStart Inc Q1 2025 Earnings"

    doc_chain = _build_specialist_chain("extract_document_text", "Extract Document")
    web_chain = _build_specialist_chain("search_web", "Web Search")
    meta_chain = _build_specialist_chain("extract_metadata", "Extract Metadata")

    chain = ReasoningChain(
        steps=[
            AgentHandoffStepDescription(
                number=1,
                title="Document Extractor",
                sub_chain=doc_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="doc_findings",
                ),
            ),
            AgentHandoffStepDescription(
                number=2,
                title="Web Searcher",
                sub_chain=web_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="web_findings",
                ),
            ),
            AgentHandoffStepDescription(
                number=3,
                title="Metadata Analyzer",
                sub_chain=meta_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="meta_findings",
                ),
            ),
            ToolStepDescription(
                number=4,
                title="Synthesize Research",
                dependencies=[1, 2, 3],
                config=ToolStepConfig(
                    tool_name="synthesize_results",
                    input_mapping={
                        "doc_result": "$memory.handoff.doc_findings",
                        "web_result": "$memory.handoff.web_findings",
                        "meta_result": "$memory.handoff.meta_findings",
                    },
                ),
            ),
        ]
    )

    ctx = ReasoningContext(
        outer_context=f"Quarterly earnings research: {topic}",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("extract_document_text", extract_document_text)
    ctx.register_tool("search_web", search_web)
    ctx.register_tool("extract_metadata", extract_metadata)
    ctx.register_tool("synthesize_results", synthesize_results)

    result = await chain.execute_async(ctx)

    print_execution_summary(result, label="\nExecution")
    print(f"Steps completed: {len([s for s in result.step_results if s.success])}")
    print(f"History entries: {len(result.history)}")

    # Print the synthesis output
    synthesis = ctx.memory.get("handoff", {})
    if synthesis:
        print(f"\nKeys in memory.handoff: {list(synthesis.keys())}")

    if result.success:
        print("\n--- Synthesis Output ---")
        print(result.history[-1])

    return result


# ============================================================================
# Example 3: Failure isolation — one branch fails, others succeed
# ============================================================================


async def example_failure_isolation():
    """
    One branch raises an exception; the others complete normally.
    The parent step for the failed branch is marked failed, but siblings
    still run since they have no dependency on the failing branch.
    The synthesis step depends on all three, so it is skipped/failed when
    one of its required inputs is missing.

    To show resilience, the synthesis in this example accepts a fallback.
    """
    print("\n" + "=" * 60)
    print("Example 3: Failure Isolation")
    print("=" * 60)

    topic = "Failure isolation demo"

    def failing_tool(topic: str) -> str:
        raise RuntimeError(f"Simulated network failure for topic: {topic!r}")

    def resilient_synthesize(
        doc_result: str, meta_result: str, web_result: str = "[unavailable]"
    ) -> str:
        return (
            f"=== PARTIAL SYNTHESIS ===\n"
            f"doc: {doc_result[:60]}...\n"
            f"web: {web_result}\n"
            f"meta: {meta_result[:60]}..."
        )

    doc_chain = _build_specialist_chain("extract_document_text", "Extract Document")
    failing_chain = _build_specialist_chain("failing_tool", "Failing Web Search")
    meta_chain = _build_specialist_chain("extract_metadata", "Extract Metadata")

    chain = ReasoningChain(
        steps=[
            AgentHandoffStepDescription(
                number=1,
                title="Document Extractor",
                sub_chain=doc_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="doc_findings",
                ),
            ),
            AgentHandoffStepDescription(
                number=2,
                title="Web Searcher (will fail)",
                sub_chain=failing_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="web_findings",
                    propagate_failure=False,  # ← don't mark parent step as failed
                ),
            ),
            AgentHandoffStepDescription(
                number=3,
                title="Metadata Analyzer",
                sub_chain=meta_chain,
                config=AgentHandoffStepConfig(
                    input_mapping={"input.topic": f"'{topic}'"},
                    output_memory_key="meta_findings",
                ),
            ),
            ToolStepDescription(
                number=4,
                title="Resilient Synthesize",
                dependencies=[1, 2, 3],
                config=ToolStepConfig(
                    tool_name="resilient_synthesize",
                    input_mapping={
                        "doc_result": "$memory.handoff.doc_findings",
                        "meta_result": "$memory.handoff.meta_findings",
                        # web_result omitted → uses default "[unavailable]"
                    },
                ),
            ),
        ]
    )

    ctx = ReasoningContext(
        outer_context=f"Research topic: {topic}",
        api=MockClient(),
        language=Language.ENGLISH,
    )
    ctx.register_tool("extract_document_text", extract_document_text)
    ctx.register_tool("failing_tool", failing_tool)
    ctx.register_tool("extract_metadata", extract_metadata)
    ctx.register_tool("resilient_synthesize", resilient_synthesize)

    result = await chain.execute_async(ctx)

    succeeded = [s for s in result.step_results if s.success]
    failed = [s for s in result.step_results if not s.success and not s.skipped]

    print_execution_summary(result, label="\nOverall")
    print(f"Succeeded: {[s.step_title for s in succeeded]}")
    print(f"Failed:    {[s.step_title for s in failed]}")
    print(f"\nKeys in memory.handoff: {list(ctx.memory.get('handoff', {}).keys())}")

    if result.history:
        print("\n--- Partial Synthesis ---")
        print(result.history[-1])

    return result


# ============================================================================
# Main
# ============================================================================


async def main():
    print("CARL Parallel Branches Example")
    print("=" * 60)
    print("Demonstrates fan-out / fan-in with output_memory_key.")
    print("No LLM or API key required.\n")

    await example_parallel_branches()
    await example_full_synthesis()
    await example_failure_isolation()

    print("\n" + "=" * 60)
    print("Parallel branches examples completed!")
    print("=" * 60)
    print("\nKey patterns:")
    print("  AgentHandoffStepConfig(output_memory_key='key')  → memory.handoff.key")
    print("  ToolStepConfig(input_mapping={'arg': '$memory.handoff.key'})")
    print("  dependencies=[1, 2, 3]  → fan-in gate (step 4 waits for all)")
    print("  propagate_failure=False → branch failure doesn't fail parent step")
    print("\nVariants:")
    print("  Replace AgentHandoffStepDescription with AgentSkillStepDescription")
    print("  for real skills (PDF extraction, web search, etc.)")
    print("  See examples/agent_skill_example.py for skill setup.")


if __name__ == "__main__":
    asyncio.run(main())
