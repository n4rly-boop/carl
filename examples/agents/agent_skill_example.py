"""
Document Analyzer with Slides Generation — AgentSkill Example

Demonstrates how to use AgentSkillStep in a CARL reasoning chain to:
  1. Extract content from a PDF using the "pdf" AgentSkill (SUBAGENT mode)
  2. Analyze and reason about the document (LLM steps)
  3. Build a .pptx presentation using the "pptx" AgentSkill (SCRIPT mode)

Execution modes:
  - Step 1 (PDF): SUBAGENT — extract_text.py runs first (real extraction via
    pdfplumber/pypdf), then LLM interprets the extracted content with PDF skill
    knowledge. The LLM receives actual document text, not hallucinated content.
  - Steps 2–4: standard LLMStep reasoning (analyze → plan → draft slides).
  - Step 5 (PPTX): SCRIPT — create_from_markdown.py parses the LLM-generated
    Markdown slides and builds a real .pptx file using python-pptx. The output
    path is stored in memory and printed at the end.

Prerequisites:
  Skills installed at ~/.agents/skills/ (auto-installed by make example-agent-skill):
    pdf/SKILL.md + scripts/extract_text.py
    pptx/SKILL.md + scripts/create_from_markdown.py
  Python packages: pdfplumber (or pypdf), python-pptx
    install with: pip install 'mmar-carl[skills]'  (or [pdf] + [pptx])
  API key: OPENAI_API_KEY or --api-key

Usage:
  make example-agent-skill                     # auto-generates test PDF
  make example-agent-skill PDF=report.pdf      # use your own PDF
  uv run python examples/agent_skill_example.py --pdf report.pdf [--output deck.pptx]
"""

import argparse
import asyncio
import os
import sys


def build_chain(pdf_path: str, output_pptx: str | None = None):
    """Build the document analysis → slides chain.

    Args:
        pdf_path: Path to the PDF to analyze.
        output_pptx: Where to save the .pptx file. Defaults to a temp file.
    """
    from mmar_carl import (
        ReasoningChain,
        AgentSkillStepDescription,
        AgentSkillStepConfig,
        AgentSkillExecutionMode,
        LLMStepDescription,
        MemoryStepDescription,
        MemoryStepConfig,
        MemoryOperation,
    )

    # Build extra script args for PPTX step
    pptx_title = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ").title()
    pptx_script_args: dict[str, str] = {"title": pptx_title}
    if output_pptx:
        pptx_script_args["output"] = output_pptx

    return ReasoningChain(
        trace_name="Document Analyzer with Slides Generation",
        steps=[
            # Step 0: Write PDF path to memory so subsequent steps can reference it
            MemoryStepDescription(
                number=0,
                title="Store PDF path",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="pdf_path",
                    namespace="input",
                    value_source=f'"{pdf_path}"',
                ),
            ),
            # Step 1: SUBAGENT — extract_text.py extracts real text, LLM interprets
            AgentSkillStepDescription(
                number=1,
                title="Extract and interpret PDF content",
                dependencies=[0],
                config=AgentSkillStepConfig(
                    skill="pdf",
                    task=(
                        "You have received the extracted text from the PDF below. "
                        "Provide a structured summary: document type, main sections, "
                        "key data points, and any tables. Preserve all numbers and facts."
                    ),
                    input_mapping={"pdf_path": "$memory.input.pdf_path"},
                    execution_mode=AgentSkillExecutionMode.SUBAGENT,
                    script_name="scripts/extract_text.py",
                    timeout=120.0,
                ),
            ),
            # Step 2: Analyze extracted content
            LLMStepDescription(
                number=2,
                title="Analyze document structure and insights",
                dependencies=[1],
                aim=(
                    "Identify the main themes, key findings, important data points, "
                    "and overall structure of the document."
                ),
                reasoning_questions=(
                    "What are the 3-5 most important findings? "
                    "What data supports these findings? "
                    "Who is the target audience?"
                ),
            ),
            # Step 3: Plan the presentation
            LLMStepDescription(
                number=3,
                title="Plan presentation slides",
                dependencies=[2],
                aim=(
                    "Determine what sections and key points the slides should cover. "
                    "Decide how many slides are needed and what each one should say."
                ),
                reasoning_questions=(
                    "How many slides fit the content? "
                    "What goes on the title slide? "
                    "Which findings deserve their own slide?"
                ),
            ),
            # Step 4: Draft polished slide content in Markdown
            LLMStepDescription(
                number=4,
                title="Draft slide content",
                dependencies=[2, 3],
                aim=(
                    "Write complete, polished content for each slide: title, "
                    "bullet points (max 5), and speaker notes."
                ),
                stage_action=(
                    "Format as Markdown with slide boundaries:\n"
                    "## Slide N: [Title]\n"
                    "- bullet 1\n"
                    "- bullet 2\n"
                    "**Speaker Notes**: ..."
                ),
            ),
            # Step 5: SCRIPT — create_from_markdown.py builds the real .pptx
            AgentSkillStepDescription(
                number=5,
                title="Build PPTX presentation",
                dependencies=[4],
                config=AgentSkillStepConfig(
                    skill="pptx",
                    task="Build a PPTX presentation from the slide Markdown content.",
                    input_mapping={"content": "$history[-1]"},
                    execution_mode=AgentSkillExecutionMode.SCRIPT,
                    script_name="scripts/create_from_markdown.py",
                    script_args=pptx_script_args,
                    output_file_key="presentation_path",
                    timeout=120.0,
                ),
            ),
        ],
    )


async def run_example(
    pdf_path: str,
    api_key: str | None = None,
    output_pptx: str | None = None,
) -> None:
    """Run the document analysis chain."""
    from mmar_carl import ReasoningContext, create_openai_client

    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_api_key:
        print("ERROR: No API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("DEFAULT_EXAMPLES_MODEL", "openai/gpt-4o")
    llm_client = create_openai_client(api_key=resolved_api_key, model=model)
    context = ReasoningContext(
        outer_context=f"Analyzing PDF: {os.path.basename(pdf_path)}",
        api=llm_client,
        model=model,
    )

    chain = build_chain(pdf_path, output_pptx)

    pdf_size = os.path.getsize(pdf_path)
    print("\nDocument Analyzer with Slides Generation")
    print(f"  PDF:   {pdf_path} ({pdf_size:,} bytes)")
    print(f"  Model: {model}")
    print("  Steps: Memory → PDF[SUBAGENT] → Analyze → Plan → Draft → PPTX[SCRIPT]")
    print("=" * 60)

    result = await chain.execute_async(context)

    print("\n" + "=" * 60)
    if result.success:
        total = result.total_execution_time or sum(
            sr.execution_time or 0.0 for sr in result.step_results
        )
        print(f"Chain completed — {len(result.step_results)} steps in {total:.1f}s\n")

        for sr in result.step_results:
            t = f"{sr.execution_time:.1f}s" if sr.execution_time else "—"
            mode_tag = ""
            if sr.result_data and sr.result_data.get("execution_mode"):
                mode_tag = f" [{sr.result_data['execution_mode']}]"
            print(f"  [OK] Step {sr.step_number}. {sr.step_title}{mode_tag} ({t})")

        pptx_path = context.memory_read("presentation_path", namespace="agent_skill")
        if pptx_path and os.path.isfile(pptx_path):
            size_kb = os.path.getsize(pptx_path) / 1024
            print(f"\nPresentation: {pptx_path} ({size_kb:.1f} KB)")
        elif pptx_path:
            print(f"\nPresentation path: {pptx_path}")

        print("\n--- Slide draft preview ---")
        for entry in context.history[-2:]:
            preview = entry[:500] + ("..." if len(entry) > 500 else "")
            print(f"\n{preview}")
    else:
        print("Chain failed:")
        for sr in result.get_failed_steps():
            print(f"  Step {sr.step_number} ({sr.step_title}): {sr.error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Document Analyzer with Slides Generation using AgentSkills"
    )
    parser.add_argument("--pdf", required=True, help="PDF file to analyze")
    parser.add_argument("--output", default=None, help="Output .pptx path (default: temp file)")
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    asyncio.run(run_example(args.pdf, args.api_key, args.output))


if __name__ == "__main__":
    main()
