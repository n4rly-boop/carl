"""
SkillResolver & LLM_AGENT Mode — New Features Example

Demonstrates the new AgentSkill features introduced in CARL v0.3:

  1. Programmatic skill creation + resolve_skill() with a local:// URI
  2. URI parsing / coercion demo (github://) without network access
  3. CARL chain using LLM_AGENT mode — iterative tool-calling loop:
       • LLM calls list_resources  → discovers analyze.py
       • LLM calls run_script      → counts words/lines in a text file
       • LLM calls write_file      → saves a Markdown report to workspace/out
       • LLM returns final answer
  4. Output file collection from the workspace

Prerequisites:
  OPENAI_API_KEY set (OpenRouter-compatible)
  Model: DEFAULT_EXAMPLES_MODEL (default qwen/qwen3-8b) must support tool calls

Usage:
  make example-skill-resolver
  PYTHONPATH=$(pwd) uv run python examples/skill_resolver_example.py
  PYTHONPATH=$(pwd) uv run python examples/skill_resolver_example.py --text "your text here"
"""

import argparse
import asyncio
import os
import sys
import textwrap


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build a temporary local skill
# ─────────────────────────────────────────────────────────────────────────────

def create_local_skill(skill_dir: str) -> None:
    """Write a minimal 'text-analyzer' AgentSkill to ``skill_dir``."""
    import pathlib

    root = pathlib.Path(skill_dir)
    root.mkdir(parents=True, exist_ok=True)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # SKILL.md
    (root / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: text-analyzer
        description: Analyzes text files and produces word/line/character statistics.
        license: MIT
        ---

        # Text Analyzer Skill

        This skill analyzes plain text files and reports statistics.

        ## What you can do

        - Count words, lines, and characters in a text file
        - Find the most frequent words
        - Generate a Markdown report saved to /workspace/out/report.md

        ## Workflow

        1. Call `list_resources` to see available scripts.
        2. Call `run_script` with `scripts/analyze.py` and the input file path
           from `/workspace/in/`.  Pass the output path as second argument.
        3. Call `read_file` on the output if you need the raw stats JSON.
        4. Write a final Markdown summary with `write_file`.
    """), encoding="utf-8")

    # scripts/analyze.py
    (scripts_dir / "analyze.py").write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"
        Analyze a text file and print JSON statistics.
        Usage: analyze.py <input_path> [output_path]
        \"\"\"
        import json
        import re
        import sys
        from collections import Counter
        from pathlib import Path

        if len(sys.argv) < 2:
            print(json.dumps({"error": "No input path provided"}))
            sys.exit(1)

        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        try:
            text = Path(input_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(1)

        lines = text.splitlines()
        words = re.findall(r"\\b\\w+\\b", text.lower())
        top_words = Counter(words).most_common(5)

        stats = {
            "chars": len(text),
            "words": len(words),
            "lines": len(lines),
            "unique_words": len(set(words)),
            "top_words": [{"word": w, "count": c} for w, c in top_words],
        }

        result = json.dumps(stats, indent=2)
        print(result)

        if output_path:
            Path(output_path).write_text(result, encoding="utf-8")
    """), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Demo: resolve_skill() + ResolvedSkill introspection
# ─────────────────────────────────────────────────────────────────────────────

def demo_resolve_skill(skill_dir: str) -> None:
    from mmar_carl import resolve_skill, SkillResolverRegistry

    print("\n── Part 1: resolve_skill() ──────────────────────────────")
    skill = resolve_skill(f"local://{skill_dir}")
    print(f"  Name:     {skill.name}")
    print(f"  SHA256:   {skill.sha256[:16]}…")
    print(f"  Version:  {skill.resolved_version}")
    print(f"  Scripts:  {[s.name for s in skill.scripts]}")
    print(f"  Instructions preview:\n    {skill.instructions[:200].strip()!r}")

    print("\n── Part 2: URI coercion (no download) ──────────────────")
    from mmar_carl.skill_resolver import _parse_github_uri
    uri = "github://anthropics/skills/skills/pdf@main"
    owner, repo, subpath, ref = _parse_github_uri(uri)
    print(f"  URI:     {uri}")
    print(f"  owner:   {owner}  repo:    {repo}")
    print(f"  subpath: {subpath}  ref:     {ref}")

    # Also show SkillResolverRegistry dispatch
    registry = SkillResolverRegistry()
    print(f"\n  Registry schemes: local/github/https/module → {type(registry._local).__name__}, "
          f"{type(registry._github).__name__}, {type(registry._https).__name__}, "
          f"{type(registry._module).__name__}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build the CARL chain using LLM_AGENT mode
# ─────────────────────────────────────────────────────────────────────────────

def build_chain(skill_dir: str, text_file: str):
    from mmar_carl import (
        ReasoningChain,
        AgentSkillStepDescription,
        AgentSkillStepConfig,
        AgentSkillExecutionMode,
        MemoryStepDescription,
        MemoryStepConfig,
        MemoryOperation,
        LLMStepDescription,
    )

    return ReasoningChain(
        trace_name="SkillResolver + LLM_AGENT Demo",
        steps=[
            # Step 0: store file path in memory
            MemoryStepDescription(
                number=0,
                title="Store text file path",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="text_file",
                    namespace="input",
                    value_source=f'"{text_file}"',
                ),
            ),

            # Step 1: LLM_AGENT mode — iterative tool-calling loop
            # The LLM decides when to call scripts, read/write files, and when to stop.
            AgentSkillStepDescription(
                number=1,
                title="Analyze text with LLM_AGENT mode",
                dependencies=[0],
                config=AgentSkillStepConfig(
                    # Use local:// URI directly in the config string
                    skill=f"local://{skill_dir}",
                    task=(
                        "Analyze the text file at the provided path. "
                        "Use list_resources to see available scripts, "
                        "then run scripts/analyze.py passing the input file "
                        "as the first argument and '/workspace/out/stats.json' as the second. "
                        "After getting the statistics, write a human-readable Markdown summary "
                        "to /workspace/out/report.md using write_file. "
                        "Input file: {text_file}"
                    ),
                    input_mapping={"text_file": "$memory.input.text_file"},
                    execution_mode=AgentSkillExecutionMode.LLM_AGENT,
                    llm_max_iterations=10,
                    output_capture="both",
                    output_files_glob=["*.json", "*.md"],
                    timeout=120.0,
                ),
            ),

            # Step 2: Reason about the stats with a regular LLM step
            LLMStepDescription(
                number=2,
                title="Summarize analysis results",
                dependencies=[1],
                aim=(
                    "Provide a concise interpretation of the text analysis statistics. "
                    "Comment on the writing style, vocabulary richness, and any observations."
                ),
                reasoning_questions=(
                    "What does the word count suggest about document length? "
                    "Is the vocabulary rich or repetitive (unique_words/words ratio)? "
                    "What can the top words tell us about the document's focus?"
                ),
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_example(text: str, api_key: str) -> None:
    import pathlib
    import tempfile

    from mmar_carl import ReasoningContext, create_openai_client

    model = os.environ.get("DEFAULT_EXAMPLES_MODEL", "qwen/qwen3-8b")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    print("=" * 60)
    print("CARL v0.3 — SkillResolver & LLM_AGENT Mode Demo")
    print(f"  Model: {model}")
    print("=" * 60)

    # Create temp skill dir and input text file
    with tempfile.TemporaryDirectory(prefix="carl_skill_demo_") as tmpdir:
        skill_dir = os.path.join(tmpdir, "text-analyzer")
        create_local_skill(skill_dir)
        print(f"\nCreated skill at: {skill_dir}")

        text_file = os.path.join(tmpdir, "input.txt")
        pathlib.Path(text_file).write_text(text, encoding="utf-8")
        print(f"Created text file: {text_file} ({len(text)} chars)")

        # Part 1 & 2: synchronous resolver demos
        demo_resolve_skill(skill_dir)

        # Part 3: run the CARL chain
        print("\n── Part 3: CARL chain with LLM_AGENT mode ──────────────")

        llm_client = create_openai_client(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=0.3,
        )
        context = ReasoningContext(
            outer_context=f"Text analysis demo — {len(text)} chars of sample text.",
            api=llm_client,
            model=model,
        )

        chain = build_chain(skill_dir, text_file)
        print("  Steps: Memory → AgentSkill[LLM_AGENT] → LLM summary")
        print("  Running chain …\n")

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
                if sr.result_data and isinstance(sr.result_data, dict):
                    em = sr.result_data.get("execution_mode", "")
                    if em:
                        mode_tag = f" [{em}]"
                    iters = sr.result_data.get("iterations")
                    if iters:
                        mode_tag += f" {iters} iters"
                    tc = sr.result_data.get("tool_calls_made")
                    if tc:
                        mode_tag += f" / {tc} tool calls"
                print(f"  [OK] Step {sr.step_number}. {sr.step_title}{mode_tag} ({t})")

            # Show agent skill result details
            agent_step = next(
                (sr for sr in result.step_results if sr.step_number == 1), None
            )
            if agent_step and agent_step.result_data:
                rd = agent_step.result_data
                out_files = rd.get("output_files", [])
                if out_files:
                    print(f"\nOutput files from workspace ({len(out_files)}):")
                    for f in out_files:
                        print(f"  {f['name']} — {f.get('size', '?')} bytes at {f['path']}")

                    # Print the report if it exists
                    report = next((f for f in out_files if f["name"] == "report.md"), None)
                    if report and os.path.isfile(report["path"]):
                        content = pathlib.Path(report["path"]).read_text()
                        print("\n--- report.md ---")
                        print(content[:1000] + ("…" if len(content) > 1000 else ""))

            # LLM summary
            if context.history:
                print("\n--- LLM Summary (Step 2) ---")
                last = context.history[-1]
                print(last[:600] + ("…" if len(last) > 600 else ""))
        else:
            print("Chain FAILED:")
            for sr in result.get_failed_steps():
                print(f"  Step {sr.step_number} ({sr.step_title}): {sr.error_message}")
            sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CARL v0.3 — SkillResolver & LLM_AGENT mode demo"
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to analyze (default: built-in sample paragraph)",
    )
    parser.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: No API key. Set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    sample_text = args.text or textwrap.dedent("""\
        CARL is a Python library for building sophisticated chain-of-thought reasoning systems
        with DAG-based parallel execution. It enables developers to create expert reasoning chains
        that can execute reasoning steps in parallel where dependencies allow, with automatic
        RAG-like context extraction for each step.

        The library supports multiple step types: LLM reasoning, Tool calls, MCP protocol,
        Memory operations, Data transformations, Conditional branching, and now AgentSkills —
        portable skill folders following the open AgentSkills standard.

        AgentSkills can be loaded from local paths, GitHub repositories via tarball download,
        generic HTTPS archives, or installed Python packages. The new LLM_AGENT execution mode
        enables iterative tool-calling loops where the LLM orchestrates script execution,
        file I/O, and resource inspection within an isolated workspace.
    """).strip()

    asyncio.run(run_example(sample_text, api_key))


if __name__ == "__main__":
    main()
