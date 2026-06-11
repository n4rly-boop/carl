"""Regenerate per-topic README.md files under tests/ and examples/.

Each topic-grouped submodule (e.g. ``tests/agents/``) gets a one-page
README with a curated topic blurb plus an auto-generated file index that
reads each module's docstring for the one-line summary. Running this
script is idempotent — re-running it picks up new files, removed files,
and changed docstrings without manual editing.

Invoked by ``make docs-topic-index``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from textwrap import dedent

# Topic blurbs — one paragraph per submodule, keyed by directory name.
TEST_TOPICS: dict[str, str] = {
    "agents": (
        "Multi-agent orchestration step types — supervisor routing, "
        "debate (round-robin LLM-vs-LLM judging), parallel sampling, "
        "AgentSkill execution (LLM / SCRIPT / HYBRID / SUBAGENT / "
        "LLM_AGENT modes), agent handoff (sub-chain delegation), and "
        "human-in-the-loop input steps."
    ),
    "chain_lifecycle": (
        "Chain construction, validation, serialization, reflection, "
        "cancellation, evolutionary search (`ChainEvolver`), "
        "checkpoint/resume, and the test-harness API."
    ),
    "evaluation": (
        "Step metrics, dataset evaluators, evaluation step types, and "
        "result formatters — including the visualization methods on "
        "`ReasoningResult` and `DatasetEvaluationReport` "
        "(cost-by-model, failure heatmap, step-metric heatmap, "
        "profiling tables, token pies)."
    ),
    "llm_inference": (
        "LLM client behaviour: introspection typing, retry policies, "
        "streaming, structured output, council/multi-model voting, "
        "execution modes (FAST vs SELF_CRITIC), token budgets, cost "
        "estimation, record-and-replay cassettes."
    ),
    "mcp": (
        "Model Context Protocol step integration — MCP tool calls, "
        "resource fetches, transport plumbing (stdio / SSE / HTTP)."
    ),
    "memory": (
        "Memory subsystem: namespaced read/write, COW-isolated parallel "
        "writes, lazy memory values, long-term memory backends, "
        "history truncation, parallel-step memory isolation."
    ),
    "orchestration": (
        "DAG execution engine: dependency resolution, parallel "
        "batches, conditional branching, loops, tool steps, transform "
        "steps, step caching, auto-workers, builder helpers."
    ),
    "replan": (
        "Runtime replanning: rule-based and LLM-based checkers, "
        "aggregation strategies (majority / unanimous), budget guards, "
        "checkpoint rollback."
    ),
    "tool_calling": (
        "Tool step execution: registration, dynamic input mapping, "
        "error recovery, tool discovery, advanced cases (async tools, "
        "tools that raise, tools with complex signatures)."
    ),
}

EXAMPLE_TOPICS: dict[str, str] = {
    "agents": (
        "Multi-agent patterns: supervisor routing, debate, parallel "
        "sampling, LLM council voting, human-in-the-loop, agent skills "
        "from the AgentSkills spec."
    ),
    "evaluation": (
        "Reflection workflows, dataset evaluators, custom metrics, "
        "step-level metrics, and the structured-output step type."
    ),
    "llm_inference": (
        "Direct LLM examples: OpenRouter / Azure / OpenAI-compatible "
        "configurations, execution modes (FAST vs SELF_CRITIC), "
        "streaming, council voting across multiple models."
    ),
    "orchestration": (
        "Core chain mechanics: basic chains, parallel branches, "
        "conditional routing, loops, transforms — the entry-point set "
        "for new users."
    ),
    "replan": (
        "Runtime replanning variations: deterministic rule-based, "
        "LLM-based, voting-based, checkpoint rollback, and budget "
        "limits."
    ),
    "skills": (
        "AgentSkill integration examples — loading skills from "
        "`github://`, `local://`, and `module://` URIs."
    ),
    "tool_calling": (
        "Tool-step usage: registering Python functions, dynamic "
        "argument mapping, tool error recovery, tool discovery."
    ),
}


def extract_first_paragraph(file_path: Path) -> str:
    """Return the first non-empty paragraph of the module docstring.

    Falls back to the empty string when the file has no docstring or
    cannot be parsed. Strips trailing punctuation noise but otherwise
    preserves the original wording.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    # Take the first paragraph (split on blank line).
    first = doc.strip().split("\n\n", 1)[0]
    # Collapse internal whitespace + line breaks within the paragraph.
    return " ".join(first.split())


# Title-case overrides for acronyms and stylised names.
TITLE_OVERRIDES: dict[str, str] = {
    "mcp": "MCP",
    "llm_inference": "LLM Inference",
}


def _format_title(name: str) -> str:
    if name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[name]
    return name.replace("_", " ").title()


def render_readme(topic_dir: Path, blurb: str, kind: str) -> str:
    """Build the README body for one topic directory."""
    title = _format_title(topic_dir.name)
    lines: list[str] = [
        f"# {title} {kind}",
        "",
        blurb,
        "",
        "## Files",
        "",
    ]
    py_files = sorted(
        p for p in topic_dir.glob("*.py")
        if p.name != "__init__.py" and not p.name.startswith("_")
    )
    if not py_files:
        lines.append("_(no files yet)_")
    else:
        for f in py_files:
            summary = extract_first_paragraph(f) or "(no description)"
            lines.append(f"- `{f.name}` — {summary}")
    lines.append("")
    lines.append(
        "_This file is generated by `scripts/generate_topic_readmes.py`. "
        "Run `make docs-topic-index` after adding or removing files._"
    )
    return "\n".join(lines) + "\n"


def regenerate(root: Path, topics: dict[str, str], kind: str) -> list[Path]:
    """Write README.md into every topic directory; return touched paths."""
    written: list[Path] = []
    for topic_name, blurb in topics.items():
        topic_dir = root / topic_name
        if not topic_dir.is_dir():
            print(
                f"warning: {kind.lower()} topic '{topic_name}' not found at "
                f"{topic_dir} — skipping"
            )
            continue
        readme = topic_dir / "README.md"
        body = render_readme(topic_dir, blurb, kind)
        readme.write_text(body, encoding="utf-8")
        written.append(readme)
    return written


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    tests_root = repo_root / "tests"
    examples_root = repo_root / "examples"

    print(dedent(
        """
        Regenerating per-topic READMEs…
        """
    ).strip())
    test_files = regenerate(tests_root, TEST_TOPICS, kind="Tests")
    example_files = regenerate(examples_root, EXAMPLE_TOPICS, kind="Examples")

    for p in test_files + example_files:
        print(f"  wrote {p.relative_to(repo_root)}")

    print(
        f"\nDone — {len(test_files)} test READMEs, "
        f"{len(example_files)} example READMEs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
