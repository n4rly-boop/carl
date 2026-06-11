"""Build ``notebooks/02_visualizations_demo.ipynb`` programmatically.

Authoring a notebook through a Python script keeps the source diffable,
lets us regenerate it from a single place after a viz API change, and
avoids the noisy ``execution_count``/``metadata`` churn that comes from
re-saving an interactively-edited notebook.

Invoked by ``make notebook-build``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/02_visualizations_demo.ipynb")


_cell_counter = 0


def _next_id() -> str:
    global _cell_counter
    _cell_counter += 1
    # nbformat 5.1+ requires a unique id per cell; deterministic IDs keep
    # rebuilds stable in git.
    return f"cell-{_cell_counter:03d}"


def md(*lines: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": "\n".join(lines),
    }


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "\n".join(lines),
    }


CELLS: list[dict] = [
    md(
        "# CARL Visualizations Demo",
        "",
        "Walks through every visualization shipped in CARL by running",
        "a real chain (or replaying from a cassette) and showing both the",
        "**text** form (committed to git so this notebook reads on GitHub",
        "without dependency installs) and the **mermaid** form (rendered in",
        "Jupyter when `mmar-carl[viz]` is installed).",
        "",
        "**Design**: each section is self-contained — you can jump to any",
        "cell and run from there, provided the **Setup** cell has been run.",
    ),

    md(
        "## Setup",
        "",
        "Set `RUN_LIVE = True` to hit OpenRouter (requires `OPENAI_API_KEY`),",
        "or leave `False` to replay from `notebooks/cassettes/*.jsonl`.",
        "Cassette mode runs in <1 s and costs nothing — perfect for CI and",
        "for readers who don't have an API key handy.",
    ),
    code(
        "import os",
        "from pathlib import Path",
        "",
        "from IPython.display import Markdown, display",
        "",
        "from mmar_carl import (",
        "    ChainVisualizer,",
        "    LLMStepDescription,",
        "    OpenAIClientConfig,",
        "    OpenAICompatibleClient,",
        "    PlayingLLMClient,",
        "    ReasoningChain,",
        "    ReasoningContext,",
        "    RecordingLLMClient,",
        ")",
        "",
        "",
        "def show_mermaid(src: str) -> None:",
        "    \"\"\"Render a Mermaid diagram inline.",
        "",
        "    Emits a markdown cell with a fenced ```mermaid block — JupyterLab",
        "    4+, GitHub, and nbviewer all render this natively. Falls back to",
        "    plain text in viewers without Mermaid support.\"\"\"",
        "    display(Markdown(f'```mermaid\\n{src}\\n```'))",
        "",
        "",
        "RUN_LIVE = False  # flip to True to hit the real API",
        "MODEL = 'qwen/qwen3-8b'",
        "CASSETTES = Path('cassettes')",
        "CASSETTES.mkdir(exist_ok=True)",
        "",
        "",
        "def build_client(cassette_name: str):",
        "    \"\"\"Return a client wired for record/replay based on RUN_LIVE.\"\"\"",
        "    cassette = CASSETTES / cassette_name",
        "    if RUN_LIVE:",
        "        api_key = os.environ.get('OPENAI_API_KEY')",
        "        if not api_key:",
        "            raise RuntimeError('Set OPENAI_API_KEY or use RUN_LIVE=False')",
        "        real = OpenAICompatibleClient(OpenAIClientConfig(",
        "            model=MODEL,",
        "            api_key=api_key,",
        "            base_url=os.environ.get('OPENAI_BASE_URL'),",
        "        ))",
        "        return RecordingLLMClient(real, cassette, overwrite=True)",
        "    if not cassette.exists():",
        "        raise FileNotFoundError(",
        "            f'Cassette {cassette} not found — re-run this notebook with '",
        "            'RUN_LIVE=True to record it first.'",
        "        )",
        "    return PlayingLLMClient(cassette)",
        "",
        "print(f'RUN_LIVE = {RUN_LIVE}  |  MODEL = {MODEL!r}')",
        "print(f'Cassettes dir: {CASSETTES.resolve()}')",
    ),

    md(
        "## Cost & tokens",
        "",
        "A 2-step chain: a cheap planner followed by a synth step. The",
        "visualizations summarise where tokens went, the prompt/completion",
        "split, and (with a `pricing=` map) the dollar cost.",
    ),
    code(
        "chain_cost = ReasoningChain(steps=[",
        "    LLMStepDescription(",
        "        number=1, title='Outline',",
        "        aim='List 3 reasons to write unit tests.',",
        "    ),",
        "    LLMStepDescription(",
        "        number=2, title='Synthesise',",
        "        aim='Summarise the outline in one sentence.',",
        "        dependencies=[1],",
        "    ),",
        "])",
        "",
        "client_cost = build_client('cost_demo.jsonl')",
        "ctx_cost = ReasoningContext(outer_context='N/A', api=client_cost)",
        "# Top-level await works in modern Jupyter; outside Jupyter use asyncio.run().",
        "result_cost = await chain_cost.execute_async(ctx_cost)",
        "print(f'success = {result_cost.success}')",
        "print(f'token usage = {result_cost.token_usage}')",
    ),
    code(
        "# Token pie — text (GitHub-friendly)",
        "print(result_cost.format_token_pie())",
    ),
    code(
        "# Token pie — Mermaid (rendered inline below)",
        "show_mermaid(result_cost.format_token_pie(format='mermaid'))",
    ),
    code(
        "# Prompt vs completion breakdown",
        "print(result_cost.format_prompt_completion_breakdown())",
    ),
    code(
        "# Per-model cost breakdown with a pricing map (OpenRouter list-prices in $/1k tokens)",
        "print(result_cost.format_cost_by_model(pricing={",
        "    'qwen/qwen3-8b': (0.00002, 0.00006),",
        "}))",
    ),

    md(
        "## Execution timing",
        "",
        "Same chain, viewed through `ExecutionTrace.format_gantt`. The",
        "Gantt chart reconstructs each step's `(start, end)` from",
        "`batch_index` + `execution_time` and lays them out so parallel",
        "batches are visually obvious.",
    ),
    code(
        "trace = result_cost.trace",
        "print(trace.format_gantt())",
    ),
    code(
        "# Same data as a Mermaid Gantt block — rendered inline",
        "show_mermaid(trace.format_gantt(format='mermaid'))",
    ),

    md(
        "## DAG enhancements",
        "",
        "Overlay tokens / latency / cost on the static DAG via",
        "`ReasoningChain.to_mermaid_heatmap(result, metric=...)`. Useful",
        "for instantly spotting hot edges in a multi-step pipeline.",
    ),
    code(
        "show_mermaid(chain_cost.to_mermaid_heatmap(result_cost, metric='tokens'))",
    ),
    code(
        "show_mermaid(chain_cost.to_mermaid_heatmap(result_cost, metric='latency'))",
    ),

    md(
        "## ChainVisualizer — composable framework",
        "",
        "`ChainVisualizer` is the **fluent facade** that lets a user build",
        "a single report from many viz methods in one call chain. Each",
        "builder appends a section to an internal buffer; `.render()`",
        "returns the concatenated string and `.print()` writes it to",
        "stdout.",
    ),
    code(
        "viz = (ChainVisualizer(result_cost, chain=chain_cost)",
        "       .token_pie()",
        "       .prompt_completion()",
        "       .gantt()",
        "       .heatmap(metric='tokens'))",
        "print('Sections accumulated:', viz.view_titles)",
        "print()",
        "viz.print()",
    ),

    md(
        "## Total spend",
        "",
        "Sums `result.token_usage` across every chain executed above so the",
        "reader knows what the demo cost (roughly $0 in cassette mode).",
    ),
    code(
        "totals = {'prompt': 0, 'completion': 0, 'total': 0}",
        "for r in [result_cost]:",
        "    for k in totals:",
        "        totals[k] += r.token_usage.get(k, 0)",
        "",
        "# Approximate cost using qwen/qwen3-8b list price.",
        "cost = totals['prompt'] / 1000 * 0.00002 + totals['completion'] / 1000 * 0.00006",
        "print(f'Total tokens: {totals}')",
        "print(f'Est. cost (qwen/qwen3-8b list price): ${cost:.4f}')",
    ),

    md(
        "## What's next",
        "",
        "Sections not yet exercised here but covered by their respective",
        "viz methods elsewhere in the codebase:",
        "",
        "- **Evolution** — `EvolutionResult.format_score_evolution`,",
        "  `format_pareto`, `format_mutation_effectiveness`. Demo in",
        "  `scripts/evolution_benchmark.py`.",
        "- **Quality** — `DatasetEvaluationReport.format_failure_heatmap`,",
        "  `format_step_metric_heatmap`. Demo in",
        "  `scripts/validate_step_metric_heatmap.py`.",
        "- **trace-to-html** — `ExecutionTrace.to_html()` produces a",
        "  static HTML page you can drop into any reviewer-facing dashboard.",
        "",
        "Re-run this notebook with `RUN_LIVE = True` once an `OPENAI_API_KEY`",
        "(OpenRouter-compatible) is set to refresh the cassettes.",
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12",
        },
        # Consumed by scripts/generate_notebooks_readme.py — keep keys stable.
        "demo_info": {
            "title": "CARL Visualizations Demo",
            "summary": (
                "Walks through every visualization shipped in CARL by "
                "running a real chain (or replaying from a cassette) and "
                "showing both the text form and the rendered Mermaid form."
            ),
            "runtime_offline": "<5 s (cassette mode)",
            "runtime_live": "~15 s (live OpenRouter, qwen/qwen3-8b)",
            "estimated_cost": "~$0.0001 per live run",
            "prerequisites": [
                "mmar-carl",
                "ipykernel, nbformat, nbclient (for `make notebook-smoke`)",
                "OPENAI_API_KEY (only for live mode; cassette mode needs nothing)",
            ],
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / NOTEBOOK_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out_path.relative_to(repo_root)} "
          f"({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
