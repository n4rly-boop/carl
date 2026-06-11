"""Live validation for the Jupyter rich-display protocol.

Runs a real OpenRouter chain, prints the ``_repr_markdown_`` outputs
for ``ReasoningResult`` and ``ChainVisualizer``, and verifies that a
notebook-style rendering pipeline (``IPython.display.Markdown``)
would receive valid Markdown content.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    ChainVisualizer,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
)


async def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping live test")
        return 0

    client = OpenAICompatibleClient(OpenAIClientConfig(
        model="qwen/qwen3-8b",
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    ))
    chain = ReasoningChain(steps=[
        LLMStepDescription(number=1, title="Plan",
                            aim="State the steps to solve a sample task."),
        LLMStepDescription(number=2, title="Solve",
                            aim="Produce a final answer.", dependencies=[1]),
    ])
    ctx = ReasoningContext(outer_context="What is 12 * 11?", api=client)
    result = await chain.execute_async(ctx)
    print(f"chain success={result.success}, tokens={result.token_usage}")
    print()

    print("=== ReasoningResult._repr_markdown_ ===")
    print(result._repr_markdown_())
    print()

    print("=== ChainVisualizer._repr_markdown_ (token_pie + gantt + heatmap) ===")
    viz = (
        ChainVisualizer(result, chain=chain)
        .token_pie(format="mermaid")
        .gantt(format="mermaid")
        .heatmap(metric="tokens")
    )
    print(viz._repr_markdown_())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
