"""Live validation for streaming-first chain execution.

Runs a 3-step linear chain against OpenRouter ``qwen/qwen3-8b``
via the new ``chain.stream_async`` async-generator and prints the
wall-clock arrival time of each ``StepExecutionResult``. Confirms
that step 1's output is visible to the consumer long before the
chain's terminal ``ReasoningResult`` arrives.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.results import ReasoningResult, StepExecutionResult  # noqa: E402


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
        LLMStepDescription(
            number=1, title="Outline",
            aim="List 3 ideas for a children's storybook.",
        ),
        LLMStepDescription(
            number=2, title="Refine",
            aim="Pick the best idea and flesh out the plot.",
            dependencies=[1],
        ),
        LLMStepDescription(
            number=3, title="Synthesise",
            aim="Write a 3-paragraph opening for the chosen story.",
            dependencies=[2],
        ),
    ])
    ctx = ReasoningContext(outer_context="N/A", api=client)

    print("Streaming a 3-step chain via chain.stream_async()…")
    print("(step results should arrive incrementally, not all at the end)")
    print()

    start = time.time()
    step_arrivals: list[tuple[int, float, int]] = []
    final: ReasoningResult | None = None

    async for item in chain.stream_async(ctx):
        elapsed = time.time() - start
        if isinstance(item, StepExecutionResult):
            tokens = item.token_usage.get("total", 0)
            print(f"  t={elapsed:5.2f}s  step {item.step_number} '{item.step_title}' "
                  f"done — {tokens} tokens")
            step_arrivals.append((item.step_number, elapsed, tokens))
        else:
            final = item
            print(f"  t={elapsed:5.2f}s  FINAL ReasoningResult success={final.success} "
                  f"total_tokens={final.token_usage.get('total', 0)}")

    print()
    if step_arrivals and final:
        first_step_at = step_arrivals[0][1]
        final_at = (time.time() - start)
        ratio = first_step_at / final_at if final_at else 0
        print("=== Streaming analysis ===")
        print(f"  step 1 visible at:   {first_step_at:.2f}s")
        print(f"  final result at:     {final_at:.2f}s")
        print(f"  step 1 visible at {ratio:.0%} of total wall time — "
              f"vs. 100% for non-streaming execute_async()")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
