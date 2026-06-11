"""Live validation for ``TraceAggregator``.

Runs the same 2-step chain N times against OpenRouter ``qwen/qwen3-8b``
and feeds every ``ReasoningResult.trace`` into ``TraceAggregator`` to
produce a per-step latency + token-usage report.
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
    TraceAggregator,
)
from mmar_carl.execution_trace import ExecutionTrace  # noqa: E402

N_RUNS = 5


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
            aim="List 3 reasons to write unit tests.",
        ),
        LLMStepDescription(
            number=2, title="Synth",
            aim="Summarise the outline in one sentence.",
            dependencies=[1],
        ),
    ])

    print(f"Running chain {N_RUNS} times against qwen/qwen3-8b...")
    traces: list[ExecutionTrace] = []
    for i in range(N_RUNS):
        t0 = time.time()
        ctx = ReasoningContext(outer_context="N/A", api=client)
        result = await chain.execute_async(ctx)
        wall = time.time() - t0
        ok = "✓" if result.success else "✗"
        tokens = result.token_usage.get("total", 0)
        print(f"  run {i + 1}: {ok}  wall={wall:.2f}s  tokens={tokens}")
        if result.success:
            traces.append(result.trace)

    if not traces:
        print("All runs failed — nothing to aggregate.")
        return 1

    agg = TraceAggregator(traces)

    print()
    print("=== TraceAggregator.format_text() ===")
    print(agg.format_text())

    print()
    print("=== Programmatic access ===")
    for n in agg.step_numbers:
        lat = agg.latency_ms.get(n, {})
        tok = agg.tokens.get(n, {})
        print(f"  step {n}: latency p50={lat.get('p50', 0):.0f}ms "
              f"p95={lat.get('p95', 0):.0f}ms max={lat.get('max', 0):.0f}ms "
              f"| tokens p50={tok.get('p50', 0):.0f} p95={tok.get('p95', 0):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
