"""Live validation for `format_cost_by_model` against OpenRouter."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    LLMStepConfig,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
)


async def main() -> int:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping live test")
        return 0

    cheap_model = "qwen/qwen3-8b"
    pricey_model = "deepseek/deepseek-chat-v3.1"

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1,
                title="Plan (cheap)",
                aim="List three benefits of unit tests.",
                llm_config=LLMStepConfig(model=cheap_model),
            ),
            LLMStepDescription(
                number=2,
                title="Synth (pricey)",
                aim="Summarize the previous step in one sentence.",
                dependencies=[1],
                llm_config=LLMStepConfig(model=pricey_model),
            ),
        ],
    )

    client = OpenAICompatibleClient(OpenAIClientConfig(
        model=cheap_model,
        api_key=api_key,
        base_url=base_url,
    ))

    ctx = ReasoningContext(outer_context="N/A", api=client)
    result = await chain.execute_async(ctx)

    print("=== Step models ===")
    for sr in result.step_results:
        print(f"  step {sr.step_number} '{sr.step_title}' success={sr.success} -> "
              f"model={sr.model!r} tokens={sr.token_usage}")

    if not result.success:
        print("Chain failed:", [s.error_message for s in result.get_failed_steps()])
        # Continue rendering with whatever we have

    print()
    print("=== format_cost_by_model (text) ===")
    print(result.format_cost_by_model(pricing={
        cheap_model: (0.00002, 0.00006),
        pricey_model: (0.00015, 0.0006),
    }))

    print()
    print("=== format_cost_by_model (mermaid) ===")
    print(result.format_cost_by_model(format="mermaid"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
