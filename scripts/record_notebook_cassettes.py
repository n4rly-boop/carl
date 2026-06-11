"""Record cassettes consumed by ``notebooks/02_visualizations_demo.ipynb``.

Run once with an OpenRouter API key set; commits one JSONL per chain
under ``notebooks/cassettes/``. The notebook then replays from those
files in cassette mode (``RUN_LIVE = False``).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    RecordingLLMClient,
)

MODEL = "qwen/qwen3-8b"
CASSETTES = Path(__file__).resolve().parents[1] / "notebooks" / "cassettes"


def real_client() -> OpenAICompatibleClient:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set — aborting.")
        sys.exit(1)
    return OpenAICompatibleClient(OpenAIClientConfig(
        model=MODEL,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    ))


async def record_cost_demo() -> None:
    chain = ReasoningChain(steps=[
        LLMStepDescription(
            number=1, title="Outline",
            aim="List 3 reasons to write unit tests.",
        ),
        LLMStepDescription(
            number=2, title="Synthesise",
            aim="Summarise the outline in one sentence.",
            dependencies=[1],
        ),
    ])
    CASSETTES.mkdir(parents=True, exist_ok=True)
    cassette = CASSETTES / "cost_demo.jsonl"
    rec = RecordingLLMClient(real_client(), cassette, overwrite=True)
    ctx = ReasoningContext(outer_context="N/A", api=rec)
    result = await chain.execute_async(ctx)
    if not result.success:
        print("cost_demo recording failed:", result.get_failed_steps())
        sys.exit(2)
    print(f"recorded {cassette.name}: {rec.cassette_size} entries, "
          f"tokens={result.token_usage}")


async def main() -> None:
    await record_cost_demo()
    print("\nAll cassettes recorded.")


if __name__ == "__main__":
    asyncio.run(main())
