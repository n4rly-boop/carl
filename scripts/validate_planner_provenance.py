"""Live validation for planner-provenance capture.

Generates a chain from a natural-language description against
OpenRouter ``qwen/qwen3-8b`` and prints back the provenance captured
in ``chain.metadata``: the planner prompt (truncated), the raw reply,
and the per-attempt log so a failure can be diagnosed offline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    ChainBuilder,
    OpenAIClientConfig,
    OpenAICompatibleClient,
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

    task = (
        "Build a 2-step chain that (a) outlines the key arguments in "
        "the input text and (b) condenses them into a single paragraph."
    )
    print(f"Planning chain for task: {task[:70]}…")
    chain = await ChainBuilder.from_description(
        task=task, llm_client=client, max_steps=4, max_retries=2,
    )

    print(f"\nGenerated chain: {len(chain.steps)} step(s)")
    for s in chain.steps:
        title = getattr(s, "title", "?")
        print(f"  step {s.number}: {title}")

    md = chain.metadata
    print()
    print("=== Captured provenance keys ===")
    for key in sorted(md.keys()):
        v = md[key]
        if isinstance(v, list):
            print(f"  {key}: list of {len(v)} item(s)")
        elif isinstance(v, str):
            print(f"  {key}: {len(v)} chars")
        else:
            print(f"  {key}: {type(v).__name__}")

    print()
    print("=== planner_prompt (first 200 chars) ===")
    print(md["planner_prompt"][:200])

    print()
    print("=== planner_reply (first 200 chars) ===")
    print(md["planner_reply"][:200])

    print()
    print("=== planner_attempts ===")
    for a in md["planner_attempts"]:
        err = a["error"] or "(none)"
        print(f"  attempt {a['attempt']}: error={err[:60]} reply_len={len(a['reply'])}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
