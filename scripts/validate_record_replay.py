"""Live validation for RecordingLLMClient + PlayingLLMClient.

Phase 1: run a chain against OpenRouter, recording to a temp cassette.
Phase 2: re-run the same chain backed only by the cassette (no API key
required) and confirm outputs are byte-identical.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    PlayingLLMClient,
    ReasoningChain,
    ReasoningContext,
    RecordingLLMClient,
)


def _make_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(
            number=1, title="Brainstorm",
            aim="List 3 reasons to use unit tests.",
        ),
        LLMStepDescription(
            number=2, title="Summarize",
            aim="Summarize the previous list in one sentence.",
            dependencies=[1],
        ),
    ])


async def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping live test")
        return 0

    model = "qwen/qwen3-8b"
    base_url = os.environ.get("OPENAI_BASE_URL")
    cassette = Path(tempfile.mkdtemp()) / "validate.jsonl"

    # Phase 1: real call wrapped in RecordingLLMClient
    real = OpenAICompatibleClient(OpenAIClientConfig(
        model=model, api_key=api_key, base_url=base_url,
    ))
    rec = RecordingLLMClient(real, cassette, overwrite=True)

    t0 = time.time()
    ctx_rec = ReasoningContext(outer_context="N/A", api=rec)
    result_rec = await _make_chain().execute_async(ctx_rec)
    t_record = time.time() - t0

    if not result_rec.success:
        print("Recording phase failed:", result_rec.get_failed_steps())
        return 1

    print(f"=== Phase 1: recording ({t_record:.1f}s) ===")
    print(f"  cassette: {cassette}")
    print(f"  cassette_size: {rec.cassette_size} entries")
    print(f"  final output (truncated): {result_rec.get_final_output()[:120]!r}")
    print(f"  total tokens: {result_rec.token_usage}")

    # Phase 2: replay, no API key needed
    play = PlayingLLMClient(cassette)
    t0 = time.time()
    ctx_play = ReasoningContext(outer_context="N/A", api=play)
    result_play = await _make_chain().execute_async(ctx_play)
    t_replay = time.time() - t0

    print()
    print(f"=== Phase 2: replay ({t_replay:.3f}s) ===")
    print(f"  cassette_size: {play.cassette_size} entries")
    print(f"  final output (truncated): {result_play.get_final_output()[:120]!r}")
    print(f"  speedup: {t_record / max(t_replay, 1e-6):.0f}x")

    # Byte-identical outputs
    same = result_rec.get_final_output() == result_play.get_final_output()
    print()
    print("=== Determinism check ===")
    print(f"  identical final outputs: {same}")
    print(f"  identical step outputs:  "
          f"{all(a.result == b.result for a, b in zip(result_rec.step_results, result_play.step_results))}")
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
