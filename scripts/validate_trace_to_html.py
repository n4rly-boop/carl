"""Live validation for ``ExecutionTrace.to_html``.

Runs a real chain against OpenRouter, persists the animated HTML
playback, and reports basic shape stats (file size, embedded events
count). The resulting file can be opened directly in any browser —
no server, no external dependencies.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
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
            number=2, title="Refine",
            aim="Sharpen the wording of each bullet.",
            dependencies=[1],
        ),
        LLMStepDescription(
            number=3, title="Synthesise",
            aim="Summarise the refined list in one sentence.",
            dependencies=[2],
        ),
    ])
    ctx = ReasoningContext(outer_context="N/A", api=client)
    result = await chain.execute_async(ctx)
    print(f"chain success={result.success}, tokens={result.token_usage}")

    out = Path(tempfile.gettempdir()) / "carl_playback_demo.html"
    html = result.trace.to_html(out)

    # Shape report
    n_events_match = re.search(r"const EVENTS = (\[[\s\S]*?\]);", html)
    assert n_events_match, "EVENTS payload missing from HTML"
    events_blob = n_events_match.group(1)
    print()
    print("=== ExecutionTrace.to_html() ===")
    print(f"  output file: {out}")
    print(f"  file size:   {out.stat().st_size:,} bytes")
    print(f"  HTML lines:  {html.count(chr(10)):,}")
    print(f"  events payload: {len(events_blob):,} chars")
    # Sanity check: each step's title appears in the HTML
    for sr in result.step_results:
        assert sr.step_title in html, f"missing {sr.step_title!r} in HTML"
    print(f"  contains all {len(result.step_results)} step titles ✓")
    print(f"\nOpen with: open {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
