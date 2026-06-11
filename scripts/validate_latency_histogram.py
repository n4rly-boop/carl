"""Live validation for ``DatasetEvaluationReport.format_latency_histogram``.

Runs a 2-step chain over a small dataset against OpenRouter
``qwen/qwen3-8b`` and renders per-step latency histograms across the
real runs.
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
)
from mmar_carl.dataset_evaluator import DatasetEvaluator  # noqa: E402
from mmar_carl.metrics import MetricBase  # noqa: E402
from mmar_carl.models.dataset import (  # noqa: E402
    DataCase,
    SimpleDataset,
    ThresholdStrategy,
)
from mmar_carl.models.results import ReasoningResult  # noqa: E402


class TrivialMetric(MetricBase):
    @property
    def name(self) -> str:
        return "trivial"

    async def compute_async(self, output) -> float:  # type: ignore[override]
        if isinstance(output, ReasoningResult):
            return 1.0 if output.success else 0.0
        return 1.0


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
            number=1, title="Plan",
            aim="List 2-3 bullet-point steps to solve the question.",
        ),
        LLMStepDescription(
            number=2, title="Solve",
            aim="Produce the final answer.",
            dependencies=[1],
        ),
    ])

    questions = [
        "What is the capital of France?",
        "What is 12 * 11?",
        "Name a planet in our solar system.",
        "Spell the word 'hello' backwards.",
        "What year did WW2 end?",
        "Convert 100 °C to °F.",
        "How many continents are there?",
        "What is the square root of 144?",
    ]
    dataset = SimpleDataset([
        DataCase(input=q, label=f"q{i}") for i, q in enumerate(questions)
    ])

    evaluator = DatasetEvaluator(
        chain=chain, dataset=dataset, metric=TrivialMetric(),
        strategy=ThresholdStrategy(threshold=0.0),
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    print(f"Evaluating {len(questions)} cases against qwen/qwen3-8b…")
    report = await evaluator.evaluate_async(factory)

    print()
    print("=== Per-case step latencies (ms) ===")
    for cr in report.all_results:
        lats = " | ".join(
            f"step {n}={ms:.0f}ms"
            for n, ms in sorted(cr.step_latencies_ms.items())
        )
        print(f"  {cr.case.label}: {lats}")

    print()
    print("=== format_latency_histogram (default 16 bins) ===")
    print(report.format_latency_histogram())

    print()
    print("=== format_latency_histogram (8 bins) ===")
    print(report.format_latency_histogram(bins=8))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
