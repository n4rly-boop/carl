"""Live validation for `format_step_metric_heatmap` against OpenRouter.

Runs a 2-step chain (plan -> draft) across 3 cases of varying difficulty.
A simple LengthBonus metric scores each step's output; the heatmap should
show step 2 (draft, longer outputs) scoring higher than step 1 (plan).
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
from mmar_carl.models.results import ReasoningResult, StepExecutionResult  # noqa: E402


class LengthScore(MetricBase):
    """Score = min(1.0, len(output) / 400). Trivial but lets us see
    real per-step variation in the heatmap."""

    @property
    def name(self) -> str:
        return "length"

    async def compute_async(self, output) -> float:  # type: ignore[override]
        if isinstance(output, StepExecutionResult):
            text = output.result or ""
        elif isinstance(output, ReasoningResult):
            text = output.get_final_output() or ""
        else:
            text = str(output)
        return min(1.0, len(text) / 400.0)


async def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set, skipping live test")
        return 0

    model = "qwen/qwen3-8b"
    client = OpenAICompatibleClient(OpenAIClientConfig(
        model=model,
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL"),
    ))

    chain = ReasoningChain(steps=[
        LLMStepDescription(
            number=1, title="Plan",
            aim="Outline the key points (3-5 bullets only).",
            metrics=[LengthScore()],
        ),
        LLMStepDescription(
            number=2, title="Draft",
            aim="Write a paragraph based on the outline.",
            dependencies=[1],
            metrics=[LengthScore()],
        ),
    ])

    dataset = SimpleDataset([
        DataCase(input="Explain why unit tests matter.", label="tests"),
        DataCase(input="Describe the Pythagorean theorem.", label="pythag"),
        DataCase(input="Summarise the plot of Hamlet in three sentences.", label="hamlet"),
    ])

    evaluator = DatasetEvaluator(
        chain=chain,
        dataset=dataset,
        metric=LengthScore(),
        strategy=ThresholdStrategy(threshold=0.5),
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    report = await evaluator.evaluate_async(factory)

    print("=== step_metrics per case ===")
    for cr in report.all_results:
        print(f"  case '{cr.case.label}' success={cr.success} score={cr.score:.2f}")
        for step_num, mmap in sorted(cr.step_metrics.items()):
            print(f"    step {step_num}: {mmap}")

    print()
    print("=== format_step_metric_heatmap (autorange) ===")
    print(report.format_step_metric_heatmap("length"))

    print()
    print("=== format_step_metric_heatmap (scale 0..1) ===")
    print(report.format_step_metric_heatmap("length", scale_min=0.0, scale_max=1.0))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
