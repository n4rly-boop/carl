"""Live validation for ``DatasetEvaluationReport.format_score_distribution``.

Runs a one-step LLM chain over 8 arithmetic problems of varying
difficulty using OpenRouter ``qwen/qwen3-8b`` and scores each with an
exact-answer metric. The resulting score distribution exercises the
box-plot path with a real (non-uniform) score spread.
"""

from __future__ import annotations

import asyncio
import os
import re
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


class ExactAnswerMetric(MetricBase):
    """Scores 1.0 if the expected answer appears as a digit run in the
    final output, else 0.0. Tolerant of surrounding prose."""

    @property
    def name(self) -> str:
        return "exact_answer"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        if isinstance(output, ReasoningResult):
            text = output.get_final_output() or ""
        else:
            text = str(output)
        if case is None:
            return 0.0
        expected = case.label
        nums = re.findall(r"-?\d+", text)
        return 1.0 if expected in nums else 0.0


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
            number=1, title="Solve",
            aim=("Solve the math problem in the outer context. "
                 "Reply with the integer answer only — no words, no units."),
        ),
    ])

    # Mix of trivial / medium / hard problems that produce a real score
    # spread (Qwen-8b solves easy arithmetic but struggles with deeper
    # multi-step or tricky-format problems).
    dataset = SimpleDataset([
        # Easy — Qwen should solve all of these
        DataCase(input="What is 2 + 2?", label="4"),
        DataCase(input="What is 9 - 5?", label="4"),
        # Medium
        DataCase(input="What is 13 * 17?", label="221"),
        DataCase(input="What is 256 / 16?", label="16"),
        # Hard — chains of operations where Qwen often slips
        DataCase(
            input=(
                "A train leaves Boston at 8am going 60 mph. "
                "Another leaves NYC at 9am going 70 mph toward Boston. "
                "If Boston and NYC are 200 miles apart, how many miles "
                "from Boston do they meet? Reply with the integer only."
            ),
            label="111",  # roughly — model will likely overshoot
        ),
        DataCase(
            input=(
                "A baker uses 3 eggs per cake. She made cakes for 12 "
                "days, increasing by 2 cakes each day. She started with "
                "5 cakes on day 1. How many eggs did she use in total? "
                "Reply with the integer only."
            ),
            label="612",
        ),
        DataCase(
            input=(
                "Compute the sum of all even numbers from 2 to 100 "
                "inclusive. Reply with the integer only."
            ),
            label="2550",
        ),
        DataCase(
            input=(
                "What is the 7th prime number? "
                "Reply with the integer only."
            ),
            label="17",
        ),
    ])

    evaluator = DatasetEvaluator(
        chain=chain, dataset=dataset, metric=ExactAnswerMetric(),
        strategy=ThresholdStrategy(threshold=0.5),
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    report = await evaluator.evaluate_async(factory)

    print("=== Per-case scores ===")
    for r in report.all_results:
        print(f"  {r.case.label:>4} ← {r.case.input[:36]:<38}  score={r.score:.2f}")

    print()
    print("=== format_score_distribution (default 40-char canvas) ===")
    print(report.format_score_distribution())

    print()
    print("=== format_score_distribution (60-char canvas) ===")
    print(report.format_score_distribution(canvas_width=60))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
