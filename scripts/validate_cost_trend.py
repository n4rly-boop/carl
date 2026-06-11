"""Live validation for ``DatasetEvaluationReport.format_cost_trend``.

Evaluates a 1-step chain over 6 prompts where one prompt is much
longer than the rest — the cost-trend sparkline should highlight that
run and the regression detector should flag it.
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
        return 1.0 if isinstance(output, ReasoningResult) and output.success else 0.0


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
            number=1, title="Answer",
            aim="Answer the user's question briefly.",
        ),
    ])

    # 5 short prompts + 1 deliberately long prompt to trigger the
    # regression detector. The long prompt should inflate prompt tokens
    # and likely also completion tokens (the model has more to chew on).
    long_prompt = (
        "Below is a list of trivia categories. Pick three and write one "
        "two-sentence question and answer pair for each. Categories: "
        + ", ".join([f"Topic-{i}" for i in range(30)])
        + ". Be thorough."
    )
    questions = [
        ("q1", "What is 2 + 2?"),
        ("q2", "What is the capital of France?"),
        ("q3", "Name a primary color."),
        ("q4", "What year did WW2 end?"),
        ("longy", long_prompt),
        ("q5", "Spell 'cat' backwards."),
    ]
    dataset = SimpleDataset([
        DataCase(input=q, label=lbl) for lbl, q in questions
    ])

    evaluator = DatasetEvaluator(
        chain=chain, dataset=dataset, metric=TrivialMetric(),
        strategy=ThresholdStrategy(threshold=0.0),
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    print(f"Evaluating {len(questions)} cases against {model}…")
    report = await evaluator.evaluate_async(factory)

    print()
    print("=== Per-case token usage ===")
    for cr in report.all_results:
        u = cr.token_usage
        print(f"  {cr.case.label}: prompt={u.get('prompt', 0)} "
              f"completion={u.get('completion', 0)} total={u.get('total', 0)}")

    print()
    print("=== format_cost_trend (tokens-only) ===")
    print(report.format_cost_trend())

    print()
    print("=== format_cost_trend (with pricing) ===")
    print(report.format_cost_trend(
        pricing={model: (0.00002, 0.00006)},
        default_model=model,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
