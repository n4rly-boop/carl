"""Live validation for multi-objective ChainEvolver.

Runs a small 2-gen × 2-pop evolution against OpenRouter
``qwen/qwen3-8b`` with TWO metrics:
  - ``accuracy`` — 1.0 when the answer contains the expected digit.
  - ``brevity`` — 1.0 when the answer is ≤ 60 tokens, decaying past that.

The composite fitness rewards accurate AND concise answers, exposing
the speed/quality trade-off that single-metric evolution can't surface.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from mmar_carl import (  # noqa: E402
    ChainEvolver,
    LLMStepDescription,
    OpenAIClientConfig,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.chain_evolution import ChainMutator, MutationKind  # noqa: E402
from mmar_carl.metrics import MetricBase  # noqa: E402
from mmar_carl.models.dataset import DataCase, SimpleDataset  # noqa: E402
from mmar_carl.models.results import ReasoningResult  # noqa: E402


class AccuracyMetric(MetricBase):
    @property
    def name(self) -> str:
        return "accuracy"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        if not isinstance(output, ReasoningResult) or not output.success:
            return 0.0
        text = output.get_final_output() or ""
        if case is None:
            return 0.0
        nums = re.findall(r"-?\d+", text)
        return 1.0 if case.label in nums else 0.0


class BrevityMetric(MetricBase):
    @property
    def name(self) -> str:
        return "brevity"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        if not isinstance(output, ReasoningResult) or not output.success:
            return 0.0
        completion = int(output.token_usage.get("completion", 0))
        # 1.0 for ≤60 completion tokens; exponential decay past that.
        if completion <= 60:
            return 1.0
        return math.exp(-(completion - 60) / 120.0)


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

    base = ReasoningChain(steps=[
        LLMStepDescription(
            number=1, title="Solve",
            aim="Answer the math question. Reply with the integer only.",
        ),
    ])
    dataset = SimpleDataset([
        DataCase(input="What is 7 * 8?", label="56"),
        DataCase(input="What is 100 - 37?", label="63"),
    ])

    # Composite: 70% accuracy, 30% brevity.
    def fitness(scores: dict[str, float]) -> float:
        return 0.7 * scores["accuracy"] + 0.3 * scores["brevity"]

    evolver = ChainEvolver(
        base_chain=base,
        dataset=dataset,
        metric=[AccuracyMetric(), BrevityMetric()],
        fitness_fn=fitness,
        mutator=ChainMutator(
            aim_suffix_pool=[" Be brief.", " Reply with only the number."],
            enabled_kinds=[MutationKind.PROMPT_REWRITE],
        ),
        population_size=2,
        generations=2,
        elitism=1,
        rng=random.Random(0),
        smoke_check=False,
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    print("Running 2-gen × 2-pop multi-objective evolution against qwen/qwen3-8b…")
    result = await evolver.evolve(context_factory=factory)
    print(f"best fitness: {result.best_score:.3f} @ generation {result.best_generation}")

    print()
    print("=== Per-generation per-individual breakdown ===")
    for gen in result.history:
        print(f"\nGeneration {gen.generation}")
        for i, im in enumerate(gen.population_metrics):
            sbm = im.scores_by_metric
            print(
                f"  ind#{i}  fitness={im.score:.3f}  "
                f"acc={sbm.get('accuracy', 0):.2f}  "
                f"brev={sbm.get('brevity', 0):.2f}  "
                f"tokens={im.total_tokens}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
