"""Live validation for ``EvolutionResult.to_lineage_mermaid``.

Runs a small 2-generation × 2-individual evolution against OpenRouter
``qwen/qwen3-8b`` and renders the resulting lineage tree. Keeps the
total cost small (~4 chain runs) so the script finishes in a couple
of minutes.
"""

from __future__ import annotations

import asyncio
import os
import random
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
from mmar_carl.chain_evolution import (  # noqa: E402
    ChainEvolver, ChainMutator, MutationKind,
)
from mmar_carl.metrics import MetricBase  # noqa: E402
from mmar_carl.models.dataset import DataCase, SimpleDataset  # noqa: E402
from mmar_carl.models.results import ReasoningResult  # noqa: E402


class HasDigitMetric(MetricBase):
    """1.0 when the final output contains the expected digit string."""

    @property
    def name(self) -> str:
        return "digit_match"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        if not isinstance(output, ReasoningResult) or not output.success:
            return 0.0
        text = output.get_final_output() or ""
        if case is None:
            return 0.0
        return 1.0 if case.label in text else 0.0


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
            aim="Reply with the integer answer only. No words.",
        ),
    ])
    dataset = SimpleDataset([
        DataCase(input="What is 2 + 2?", label="4"),
        DataCase(input="What is 5 * 6?", label="30"),
    ])

    evolver = ChainEvolver(
        base_chain=base,
        dataset=dataset,
        metric=HasDigitMetric(),
        mutator=ChainMutator(
            temperature_pool=[0.1, 0.5, 0.9],
            aim_suffix_pool=[" Be precise.", " Be brief."],
            enabled_kinds=[
                MutationKind.PROMPT_REWRITE, MutationKind.TEMPERATURE_SWAP,
            ],
        ),
        population_size=2,
        generations=2,
        elitism=1,
        rng=random.Random(0),
        smoke_check=False,
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    print("Running 2-gen × 2-pop evolution against qwen/qwen3-8b…")
    result = await evolver.evolve(context_factory=factory)
    print(f"best score: {result.best_score:.3f} @ generation {result.best_generation}")
    print(f"history: {len(result.history)} generations")

    print()
    print("=== to_lineage_mermaid() ===")
    print(result.to_lineage_mermaid())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
