"""Live validation for structural mutations.

Runs a small 2-gen × 3-pop evolution where the mutator can either
delete a leaf step or insert a verification step from a template
pool. Reports the per-individual mutation kind to confirm both new
operators fire end-to-end without breaking chain validation.
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


class SuccessMetric(MetricBase):
    """Fitness = 1.0 if chain succeeded and produced any output."""

    @property
    def name(self) -> str:
        return "success"

    async def compute_async(self, output, case=None) -> float:  # type: ignore[override]
        if not isinstance(output, ReasoningResult) or not output.success:
            return 0.0
        return 1.0 if (output.get_final_output() or "").strip() else 0.0


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

    # Three-step base chain so DELETE_STEP has a leaf to remove.
    base = ReasoningChain(steps=[
        LLMStepDescription(number=1, title="Outline", aim="Outline a plan."),
        LLMStepDescription(number=2, title="Solve", aim="Produce the answer.",
                            dependencies=[1]),
        LLMStepDescription(number=3, title="Polish", aim="Refine the answer.",
                            dependencies=[2]),
    ])
    dataset = SimpleDataset([DataCase(input="What is 2+2?", label="4")])

    mutator = ChainMutator(
        step_template_pool=[{
            "step_type": "llm",
            "title": "Verify",
            "aim": "Sanity-check the previous answer.",
        }],
        allow_step_deletion=True,
        enabled_kinds=[MutationKind.INSERT_STEP, MutationKind.DELETE_STEP],
    )
    evolver = ChainEvolver(
        base_chain=base, dataset=dataset,
        metric=SuccessMetric(), mutator=mutator,
        population_size=3, generations=2, elitism=1,
        rng=random.Random(7),
        smoke_check=False,
    )

    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    print("Running 2-gen × 3-pop evolution with structural mutations…")
    result = await evolver.evolve(context_factory=factory)
    print(f"best score: {result.best_score:.3f} @ generation {result.best_generation}")

    print()
    print("=== Per-individual mutation kinds ===")
    for gen in result.history:
        print(f"\nGeneration {gen.generation}")
        for i, im in enumerate(gen.population_metrics):
            kind = im.mutation_kind or "seed/clone"
            print(f"  ind#{i}  mutation={kind:<18}  score={im.score:.2f}  tokens={im.total_tokens}")

    # Confirm both structural kinds fired
    seen_kinds = set()
    for gen in result.history:
        for im in gen.population_metrics:
            if im.mutation_kind:
                seen_kinds.add(im.mutation_kind)
    print(f"\nMutation kinds observed across the run: {sorted(seen_kinds)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
