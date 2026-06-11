#!/usr/bin/env python3
"""Live benchmark for ChainBuilder.from_description + ChainEvolver.

Goal: validate that (a) a chain generated from a natural-language description
actually executes, and (b) the evolutionary loop finds prompt/model/parameter
variants that *measurably* improve a score against a small benchmark.

This is intentionally cheap: ~5 cases x ~4 individuals x ~2 generations on
qwen3-8b through OpenRouter.

Output: per-stage timing, the generated chain spec, evolution history,
baseline vs evolved score, and a list of observed friction points.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

from mmar_carl import (
    ChainEvolver,
    ChainMutator,
    DataCase,
    LLMStepDescription,
    MetricBase,
    OpenAICompatibleClient,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    create_openai_client,
)
from mmar_carl.chain import ChainBuilder

OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Benchmark — simple arithmetic word problems with extractable numeric answers
# ---------------------------------------------------------------------------


BENCHMARK: list[tuple[str, float]] = [
    ("Alice has 12 apples and gives 5 to Bob. How many apples does she have left?", 7.0),
    ("A train travels 60 km in 1 hour. How many km does it travel in 3.5 hours?", 210.0),
    ("If a shirt costs 20 dollars and is discounted by 25%, what is the final price in dollars?", 15.0),
    ("3 friends split a 90-dollar bill equally. How much does each pay?", 30.0),
    ("A rectangle is 8 by 5. What is its area?", 40.0),
]


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_number(text: str) -> float | None:
    """Extract the FIRST numeric token from text — robust to LLM verbosity."""
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])  # take the LAST number — usually the answer
    except ValueError:
        return None


class AnswerCorrectnessMetric(MetricBase):
    """1.0 if final output's last number equals the expected answer (±1%)."""

    def __init__(self, expected_by_label: dict[str, float]) -> None:
        self._expected = expected_by_label

    @property
    def name(self) -> str:
        return "answer_correctness"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        # output is a ReasoningResult; use the input as the label key.
        try:
            text = output.get_final_output()
        except AttributeError:
            return 0.0
        # We need to recover the case via the outer_context (no clean API today).
        # Re-extract by scanning the per-step results for the question.
        question = ""
        for step in output.step_results:
            if "Question:" in (step.result or ""):
                question = step.result
                break
        # Fallback: just check if any expected number appears.
        expected = self._expected.get(question)
        if expected is None:
            # Pick the closest match from the keys
            expected = next(iter(self._expected.values()))  # cheap fallback
        got = extract_first_number(text)
        if got is None:
            return 0.0
        # Allow 1% tolerance
        if expected == 0:
            return 1.0 if got == 0 else 0.0
        return 1.0 if abs(got - expected) / abs(expected) <= 0.01 else 0.0


# ---------------------------------------------------------------------------
# Simpler approach: extract answer from the chain's final step output and
# attach the expected answer via case.metadata so the metric can score
# directly. Avoids fragile question-matching.
# ---------------------------------------------------------------------------


class ExpectedNumberMetric(MetricBase):
    """Reads expected answer from `output.step_results[0]`'s prefixed marker."""

    @property
    def name(self) -> str:
        return "answer_correctness"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        # ReasoningResult: scan metadata for our injected expected answer.
        expected = None
        if hasattr(output, "metadata") and isinstance(output.metadata, dict):
            expected = output.metadata.get("__expected_answer")
        if expected is None:
            # Look at the context we stashed via memory.
            return 0.0
        try:
            text = output.get_final_output()
        except AttributeError:
            text = ""
        got = extract_first_number(text or "")
        if got is None:
            return 0.0
        if expected == 0:
            return 1.0 if got == 0 else 0.0
        return 1.0 if abs(got - expected) / abs(expected) <= 0.01 else 0.0


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def make_client(model: str) -> OpenAICompatibleClient:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ["OPENROUTER_API_KEY"]
    return create_openai_client(
        api_key=api_key,
        model=model,
        base_url=OPENROUTER_BASE,
        temperature=0.3,
    )


def make_dataset() -> SimpleDataset:
    return SimpleDataset(
        [
            DataCase(input=q, label=f"q{i + 1}", expected=str(a), metadata={"answer": a})
            for i, (q, a) in enumerate(BENCHMARK)
        ]
    )


def make_baseline_chain(client) -> ReasoningChain:
    """A 2-step chain: first 'plan the solution', then 'compute the answer'."""
    return ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1,
                title="Plan",
                aim="Identify the arithmetic operation needed to answer the question.",
                reasoning_questions="What numbers and operation does this problem call for?",
                stage_action="List the relevant quantities and the operation.",
                example_reasoning="Subtraction: 12 - 5.",
            ),
            LLMStepDescription(
                number=2,
                title="Compute",
                aim="Compute the answer. End your response with the final number and nothing else.",
                reasoning_questions="What is the numeric answer?",
                stage_action="Evaluate the operation from step 1. End with: 'Answer: <number>'.",
                example_reasoning="12 - 5 = 7. Answer: 7",
                dependencies=[1],
            ),
        ],
        max_workers=2,
        trace_name="baseline",
    )


async def evaluate_chain(chain: ReasoningChain, client) -> tuple[float, float]:
    """Returns (mean_correctness, total_seconds)."""
    correct = 0
    n = 0
    start = time.perf_counter()
    for question, expected in BENCHMARK:
        ctx = ReasoningContext(outer_context=question, api=client, model="unused")
        ctx.metadata["__expected_answer"] = expected
        try:
            result = await chain.execute_async(ctx)
            text = result.get_final_output() if result.success else ""
            got = extract_first_number(text or "")
            ok = got is not None and abs(got - expected) / max(abs(expected), 1) <= 0.01
        except Exception as e:  # noqa: BLE001
            print(f"  [error] case '{question[:50]}': {e}")
            ok = False
        correct += 1 if ok else 0
        n += 1
        print(f"  case: {question[:50]:<55}  expected={expected}  ok={ok}")
    elapsed = time.perf_counter() - start
    return (correct / n if n else 0.0, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    cheap_model = os.environ.get("CHEAP_MODEL", "qwen/qwen3-8b")
    print(f"=== Live benchmark: model={cheap_model} ===\n")
    client = make_client(cheap_model)

    # --------------------------------------------------------------------
    # 1. Baseline chain
    # --------------------------------------------------------------------
    print("\n[1] Baseline 2-step chain")
    baseline = make_baseline_chain(client)
    base_score, base_time = await evaluate_chain(baseline, client)
    print(f"  → baseline score: {base_score:.2f}, time: {base_time:.1f}s")

    # --------------------------------------------------------------------
    # 2. Chain generation via from_description
    # --------------------------------------------------------------------
    print("\n[2] Chain generated via ChainBuilder.from_description() — NO list-format warning in extra_instructions (relying on the new coercion)")
    gen_start = time.perf_counter()
    try:
        generated = await ChainBuilder.from_description(
            task=(
                "Solve a one-step arithmetic word problem. Output the final number "
                "preceded by 'Answer:' at the end of the last step."
            ),
            llm_client=client,
            available_tools=[],
            max_steps=4,
            extra_instructions=(
                "Use only LLM steps (step_type='llm'). Each step needs aim + reasoning_questions + "
                "stage_action + example_reasoning. The LAST step must end its response with "
                "'Answer: <number>'."
            ),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  [error] from_description failed: {e}")
        return
    gen_time = time.perf_counter() - gen_start
    print(f"  → generated {len(generated.steps)} steps in {gen_time:.1f}s")
    for s in generated.steps:
        print(f"     step {s.number}: {s.title} ({s.step_type})  deps={s.dependencies}")

    gen_score, gen_time_run = await evaluate_chain(generated, client)
    print(f"  → generated score: {gen_score:.2f}, time: {gen_time_run:.1f}s")

    # --------------------------------------------------------------------
    # 3. Evolution
    # --------------------------------------------------------------------
    print("\n[3] ChainEvolver on the generated chain")
    mutator = ChainMutator(
        aim_suffix_pool=[
            "Think step by step.",
            "Be concise.",
            "Show your reasoning before the final answer.",
        ],
        temperature_pool=[0.0, 0.3, 0.7],
        max_workers_pool=[1, 2],
    )
    dataset = make_dataset()
    metric = ExpectedNumberMetric()

    def context_factory(case: DataCase) -> ReasoningContext:
        ctx = ReasoningContext(outer_context=case.input, api=client, model="unused")
        ctx.metadata["__expected_answer"] = float(case.metadata["answer"])
        return ctx

    evolver = ChainEvolver(
        base_chain=generated,
        dataset=dataset,
        metric=metric,
        mutator=mutator,
        population_size=3,
        generations=2,
        elitism=1,
    )
    evo_start = time.perf_counter()
    try:
        result = await evolver.evolve(context_factory)
    except Exception as e:  # noqa: BLE001
        print(f"  [error] evolve failed: {e}")
        return
    evo_time = time.perf_counter() - evo_start
    print(f"  → evolved best score: {result.best_score:.2f} (gen {result.best_generation})")
    print(f"  → evolution wall time: {evo_time:.1f}s")
    for stats in result.history:
        print(
            f"     gen {stats.generation}: best={stats.best_score:.2f} "
            f"mean={stats.mean_score:.2f}  scores={stats.population_scores}"
        )

    # --------------------------------------------------------------------
    # 4. Summary
    # --------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"  baseline:  score={base_score:.2f}")
    print(f"  generated: score={gen_score:.2f}")
    print(f"  evolved:   score={result.best_score:.2f}")
    delta = result.best_score - gen_score
    print(f"  evolution gain vs generated baseline: {delta:+.2f}")


if __name__ == "__main__":
    asyncio.run(main())
