#!/usr/bin/env python3
"""
Example: token usage and cost tracking in CARL.

Demonstrates three granularities of token consumption:
  1. Pre-flight — ``chain.estimate_cost(context, pricing=...)`` projects
     input/output tokens and USD cost before any LLM call is made.
  2. Single run — after ``chain.execute_async(context)``, read per-step
     ``StepExecutionResult.token_usage`` and chain-total ``result.token_usage``.
     Combined with ``result.get_profiling_summary()`` for byte-level memory
     stats.
  3. Batch run — loop a chain over a dataset and aggregate token usage
     across runs (min / median / p95 / max / total + USD).

Uses a mock OpenAI-compatible LLM client that returns realistic ``usage``
numbers, so the example runs without an API key.

Run:
    python examples/token_usage_example.py
    PYTHONPATH=$(pwd) python examples/token_usage_example.py
"""

import asyncio
import statistics

from mmar_carl import (
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from examples.utils import format_status, print_execution_summary


# ---------------------------------------------------------------------------
# Mock OpenAI-compatible LLM with usage reporting
# ---------------------------------------------------------------------------


class MockUsageLLM(LLMClientBase):
    """Returns canned responses plus realistic token usage numbers.

    The mock estimates prompt tokens as ``len(prompt) // 4`` (the same
    heuristic CARL uses for cost estimation) and reports completion tokens
    drawn from a small repeating sequence so batch runs see variation.
    """

    def __init__(self, completion_tokens_cycle: list[int] | None = None) -> None:
        self.completion_tokens_cycle = completion_tokens_cycle or [80, 120, 95, 150, 60]
        self._calls = 0

    async def get_response(self, prompt: str) -> str:
        self._calls += 1
        return f"response-{self._calls}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        text = await self.get_response_with_retries(prompt, retries=retries)
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = self.completion_tokens_cycle[
            (self._calls - 1) % len(self.completion_tokens_cycle)
        ]
        usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }
        return text, usage


def make_chain() -> ReasoningChain:
    """A simple 3-step linear chain we use for all three scenarios."""
    return ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1,
                title="extract facts",
                aim="Pull out key claims.",
                llm_config=LLMStepConfig(model="gpt-4o-mini", max_tokens=200),
            ),
            LLMStepDescription(
                number=2,
                title="verify claims",
                aim="Verify each claim.",
                dependencies=[1],
                llm_config=LLMStepConfig(model="gpt-4o", max_tokens=400),
            ),
            LLMStepDescription(
                number=3,
                title="synthesise report",
                aim="Combine into a report.",
                dependencies=[2],
                llm_config=LLMStepConfig(model="gpt-4o", max_tokens=500),
            ),
        ],
        max_workers=1,
    )


# Approximate prices in USD per 1k tokens (Nov 2025 list-price snapshot).
PRICING = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


# ---------------------------------------------------------------------------
# 1. Pre-flight estimate
# ---------------------------------------------------------------------------


def example_preflight_estimate() -> None:
    print("=" * 72)
    print("Example 1: Pre-flight cost estimate (no LLM calls made)")
    print("=" * 72)
    chain = make_chain()
    ctx = ReasoningContext(
        outer_context="A medium-sized document with several factual claims. " * 30,
        api=MockUsageLLM(),
    )
    estimate = chain.estimate_cost(ctx, pricing=PRICING)
    print(estimate.format_table())
    print(f"\nProjected total: ${estimate.total_cost_usd:.6f} for "
          f"{estimate.total_tokens} tokens "
          f"({len(estimate.llm_call_steps)} LLM-calling steps).")


# ---------------------------------------------------------------------------
# 2. Single-run actuals
# ---------------------------------------------------------------------------


def _row(label: str, prompt: int, completion: int, total: int, cost: str = "") -> str:
    return f"  {label:<24}  prompt={prompt:>5}  completion={completion:>5}  total={total:>5}  {cost}"


async def example_single_run_actuals() -> None:
    print("\n" + "=" * 72)
    print("Example 2: Single-run actuals (per-step + chain totals)")
    print("=" * 72)
    chain = make_chain()
    ctx = ReasoningContext(
        outer_context="A medium-sized document with several factual claims. " * 30,
        api=MockUsageLLM(),
    )
    result = await chain.execute_async(ctx)
    print_execution_summary(result, label="\nSuccess")
    print("\nPer-step token usage:")
    print("  " + "-" * 68)
    chain_cost = 0.0
    for sr in result.step_results:
        usage = sr.token_usage or {}
        prompt = usage.get("prompt", 0)
        completion = usage.get("completion", 0)
        total = usage.get("total", 0)
        # Resolve the model for cost calculation
        model = (sr.metadata or {}).get("model") if hasattr(sr, "metadata") else None
        if model is None:
            # Fall back to the chain's per-step config we just built
            cfg = chain.steps[sr.step_number - 1].llm_config
            model = cfg.model if cfg else None
        cost_str = ""
        if model and model in PRICING:
            in_per_1k, out_per_1k = PRICING[model]
            step_cost = (prompt / 1000) * in_per_1k + (completion / 1000) * out_per_1k
            chain_cost += step_cost
            cost_str = f"  cost=${step_cost:.6f}  ({model})"
        print(_row(f"step {sr.step_number} ({sr.step_title[:14]})",
                   prompt, completion, total, cost_str))
    print("  " + "-" * 68)
    totals = result.token_usage or {}
    print(_row(
        "CHAIN TOTAL",
        totals.get("prompt", 0),
        totals.get("completion", 0),
        totals.get("total", 0),
        f"  cost=${chain_cost:.6f}",
    ))

    # Profiling summary for byte-level memory stats
    profiling = result.get_profiling_summary()
    print("\nProfiling summary:")
    print(f"  peak memory bytes:    {profiling.get('peak_memory_bytes', 0)}")
    print(f"  total history bytes:  {profiling.get('total_history_bytes', 0)}")
    print(f"  total execution time: {profiling.get('total_execution_time_s', 0):.4f}s")


# ---------------------------------------------------------------------------
# 3. Batch aggregation
# ---------------------------------------------------------------------------


async def example_batch_aggregation() -> None:
    print("\n" + "=" * 72)
    print("Example 3: Batch aggregation across multiple runs")
    print("=" * 72)
    chain = make_chain()
    # Five documents of varying sizes — input length drives prompt-token cost
    documents = [
        "Short doc. Two sentences. " * 5,
        "Medium doc with more detail. " * 20,
        "Long doc with extensive content and many claims to verify. " * 40,
        "Tiny prompt.",
        "Final test document with moderate length and clear structure. " * 25,
    ]

    per_run_totals: list[int] = []
    per_run_costs: list[float] = []
    per_step_totals: dict[int, list[int]] = {1: [], 2: [], 3: []}

    for i, doc in enumerate(documents, start=1):
        ctx = ReasoningContext(outer_context=doc, api=MockUsageLLM())
        result = await chain.execute_async(ctx)
        usage = result.token_usage or {}
        run_total = usage.get("total", 0)
        per_run_totals.append(run_total)

        run_cost = 0.0
        for sr in result.step_results:
            sr_usage = sr.token_usage or {}
            sr_total = sr_usage.get("total", 0)
            per_step_totals[sr.step_number].append(sr_total)
            cfg = chain.steps[sr.step_number - 1].llm_config
            model = cfg.model if cfg else None
            if model and model in PRICING:
                in_per_1k, out_per_1k = PRICING[model]
                run_cost += (
                    (sr_usage.get("prompt", 0) / 1000) * in_per_1k
                    + (sr_usage.get("completion", 0) / 1000) * out_per_1k
                )
        per_run_costs.append(run_cost)
        print(f"  run {i}: doc-len={len(doc):>5}  tokens={run_total:>5}  "
              f"cost=${run_cost:.6f}")

    print()
    print("  Per-run token totals (n={}):".format(len(per_run_totals)))
    print(f"    min:    {min(per_run_totals)}")
    print(f"    median: {int(statistics.median(per_run_totals))}")
    print(f"    p95:    {_percentile(per_run_totals, 0.95)}")
    print(f"    max:    {max(per_run_totals)}")
    print(f"    total:  {sum(per_run_totals)}")

    print()
    print("  Per-step token spread (median / max across runs):")
    for step_num in sorted(per_step_totals):
        vals = per_step_totals[step_num]
        cfg = chain.steps[step_num - 1].llm_config
        model = cfg.model if cfg else "?"
        print(f"    step {step_num} ({model}):  median={int(statistics.median(vals)):>4}  "
              f"max={max(vals):>4}")

    print()
    print(f"  Batch total cost: ${sum(per_run_costs):.6f} "
          f"(mean per run: ${statistics.mean(per_run_costs):.6f})")


def _percentile(values: list[int], q: float) -> int:
    """Inclusive percentile (avoids the numpy dependency)."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    index = int(q * (len(sorted_vals) - 1))
    return sorted_vals[index]


# ---------------------------------------------------------------------------
# 4. Token-budget warning demo (bonus)
# ---------------------------------------------------------------------------


async def example_token_budget_warning() -> None:
    print("\n" + "=" * 72)
    print("Example 4: Per-step token_budget_warning")
    print("=" * 72)
    import warnings

    chain = ReasoningChain(
        steps=[
            LLMStepDescription(
                number=1,
                title="bounded step",
                aim="x",
                llm_config=LLMStepConfig(
                    model="gpt-4o-mini",
                    max_tokens=200,
                    token_budget_warning=50,  # very low — will trigger
                ),
            ),
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(
        outer_context="x" * 100,
        api=MockUsageLLM(completion_tokens_cycle=[100]),  # 100 > 50 budget
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await chain.execute_async(ctx)

    budget_warnings = [w for w in caught if "budget warning" in str(w.message).lower()]
    print(f"  step success: {format_status(result.step_results[0].success)}")
    print(f"  warnings raised: {len(budget_warnings)}")
    if budget_warnings:
        print(f"  first warning: {budget_warnings[0].message}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    example_preflight_estimate()
    await example_single_run_actuals()
    await example_batch_aggregation()
    await example_token_budget_warning()
    print("\n" + "=" * 72)
    print("All token-usage examples completed.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
