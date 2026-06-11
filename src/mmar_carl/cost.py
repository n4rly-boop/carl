"""
Pre-execution cost estimation for ``ReasoningChain``.

Provides ``chain.estimate_cost(context)`` — a dry-run analyser that walks each
step in the chain, identifies LLM-calling steps, and returns a structured
report with per-step and chain-level token/cost estimates. No LLM calls are
made.

Tokens are estimated using a simple character-per-token heuristic
(``char_per_token=4`` by default — a reasonable proxy for English text).
This is intentionally rough: the goal is to spot a chain that's about to
burn $50, not to predict the bill to the cent. Provide a real tokenizer
externally if you need precision.

Pricing is opt-in. ``pricing`` is a dict of ``{model_name: (input_per_1k, output_per_1k)}``
in USD. If a step's resolved model isn't in the dict, its cost is ``0.0`` and
``pricing_missing=True`` is flagged on the row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

from .models.config import LLMStepConfig
from .models.enums import StepType

if TYPE_CHECKING:
    from .models.context import ReasoningContext
    from .models.steps import StepDescriptionBase


# Step types whose executors call the LLM at least once per invocation.
_LLM_CALLING_STEP_TYPES: frozenset[StepType] = frozenset(
    {
        StepType.LLM,
        StepType.STRUCTURED_OUTPUT,
        # EVALUATION calls an LLM only when evaluation_method == "llm" — handled below.
        StepType.EVALUATION,
        # PARALLEL_SAMPLING fans out N copies of an LLM step — call count = n_samples.
        StepType.PARALLEL_SAMPLING,
    }
)


class StepCostEstimate(BaseModel):
    """Per-step row in a :class:`CostEstimate`."""

    step_number: int
    step_title: str
    step_type: StepType
    calls_llm: bool = Field(description="Whether this step will invoke an LLM at runtime")
    model: Optional[str] = Field(default=None, description="Resolved model name (None when unresolved)")
    estimated_calls: int = Field(default=0, ge=0, description="Number of LLM calls this step will make")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    input_cost_usd: float = Field(default=0.0, ge=0.0)
    output_cost_usd: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    pricing_missing: bool = Field(
        default=False,
        description="True when this step calls an LLM but no pricing entry was found for its model.",
    )
    note: str = Field(default="", description="Free-text caveat (e.g. 'LLM_AGENT iterations not modelled').")


class CostEstimate(BaseModel):
    """Aggregated dry-run cost estimate for an entire chain."""

    steps: list[StepCostEstimate]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    char_per_token: int = 4
    currency: str = "USD"

    @property
    def llm_call_steps(self) -> list[StepCostEstimate]:
        """Subset of step rows that will invoke an LLM."""
        return [s for s in self.steps if s.calls_llm]

    def _repr_markdown_(self) -> str:
        """Rich-display protocol for Jupyter.

        Markdown headline (total cost, total tokens, currency) + the
        existing per-step table wrapped in a fenced code block.
        """
        n_llm = len(self.llm_call_steps)
        n = len(self.steps)
        lines = [
            f"**CostEstimate** — total: **${self.total_cost_usd:.4f} "
            f"{self.currency}** · {self.total_tokens:,} tokens "
            f"({self.total_input_tokens:,} in / {self.total_output_tokens:,} out) "
            f"across {n} step{'s' if n != 1 else ''} "
            f"({n_llm} LLM)",
            "",
            "```text",
            self.format_table(),
            "```",
        ]
        return "\n".join(lines)

    def format_table(self) -> str:
        """One-line-per-step human-readable summary suitable for stdout."""
        if not self.steps:
            return "(empty chain)"
        header = f"{'#':>3}  {'title':<28}  {'type':<18}  {'calls':>5}  {'in':>6}  {'out':>6}  {'cost':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for row in self.steps:
            cost_str = f"${row.total_cost_usd:.4f}" if row.calls_llm else "-"
            if row.pricing_missing:
                cost_str = "(no price)"
            lines.append(
                f"{row.step_number:>3}  {row.step_title[:28]:<28}  {row.step_type.value:<18}  "
                f"{row.estimated_calls:>5}  {row.input_tokens:>6}  {row.output_tokens:>6}  {cost_str:>10}"
            )
        lines.append(sep)
        lines.append(
            f"TOTAL  {'':<28}  {'':<18}  {sum(s.estimated_calls for s in self.steps):>5}  "
            f"{self.total_input_tokens:>6}  {self.total_output_tokens:>6}  ${self.total_cost_usd:>9.4f}"
        )
        return "\n".join(lines)


def _resolve_step_llm_config(
    step: "StepDescriptionBase",
    chain_default: Optional[LLMStepConfig],
) -> LLMStepConfig:
    """
    Compose effective :class:`LLMStepConfig` for *step*.

    Resolution order (highest → lowest): per-step ``llm_config`` set fields →
    ``chain_default`` set fields → built-in defaults. Mirrors the runtime
    precedence in :py:meth:`ReasoningContext.get_llm_client_for_step`.
    """
    step_cfg: Optional[LLMStepConfig] = getattr(step, "llm_config", None)

    def _set_fields(cfg: Optional[LLMStepConfig]) -> dict[str, Any]:
        if cfg is None:
            return {}
        return {name: getattr(cfg, name) for name in cfg.model_fields_set}

    merged_kwargs: dict[str, Any] = {}
    merged_kwargs.update(_set_fields(chain_default))
    merged_kwargs.update(_set_fields(step_cfg))  # step wins
    return LLMStepConfig(**merged_kwargs) if merged_kwargs else LLMStepConfig()


def _calls_llm_for_step(step: "StepDescriptionBase") -> tuple[bool, int, str]:
    """
    Decide whether *step* invokes an LLM and how many calls it makes.

    Returns ``(calls_llm, estimated_calls, note)``.
    """
    if step.step_type == StepType.LLM:
        return True, 1, ""
    if step.step_type == StepType.STRUCTURED_OUTPUT:
        return True, 1, ""
    if step.step_type == StepType.EVALUATION:
        cfg = getattr(step, "config", None)
        method = getattr(cfg, "evaluation_method", "rule") if cfg else "rule"
        if method == "llm":
            # Worst case: 1 judge call + up to max_retries improvement calls.
            max_retries = getattr(cfg, "max_retries", 0) or 0
            return True, 1 + max_retries, "includes up to max_retries retry calls"
        return False, 0, "rule-based evaluation — no LLM call"
    if step.step_type == StepType.PARALLEL_SAMPLING:
        cfg = getattr(step, "config", None)
        n_samples = getattr(cfg, "n_samples", 1) if cfg else 1
        agg = getattr(cfg, "aggregation", None) if cfg else None
        note = "parallel sampling"
        if agg is not None and getattr(agg, "value", "") in ("best_of_n", "llm_judge"):
            return True, int(n_samples) + 1, note + " + 1 judge call"
        return True, int(n_samples), note
    return False, 0, ""


def _format_note(base_note: str, step_type: StepType) -> str:
    """Add caveats for step types whose cost isn't fully modelled."""
    extra = ""
    if step_type == StepType.AGENT_SKILL:
        extra = "AgentSkill LLM_AGENT iterations not modelled — estimate is lower bound"
    elif step_type == StepType.AGENT_HANDOFF:
        extra = "sub-chain LLM calls not recursively estimated — call estimate_cost on the sub-chain"
    if base_note and extra:
        return f"{base_note}; {extra}"
    return base_note or extra


def estimate_chain_cost(
    chain: Any,
    context: "ReasoningContext",
    *,
    pricing: Optional[dict[str, tuple[float, float]]] = None,
    default_output_tokens: int = 512,
    char_per_token: int = 4,
) -> CostEstimate:
    """
    Walk *chain*'s steps and produce a :class:`CostEstimate`.

    See :py:meth:`mmar_carl.ReasoningChain.estimate_cost` for the public docstring.
    """
    if char_per_token <= 0:
        raise ValueError("char_per_token must be positive")
    if default_output_tokens < 0:
        raise ValueError("default_output_tokens must be >= 0")

    chain_default: Optional[LLMStepConfig] = getattr(chain, "default_llm_config", None)
    context_model: Optional[str] = getattr(context, "model", None)
    if context_model in (None, "", "default"):
        context_model = None

    rows: list[StepCostEstimate] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0

    # Rough proxy for "input context size at this step":
    # outer_context + accumulated short history estimate. We model the history
    # as ~150 chars per prior LLM-style step's expected output, which is a
    # deliberately coarse upper bound.
    outer_ctx_len = len(getattr(context, "outer_context", "") or "")
    history_chars_so_far = 0

    for step in chain.steps:
        calls_llm, est_calls, base_note = _calls_llm_for_step(step)
        note = _format_note(base_note, step.step_type)

        if not calls_llm:
            rows.append(
                StepCostEstimate(
                    step_number=step.number,
                    step_title=step.title,
                    step_type=step.step_type,
                    calls_llm=False,
                    note=note,
                )
            )
            continue

        effective_cfg = _resolve_step_llm_config(step, chain_default)
        model = effective_cfg.model or context_model

        aim_chars = len(getattr(step, "aim", "") or "")
        title_chars = len(step.title or "")
        # Per-call input: outer context + aim/title + accumulated history.
        per_call_input_chars = outer_ctx_len + aim_chars + title_chars + history_chars_so_far
        per_call_input_tokens = max(1, per_call_input_chars // char_per_token)
        input_tokens = per_call_input_tokens * est_calls

        per_call_output_tokens = effective_cfg.max_tokens or default_output_tokens
        output_tokens = per_call_output_tokens * est_calls

        input_cost = 0.0
        output_cost = 0.0
        pricing_missing = False
        if pricing is not None:
            entry = pricing.get(model) if model else None
            if entry is None:
                pricing_missing = True
            else:
                in_per_1k, out_per_1k = entry
                input_cost = (input_tokens / 1000.0) * in_per_1k
                output_cost = (output_tokens / 1000.0) * out_per_1k

        step_total = input_cost + output_cost
        rows.append(
            StepCostEstimate(
                step_number=step.number,
                step_title=step.title,
                step_type=step.step_type,
                calls_llm=True,
                model=model,
                estimated_calls=est_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost_usd=round(input_cost, 6),
                output_cost_usd=round(output_cost, 6),
                total_cost_usd=round(step_total, 6),
                pricing_missing=pricing_missing,
                note=note,
            )
        )
        total_input += input_tokens
        total_output += output_tokens
        total_cost += step_total

        # Advance the history proxy by the expected output of this LLM-style step.
        # Cap at 4000 chars to keep estimates from running away on deep chains.
        history_chars_so_far = min(history_chars_so_far + per_call_output_tokens * char_per_token, 4000)

    return CostEstimate(
        steps=rows,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_usd=round(total_cost, 6),
        char_per_token=char_per_token,
    )
