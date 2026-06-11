"""
Tests for compile-time reference validation in ReasoningChain.

_validate_references() is called from _validate_steps() during chain
construction and issues UserWarning for:
- $history[N] references when history is empty or too shallow at that batch
- $memory.ns.key references that no prior step writes
"""

import warnings

from mmar_carl import ReasoningChain
from mmar_carl.models.steps import (
    LLMStepDescription,
    MemoryStepDescription,
    ToolStepDescription,
    TransformStepDescription,
    ConditionalStepDescription,
)
from mmar_carl.models.config import (
    MemoryStepConfig,
    ToolStepConfig,
    TransformStepConfig,
    ConditionalStepConfig,
)
from mmar_carl.models.enums import MemoryOperation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_chain(*steps) -> tuple[ReasoningChain, list[warnings.WarningMessage]]:
    """Build chain and capture all UserWarnings emitted during construction."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        chain = ReasoningChain(steps=list(steps))
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    return chain, user_warnings


def warning_messages(w: list[warnings.WarningMessage]) -> list[str]:
    return [str(x.message) for x in w]


# ---------------------------------------------------------------------------
# 1. $history[N] — empty / shallow history
# ---------------------------------------------------------------------------

class TestHistoryReferenceValidation:
    """Validator warns when $history[N] will be out-of-range at runtime."""

    def test_no_warning_when_history_is_deep_enough(self):
        """
        Step 2 depends on step 1 (LLM). history_depth = 1 when batch 2 starts.
        $history[0] (idx=0) is in range → no warning.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            TransformStepDescription(
                number=2, title="T2", dependencies=[1],
                config=TransformStepConfig(
                    transform_type="format",
                    input_key="$history[0]",
                    output_format="{value}",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert not any("history" in m for m in msgs), f"Unexpected warning: {msgs}"

    def test_warns_on_history_ref_in_first_batch(self):
        """
        Step 1 is in the first batch (history_depth=0).
        $history[-1] resolves to None — validator warns.
        """
        _, w = build_chain(
            TransformStepDescription(
                number=1, title="T1",
                config=TransformStepConfig(
                    transform_type="format",
                    input_key="$history[-1]",
                    output_format="{value}",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert any("history will be empty" in m for m in msgs), (
            f"Expected empty-history warning, got: {msgs}"
        )

    def test_warns_on_positive_index_out_of_range(self):
        """
        Step 2 in second batch: history_depth=1.
        $history[1] (index 1, depth 1) is out of range → warning.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            TransformStepDescription(
                number=2, title="T2", dependencies=[1],
                config=TransformStepConfig(
                    transform_type="format",
                    input_key="$history[1]",  # needs depth>=2 but only 1 available
                    output_format="{value}",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert any("out of range" in m for m in msgs), (
            f"Expected out-of-range warning, got: {msgs}"
        )

    def test_no_warning_for_default_history_ref_after_prior_step(self):
        """
        Conditional step 2 uses default condition_context_key="$history[-1]".
        Batch 1 (step 1) runs first → history_depth=1 → no warning.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            ConditionalStepDescription(
                number=2, title="C2", dependencies=[1],
                config=ConditionalStepConfig(
                    condition_expression="'x' in value",
                    condition_context_key="$history[-1]",
                    branches=[],
                    default_next_step=None,
                ),
            ),
        )
        msgs = warning_messages(w)
        assert not any("history" in m for m in msgs), f"Unexpected warning: {msgs}"


# ---------------------------------------------------------------------------
# 2. $memory.ns.key — no prior writer
# ---------------------------------------------------------------------------

class TestMemoryReferenceValidation:
    """Validator warns when $memory.ns.key is read before any step writes it."""

    def test_no_warning_when_prior_step_writes_key(self):
        """
        Step 1 writes 'data', step 2 reads it via input_mapping.
        No warning expected.
        """
        _, w = build_chain(
            MemoryStepDescription(
                number=1, title="Write",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="data",
                    value_source='"hello"',
                ),
            ),
            ToolStepDescription(
                number=2, title="Use", dependencies=[1],
                config=ToolStepConfig(
                    tool_name="my_tool",
                    input_mapping={"x": "$memory.default.data"},
                    output_key="result",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert not any("memory key" in m for m in msgs), f"Unexpected warning: {msgs}"

    def test_warns_when_no_step_writes_key(self):
        """
        Step 1 reads '$memory.default.missing' but nothing writes it.
        Validator should warn.
        """
        _, w = build_chain(
            ToolStepDescription(
                number=1, title="Use missing key",
                config=ToolStepConfig(
                    tool_name="my_tool",
                    input_mapping={"x": "$memory.default.missing"},
                    output_key="result",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert any("memory key" in m and "missing" in m for m in msgs), (
            f"Expected missing-key warning, got: {msgs}"
        )

    def test_warns_when_reading_before_write_in_same_batch(self):
        """
        Step 2 writes 'key'; step 3 (parallel sibling of step 2) reads 'key'.
        Siblings can't see each other's writes → validator warns.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            MemoryStepDescription(
                number=2, title="Writer", dependencies=[1],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="shared",
                    value_source='"v"',
                ),
            ),
            ToolStepDescription(
                number=3, title="Reader (parallel sibling)", dependencies=[1],
                config=ToolStepConfig(
                    tool_name="t",
                    input_mapping={"x": "$memory.default.shared"},
                    output_key="out",
                ),
            ),
        )
        msgs = warning_messages(w)
        # Step 3 runs in the same batch as step 2 — step 2's write not yet merged
        assert any("memory key" in m for m in msgs), (
            f"Expected memory warning for parallel sibling read, got: {msgs}"
        )

    def test_no_warning_for_memory_ref_written_in_prior_batch(self):
        """
        Step 2 writes 'key'; step 4 reads it (step 4 depends on step 2).
        Step 4 is in a later batch → no warning.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            MemoryStepDescription(
                number=2, title="Write key", dependencies=[1],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="key",
                    value_source='"hello"',
                ),
            ),
            LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[1]),
            ToolStepDescription(
                number=4, title="Read key", dependencies=[2, 3],
                config=ToolStepConfig(
                    tool_name="t",
                    input_mapping={"x": "$memory.default.key"},
                    output_key="out",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert not any("memory key" in m for m in msgs), f"Unexpected warning: {msgs}"

    def test_warns_for_memory_value_source_with_unknown_key(self):
        """
        MemoryStepDescription with WRITE and value_source='$memory.default.missing'.
        No step writes 'missing' before this step → warning.
        """
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            MemoryStepDescription(
                number=2, title="Copy from missing", dependencies=[1],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="dest",
                    value_source="$memory.default.missing",
                ),
            ),
        )
        msgs = warning_messages(w)
        assert any("memory key" in m and "missing" in m for m in msgs), (
            f"Expected warning for unwritten source key, got: {msgs}"
        )


# ---------------------------------------------------------------------------
# 3. Clean chains produce no warnings
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    """Well-formed chains must produce no spurious reference warnings."""

    def test_linear_llm_chain_no_warnings(self):
        """A straightforward linear LLM chain should emit no reference warnings."""
        _, w = build_chain(
            LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
            LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
            LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
        )
        assert not w, f"Unexpected warnings: {warning_messages(w)}"

    def test_tool_step_without_memory_refs_no_warnings(self):
        """Tool step with literal input mapping should not warn."""
        _, w = build_chain(
            ToolStepDescription(
                number=1, title="Tool",
                config=ToolStepConfig(
                    tool_name="t",
                    input_mapping={"x": '"literal"'},
                    output_key="out",
                ),
            ),
        )
        assert not w, f"Unexpected warnings: {warning_messages(w)}"
