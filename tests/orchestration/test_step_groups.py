"""
Tests for ``StepGroup`` bulk LLM-config overrides.

A ``StepGroup`` propagates *explicitly-set* fields from its ``llm_config``
into each member step's ``llm_config`` at ``ReasoningChain`` construction
time. Per-step config wins; tool/memory/etc. steps without an ``llm_config``
attribute are skipped silently so groups may list mixed step numbers.
"""

from typing import Any

import pytest

from mmar_carl import (
    ExecutionMode,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepGroup,
    ToolStepConfig,
    ToolStepDescription,
)


class _NoopLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _llm(number: int, **kw: Any) -> LLMStepDescription:
    return LLMStepDescription(number=number, title=f"s{number}", aim="x", **kw)


# --------------------------------------------------------------------------- #
# Basic application
# --------------------------------------------------------------------------- #


def test_group_config_fills_unset_fields_on_member_steps() -> None:
    chain = ReasoningChain(
        steps=[_llm(1), _llm(2), _llm(3)],
        step_groups=[
            StepGroup(
                name="analysis",
                steps=[1, 2],
                llm_config=LLMStepConfig(temperature=0.0, model="gpt-4o"),
            )
        ],
        max_workers=1,
    )
    s1, s2, s3 = chain.steps
    assert s1.llm_config is not None
    assert s1.llm_config.temperature == 0.0
    assert s1.llm_config.model == "gpt-4o"
    assert s2.llm_config is not None
    assert s2.llm_config.temperature == 0.0
    assert s2.llm_config.model == "gpt-4o"
    # Step 3 was not in the group → no llm_config gets created
    assert s3.llm_config is None


def test_per_step_config_wins_over_group_config() -> None:
    chain = ReasoningChain(
        steps=[
            _llm(1, llm_config=LLMStepConfig(temperature=0.9)),
            _llm(2),
        ],
        step_groups=[
            StepGroup(
                name="analysis",
                steps=[1, 2],
                llm_config=LLMStepConfig(temperature=0.0, model="gpt-4o"),
            )
        ],
        max_workers=1,
    )
    s1, s2 = chain.steps
    # Step 1: its temperature wins, but unset 'model' is filled from group
    assert s1.llm_config.temperature == 0.9
    assert s1.llm_config.model == "gpt-4o"
    # Step 2: both come from group
    assert s2.llm_config.temperature == 0.0
    assert s2.llm_config.model == "gpt-4o"


def test_unset_group_fields_do_not_overwrite_step_defaults() -> None:
    """Group has only ``model`` set → step retains its own ``temperature``."""
    chain = ReasoningChain(
        steps=[_llm(1, llm_config=LLMStepConfig(temperature=0.3))],
        step_groups=[
            StepGroup(
                name="g",
                steps=[1],
                llm_config=LLMStepConfig(model="gpt-4o"),  # only model set
            )
        ],
        max_workers=1,
    )
    cfg = chain.steps[0].llm_config
    assert cfg.model == "gpt-4o"
    assert cfg.temperature == 0.3


def test_group_only_propagates_explicitly_set_fields() -> None:
    """A group that sets only ``temperature`` must not propagate
    ``execution_mode`` (its default) onto a step that has none."""
    chain = ReasoningChain(
        steps=[_llm(1)],
        step_groups=[
            StepGroup(name="g", steps=[1], llm_config=LLMStepConfig(temperature=0.2)),
        ],
        max_workers=1,
    )
    cfg = chain.steps[0].llm_config
    assert cfg.temperature == 0.2
    # execution_mode was a default on the group config, so it should NOT be in
    # the propagated set on the step.
    assert "execution_mode" not in cfg.model_fields_set


def test_group_includes_tool_step_does_not_crash() -> None:
    """Non-LLM steps in a group are silently skipped."""
    chain = ReasoningChain(
        steps=[
            _llm(1),
            ToolStepDescription(
                number=2,
                title="emit",
                config=ToolStepConfig(tool_name="x", parameters=[], input_mapping={}),
            ),
            MemoryStepDescription(
                number=3,
                title="store",
                config=MemoryStepConfig(operation=MemoryOperation.WRITE, memory_key="k", value_source="'\"v\"'"),
            ),
        ],
        step_groups=[
            StepGroup(name="all", steps=[1, 2, 3], llm_config=LLMStepConfig(temperature=0.0)),
        ],
        max_workers=1,
    )
    assert chain.steps[0].llm_config is not None
    assert chain.steps[0].llm_config.temperature == 0.0
    # Tool / Memory steps don't grow an llm_config attribute
    assert not hasattr(chain.steps[1], "llm_config") or chain.steps[1].llm_config is None  # type: ignore[attr-defined]
    assert not hasattr(chain.steps[2], "llm_config") or chain.steps[2].llm_config is None  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_group_referencing_nonexistent_step_raises() -> None:
    with pytest.raises(ValueError, match="non-existent step 99"):
        ReasoningChain(
            steps=[_llm(1)],
            step_groups=[
                StepGroup(name="g", steps=[99], llm_config=LLMStepConfig(temperature=0.0)),
            ],
            max_workers=1,
        )


def test_step_in_multiple_groups_raises() -> None:
    with pytest.raises(ValueError, match="Step 1 listed in multiple groups"):
        ReasoningChain(
            steps=[_llm(1), _llm(2)],
            step_groups=[
                StepGroup(name="a", steps=[1, 2], llm_config=LLMStepConfig(temperature=0.0)),
                StepGroup(name="b", steps=[1], llm_config=LLMStepConfig(model="x")),
            ],
            max_workers=1,
        )


def test_step_group_rejects_empty_steps_list() -> None:
    with pytest.raises(ValueError):
        StepGroup(name="g", steps=[], llm_config=LLMStepConfig())


def test_step_group_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        StepGroup(name="", steps=[1], llm_config=LLMStepConfig())


# --------------------------------------------------------------------------- #
# Multiple groups, overlapping fields
# --------------------------------------------------------------------------- #


def test_multiple_groups_each_apply_to_their_members() -> None:
    chain = ReasoningChain(
        steps=[_llm(1), _llm(2), _llm(3), _llm(4)],
        step_groups=[
            StepGroup(name="A", steps=[1, 2], llm_config=LLMStepConfig(temperature=0.0)),
            StepGroup(name="B", steps=[3, 4], llm_config=LLMStepConfig(temperature=1.5, model="m")),
        ],
        max_workers=1,
    )
    assert chain.steps[0].llm_config.temperature == 0.0
    assert chain.steps[1].llm_config.temperature == 0.0
    assert chain.steps[2].llm_config.temperature == 1.5
    assert chain.steps[2].llm_config.model == "m"
    assert chain.steps[3].llm_config.temperature == 1.5
    assert chain.steps[3].llm_config.model == "m"


# --------------------------------------------------------------------------- #
# Integration: groups + chain default
# --------------------------------------------------------------------------- #


def test_precedence_per_step_then_group_then_chain_default() -> None:
    """End-to-end precedence: step → group → chain default."""
    chain_default = LLMStepConfig(model="default-model", temperature=0.5, max_tokens=2048)
    group_cfg = LLMStepConfig(temperature=0.0, max_tokens=512)

    chain = ReasoningChain(
        steps=[
            _llm(1, llm_config=LLMStepConfig(temperature=0.9)),  # per-step temp wins
            _llm(2),  # all from group + chain default
            _llm(3),  # no group → from chain default only
        ],
        step_groups=[StepGroup(name="g", steps=[1, 2], llm_config=group_cfg)],
        default_llm_config=chain_default,
        max_workers=1,
    )

    # After init: per-step + group are baked in. chain_default is applied at
    # *runtime* by context.get_llm_client_for_step(), not here.
    s1, s2, s3 = chain.steps
    assert s1.llm_config.temperature == 0.9  # per-step wins
    assert s1.llm_config.max_tokens == 512   # filled from group
    assert s2.llm_config.temperature == 0.0  # group wins
    assert s2.llm_config.max_tokens == 512
    assert s3.llm_config is None  # chain default still applies at runtime


# --------------------------------------------------------------------------- #
# End-to-end execution proves groups don't break chain runs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_grouped_chain_actually_executes() -> None:
    """A real chain run with a StepGroup completes successfully and the LLM
    sees the resolved (per-step) config — verified by the fact that no
    exceptions arise from the merge logic during real execution."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="analyse", aim="A"),
            LLMStepDescription(number=2, title="report", aim="B", dependencies=[1]),
        ],
        step_groups=[
            StepGroup(
                name="all",
                steps=[1, 2],
                llm_config=LLMStepConfig(
                    temperature=0.0,
                    execution_mode=ExecutionMode.FAST,
                ),
            )
        ],
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="data", api=_NoopLLM())
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    # The merge result is observable on the chain itself:
    assert chain.steps[0].llm_config.temperature == 0.0
    assert chain.steps[1].llm_config.temperature == 0.0
