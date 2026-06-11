"""Round-trip every ``StepType`` value through
``chain.to_dict()`` → ``ReasoningChain.from_dict(use_typed_steps=True)``.

This test is the exhaustive companion to the focused per-step tests
elsewhere — every value of :class:`StepType` is exercised so that adding
a new step type without extending the reconstruction logic surfaces
immediately.

Runtime-only fields are intentionally *not* round-tripped:

- :attr:`AgentHandoffStepDescription.sub_chain`
- :attr:`SupervisorStepDescription.agents`
- :attr:`ParallelSamplingStepDescription.base_step`
- ``metrics``, ``cache`` runtime-only attachments

Reconstruction inserts ``__placeholder__`` sentinels so the chain spec is
re-instantiable; CARE rebuilds the real values from gigaevo-memory entity
refs before execution.
"""

from __future__ import annotations

import json

import pytest

from mmar_carl import (
    AgentHandoffStepDescription,
    AgentSkillStepDescription,
    ConditionalStepDescription,
    DebateStepDescription,
    EvaluationStepDescription,
    HumanInputStepDescription,
    LLMStepDescription,
    MCPResourceStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    ParallelSamplingStepDescription,
    ReasoningChain,
    StructuredOutputStepDescription,
    SupervisorStepDescription,
    ToolDiscoveryStepDescription,
    ToolStepDescription,
    TransformStepDescription,
)
from mmar_carl.models.agent_skill import (
    AgentSkillExecutionMode,
    AgentSkillStepConfig,
)
from mmar_carl.models.config import (
    AgentHandoffStepConfig,
    ConditionalBranch,
    ConditionalStepConfig,
    DebateStepConfig,
    EvaluationStepConfig,
    HumanInputStepConfig,
    MCPResourceStepConfig,
    MCPServerConfig,
    MCPStepConfig,
    MemoryStepConfig,
    ModuleToolSource,
    ParallelSamplingAggregation,
    ParallelSamplingStepConfig,
    StructuredOutputStepConfig,
    SupervisorStepConfig,
    ToolDiscoveryStepConfig,
    ToolStepConfig,
    TransformStepConfig,
)
from mmar_carl.models.enums import MemoryOperation, StepType


# ---------------------------------------------------------------------------
# Fixtures — one minimal chain per StepType. The step under test is always
# the LAST entry; preceding steps (when present) exist solely to satisfy
# in-chain dependencies (e.g. EvaluationStep needs a prior step to evaluate).
# ---------------------------------------------------------------------------


def _make_steps(step_type: StepType) -> list:
    if step_type is StepType.LLM:
        return [LLMStepDescription(number=1, title="LLM step", aim="think")]

    if step_type is StepType.TOOL:
        return [ToolStepDescription(
            number=1, title="Tool step",
            config=ToolStepConfig(tool_name="lookup", timeout=12.5),
        )]

    if step_type is StepType.MCP:
        return [MCPStepDescription(
            number=1, title="MCP step",
            config=MCPStepConfig(
                server=MCPServerConfig(server_name="srv", transport="stdio"),
                tool_name="ping",
            ),
        )]

    if step_type is StepType.MCP_RESOURCE:
        return [MCPResourceStepDescription(
            number=1, title="MCP resource step",
            config=MCPResourceStepConfig(
                server=MCPServerConfig(server_name="srv", transport="stdio"),
                resource_uri="docs://x.md",
                output_memory_key="docs",
            ),
        )]

    if step_type is StepType.MEMORY:
        return [MemoryStepDescription(
            number=1, title="Memory step",
            config=MemoryStepConfig(
                operation=MemoryOperation.WRITE,
                memory_key="k",
                value_source="'v'",
                namespace="ns",
            ),
        )]

    if step_type is StepType.TRANSFORM:
        return [TransformStepDescription(
            number=1, title="Transform step",
            config=TransformStepConfig(
                transform_type="extract",
                input_key="$history[-1]",
                expression="[a-z]+",
            ),
        )]

    if step_type is StepType.CONDITIONAL:
        return [ConditionalStepDescription(
            number=1, title="Conditional step",
            config=ConditionalStepConfig(
                branches=[ConditionalBranch(condition="x > 0", next_step=2)],
                default_step=3,
            ),
        )]

    if step_type is StepType.STRUCTURED_OUTPUT:
        return [StructuredOutputStepDescription(
            number=1, title="Structured step",
            config=StructuredOutputStepConfig(
                output_schema={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
                schema_name="Out",
            ),
        )]

    if step_type is StepType.AGENT_SKILL:
        return [AgentSkillStepDescription(
            number=1, title="AgentSkill step",
            config=AgentSkillStepConfig(
                skill="example-skill",
                task="do the task",
                execution_mode=AgentSkillExecutionMode.LLM,
            ),
        )]

    if step_type is StepType.EVALUATION:
        return [
            LLMStepDescription(number=1, title="LLM prereq", aim="produce output"),
            EvaluationStepDescription(
                number=2, title="Eval step", dependencies=[1],
                config=EvaluationStepConfig(
                    evaluates_step=1,
                    criteria=["min_words:5"],
                ),
            ),
        ]

    if step_type is StepType.AGENT_HANDOFF:
        # Sub-chain is runtime-only; provide a fake one purely so we can
        # construct the original step.
        sub = ReasoningChain(steps=[LLMStepDescription(
            number=1, title="sub", aim="sub-aim",
        )])
        return [AgentHandoffStepDescription(
            number=1, title="Handoff step",
            sub_chain=sub,
            config=AgentHandoffStepConfig(
                output_memory_key="result",
                output_namespace="handoff",
            ),
        )]

    if step_type is StepType.SUPERVISOR:
        sub = ReasoningChain(steps=[LLMStepDescription(
            number=1, title="sub", aim="sub-aim",
        )])
        return [SupervisorStepDescription(
            number=1, title="Supervisor step",
            agents={"alpha": sub, "beta": sub},
            config=SupervisorStepConfig(
                routing_prompt="Pick one for {task}. Options: {agents}",
                output_memory_key="winner",
            ),
        )]

    if step_type is StepType.DEBATE:
        return [DebateStepDescription(
            number=1, title="Debate step",
            config=DebateStepConfig(
                roles=["pro", "con"],
                rounds=1,
                judge_prompt="Synthesise: {task}\n{transcript}",
            ),
        )]

    if step_type is StepType.PARALLEL_SAMPLING:
        return [ParallelSamplingStepDescription(
            number=1, title="ParallelSampling step",
            base_step=LLMStepDescription(number=1, title="b", aim="b-aim"),
            config=ParallelSamplingStepConfig(
                n_samples=3,
                aggregation=ParallelSamplingAggregation.MAJORITY_VOTE,
            ),
        )]

    if step_type is StepType.TOOL_DISCOVERY:
        return [ToolDiscoveryStepDescription(
            number=1, title="ToolDiscovery step",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="myapp.tools", name_prefix="t_"),
                output_memory_key="loaded",
            ),
        )]

    if step_type is StepType.HUMAN_INPUT:
        return [HumanInputStepDescription(
            number=1, title="HumanInput step",
            config=HumanInputStepConfig(
                prompt="What now?",
                fallback_value="[skip]",
                output_memory_key="user_response",
            ),
        )]

    raise AssertionError(f"unhandled step type: {step_type!r}")


# Runtime-only fields that don't survive the round trip — caller must
# rebuild from external state. The test asserts a placeholder sentinel
# value, not the original.
_RUNTIME_ONLY: dict[StepType, set[str]] = {
    StepType.AGENT_HANDOFF: {"sub_chain"},
    StepType.SUPERVISOR: {"agents"},
    StepType.PARALLEL_SAMPLING: {"base_step"},
}


# ---------------------------------------------------------------------------
# Exhaustive round-trip — one parametrised case per StepType.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step_type",
    list(StepType),
    ids=lambda s: s.value,
)
def test_step_type_round_trips(step_type: StepType) -> None:
    steps = _make_steps(step_type)
    original = ReasoningChain(steps=steps)

    data = original.to_dict()
    restored = ReasoningChain.from_dict(data, use_typed_steps=True)

    assert len(restored.steps) == len(steps)
    target_original = steps[-1]
    target_restored = restored.steps[-1]

    # Step type matches.
    assert target_restored.step_type == step_type
    # Class type matches (typed step factory chose correctly).
    assert type(target_restored) is type(target_original), (
        f"got {type(target_restored).__name__}, "
        f"expected {type(target_original).__name__}"
    )
    # Core scalar fields preserved.
    assert target_restored.number == target_original.number
    assert target_restored.title == target_original.title
    assert target_restored.dependencies == target_original.dependencies

    # Step-specific config — compare via model_dump for typed configs.
    original_config = getattr(target_original, "config", None)
    restored_config = getattr(target_restored, "config", None)
    if original_config is not None:
        assert restored_config is not None
        assert type(restored_config) is type(original_config)
        original_dump = original_config.model_dump(mode="json")
        restored_dump = restored_config.model_dump(mode="json")
        assert restored_dump == original_dump, (
            f"config mismatch for {step_type.value}: "
            f"\noriginal={original_dump}\nrestored={restored_dump}"
        )

    # Runtime-only fields are *not* preserved — placeholders are inserted.
    for field_name in _RUNTIME_ONLY.get(step_type, set()):
        original_value = getattr(target_original, field_name)
        restored_value = getattr(target_restored, field_name)
        assert restored_value is not original_value


# ---------------------------------------------------------------------------
# Exhaustiveness guard — fail loudly if a new StepType is added.
# ---------------------------------------------------------------------------


def test_all_step_types_have_fixtures() -> None:
    for step_type in StepType:
        steps = _make_steps(step_type)
        assert steps[-1].step_type == step_type, (
            f"_make_steps({step_type!r}) returned terminal step with "
            f"step_type={steps[-1].step_type!r}"
        )


# ---------------------------------------------------------------------------
# Runtime-only field documentation — sanity-check the placeholders.
# ---------------------------------------------------------------------------


class TestRuntimeOnlyPlaceholders:
    """Documents the rebuild contract: when a chain with runtime-only
    fields is reconstructed from a dict, those fields are populated with
    a placeholder. CARE must inject the real values from gigaevo-memory
    before executing the chain.
    """

    def test_agent_handoff_sub_chain_is_placeholder(self) -> None:
        chain = ReasoningChain(steps=_make_steps(StepType.AGENT_HANDOFF))
        restored = ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
        restored_step = restored.steps[-1]
        # Placeholder is a chain with a single `__placeholder__` LLM step.
        assert isinstance(restored_step.sub_chain, ReasoningChain)
        assert len(restored_step.sub_chain.steps) == 1
        assert restored_step.sub_chain.steps[0].title == "__placeholder__"

    def test_supervisor_agents_is_empty_dict(self) -> None:
        chain = ReasoningChain(steps=_make_steps(StepType.SUPERVISOR))
        restored = ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
        restored_step = restored.steps[-1]
        assert restored_step.agents == {}
        # But the routing prompt + agent-naming config DID survive.
        assert "{task}" in restored_step.config.routing_prompt

    def test_parallel_sampling_base_step_is_placeholder(self) -> None:
        chain = ReasoningChain(steps=_make_steps(StepType.PARALLEL_SAMPLING))
        restored = ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
        restored_step = restored.steps[-1]
        assert isinstance(restored_step.base_step, LLMStepDescription)
        assert restored_step.base_step.title == "__placeholder__"
        # But the sampling config DID survive.
        assert restored_step.config.n_samples == 3


# ---------------------------------------------------------------------------
# JSON round-trip — guards file-based persistence.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "step_type",
    list(StepType),
    ids=lambda s: s.value,
)
def test_step_type_json_round_trip(step_type: StepType) -> None:
    steps = _make_steps(step_type)
    original = ReasoningChain(steps=steps)

    json_str = original.to_json()
    restored = ReasoningChain.from_dict(json.loads(json_str), use_typed_steps=True)
    assert restored.steps[-1].step_type == step_type
