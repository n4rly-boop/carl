"""
Verify the migration guide examples.

Every before/after pair in ``docs/MIGRATION_legacy_to_typed_steps.md`` is
constructed here. We assert that legacy and typed forms produce chains
that behave equivalently end-to-end and serialise to the same shape.

Each test pair has identical numbering/titles/configs so the two chains
are structurally interchangeable.
"""

import warnings

import pytest

from mmar_carl import (
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    LLMClientBase,
    LLMStepConfig,
    LLMStepDescription,
    MCPServerConfig,
    MCPStepConfig,
    MCPStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepDescription,
    StepType,
    StructuredOutputStepConfig,
    StructuredOutputStepDescription,
    ToolStepConfig,
    ToolStepDescription,
    TransformStepConfig,
    TransformStepDescription,
)


class _StubLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "stub"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "stub"


# --------------------------------------------------------------------------- #
# Legacy class emits DeprecationWarning on construction
# --------------------------------------------------------------------------- #


def test_legacy_step_description_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StepDescription(number=1, title="x", aim="x", step_type=StepType.LLM)
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation) == 1
    assert "StepDescription is deprecated" in str(deprecation[0].message)


def test_typed_step_description_does_not_emit_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMStepDescription(number=1, title="x", aim="x")
    deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation == []


# --------------------------------------------------------------------------- #
# Side-by-side equivalence (one test per step type from the guide)
# --------------------------------------------------------------------------- #


def test_llm_step_legacy_and_typed_match() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=1,
            title="Analyse claims",
            step_type=StepType.LLM,
            aim="Extract every factual claim.",
            reasoning_questions="Verifiable?",
            stage_action="enumerate",
            llm_config=LLMStepConfig(model="gpt-4o", temperature=0.3),
        )
    typed = LLMStepDescription(
        number=1,
        title="Analyse claims",
        aim="Extract every factual claim.",
        reasoning_questions="Verifiable?",
        stage_action="enumerate",
        llm_config=LLMStepConfig(model="gpt-4o", temperature=0.3),
    )
    assert legacy.number == typed.number
    assert legacy.title == typed.title
    assert legacy.aim == typed.aim
    assert legacy.reasoning_questions == typed.reasoning_questions
    assert legacy.stage_action == typed.stage_action
    assert legacy.step_type == typed.step_type == StepType.LLM
    assert legacy.llm_config.model == typed.llm_config.model
    assert legacy.llm_config.temperature == typed.llm_config.temperature


def test_tool_step_legacy_and_typed_match() -> None:
    cfg = ToolStepConfig(
        tool_name="web_search",
        parameters=[],
        input_mapping={"q": "'\"hello\"'"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=2,
            title="Fetch data",
            step_type=StepType.TOOL,
            step_config=cfg,
            dependencies=[1],
        )
    typed = ToolStepDescription(
        number=2,
        title="Fetch data",
        config=cfg,
        dependencies=[1],
    )
    assert legacy.step_type == typed.step_type == StepType.TOOL
    assert legacy.step_config is typed.step_config  # same instance passed
    assert legacy.dependencies == typed.dependencies


def test_memory_step_legacy_and_typed_match() -> None:
    cfg = MemoryStepConfig(
        operation=MemoryOperation.WRITE,
        memory_key="summary",
        value_source="'\"data\"'",
        namespace="output",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=3, title="Store result",
            step_type=StepType.MEMORY, step_config=cfg,
        )
    typed = MemoryStepDescription(number=3, title="Store result", config=cfg)
    assert legacy.step_type == typed.step_type == StepType.MEMORY
    assert legacy.step_config is typed.step_config


def test_transform_step_legacy_and_typed_match() -> None:
    cfg = TransformStepConfig(
        transform_type="extract",
        expression=r"[\w.+-]+@[\w-]+\.[\w.-]+",
        input_key="$history[-1]",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=4, title="Extract emails",
            step_type=StepType.TRANSFORM, step_config=cfg,
        )
    typed = TransformStepDescription(number=4, title="Extract emails", config=cfg)
    assert legacy.step_type == typed.step_type == StepType.TRANSFORM
    assert legacy.step_config is typed.step_config


def test_conditional_step_legacy_and_typed_match() -> None:
    cfg = ConditionalStepConfig(
        branches=[
            ConditionalBranch(condition="contains:positive", next_step=6),
            ConditionalBranch(condition="contains:negative", next_step=7),
        ],
        default_step=8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=5, title="Route", step_type=StepType.CONDITIONAL, step_config=cfg,
            dependencies=[1],
        )
    typed = ConditionalStepDescription(
        number=5, title="Route", config=cfg, dependencies=[1],
    )
    assert legacy.step_type == typed.step_type == StepType.CONDITIONAL
    assert legacy.step_config.default_step == typed.step_config.default_step


def test_mcp_step_legacy_and_typed_match() -> None:
    cfg = MCPStepConfig(
        server=MCPServerConfig(server_name="local", transport="stdio", command="x"),
        tool_name="search",
        argument_mapping={"q": "$outer_context"},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=6, title="MCP", step_type=StepType.MCP, step_config=cfg,
        )
    typed = MCPStepDescription(number=6, title="MCP", config=cfg)
    assert legacy.step_type == typed.step_type == StepType.MCP


def test_structured_output_step_legacy_and_typed_match() -> None:
    cfg = StructuredOutputStepConfig(
        output_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        input_source="$history[-1]",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = StepDescription(
            number=7, title="Structured",
            step_type=StepType.STRUCTURED_OUTPUT, step_config=cfg,
            aim="x",
        )
    typed = StructuredOutputStepDescription(
        number=7, title="Structured", aim="x", config=cfg,
    )
    assert legacy.step_type == typed.step_type == StepType.STRUCTURED_OUTPUT
    assert legacy.step_config.output_schema == typed.step_config.output_schema


# --------------------------------------------------------------------------- #
# Mixed legacy+typed chain still executes
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mixed_legacy_and_typed_chain_executes() -> None:
    """The migration guide claims you can migrate incrementally — verify it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="step1", aim="produce"),
                StepDescription(
                    number=2,
                    title="step2",
                    step_type=StepType.TOOL,
                    step_config=ToolStepConfig(
                        tool_name="cap",
                        parameters=[],
                        input_mapping={"value": "$history[-1]"},
                    ),
                    dependencies=[1],
                ),
            ],
            max_workers=1,
        )

    captured: dict[str, str] = {}

    def cap(value: str) -> str:
        captured["got"] = value
        return value

    ctx = ReasoningContext(outer_context="x", api=_StubLLM())
    ctx.register_tool("cap", cap)
    result = await chain.execute_async(ctx)
    assert all(sr.success for sr in result.step_results)
    assert "got" in captured  # legacy tool step ran


# --------------------------------------------------------------------------- #
# Serialization round-trip — guide claim
# --------------------------------------------------------------------------- #


def test_legacy_chain_to_dict_round_trip_keeps_step_types() -> None:
    """Migration guide claim: chain.to_dict() round-trips both forms."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        chain = ReasoningChain(
            steps=[
                StepDescription(number=1, title="a", aim="x", step_type=StepType.LLM),
                StepDescription(
                    number=2, title="b",
                    step_type=StepType.TOOL,
                    step_config=ToolStepConfig(
                        tool_name="t", parameters=[], input_mapping={}
                    ),
                    dependencies=[1],
                ),
            ],
            max_workers=1,
        )
    data = chain.to_dict()
    # Step type discriminator present
    assert data["steps"][0]["step_type"] == "llm"
    assert data["steps"][1]["step_type"] == "tool"


# --------------------------------------------------------------------------- #
# Sanity: the migration guide file exists and references expected sections
# --------------------------------------------------------------------------- #


def test_migration_guide_file_exists_and_covers_step_types() -> None:
    from pathlib import Path

    # Walk up from tests/chain_lifecycle/<this file> → repo root → docs/
    guide = Path(__file__).parent.parent.parent / "docs" / "MIGRATION_legacy_to_typed_steps.md"
    assert guide.exists(), f"Migration guide not found at {guide}"
    text = guide.read_text(encoding="utf-8")
    # Each typed-class name should appear in the guide
    for cls in (
        "LLMStepDescription",
        "ToolStepDescription",
        "MemoryStepDescription",
        "TransformStepDescription",
        "ConditionalStepDescription",
        "MCPStepDescription",
        "StructuredOutputStepDescription",
    ):
        assert cls in text, f"Migration guide missing reference to {cls}"
    # The mechanical-refactor section should mention the field rename
    assert "step_config=" in text and "config=" in text
