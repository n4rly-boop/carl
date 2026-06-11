"""
Tests for AgentSkill LLM_AGENT structured output schema.

Covers the standalone validator + integration with ``AgentSkillStepExecutor``.
The integration test uses ``ChainTestHarness`` to drive a fake LLM_AGENT
final response through the executor without hitting a real LLM.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_carl import (
    SkillOutputSchemaError,
    parse_and_validate_skill_output,
)
from mmar_carl.skill_output_schema import _strip_code_fences


# --------------------------------------------------------------------------- #
# parse_and_validate_skill_output — happy paths
# --------------------------------------------------------------------------- #


def test_valid_object_matches_schema() -> None:
    schema = {
        "type": "object",
        "required": ["name", "count"],
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    out = parse_and_validate_skill_output('{"name": "alice", "count": 7}', schema)
    assert out == {"name": "alice", "count": 7}


def test_array_with_items_schema() -> None:
    schema = {"type": "array", "items": {"type": "string"}}
    out = parse_and_validate_skill_output('["a", "b", "c"]', schema)
    assert out == ["a", "b", "c"]


def test_nested_object_validation() -> None:
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
            },
            "items": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["user"],
    }
    out = parse_and_validate_skill_output(
        '{"user": {"id": 42, "name": "x"}, "items": [1, 2, 3]}',
        schema,
    )
    assert out["user"]["id"] == 42


def test_optional_property_can_be_absent() -> None:
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
    }
    out = parse_and_validate_skill_output('{"a": "x"}', schema)
    assert out == {"a": "x"}


def test_enum_constraint_accepts_member() -> None:
    schema = {"type": "string", "enum": ["red", "green", "blue"]}
    out = parse_and_validate_skill_output('"green"', schema)
    assert out == "green"


def test_no_type_in_schema_skips_type_check() -> None:
    """A schema without 'type' is permissive about the value's outer type."""
    out = parse_and_validate_skill_output('"anything"', {})
    assert out == "anything"
    out = parse_and_validate_skill_output("123", {})
    assert out == 123


# --------------------------------------------------------------------------- #
# Code-fence stripping
# --------------------------------------------------------------------------- #


def test_strip_code_fences_with_language_tag() -> None:
    text = '```json\n{"a": 1}\n```'
    assert _strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_without_language_tag() -> None:
    text = '```\n{"a": 1}\n```'
    assert _strip_code_fences(text) == '{"a": 1}'


def test_strip_code_fences_keeps_plain_json_unchanged() -> None:
    assert _strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_validate_code_fenced_output() -> None:
    out = parse_and_validate_skill_output(
        '```json\n{"k": "v"}\n```',
        {"type": "object"},
    )
    assert out == {"k": "v"}


# --------------------------------------------------------------------------- #
# Failure cases
# --------------------------------------------------------------------------- #


def test_empty_response_fails() -> None:
    with pytest.raises(SkillOutputSchemaError, match="Empty"):
        parse_and_validate_skill_output("", {"type": "object"})
    with pytest.raises(SkillOutputSchemaError, match="Empty"):
        parse_and_validate_skill_output("   \n\t  ", {"type": "object"})


def test_invalid_json_fails() -> None:
    with pytest.raises(SkillOutputSchemaError, match="not valid JSON"):
        parse_and_validate_skill_output("{not json}", {"type": "object"})


def test_wrong_top_level_type_fails() -> None:
    with pytest.raises(SkillOutputSchemaError, match="expected type object"):
        parse_and_validate_skill_output('"a string"', {"type": "object"})


def test_missing_required_key_fails() -> None:
    schema = {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
    with pytest.raises(SkillOutputSchemaError, match="missing required key 'b'"):
        parse_and_validate_skill_output('{"a": "x"}', schema)


def test_wrong_nested_property_type_fails_with_path() -> None:
    schema = {
        "type": "object",
        "properties": {"user": {"type": "object", "properties": {"id": {"type": "integer"}}}},
    }
    with pytest.raises(SkillOutputSchemaError, match=r"\$.user.id: expected type integer"):
        parse_and_validate_skill_output('{"user": {"id": "not-an-int"}}', schema)


def test_array_item_type_violation_fails_with_index_path() -> None:
    schema = {"type": "array", "items": {"type": "integer"}}
    with pytest.raises(SkillOutputSchemaError, match=r"\$\[1\]: expected type integer"):
        parse_and_validate_skill_output('[1, "oops", 3]', schema)


def test_enum_violation_fails() -> None:
    schema = {"type": "string", "enum": ["a", "b"]}
    with pytest.raises(SkillOutputSchemaError, match="not in enum"):
        parse_and_validate_skill_output('"c"', schema)


# --------------------------------------------------------------------------- #
# Type quirks
# --------------------------------------------------------------------------- #


def test_bool_does_not_satisfy_integer_or_number_type() -> None:
    """Python bool is a subclass of int, but JSON schema treats them as distinct."""
    with pytest.raises(SkillOutputSchemaError, match="expected type integer, got bool"):
        parse_and_validate_skill_output("true", {"type": "integer"})
    with pytest.raises(SkillOutputSchemaError, match="expected type number, got bool"):
        parse_and_validate_skill_output("false", {"type": "number"})


def test_integer_satisfies_number_type() -> None:
    parse_and_validate_skill_output("42", {"type": "number"})


def test_float_does_not_satisfy_integer_type() -> None:
    with pytest.raises(SkillOutputSchemaError, match="expected type integer, got float"):
        parse_and_validate_skill_output("3.14", {"type": "integer"})


def test_null_type_accepts_none() -> None:
    out = parse_and_validate_skill_output("null", {"type": "null"})
    assert out is None


def test_union_type_via_type_list() -> None:
    """Schema type as a list of strings means 'any of these'."""
    schema = {"type": ["string", "null"]}
    assert parse_and_validate_skill_output('"hello"', schema) == "hello"
    assert parse_and_validate_skill_output("null", schema) is None
    with pytest.raises(SkillOutputSchemaError, match="expected type string | null"):
        parse_and_validate_skill_output("42", schema)


# --------------------------------------------------------------------------- #
# Unknown keys are ignored (forward compatibility)
# --------------------------------------------------------------------------- #


def test_unknown_schema_keys_are_silently_ignored() -> None:
    """Extra schema keys (description, examples, format, etc.) don't break validation."""
    schema = {
        "type": "object",
        "description": "ignored",
        "additionalProperties": False,  # not enforced
        "properties": {"x": {"type": "integer", "minimum": 0}},  # minimum not enforced
    }
    out = parse_and_validate_skill_output('{"x": -5, "extra": "allowed"}', schema)
    assert out == {"x": -5, "extra": "allowed"}


# --------------------------------------------------------------------------- #
# Integration with AgentSkillStepConfig
# --------------------------------------------------------------------------- #


def test_agent_skill_step_config_accepts_output_schema_fields() -> None:
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillStepConfig,
    )

    cfg = AgentSkillStepConfig(
        skill="local:///nonexistent",
        task="x",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        output_schema={"type": "object"},
        output_schema_strict=False,
    )
    assert cfg.output_schema == {"type": "object"}
    assert cfg.output_schema_strict is False


def test_agent_skill_step_config_default_output_schema_is_none() -> None:
    from mmar_carl.models.agent_skill import AgentSkillStepConfig

    cfg = AgentSkillStepConfig(skill="local:///x", task="x")
    assert cfg.output_schema is None
    assert cfg.output_schema_strict is True


# --------------------------------------------------------------------------- #
# End-to-end with AgentSkillStepExecutor + a mocked LLM
# --------------------------------------------------------------------------- #


def _make_mock_context():
    """Same context shape as tests/test_agent_skill_step.py uses."""
    ctx = MagicMock()
    ctx.outer_context = ""
    ctx.memory = {}
    ctx.history = []
    ctx.metadata = {}
    ctx.language = "en"
    ctx.retry_max = 1
    ctx.api = MagicMock()
    ctx.memory_schema = None
    # MagicMock's default makes is_cancelled() return a truthy MagicMock;
    # intra-step polls would short-circuit. Force False.
    ctx.is_cancelled = MagicMock(return_value=False)
    return ctx


def _make_skill_dir(tmp_path) -> str:
    skill_dir = tmp_path / "schema-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: schema-skill\ndescription: produces structured output\n---\n"
        "Return structured JSON output for the task.\n"
    )
    return str(skill_dir)


@pytest.mark.asyncio
async def test_llm_agent_valid_schema_output_succeeds(tmp_path) -> None:
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce JSON",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
        output_schema={
            "type": "object",
            "required": ["status", "count"],
            "properties": {
                "status": {"type": "string"},
                "count": {"type": "integer"},
            },
        },
    )
    step = MagicMock(number=1, title="schema step", step_config=config)
    ctx = _make_mock_context()
    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = AsyncMock(
        return_value=('{"status": "ok", "count": 3}', [])
    )
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is True
    assert result.result_data.get("schema_validated") is True
    assert result.result_data.get("parsed_output") == {"status": "ok", "count": 3}


@pytest.mark.asyncio
async def test_llm_agent_invalid_json_strict_fails(tmp_path) -> None:
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce JSON",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
        output_schema={"type": "object", "required": ["k"]},
        output_schema_strict=True,
    )
    step = MagicMock(number=1, title="bad json", step_config=config)
    ctx = _make_mock_context()
    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = AsyncMock(
        return_value=("this is not valid JSON at all", [])
    )
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is False
    assert "not valid JSON" in result.error_message


@pytest.mark.asyncio
async def test_llm_agent_invalid_schema_strict_fails(tmp_path) -> None:
    """Parseable JSON but wrong type → strict mode fails the step."""
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce JSON",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
        output_schema={
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
        },
        output_schema_strict=True,
    )
    step = MagicMock(number=1, title="wrong type", step_config=config)
    ctx = _make_mock_context()
    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = AsyncMock(
        return_value=('{"count": "not-a-number"}', [])
    )
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is False
    assert "expected type integer" in result.error_message


@pytest.mark.asyncio
async def test_llm_agent_invalid_schema_lenient_succeeds_with_warning(tmp_path) -> None:
    """With ``output_schema_strict=False`` the step succeeds and surfaces a warning."""
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce JSON",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
        output_schema={
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
        },
        output_schema_strict=False,
    )
    step = MagicMock(number=1, title="lenient", step_config=config)
    ctx = _make_mock_context()
    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = AsyncMock(
        return_value=("not valid json", []),
    )
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is True
    assert result.result_data.get("schema_validated") is False
    warnings = result.result_data.get("schema_warnings", [])
    assert len(warnings) == 1
    assert "not valid JSON" in warnings[0]


@pytest.mark.asyncio
async def test_llm_agent_no_schema_does_not_attempt_validation(tmp_path) -> None:
    """When ``output_schema`` is None, the executor skips validation entirely."""
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce anything",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
    )
    step = MagicMock(number=1, title="no schema", step_config=config)
    ctx = _make_mock_context()
    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = AsyncMock(
        return_value=("Free-form prose, definitely not JSON.", [])
    )
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is True
    # No schema_validated key when validation was skipped
    assert "schema_validated" not in result.result_data
    assert "schema_warnings" not in result.result_data


@pytest.mark.asyncio
async def test_llm_agent_schema_instruction_appended_to_user_prompt(tmp_path) -> None:
    """When ``output_schema`` is set, the LLM's user prompt mentions the schema."""
    from mmar_carl.models.agent_skill import (
        AgentSkillExecutionMode,
        AgentSkillSource,
        AgentSkillStepConfig,
    )
    from mmar_carl.step_executors import AgentSkillStepExecutor

    config = AgentSkillStepConfig(
        skill=AgentSkillSource(path=_make_skill_dir(tmp_path)),
        task="Produce JSON",
        execution_mode=AgentSkillExecutionMode.LLM_AGENT,
        llm_max_iterations=2,
        output_schema={"type": "object", "properties": {"k": {"type": "string"}}},
    )
    step = MagicMock(number=1, title="schema prompted", step_config=config)
    ctx = _make_mock_context()
    captured_prompts: dict[str, str] = {}

    async def fake_get_response_with_tools(**kwargs) -> tuple[str, list]:
        captured_prompts["user_prompt"] = kwargs.get("user_prompt", "")
        return '{"k": "v"}', []

    mock_llm = AsyncMock()
    mock_llm.get_response_with_tools = fake_get_response_with_tools
    ctx.get_llm_client_for_step = MagicMock(return_value=mock_llm)

    executor = AgentSkillStepExecutor()
    result = await executor.execute(step, ctx)
    assert result.success is True
    assert "Output requirement" in captured_prompts["user_prompt"]
    assert "JSON" in captured_prompts["user_prompt"]
    # The schema itself is included in the prompt
    assert '"k"' in captured_prompts["user_prompt"]
