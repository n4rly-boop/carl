"""
Tests for memory schema validation.

A memory schema is a partial contract: ``{namespace: {key: type_spec}}``.
Writes that match a declared ``(namespace, key)`` and violate its type spec
raise :class:`MemorySchemaError`. Writes to pairs not in the schema are
silently allowed.
"""

from typing import Optional, Union

import pytest

from mmar_carl import (
    LLMClientBase,
    LLMStepDescription,
    MemoryOperation,
    MemoryStepConfig,
    MemoryStepDescription,
    MemorySchemaError,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.memory_schema import _normalize_type_spec, validate_memory_write


# --------------------------------------------------------------------------- #
# Mock
# --------------------------------------------------------------------------- #


class _Stub(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


# --------------------------------------------------------------------------- #
# _normalize_type_spec
# --------------------------------------------------------------------------- #


def test_normalize_single_type() -> None:
    assert _normalize_type_spec(str) == (str,)
    assert _normalize_type_spec(int) == (int,)


def test_normalize_tuple_of_types() -> None:
    assert set(_normalize_type_spec((int, float))) == {int, float}


def test_normalize_optional() -> None:
    assert set(_normalize_type_spec(Optional[str])) == {str, type(None)}


def test_normalize_union() -> None:
    assert set(_normalize_type_spec(Union[int, str, None])) == {int, str, type(None)}


def test_normalize_parameterised_generic_collapses_to_container() -> None:
    assert _normalize_type_spec(list[int]) == (list,)
    assert _normalize_type_spec(dict[str, int]) == (dict,)


def test_normalize_unsupported_spec_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported memory_schema type spec"):
        _normalize_type_spec("not-a-type")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# validate_memory_write (low-level)
# --------------------------------------------------------------------------- #


def test_validate_none_schema_is_noop() -> None:
    validate_memory_write(None, "input", "x", 123)  # no exception


def test_validate_namespace_not_in_schema_is_noop() -> None:
    schema = {"input": {"x": str}}
    validate_memory_write(schema, "other", "x", 123)  # no exception


def test_validate_key_not_in_namespace_is_noop() -> None:
    schema = {"input": {"x": str}}
    validate_memory_write(schema, "input", "y", 123)  # no exception


def test_validate_type_match_passes() -> None:
    validate_memory_write({"input": {"x": str}}, "input", "x", "hello")


def test_validate_type_mismatch_raises_with_descriptive_message() -> None:
    schema = {"input": {"x": str}}
    with pytest.raises(MemorySchemaError) as exc_info:
        validate_memory_write(schema, "input", "x", 42)
    e = exc_info.value
    assert e.namespace == "input"
    assert e.key == "x"
    assert str in e.expected
    assert e.actual_value == 42
    assert "expected str" in str(e)
    assert "got int" in str(e)


def test_validate_optional_allows_none() -> None:
    schema = {"output": {"slides_path": Optional[str]}}
    validate_memory_write(schema, "output", "slides_path", None)
    validate_memory_write(schema, "output", "slides_path", "/tmp/x")
    with pytest.raises(MemorySchemaError):
        validate_memory_write(schema, "output", "slides_path", 123)


# --------------------------------------------------------------------------- #
# Context-level integration
# --------------------------------------------------------------------------- #


def test_context_memory_write_validates() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"input": {"pdf_path": str}},
    )
    ctx.memory_write("pdf_path", "/tmp/x.pdf", namespace="input")
    assert ctx.memory["input"]["pdf_path"] == "/tmp/x.pdf"

    with pytest.raises(MemorySchemaError, match="pdf_path"):
        ctx.memory_write("pdf_path", 123, namespace="input")
    # No partial write on failure
    assert ctx.memory["input"]["pdf_path"] == "/tmp/x.pdf"


def test_context_unspec_keys_pass_through() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"input": {"pdf_path": str}},
    )
    # 'extra' isn't in schema — silently allowed
    ctx.memory_write("extra", 123, namespace="input")
    ctx.memory_write("anything", [1, 2, 3], namespace="other_ns")
    assert ctx.memory["input"]["extra"] == 123
    assert ctx.memory["other_ns"]["anything"] == [1, 2, 3]


def test_context_optional_field_accepts_none_and_str() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"output": {"slides_path": Optional[str]}},
    )
    ctx.memory_write("slides_path", None, namespace="output")
    ctx.memory_write("slides_path", "/tmp/x.pptx", namespace="output")
    with pytest.raises(MemorySchemaError):
        ctx.memory_write("slides_path", 42, namespace="output")


def test_context_with_no_schema_writes_anything() -> None:
    ctx = ReasoningContext(outer_context="x", api=_Stub())  # no schema
    ctx.memory_write("k", 1, namespace="ns")
    ctx.memory_write("k", "now a string", namespace="ns")
    ctx.memory_write("k", None, namespace="ns")
    assert ctx.memory["ns"]["k"] is None


# --------------------------------------------------------------------------- #
# memory_append
# --------------------------------------------------------------------------- #


def test_append_against_plain_list_spec_allows_any_element() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"events": {"log": list}},
    )
    ctx.memory_append("log", "hello", namespace="events")
    ctx.memory_append("log", 42, namespace="events")
    ctx.memory_append("log", {"k": "v"}, namespace="events")
    assert ctx.memory["events"]["log"] == ["hello", 42, {"k": "v"}]


def test_append_against_parameterised_list_checks_element_type() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"events": {"log": list[str]}},
    )
    ctx.memory_append("log", "a", namespace="events")
    ctx.memory_append("log", "b", namespace="events")
    with pytest.raises(MemorySchemaError, match=r"log\[\]"):
        ctx.memory_append("log", 42, namespace="events")
    # No partial mutation on failure
    assert ctx.memory["events"]["log"] == ["a", "b"]


def test_append_against_non_list_spec_raises() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"input": {"name": str}},
    )
    with pytest.raises(MemorySchemaError, match="expected list"):
        ctx.memory_append("name", "x", namespace="input")


def test_append_against_optional_list_element_accepts_none() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"events": {"log": list[Optional[str]]}},
    )
    ctx.memory_append("log", None, namespace="events")
    ctx.memory_append("log", "hi", namespace="events")
    with pytest.raises(MemorySchemaError):
        ctx.memory_append("log", 99, namespace="events")


# --------------------------------------------------------------------------- #
# Chain-level memory_schema propagates onto context
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chain_memory_schema_pushed_onto_context_at_execution() -> None:
    """When the chain declares a schema and the context doesn't,
    the schema is pushed onto the context at chain.execute_async() entry."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="produce", aim="x"),
            MemoryStepDescription(
                number=2,
                title="store",
                dependencies=[1],
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="summary",
                    value_source="$history[-1]",
                    namespace="output",
                ),
            ),
        ],
        memory_schema={"output": {"summary": str}},
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    assert ctx.memory_schema is None
    result = await chain.execute_async(ctx)
    assert result.step_results[1].success
    # Schema was attached at execution start
    assert ctx.memory_schema == {"output": {"summary": str}}


@pytest.mark.asyncio
async def test_context_schema_takes_precedence_over_chain_schema() -> None:
    """If the user pre-sets ``context.memory_schema``, the chain doesn't overwrite it."""
    chain = ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="produce", aim="x"),
        ],
        memory_schema={"chain_ns": {"k": int}},
        max_workers=1,
    )
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"user_ns": {"k": str}},
    )
    await chain.execute_async(ctx)
    # User's schema preserved
    assert ctx.memory_schema == {"user_ns": {"k": str}}


# --------------------------------------------------------------------------- #
# Memory step → schema failure → step failure
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_memory_step_failure_when_value_violates_schema() -> None:
    """A MemoryStep writing a wrong-typed value should fail the step."""
    chain = ReasoningChain(
        steps=[
            MemoryStepDescription(
                number=1,
                title="bad write",
                config=MemoryStepConfig(
                    operation=MemoryOperation.WRITE,
                    memory_key="age",
                    value_source="'not-an-int'",  # literal string
                    namespace="user",
                ),
            ),
        ],
        memory_schema={"user": {"age": int}},
        max_workers=1,
    )
    ctx = ReasoningContext(outer_context="x", api=_Stub())
    result = await chain.execute_async(ctx)
    sr = result.step_results[0]
    assert not sr.success
    assert "expected int" in sr.error_message


# --------------------------------------------------------------------------- #
# Union / tuple specs
# --------------------------------------------------------------------------- #


def test_union_type_accepts_any_member() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"any": {"v": Union[int, str]}},
    )
    ctx.memory_write("v", 1, namespace="any")
    ctx.memory_write("v", "hello", namespace="any")
    with pytest.raises(MemorySchemaError):
        ctx.memory_write("v", [1, 2], namespace="any")


def test_tuple_spec_accepts_any_member() -> None:
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"any": {"v": (int, float)}},
    )
    ctx.memory_write("v", 1, namespace="any")
    ctx.memory_write("v", 1.5, namespace="any")
    with pytest.raises(MemorySchemaError):
        ctx.memory_write("v", "no", namespace="any")


# --------------------------------------------------------------------------- #
# Bool / int interaction (Python's bool is a subclass of int — document this)
# --------------------------------------------------------------------------- #


def test_bool_is_accepted_where_int_is_declared_python_subclass_quirk() -> None:
    """Python's ``bool`` inherits from ``int``, so ``isinstance(True, int)`` is True.

    This is documented behaviour — the schema mirrors Python's type system.
    Pinning a key to ``int`` and writing ``True`` therefore passes.
    """
    ctx = ReasoningContext(
        outer_context="x",
        api=_Stub(),
        memory_schema={"flags": {"on": int}},
    )
    ctx.memory_write("on", True, namespace="flags")
    ctx.memory_write("on", 1, namespace="flags")
    with pytest.raises(MemorySchemaError):
        ctx.memory_write("on", "yes", namespace="flags")


# --------------------------------------------------------------------------- #
# ReasoningChain constructor passes schema through
# --------------------------------------------------------------------------- #


def test_reasoning_chain_stores_memory_schema() -> None:
    schema = {"input": {"v": str}}
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="t",
                config=ToolStepConfig(tool_name="x", parameters=[], input_mapping={}),
            )
        ],
        memory_schema=schema,
        max_workers=1,
    )
    assert chain.memory_schema == schema
