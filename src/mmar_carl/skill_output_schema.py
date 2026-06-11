"""
Lightweight JSON-schema-style validator for ``AgentSkillStepConfig.output_schema``.

Honoured schema keys: ``type`` (object/array/string/number/integer/boolean/null),
``properties``, ``required``, ``items``, ``enum``. Unknown keys are ignored.

This is intentionally NOT a full JSON Schema implementation — we only support
the subset needed for AgentSkill LLM_AGENT output validation. For full schema
support, plug in `jsonschema` externally and validate the parsed dict yourself.
"""

from __future__ import annotations

import json
import re
from typing import Any


class SkillOutputSchemaError(ValueError):
    """Raised when an LLM_AGENT final response fails schema validation."""


# Map JSON Schema 'type' values to Python type tuples for isinstance() checks.
# Note: bool is checked separately because Python's bool is a subclass of int.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),  # bool excluded below
    "integer": (int,),       # bool excluded below
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
    "null": (type(None),),
}


def _strip_code_fences(text: str) -> str:
    """Remove triple-backtick code fences around JSON output.

    Some models wrap JSON in ```json ... ``` blocks despite instructions. We
    handle the common cases (with or without language tag) defensively.
    """
    stripped = text.strip()
    fence_match = re.match(
        r"^```(?:json|JSON)?\s*\n(.*?)\n```\s*$",
        stripped,
        re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def parse_and_validate_skill_output(text: str, schema: dict[str, Any]) -> Any:
    """
    Parse ``text`` as JSON and validate against ``schema``.

    Returns the parsed Python value on success. Raises
    :class:`SkillOutputSchemaError` on parse failure or schema violation.
    """
    cleaned = _strip_code_fences(text)
    if not cleaned:
        raise SkillOutputSchemaError("Empty response cannot satisfy output schema")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SkillOutputSchemaError(
            f"Final response is not valid JSON: {exc.msg} at line "
            f"{exc.lineno}, col {exc.colno}"
        ) from exc

    _validate(parsed, schema, path="$")
    return parsed


def _validate(value: Any, schema: dict[str, Any], *, path: str) -> None:
    """Recursive validator. Raises SkillOutputSchemaError on first violation."""
    if not isinstance(schema, dict):
        return  # Permissive: malformed schema, skip validation

    # type check
    expected_type = schema.get("type")
    if expected_type is not None:
        types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not _matches_any_type(value, types):
            allowed = " | ".join(types)
            raise SkillOutputSchemaError(
                f"{path}: expected type {allowed}, got {type(value).__name__}"
            )

    # enum check
    enum = schema.get("enum")
    if enum is not None:
        if value not in enum:
            raise SkillOutputSchemaError(
                f"{path}: value {value!r} not in enum {enum!r}"
            )

    # object: properties + required
    if isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                raise SkillOutputSchemaError(f"{path}: missing required key '{key}'")
        properties = schema.get("properties") or {}
        for key, sub_schema in properties.items():
            if key in value:
                _validate(value[key], sub_schema, path=f"{path}.{key}")

    # array: items
    elif isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                _validate(item, items_schema, path=f"{path}[{i}]")


def _matches_any_type(value: Any, types: list[str]) -> bool:
    """``True`` iff *value* matches at least one JSON-schema type name."""
    for t in types:
        py_types = _TYPE_MAP.get(t)
        if py_types is None:
            continue
        # Bool is an int subclass — exclude bool from number/integer.
        if t in ("number", "integer") and isinstance(value, bool):
            continue
        if isinstance(value, py_types):
            return True
    return False
