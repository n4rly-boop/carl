"""
Memory schema validation for ``ReasoningChain``.

A *memory schema* is a partial contract: a nested mapping
``{namespace: {key: type_spec}}`` declaring the expected Python type for a
specific ``(namespace, key)`` pair. Writes that match a declared
``(namespace, key)`` and violate its type spec raise
:class:`MemorySchemaError`. Writes to ``(namespace, key)`` pairs that the
schema does *not* mention are silently allowed — schemas are additive
constraints, not exhaustive lists.

Supported type specs:

- ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``, etc. — single types.
- ``typing.Optional[T]`` — ``T`` or ``None``.
- ``typing.Union[A, B, C]`` — any of A/B/C.
- ``list[X]`` / ``dict[K, V]`` — checked at the *container* level only
  (we verify the value is a ``list``/``dict``; element types are not
  recursively validated to keep this cheap).
- Tuples of types: ``(int, float)`` — matches any of them.

Example::

    schema = {
        "input": {"pdf_path": str, "language": str},
        "output": {"summary": str, "slides_path": Optional[str]},
    }
"""

from __future__ import annotations

import typing
from typing import Any, Optional


class MemorySchemaError(TypeError):
    """Raised when a memory write violates the declared schema."""

    def __init__(
        self,
        namespace: str,
        key: str,
        expected: tuple[type, ...],
        actual_value: Any,
    ) -> None:
        self.namespace = namespace
        self.key = key
        self.expected = expected
        self.actual_value = actual_value
        expected_str = " | ".join(t.__name__ for t in expected)
        actual_type = type(actual_value).__name__
        super().__init__(
            f"memory['{namespace}']['{key}'] expected {expected_str}, "
            f"got {actual_type} (value={actual_value!r})"
        )


def _normalize_type_spec(spec: Any) -> tuple[type, ...]:
    """Turn a user-facing type spec into a tuple suitable for ``isinstance``.

    Handles single types, tuples, ``Optional[X]``, ``Union[A, B, ...]``, and
    parameterised generics like ``list[X]`` / ``dict[K, V]`` (collapsing them
    to their origin container).
    """
    # Direct type already
    if isinstance(spec, type):
        return (spec,)
    # Tuple of types
    if isinstance(spec, tuple):
        flat: list[type] = []
        for item in spec:
            flat.extend(_normalize_type_spec(item))
        # Deduplicate while preserving order
        seen: set[type] = set()
        result: list[type] = []
        for t in flat:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return tuple(result)
    # typing constructs (Union / Optional / Generic alias)
    origin = typing.get_origin(spec)
    if origin is typing.Union:
        return _normalize_type_spec(tuple(typing.get_args(spec)))
    if origin is not None:
        # Parameterised generic: use the container origin (list, dict, set, ...).
        # NoneType slips through Union handling above; everything else with an
        # origin reduces to its container type.
        if isinstance(origin, type):
            return (origin,)
    # NoneType comes from Optional[...] — represented as type(None)
    if spec is None or spec is type(None):  # noqa: E721 — intentional identity test
        return (type(None),)
    raise TypeError(
        f"Unsupported memory_schema type spec: {spec!r} (origin={origin!r})"
    )


def validate_memory_write(
    schema: Optional[dict[str, dict[str, Any]]],
    namespace: str,
    key: str,
    value: Any,
) -> None:
    """
    Validate *value* against *schema* for ``schema[namespace][key]`` if declared.

    Silently returns when *schema* is ``None`` or the ``(namespace, key)`` pair
    is not present.
    """
    if schema is None:
        return
    ns_spec = schema.get(namespace)
    if ns_spec is None:
        return
    if key not in ns_spec:
        return

    expected = _normalize_type_spec(ns_spec[key])
    if not isinstance(value, expected):
        raise MemorySchemaError(namespace, key, expected, value)
