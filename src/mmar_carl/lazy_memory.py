"""
LazyMemoryValue — share-on-copy wrapper for large memory payloads.

CARL deep-copies memory in several places: when capturing/restoring RE-PLAN
checkpoints, when seeding sub-chain contexts (parallel branches, supervisor,
debate, etc.), and in some legacy paths. For large payloads (extracted PDF
text, embeddings, raw JSON dumps) these deep copies are expensive in both
time and RAM.

``LazyMemoryValue`` is an opt-in wrapper users can stash in memory whenever
the value is large and not expected to be mutated in place. Its
``__deepcopy__`` returns a *new wrapper around the same payload*, so the
payload itself is shared across snapshots/branches by reference. Mutation
is still safe — see ``mutable_copy()`` for copy-on-first-write semantics.

Usage::

    from mmar_carl import LazyMemoryValue

    context.memory_write("pdf", "text", LazyMemoryValue(big_extracted_text))
    # ...later in a step...
    payload = context.memory_read("text", namespace="pdf")
    if isinstance(payload, LazyMemoryValue):
        text = payload.value           # read-only access
    # To mutate safely without affecting sibling parallel branches:
    new_wrapper = payload.mutable_copy()
    new_wrapper.value.append(...)      # mutate the *copy*'s payload

The wrapper is transparent to ``str()``/``repr()``/``len()``/``==``/``hash()``
where it makes sense, so most read paths Just Work without explicit unwrapping.
"""

from __future__ import annotations

import copy
from typing import Any, Generic, TypeVar

__all__ = ["LazyMemoryValue", "unwrap_lazy"]

T = TypeVar("T")


class LazyMemoryValue(Generic[T]):
    """Wrapper that shares its payload across :func:`copy.deepcopy`.

    Parameters
    ----------
    payload:
        The underlying value (any Python object). Treat as read-only after
        wrapping unless you obtain a writable handle via :meth:`mutable_copy`.

    Notes
    -----
    * ``__deepcopy__`` returns a fresh ``LazyMemoryValue`` that points at the
      *same* ``payload`` object — the payload is **not** duplicated.
    * ``__copy__`` does the same (shallow copy of the wrapper, shared payload).
    * Use :meth:`mutable_copy` to obtain a wrapper whose payload is a real
      deep copy — call this once before mutating in a parallel branch to
      avoid leaking writes into sibling steps.
    """

    __slots__ = ("_payload",)

    def __init__(self, payload: T) -> None:
        self._payload = payload

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def value(self) -> T:
        """Return the wrapped payload (no copy)."""
        return self._payload

    def get(self) -> T:
        """Alias for ``self.value`` — convenience for non-property callers."""
        return self._payload

    # ------------------------------------------------------------------
    # Write access (explicit copy-on-write)
    # ------------------------------------------------------------------

    def mutable_copy(self) -> "LazyMemoryValue[T]":
        """Return a new wrapper around a deep copy of the payload.

        Call this once *before* mutating the payload inside a parallel branch
        / replan-restored step. The original wrapper is left untouched, and
        the returned wrapper owns a fresh copy of the payload that you may
        mutate freely.
        """
        return LazyMemoryValue(copy.deepcopy(self._payload))

    def replace(self, new_payload: T) -> "LazyMemoryValue[T]":
        """Return a new wrapper around ``new_payload`` — original untouched."""
        return LazyMemoryValue(new_payload)

    # ------------------------------------------------------------------
    # copy / deepcopy protocol — the whole point of the class
    # ------------------------------------------------------------------

    def __copy__(self) -> "LazyMemoryValue[T]":
        new: LazyMemoryValue[T] = LazyMemoryValue.__new__(LazyMemoryValue)
        new._payload = self._payload
        return new

    def __deepcopy__(self, memo: dict) -> "LazyMemoryValue[T]":
        new: LazyMemoryValue[T] = LazyMemoryValue.__new__(LazyMemoryValue)
        new._payload = self._payload  # shared reference — payload not copied
        memo[id(self)] = new
        return new

    # ------------------------------------------------------------------
    # Convenience dunder delegations
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        type_name = type(self._payload).__name__
        size_hint = ""
        try:
            size_hint = f", len={len(self._payload)}"  # type: ignore[arg-type]
        except TypeError:
            pass
        return f"LazyMemoryValue<{type_name}{size_hint}>"

    def __str__(self) -> str:
        return str(self._payload)

    def __len__(self) -> int:
        return len(self._payload)  # type: ignore[arg-type]

    def __bool__(self) -> bool:
        return bool(self._payload)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LazyMemoryValue):
            return self._payload == other._payload
        return self._payload == other

    def __hash__(self) -> int:
        try:
            return hash(self._payload)
        except TypeError:
            return id(self)


def unwrap_lazy(value: Any) -> Any:
    """Return ``value.value`` if it's a :class:`LazyMemoryValue`, else ``value``.

    Useful in user code that wants to be agnostic to whether a memory entry
    was lazily wrapped or stored directly.
    """
    if isinstance(value, LazyMemoryValue):
        return value.value
    return value
