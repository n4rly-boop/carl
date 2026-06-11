"""
Copy-on-write (COW) memory store for parallel step execution isolation.

Instead of ``copy.deepcopy(context.memory)`` before each parallel batch,
the executor creates a ``COWMemoryStore`` that shares read access to the
parent's memory dict without copying.  A namespace is only shallow-copied
into the local overlay on first access, so steps that never touch a
namespace add zero copy overhead for that namespace.

Key properties
--------------
* **Reads**: zero-copy when the namespace was not yet accessed.
* **Writes**: the namespace is shallow-copied into the local overlay on
  first access; subsequent reads/writes use the overlay.
* **Merge**: after the batch the executor merges only the *overlay*
  namespaces (those actually touched by the step) back into the parent
  context, fixing a pre-existing correctness issue where the deep-copy
  approach could overwrite a sibling step's writes with stale snapshot data.
* **API compatibility**: ``COWMemoryStore`` is a ``dict`` subclass so all
  existing code that treats ``context.memory`` as a plain dict continues to
  work without changes.
"""

from typing import Any, Iterator


class COWMemoryStore(dict):
    """
    Copy-on-write namespace-level memory store.

    Designed to be used as a drop-in replacement for ``dict[str, dict[str,
    Any]]`` in :class:`~mmar_carl.models.context.ReasoningContext`.

    Usage in the executor::

        from mmar_carl.cow_memory import COWMemoryStore

        cow = COWMemoryStore(base=context.memory)
        snapshot.memory = cow

        # After step execution, merge ONLY touched namespaces:
        for namespace, data in cow.overlay.items():
            context.memory.setdefault(namespace, {}).update(data)

    Attributes
    ----------
    _base : dict
        Reference to the parent memory dict.  Never mutated.
    """

    def __init__(self, base: "dict[str, dict[str, Any]]") -> None:
        super().__init__()
        # Store base as a plain attribute (not in the dict itself)
        object.__setattr__(self, "_cow_base", base)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _base(self) -> "dict[str, dict[str, Any]]":
        return object.__getattribute__(self, "_cow_base")

    def _ensure_local(self, namespace: str) -> None:
        """Copy *namespace* from base into overlay if not already there."""
        if not dict.__contains__(self, namespace):
            if namespace in self._base:
                dict.__setitem__(self, namespace, dict(self._base[namespace]))
            else:
                dict.__setitem__(self, namespace, {})

    # ------------------------------------------------------------------
    # Public property: only the locally copied/created namespaces
    # ------------------------------------------------------------------

    @property
    def overlay(self) -> "dict[str, dict[str, Any]]":
        """Return a plain dict of only the overlay (locally touched) namespaces."""
        return dict(dict.items(self))

    # ------------------------------------------------------------------
    # dict protocol overrides
    # ------------------------------------------------------------------

    def __getitem__(self, namespace: str) -> "dict[str, Any]":
        if dict.__contains__(self, namespace):
            return dict.__getitem__(self, namespace)
        if namespace in self._base:
            self._ensure_local(namespace)
            return dict.__getitem__(self, namespace)
        raise KeyError(namespace)

    def __setitem__(self, namespace: str, value: "dict[str, Any]") -> None:
        dict.__setitem__(self, namespace, value)

    def __contains__(self, item: object) -> bool:
        return dict.__contains__(self, item) or item in self._base

    def get(self, namespace: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[namespace]
        except KeyError:
            return default

    def __iter__(self) -> "Iterator[str]":
        seen: set[str] = set(dict.keys(self))
        yield from seen
        for ns in self._base:
            if ns not in seen:
                yield ns

    def __len__(self) -> int:
        return len(set(dict.keys(self)) | set(self._base.keys()))

    def items(self):  # type: ignore[override]
        seen: set[str] = set(dict.keys(self))
        yield from dict.items(self)
        for ns, data in self._base.items():
            if ns not in seen:
                yield ns, data

    def keys(self):  # type: ignore[override]
        return set(dict.keys(self)) | set(self._base.keys())

    def values(self):  # type: ignore[override]
        for _, v in self.items():
            yield v

    def setdefault(self, namespace: str, default: Any = None) -> Any:  # type: ignore[override]
        if namespace not in self:
            self[namespace] = default if default is not None else {}
        return self[namespace]

    def update(self, other=(), **kwargs):  # type: ignore[override]
        if hasattr(other, "items"):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __repr__(self) -> str:  # pragma: no cover
        overlay_keys = set(dict.keys(self))
        base_keys = set(self._base.keys())
        return (
            f"COWMemoryStore(overlay_namespaces={sorted(overlay_keys)}, "
            f"base_namespaces={sorted(base_keys - overlay_keys)})"
        )
