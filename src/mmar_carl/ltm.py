"""
Long-term memory (LTM) implementations for CARL.

Provides a pluggable long-term memory layer that persists across chain runs,
complementing the session-scoped working memory (``context.memory``).

Usage::

    from mmar_carl import InMemoryLTM, JsonFileLTM

    # In-memory (lost when process exits)
    ltm = InMemoryLTM()
    ltm.store("user_pref", "concise")
    print(ltm.retrieve("user_pref"))   # "concise"

    # JSON file (persisted to disk, survives restarts)
    ltm = JsonFileLTM("~/.carl/memory/")
    ctx = ReasoningContext(..., session_id="user-123", long_term_memory=ltm)
    await ctx.remember("last_query", "revenue growth Q3")
    hits = await ctx.recall("revenue")   # [{"key": "last_query", "value": "..."}]

Reference in steps (via input_mapping)::

    ToolStepDescription(
        input_mapping={"query": "$ltm.last_query"},
    )
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class LTMBase(ABC):
    """
    Abstract base class for long-term memory stores.

    All methods are synchronous so they can be called from both sync and async
    contexts without ceremony.  For async-only use-cases, run blocking I/O
    methods in a thread pool (``asyncio.to_thread``).

    Key design decisions:
    - Keys are plain strings; values are arbitrary JSON-serialisable objects.
    - ``session_id`` scopes entries so multiple users / chain runs can share
      one store without collisions.  An empty ``session_id`` is a valid global
      scope.
    - ``search`` uses simple substring / keyword matching by default; subclasses
      may override with vector search.
    """

    @abstractmethod
    def store(self, key: str, value: Any, *, session_id: str = "") -> None:
        """Persist *value* under *key* (optionally scoped by *session_id*)."""

    @abstractmethod
    def retrieve(self, key: str, *, session_id: str = "") -> Any:
        """Return the stored value for *key*, or ``None`` if not found."""

    @abstractmethod
    def delete(self, key: str, *, session_id: str = "") -> bool:
        """Remove *key*.  Returns ``True`` if the key existed."""

    @abstractmethod
    def keys(self, *, session_id: str = "") -> list[str]:
        """Return all keys visible in *session_id* scope."""

    def search(self, query: str, *, session_id: str = "", top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find entries whose serialised value contains *query* (case-insensitive).

        The default implementation is a simple substring scan.  Subclasses may
        override this with vector / semantic search.

        Returns a list of ``{"key": ..., "value": ..., "score": ...}`` dicts
        ordered by relevance (highest first).  Score is 1.0 for exact substring
        match, 0.5 for case-insensitive match.
        """
        query_lower = query.lower()
        hits: list[dict[str, Any]] = []
        for key in self.keys(session_id=session_id):
            value = self.retrieve(key, session_id=session_id)
            serialised = json.dumps(value, default=str)
            if query in serialised:
                hits.append({"key": key, "value": value, "score": 1.0})
            elif query_lower in serialised.lower():
                hits.append({"key": key, "value": value, "score": 0.5})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]

    def clear(self, *, session_id: str = "") -> int:
        """
        Remove all entries in *session_id* scope.

        Returns the number of keys deleted.
        """
        all_keys = self.keys(session_id=session_id)
        for key in all_keys:
            self.delete(key, session_id=session_id)
        return len(all_keys)


class InMemoryLTM(LTMBase):
    """
    Volatile in-memory long-term memory store.

    Data is lost when the Python process exits.  Useful for unit tests and for
    passing remembered facts between multiple chains within one process run.

    Example::

        ltm = InMemoryLTM()
        ltm.store("pref", "concise", session_id="user-1")
        ltm.retrieve("pref", session_id="user-1")  # "concise"
        ltm.retrieve("pref", session_id="user-2")  # None — different scope
    """

    def __init__(self) -> None:
        # _data[session_id][key] = value
        self._data: dict[str, dict[str, Any]] = {}

    def _ns(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._data:
            self._data[session_id] = {}
        return self._data[session_id]

    def store(self, key: str, value: Any, *, session_id: str = "") -> None:
        self._ns(session_id)[key] = value

    def retrieve(self, key: str, *, session_id: str = "") -> Any:
        return self._ns(session_id).get(key)

    def delete(self, key: str, *, session_id: str = "") -> bool:
        ns = self._ns(session_id)
        if key in ns:
            del ns[key]
            return True
        return False

    def keys(self, *, session_id: str = "") -> list[str]:
        return list(self._ns(session_id).keys())

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._data.values())
        return f"InMemoryLTM(sessions={len(self._data)}, total_entries={total})"


class JsonFileLTM(LTMBase):
    """
    JSON-file-backed long-term memory store.

    Each session is stored in a separate JSON file inside *directory*:
    ``<directory>/<session_id or '_global'>.json``.

    The file is read on every access so concurrent processes can safely share
    the same directory (last writer wins — no locking beyond atomic write-back).

    Example::

        ltm = JsonFileLTM("~/.carl/memory/")
        ltm.store("last_report", {"date": "2025-01", "pages": 12}, session_id="proj-abc")
        report = ltm.retrieve("last_report", session_id="proj-abc")
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        safe_name = session_id.replace("/", "_").replace("\\", "_") or "_global"
        return self._dir / f"{safe_name}.json"

    def _load(self, session_id: str) -> dict[str, Any]:
        path = self._path(session_id)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, session_id: str, data: dict[str, Any]) -> None:
        path = self._path(session_id)
        # Atomic write: write to temp then rename
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def store(self, key: str, value: Any, *, session_id: str = "") -> None:
        data = self._load(session_id)
        data[key] = value
        self._save(session_id, data)

    def retrieve(self, key: str, *, session_id: str = "") -> Any:
        return self._load(session_id).get(key)

    def delete(self, key: str, *, session_id: str = "") -> bool:
        data = self._load(session_id)
        if key not in data:
            return False
        del data[key]
        self._save(session_id, data)
        return True

    def keys(self, *, session_id: str = "") -> list[str]:
        return list(self._load(session_id).keys())

    def __repr__(self) -> str:
        return f"JsonFileLTM(directory={str(self._dir)!r})"
