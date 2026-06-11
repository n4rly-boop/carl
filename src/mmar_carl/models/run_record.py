"""RunRecord — durable persistence shape for one chain execution.

CARE writes one ``RunRecord`` per execution as a gigaevo-memory
``memory_card`` with kind ``"run_record"``. The record bundles the
chain spec, the input snapshot, the full lossless
:class:`ReasoningResult`, timestamps, and a tiny bit of runtime
provenance (CARL version, Python version, host) so a run can be
audited / replayed years later from the library view.

"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .results import ReasoningResult


def _current_runtime_info() -> dict[str, Any]:
    """Capture host / Python / CARL version at record-creation time."""
    try:
        from .. import __version__ as carl_version  # noqa: PLC0415
    except Exception:
        carl_version = "unknown"
    return {
        "carl_version": carl_version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


class RunRecord(BaseModel):
    """Audit-friendly snapshot of one chain execution.

    ``chain_dict`` is intentionally the full ``ReasoningChain.to_dict()``
    snapshot at the moment of execution — not a pointer to a Memory
    entity — so the record is **self-contained**: dropping it on a
    different machine still lets the user re-render the chain DAG and
    inspect the steps even if the original chain has since been edited
    or deleted from CARE's library.

    ``input`` mirrors the executor's input surface: ``outer_context``
    and the initial ``memory`` snapshot (the ``input`` namespace is
    what CARE typically writes to before calling ``execute_async``,
    so it's the natural replay-friendly slice).
    """

    chain_id: Optional[str] = Field(
        default=None,
        description=(
            "gigaevo-memory entity id of the source chain (when CARE "
            "saved this run from a library entry). ``None`` for ad-hoc "
            "runs of in-memory chains."
        ),
    )
    chain_version: Optional[str] = Field(
        default=None,
        description=(
            "Library version id of the chain at run time. None for "
            "chains that don't live in a versioned store."
        ),
    )
    chain_dict: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "``ReasoningChain.to_dict()`` snapshot at the time of "
            "execution — the durable image of the chain that drove "
            "this run, independent of any later library edits."
        ),
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Initial inputs: ``{'outer_context': str, 'memory': "
            "{namespace: {key: value}}, 'context_metadata': dict}``. "
            "Mirrors what was on the ``ReasoningContext`` at execution "
            "start so a replay can prime an identical context."
        ),
    )
    result: ReasoningResult = Field(
        ..., description="Lossless ReasoningResult from this run.",
    )
    started_at: datetime = Field(
        ...,
        description="UTC ISO-8601 timestamp at chain-execute start.",
    )
    finished_at: datetime = Field(
        ...,
        description="UTC ISO-8601 timestamp at chain-execute completion.",
    )
    runtime_info: dict[str, Any] = Field(
        default_factory=_current_runtime_info,
        description=(
            "Snapshot of the executing environment: CARL version, "
            "Python version, platform string, machine arch. Captured "
            "automatically when the record is constructed; pass an "
            "explicit dict to override (CARE uses this when "
            "reconstructing a record from a memory_card)."
        ),
    )

    # ---------------------------------------------------------------- #
    # Construction helpers                                              #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_run(
        cls,
        *,
        chain: Any,
        context: Any,
        result: ReasoningResult,
        started_at: datetime,
        finished_at: Optional[datetime] = None,
        chain_id: Optional[str] = None,
        chain_version: Optional[str] = None,
    ) -> "RunRecord":
        """Convenience constructor — wraps a finished
        ``(chain, context, result)`` triple into a ``RunRecord``.

        Pulls ``chain.to_dict()`` for the durable snapshot, captures
        ``outer_context`` + ``memory`` + ``context_metadata`` from
        the context, and stamps ``finished_at = now()`` when the
        caller didn't supply one.
        """
        ctx_memory: dict[str, Any] = {}
        if context is not None:
            raw_memory = getattr(context, "memory", None)
            if isinstance(raw_memory, dict):
                # Best-effort deep-ish copy via JSON-safe coercion.
                try:
                    ctx_memory = json.loads(json.dumps(raw_memory, default=str))
                except (TypeError, ValueError):
                    ctx_memory = {k: dict(v) for k, v in raw_memory.items()}
        input_payload: dict[str, Any] = {
            "outer_context": getattr(context, "outer_context", "") or "",
            "memory": ctx_memory,
        }
        ctx_md = getattr(context, "metadata", None)
        if isinstance(ctx_md, dict):
            # Drop framework-internal keys (the ``__`` prefix is the
            # documented convention for non-persistable runtime state
            # like the live Langfuse trace handle, replan feedback
            # buffers, etc.). Keeping them would (a) leak non-JSON-safe
            # objects into the durable record and (b) couple replay
            # against an in-process tracer that no longer exists.
            input_payload["context_metadata"] = {
                k: v for k, v in ctx_md.items()
                if not str(k).startswith("__")
            }

        chain_dict: dict[str, Any] = {}
        if chain is not None and hasattr(chain, "to_dict"):
            try:
                chain_dict = chain.to_dict()
            except Exception:
                chain_dict = {}

        return cls(
            chain_id=chain_id,
            chain_version=chain_version,
            chain_dict=chain_dict,
            input=input_payload,
            result=result,
            started_at=started_at,
            finished_at=finished_at or datetime.now(timezone.utc),
        )

    # ---------------------------------------------------------------- #
    # Serialisation                                                     #
    # ---------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Lossless dict representation suitable for a memory_card body.

        Datetimes are emitted as ISO-8601 strings; ``result`` is
        rendered via ``ReasoningResult.to_dict(full=True)`` so the
        record round-trips losslessly through :meth:`from_dict`.
        """
        return {
            "chain_id": self.chain_id,
            "chain_version": self.chain_version,
            "chain_dict": self.chain_dict,
            "input": self.input,
            "result": self.result.to_dict(full=True),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "runtime_info": self.runtime_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        """Inverse of :meth:`to_dict`."""

        def _parse_ts(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            raise ValueError(f"unsupported timestamp value: {value!r}")

        result_data = data.get("result")
        if isinstance(result_data, ReasoningResult):
            result = result_data
        elif isinstance(result_data, dict):
            result = ReasoningResult.from_dict(result_data)
        else:
            raise ValueError(
                "RunRecord.from_dict: missing or invalid 'result' field"
            )

        return cls(
            chain_id=data.get("chain_id"),
            chain_version=data.get("chain_version"),
            chain_dict=data.get("chain_dict") or {},
            input=data.get("input") or {},
            result=result,
            started_at=_parse_ts(data["started_at"]),
            finished_at=_parse_ts(data["finished_at"]),
            runtime_info=data.get("runtime_info") or {},
        )

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """JSON wrapper around :meth:`to_dict`. Datetimes already
        ISO-8601-formatted by ``to_dict``, so a plain ``json.dumps``
        works without a custom encoder."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "RunRecord":
        return cls.from_dict(json.loads(json_str))

    def save(self, path: "str | Path", *, indent: Optional[int] = 2) -> Path:
        """Persist to a JSON file. Auto-creates parent directories.
        Returns the resolved absolute path (mirrors
        :meth:`ReasoningResult.save` / :meth:`ReasoningChain.save`).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")
        return p.resolve()

    @classmethod
    def load(cls, path: "str | Path") -> "RunRecord":
        """Inverse of :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = ["RunRecord"]
