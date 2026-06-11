"""Tests for RunRecord durable persistence model.

`RunRecord` bundles a ``ReasoningChain.to_dict()`` snapshot, the
input slice from the context, a lossless `ReasoningResult`, timestamps,
and runtime provenance into one Pydantic model that round-trips
through ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json`` /
``save`` / ``load``. CARE writes one per execution as a gigaevo-memory
``memory_card`` with kind ``"run_record"``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from mmar_carl import (
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    RunRecord,
)
from mmar_carl.models.enums import StepType
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="A", aim="x"),
        LLMStepDescription(number=2, title="B", aim="x", dependencies=[1]),
    ])


def _make_result(success: bool = True) -> ReasoningResult:
    return ReasoningResult(
        success=success,
        history=["Step 1. A\nResult: ok\n"],
        step_results=[
            StepExecutionResult(
                step_number=1, step_title="A", step_type=StepType.LLM,
                result="ok", success=success, execution_time=0.1,
            ),
        ],
        total_execution_time=0.1,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_minimum_required_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        rec = RunRecord(
            result=_make_result(),
            started_at=ts,
            finished_at=ts,
        )
        assert rec.chain_id is None
        assert rec.chain_version is None
        assert rec.chain_dict == {}
        assert rec.input == {}

    def test_runtime_info_auto_populated(self) -> None:
        ts = datetime.now(timezone.utc)
        rec = RunRecord(
            result=_make_result(),
            started_at=ts, finished_at=ts,
        )
        for k in ("carl_version", "python_version", "platform", "machine"):
            assert k in rec.runtime_info

    def test_caller_can_override_runtime_info(self) -> None:
        ts = datetime.now(timezone.utc)
        rec = RunRecord(
            result=_make_result(),
            started_at=ts, finished_at=ts,
            runtime_info={"carl_version": "0.1.0-test"},
        )
        assert rec.runtime_info == {"carl_version": "0.1.0-test"}


# ---------------------------------------------------------------------------
# from_run convenience constructor
# ---------------------------------------------------------------------------


class TestFromRun:
    def test_chain_dict_snapshot_taken(self) -> None:
        chain = _chain()
        ctx = ReasoningContext(outer_context="hello", api=_FakeClient())
        started = datetime.now(timezone.utc)
        rec = RunRecord.from_run(
            chain=chain, context=ctx, result=_make_result(),
            started_at=started,
        )
        assert rec.chain_dict.get("steps")
        assert len(rec.chain_dict["steps"]) == 2

    def test_input_outer_context_captured(self) -> None:
        ctx = ReasoningContext(outer_context="the task", api=_FakeClient())
        rec = RunRecord.from_run(
            chain=_chain(), context=ctx, result=_make_result(),
            started_at=datetime.now(timezone.utc),
        )
        assert rec.input["outer_context"] == "the task"

    def test_input_memory_captured(self) -> None:
        ctx = ReasoningContext(
            outer_context="x", api=_FakeClient(),
            memory={"input": {"doc": "hi"}},
        )
        rec = RunRecord.from_run(
            chain=_chain(), context=ctx, result=_make_result(),
            started_at=datetime.now(timezone.utc),
        )
        assert rec.input["memory"]["input"]["doc"] == "hi"

    def test_framework_internal_metadata_keys_stripped(self) -> None:
        """``__``-prefixed keys are framework-internal (live trace
        handles, replan buffers, …) and must not survive into the
        durable record."""
        ctx = ReasoningContext(outer_context="x", api=_FakeClient())
        ctx.metadata["user_key"] = "keep"
        ctx.metadata["__langfuse_trace"] = object()  # not JSON-safe
        ctx.metadata["__replan_buffer"] = ["secret"]
        rec = RunRecord.from_run(
            chain=_chain(), context=ctx, result=_make_result(),
            started_at=datetime.now(timezone.utc),
        )
        md = rec.input.get("context_metadata", {})
        assert "user_key" in md
        assert "__langfuse_trace" not in md
        assert "__replan_buffer" not in md

    def test_finished_at_defaults_to_now(self) -> None:
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rec = RunRecord.from_run(
            chain=_chain(),
            context=ReasoningContext(outer_context="x", api=_FakeClient()),
            result=_make_result(),
            started_at=started,
        )
        assert rec.started_at == started
        # finished_at was stamped automatically; just sanity-check it's later
        assert rec.finished_at >= started

    def test_chain_id_and_version_propagate(self) -> None:
        rec = RunRecord.from_run(
            chain=_chain(),
            context=ReasoningContext(outer_context="x", api=_FakeClient()),
            result=_make_result(),
            started_at=datetime.now(timezone.utc),
            chain_id="entity-123", chain_version="v7",
        )
        assert rec.chain_id == "entity-123"
        assert rec.chain_version == "v7"


# ---------------------------------------------------------------------------
# Round-trip: dict / json / file
# ---------------------------------------------------------------------------


def _full_record() -> RunRecord:
    """A populated record useful for round-trip tests."""
    ctx = ReasoningContext(
        outer_context="hello", api=_FakeClient(),
        memory={"input": {"doc": "payload"}},
    )
    ctx.metadata["user_key"] = "keep-me"
    return RunRecord.from_run(
        chain=_chain(), context=ctx, result=_make_result(),
        started_at=datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc),
        finished_at=datetime(2026, 5, 15, 12, 0, 1, tzinfo=timezone.utc),
        chain_id="entity-x", chain_version="v1",
    )


class TestDictRoundTrip:
    def test_to_dict_keys_present(self) -> None:
        rec = _full_record()
        d = rec.to_dict()
        for k in (
            "chain_id", "chain_version", "chain_dict", "input",
            "result", "started_at", "finished_at", "runtime_info",
        ):
            assert k in d

    def test_timestamps_serialised_as_iso(self) -> None:
        rec = _full_record()
        d = rec.to_dict()
        assert isinstance(d["started_at"], str)
        assert d["started_at"].startswith("2026-05-15T12:00:00")

    def test_result_uses_full_shape(self) -> None:
        """`to_dict` calls `ReasoningResult.to_dict(full=True)` so
        history + per-step lossless data round-trip."""
        rec = _full_record()
        result_payload = rec.to_dict()["result"]
        assert result_payload["_full"] is True
        assert "history" in result_payload

    def test_from_dict_round_trip(self) -> None:
        rec = _full_record()
        d = rec.to_dict()
        rebuilt = RunRecord.from_dict(d)
        assert rebuilt.chain_id == rec.chain_id
        assert rebuilt.chain_version == rec.chain_version
        assert rebuilt.chain_dict == rec.chain_dict
        assert rebuilt.input == rec.input
        assert rebuilt.result.success == rec.result.success
        assert rebuilt.started_at == rec.started_at
        assert rebuilt.finished_at == rec.finished_at
        assert rebuilt.runtime_info == rec.runtime_info

    def test_from_dict_accepts_already_rebuilt_result(self) -> None:
        """If ``data["result"]`` is already a ReasoningResult instance
        (e.g. callers manually assembling a record), accept it."""
        rec = _full_record()
        d = rec.to_dict()
        d["result"] = rec.result  # swap in the live object
        rebuilt = RunRecord.from_dict(d)
        assert rebuilt.result.success == rec.result.success

    def test_from_dict_missing_result_raises(self) -> None:
        rec = _full_record()
        d = rec.to_dict()
        del d["result"]
        with pytest.raises(ValueError, match="result"):
            RunRecord.from_dict(d)

    def test_from_dict_accepts_iso_timestamps(self) -> None:
        d = _full_record().to_dict()
        rebuilt = RunRecord.from_dict(d)
        assert rebuilt.started_at.year == 2026


class TestJsonRoundTrip:
    def test_to_json_string(self) -> None:
        rec = _full_record()
        s = rec.to_json()
        assert isinstance(s, str)
        assert "entity-x" in s

    def test_indent_emits_newlines(self) -> None:
        s = _full_record().to_json(indent=2)
        assert "\n" in s

    def test_from_json_round_trip(self) -> None:
        rec = _full_record()
        rebuilt = RunRecord.from_json(rec.to_json())
        assert rebuilt.chain_id == rec.chain_id
        assert rebuilt.result.success == rec.result.success
        assert rebuilt.input == rec.input


class TestFileRoundTrip:
    def test_save_load(self, tmp_path: Path) -> None:
        rec = _full_record()
        out = tmp_path / "run.json"
        result_path = rec.save(out)
        assert out.exists()
        assert result_path == out.resolve()
        loaded = RunRecord.load(out)
        assert loaded.chain_id == rec.chain_id
        assert loaded.input == rec.input

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        rec = _full_record()
        deep = tmp_path / "a" / "b" / "c" / "run.json"
        rec.save(deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# End-to-end via a real chain run
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_run_persisted_and_replayed(self, tmp_path: Path) -> None:
        chain = _chain()
        ctx = ReasoningContext(
            outer_context="hello world", api=_FakeClient(),
            memory={"input": {"file.md": "# content"}},
        )
        started = datetime.now(timezone.utc)
        result = await chain.execute_async(ctx)
        assert result.success

        rec = RunRecord.from_run(
            chain=chain, context=ctx, result=result,
            started_at=started, chain_id="e1", chain_version="v1",
        )
        out = tmp_path / "run.json"
        rec.save(out)
        loaded = RunRecord.load(out)
        # Every important slice round-trips
        assert loaded.chain_id == "e1"
        assert loaded.chain_version == "v1"
        assert loaded.input["outer_context"] == "hello world"
        assert loaded.input["memory"]["input"]["file.md"] == "# content"
        assert loaded.result.success
        assert loaded.chain_dict["steps"][0]["title"] == "A"
        # Runtime info captured
        assert "carl_version" in loaded.runtime_info


# ---------------------------------------------------------------------------
# Direct construction with explicit fields
# ---------------------------------------------------------------------------


class TestDirectConstruction:
    def test_can_build_without_chain_or_context(self) -> None:
        """Callers reconstructing a record from gigaevo-memory don't
        have the live chain/context — they pass the dicts directly."""
        ts = datetime.now(timezone.utc)
        rec = RunRecord(
            chain_id="x",
            chain_dict={"steps": [{"number": 1, "title": "T", "step_type": "llm"}]},
            input={"outer_context": "hi"},
            result=_make_result(),
            started_at=ts, finished_at=ts,
        )
        assert rec.chain_id == "x"
        assert rec.chain_dict["steps"][0]["title"] == "T"
