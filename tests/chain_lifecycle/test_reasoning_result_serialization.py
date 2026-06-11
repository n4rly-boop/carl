"""Tests for ReasoningResult / StepExecutionResult
serialization round-trip.

CARE needs lossless persistence: save a `ReasoningResult` after a run,
load it back later (possibly in a different process) and reconstruct
the same Python object so the user can replay or audit the run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmar_carl.models.enums import StepType
from mmar_carl.models.replan import (
    ReplanAction,
    ReplanAggregationStrategy,
)
from mmar_carl.models.results import (
    ReasoningResult,
    ReplanAggregationOutcome,
    ReplanEvent,
    StepExecutionResult,
)


def _step(
    n: int,
    *,
    title: str = "Step",
    success: bool = True,
    result: str = "ok",
    skipped: bool = False,
    long_text: bool = False,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=n, step_title=title, step_type=StepType.LLM,
        result=("X" * 2500) if long_text else result,
        success=success, skipped=skipped,
        execution_time=0.1,
        token_usage={"prompt": 10, "completion": 5, "total": 15},
        updated_history=[f"Step {n}. {title}\nResult: {result}\n"],
        model="qwen/qwen3-8b",
    )


def _result(
    *,
    success: bool = True,
    steps: list[StepExecutionResult] | None = None,
    history: list[str] | None = None,
    replan_events: list[ReplanEvent] | None = None,
) -> ReasoningResult:
    steps = steps or [_step(1), _step(2)]
    history = history or [s.updated_history[-1] for s in steps]
    return ReasoningResult(
        success=success, history=history, step_results=steps,
        total_execution_time=0.2,
        token_usage={"prompt": 20, "completion": 10, "total": 30},
        metrics={"accuracy": 0.9},
        metadata={"trace_name": "demo"},
        context_metadata={"step_1_score": 0.85},
        replan_events=replan_events or [],
    )


# ---------------------------------------------------------------------------
# StepExecutionResult.to_dict(truncate=...) + from_dict
# ---------------------------------------------------------------------------


class TestStepResultTruncation:
    def test_default_truncates_long_result(self) -> None:
        sr = _step(1, long_text=True)
        d = sr.to_dict()
        assert d["_truncated"] is True
        assert len(d["result"]) == 1000

    def test_truncate_false_preserves_full_result(self) -> None:
        sr = _step(1, long_text=True)
        d = sr.to_dict(truncate=False)
        assert d["_truncated"] is False
        assert len(d["result"]) == 2500

    def test_truncate_false_includes_lossless_extras(self) -> None:
        sr = _step(1)
        full = sr.to_dict(truncate=False)
        for k in ("updated_history", "updated_messages", "model", "injected_steps"):
            assert k in full, f"missing lossless key: {k}"

    def test_truncate_true_omits_lossless_extras(self) -> None:
        sr = _step(1)
        trunc = sr.to_dict()
        # ``updated_history`` is intentionally absent in the log-friendly shape
        assert "updated_history" not in trunc
        assert "model" not in trunc
        assert "updated_messages" not in trunc


class TestStepResultRoundTrip:
    def test_full_roundtrip_preserves_fields(self) -> None:
        sr = _step(1, long_text=True, title="Plan")
        d = sr.to_dict(truncate=False)
        rebuilt = StepExecutionResult.from_dict(d)
        assert rebuilt.step_number == sr.step_number
        assert rebuilt.step_title == sr.step_title
        assert rebuilt.step_type == sr.step_type
        assert rebuilt.result == sr.result
        assert rebuilt.updated_history == sr.updated_history
        assert rebuilt.model == sr.model
        assert rebuilt.token_usage == sr.token_usage

    def test_truncated_roundtrip_loses_only_truncated_fields(self) -> None:
        sr = _step(1)
        d = sr.to_dict()
        rebuilt = StepExecutionResult.from_dict(d)
        # Core fields restored
        assert rebuilt.step_number == sr.step_number
        assert rebuilt.step_title == sr.step_title
        assert rebuilt.result == sr.result
        # Lossless-only fields default
        assert rebuilt.updated_history == []
        assert rebuilt.model is None

    def test_from_dict_handles_empty_result(self) -> None:
        sr = StepExecutionResult(
            step_number=1, step_title="X", step_type=StepType.TOOL,
            result="", success=True,
        )
        d = sr.to_dict()
        # ``result`` should serialise to None when empty + truncated, and
        # ``from_dict`` should restore it to an empty string.
        assert d["result"] is None
        rebuilt = StepExecutionResult.from_dict(d)
        assert rebuilt.result == ""

    def test_from_dict_resolves_step_type_string_format(self) -> None:
        """``str(StepType.LLM)`` yields ``"StepType.LLM"`` in Python; the
        round-tripper must accept that shape."""
        d = {
            "step_number": 1, "step_title": "X",
            "step_type": "StepType.LLM",  # the str() form
            "result": "ok", "success": True,
        }
        rebuilt = StepExecutionResult.from_dict(d)
        assert rebuilt.step_type == StepType.LLM


# ---------------------------------------------------------------------------
# ReasoningResult.to_dict(full=...) + from_dict
# ---------------------------------------------------------------------------


class TestReasoningResultSummaryShape:
    def test_default_emits_summary_keys(self) -> None:
        r = _result()
        d = r.to_dict()
        for k in (
            "success", "total_execution_time", "total_steps",
            "successful_steps", "failed_steps", "skipped_steps",
            "step_results", "metadata",
        ):
            assert k in d
        assert d["_full"] is False

    def test_default_does_not_include_history_or_context_metadata(self) -> None:
        r = _result()
        d = r.to_dict()
        assert "history" not in d
        assert "context_metadata" not in d

    def test_step_counts_correct(self) -> None:
        r = _result(steps=[
            _step(1, success=True),
            _step(2, success=False),
            _step(3, skipped=True),
        ])
        d = r.to_dict()
        assert d["total_steps"] == 3
        assert d["successful_steps"] == 1
        assert d["failed_steps"] == 1
        assert d["skipped_steps"] == 1


class TestReasoningResultFullShape:
    def test_full_includes_history_and_context_metadata(self) -> None:
        r = _result()
        d = r.to_dict(full=True)
        assert d["_full"] is True
        assert d["history"] == r.history
        assert d["context_metadata"] == r.context_metadata

    def test_full_uses_untruncated_step_results(self) -> None:
        r = _result(steps=[_step(1, long_text=True)])
        d = r.to_dict(full=True)
        assert d["step_results"][0]["_truncated"] is False
        assert len(d["step_results"][0]["result"]) == 2500


class TestReasoningResultRoundTrip:
    def test_full_roundtrip_lossless(self) -> None:
        r = _result(steps=[_step(1, long_text=True), _step(2)])
        d = r.to_dict(full=True)
        rebuilt = ReasoningResult.from_dict(d)
        assert rebuilt.success == r.success
        assert rebuilt.history == r.history
        assert rebuilt.metadata == r.metadata
        assert rebuilt.context_metadata == r.context_metadata
        assert rebuilt.token_usage == r.token_usage
        assert len(rebuilt.step_results) == len(r.step_results)
        # First step's long result survived intact
        assert rebuilt.step_results[0].result == r.step_results[0].result

    def test_summary_roundtrip_yields_usable_result_with_empty_history(self) -> None:
        """Summary shape is **lossy by design** — ``history`` is dropped
        (the truncated step dump doesn't carry ``updated_history`` either,
        so there's nothing to reconstruct from). ``from_dict`` returns a
        usable but history-less ReasoningResult — callers needing
        lossless persistence must pass ``full=True``."""
        r = _result()
        d = r.to_dict(full=False)
        rebuilt = ReasoningResult.from_dict(d)
        # Core fields preserved
        assert rebuilt.success == r.success
        assert len(rebuilt.step_results) == len(r.step_results)
        # History defaults to empty in the lossy round-trip
        assert rebuilt.history == []

    def test_replan_events_round_trip(self) -> None:
        ev = ReplanEvent(
            sequence=1, step_number=2, step_title="X",
            aggregation=ReplanAggregationOutcome(
                strategy=ReplanAggregationStrategy.ANY,
                triggered=False,
                trigger_count=0,
                total_count=1,
            ),
            final_action=ReplanAction.CONTINUE,
        )
        r = _result(replan_events=[ev])
        rebuilt = ReasoningResult.from_dict(r.to_dict(full=True))
        assert len(rebuilt.replan_events) == 1
        assert rebuilt.replan_events[0].sequence == 1


# ---------------------------------------------------------------------------
# JSON & file persistence
# ---------------------------------------------------------------------------


class TestJsonPersistence:
    def test_to_json_from_json_round_trip(self) -> None:
        r = _result()
        s = r.to_json()
        assert isinstance(s, str)
        rebuilt = ReasoningResult.from_json(s)
        assert rebuilt.success == r.success
        assert rebuilt.history == r.history

    def test_to_json_indent_passes_through(self) -> None:
        r = _result()
        s = r.to_json(indent=2)
        # Indented JSON contains newlines
        assert "\n" in s

    def test_to_json_full_false_emits_summary(self) -> None:
        r = _result()
        s = r.to_json(full=False)
        data = json.loads(s)
        assert data["_full"] is False
        assert "history" not in data


class TestFilePersistence:
    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        r = _result(steps=[_step(1, long_text=True), _step(2)])
        out = tmp_path / "run.json"
        result_path = r.save(out)
        assert out.exists()
        # Returned path is absolute and matches
        assert result_path == out.resolve()
        rebuilt = ReasoningResult.load(out)
        assert rebuilt.success == r.success
        assert rebuilt.history == r.history
        assert rebuilt.step_results[0].result == r.step_results[0].result

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        r = _result()
        deep = tmp_path / "a" / "b" / "c" / "run.json"
        r.save(deep)
        assert deep.exists()


# ---------------------------------------------------------------------------
# End-to-end via a real chain
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_round_trip_after_real_chain_execution(
        self, tmp_path: Path,
    ) -> None:
        """Drive a real (mocked-LLM) chain, save the result, reload from
        disk, and confirm every per-step field matches."""
        from mmar_carl import (
            LLMStepDescription, ReasoningChain, ReasoningContext,
        )
        from mmar_carl.models.llm_client_base import LLMClientBase

        class FakeClient(LLMClientBase):
            @property
            def model_name(self) -> str:
                return "fake-model"

            async def get_response(self, prompt: str) -> str:
                return "echo"

            async def get_response_with_retries(
                self, prompt: str, retries: int = 3,
            ) -> str:
                return "echo"

        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="Plan", aim="x"),
            LLMStepDescription(
                number=2, title="Synth", aim="x", dependencies=[1],
            ),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=FakeClient())
        result = await chain.execute_async(ctx)
        assert result.success

        path = tmp_path / "run.json"
        result.save(path)
        loaded = ReasoningResult.load(path)
        assert loaded.success == result.success
        assert loaded.total_execution_time == result.total_execution_time
        assert len(loaded.step_results) == len(result.step_results)
        for orig, rebuilt in zip(result.step_results, loaded.step_results):
            assert rebuilt.step_number == orig.step_number
            assert rebuilt.step_title == orig.step_title
            assert rebuilt.result == orig.result
            assert rebuilt.success == orig.success
            assert rebuilt.token_usage == orig.token_usage
            assert rebuilt.updated_history == orig.updated_history
        assert loaded.history == result.history
