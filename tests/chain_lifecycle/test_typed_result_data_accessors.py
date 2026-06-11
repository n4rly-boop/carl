"""typed views over ``StepExecutionResult.result_data``.

CARE renders per-step-type detail panes (AgentSkill output files,
Debate transcripts, Supervisor decisions, ParallelSampling candidates).
Without typed accessors it'd have to poke at the raw ``result_data:
dict[str, Any]`` and guess at field names. With them, CARE writes::

    if (skill := result.as_skill_output()):
        for f in skill.output_files:
            ...

The accessors return ``None`` (not raise) on type mismatch or
malformed payload, so the walrus pattern stays clean.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    DebateTranscript,
    DebateTurn,
    ParallelSamples,
    SkillOutput,
    StepExecutionResult,
    SupervisorDecision,
)
from mmar_carl.models.enums import StepType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _step_result(
    *,
    step_type: StepType,
    result_data: dict | None = None,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=1,
        step_title="t",
        step_type=step_type,
        result="ok",
        result_data=result_data or {},
        success=True,
    )


# ---------------------------------------------------------------------------
# AgentSkill
# ---------------------------------------------------------------------------


class TestAsSkillOutput:
    def test_returns_typed_view_for_agent_skill_step(self) -> None:
        r = _step_result(
            step_type=StepType.AGENT_SKILL,
            result_data={
                "skill_name": "pdf",
                "execution_mode": "llm_agent",
                "output_files": [
                    {"name": "out.json", "path": "/workspace/out/out.json"},
                ],
                "iterations": 3,
                "tool_calls_made": 7,
            },
        )
        view = r.as_skill_output()
        assert isinstance(view, SkillOutput)
        assert view.skill_name == "pdf"
        assert view.execution_mode == "llm_agent"
        assert view.output_files == [
            {"name": "out.json", "path": "/workspace/out/out.json"},
        ]
        assert view.iterations == 3
        assert view.tool_calls_made == 7

    def test_returns_none_for_wrong_step_type(self) -> None:
        r = _step_result(step_type=StepType.LLM, result_data={"x": 1})
        assert r.as_skill_output() is None

    def test_returns_none_for_non_dict_result_data(self) -> None:
        r = _step_result(step_type=StepType.AGENT_SKILL)
        r.result_data = ["not", "a", "dict"]  # type: ignore[assignment]
        assert r.as_skill_output() is None

    def test_extras_preserved_via_model_config(self) -> None:
        """Unknown keys produced by the executor (e.g.
        ``script_returncode``) are kept on the model since
        ``model_config = ConfigDict(extra="allow")``."""
        r = _step_result(
            step_type=StepType.AGENT_SKILL,
            result_data={
                "skill_name": "pptx",
                "script_returncode": 0,
                "stderr": "warnings",
            },
        )
        view = r.as_skill_output()
        assert view is not None
        # Extra keys are accessible via model_dump.
        dumped = view.model_dump()
        assert dumped["script_returncode"] == 0
        assert dumped["stderr"] == "warnings"

    def test_defaults_when_keys_absent(self) -> None:
        r = _step_result(step_type=StepType.AGENT_SKILL, result_data={})
        view = r.as_skill_output()
        assert view is not None
        assert view.skill_name is None
        assert view.execution_mode is None
        assert view.output_files == []

    def test_schema_validated_flag_round_trips(self) -> None:
        r = _step_result(
            step_type=StepType.AGENT_SKILL,
            result_data={
                "schema_validated": True,
                "parsed_output": {"status": "ok", "count": 3},
            },
        )
        view = r.as_skill_output()
        assert view is not None
        assert view.schema_validated is True
        assert view.parsed_output == {"status": "ok", "count": 3}


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


class TestAsDebateTranscript:
    def test_returns_typed_view_for_debate_step(self) -> None:
        r = _step_result(
            step_type=StepType.DEBATE,
            result_data={
                "verdict": "Option A wins",
                "transcript": [
                    {"round": 1, "role": "pro", "argument": "first"},
                    {"round": 1, "role": "con", "argument": "second"},
                ],
                "rounds_executed": 1,
                "role_call_count": 2,
                "topic": "Pick an option",
            },
        )
        view = r.as_debate_transcript()
        assert isinstance(view, DebateTranscript)
        assert view.verdict == "Option A wins"
        assert view.rounds_executed == 1
        assert view.role_call_count == 2
        assert view.topic == "Pick an option"
        assert len(view.transcript) == 2
        assert isinstance(view.transcript[0], DebateTurn)
        assert view.transcript[0].round == 1
        assert view.transcript[0].role == "pro"
        assert view.transcript[0].argument == "first"

    def test_returns_none_for_wrong_step_type(self) -> None:
        r = _step_result(step_type=StepType.LLM)
        assert r.as_debate_transcript() is None

    def test_empty_transcript_default(self) -> None:
        r = _step_result(step_type=StepType.DEBATE, result_data={})
        view = r.as_debate_transcript()
        assert view is not None
        assert view.verdict == ""
        assert view.transcript == []
        assert view.rounds_executed == 0

    def test_extras_preserved(self) -> None:
        r = _step_result(
            step_type=StepType.DEBATE,
            result_data={"verdict": "x", "custom_field": "hi"},
        )
        view = r.as_debate_transcript()
        assert view is not None
        assert view.model_dump()["custom_field"] == "hi"


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class TestAsSupervisorDecision:
    def test_returns_typed_view_for_supervisor_step(self) -> None:
        # `sub_result` is typed as Any (avoids circular import).
        fake_sub_result = object()
        r = _step_result(
            step_type=StepType.SUPERVISOR,
            result_data={
                "agent_selected": "alpha",
                "routing_reply": "I pick alpha",
                "sub_result": fake_sub_result,
                "sub_chain_success": True,
                "steps_executed": 5,
            },
        )
        view = r.as_supervisor_decision()
        assert isinstance(view, SupervisorDecision)
        assert view.agent_selected == "alpha"
        assert view.routing_reply == "I pick alpha"
        assert view.sub_result is fake_sub_result
        assert view.sub_chain_success is True
        assert view.steps_executed == 5

    def test_returns_none_for_wrong_step_type(self) -> None:
        r = _step_result(step_type=StepType.DEBATE)
        assert r.as_supervisor_decision() is None

    def test_partial_data_validates_with_defaults(self) -> None:
        r = _step_result(
            step_type=StepType.SUPERVISOR,
            result_data={"agent_selected": "beta"},
        )
        view = r.as_supervisor_decision()
        assert view is not None
        assert view.agent_selected == "beta"
        assert view.sub_result is None
        assert view.sub_chain_success is None


# ---------------------------------------------------------------------------
# ParallelSampling
# ---------------------------------------------------------------------------


class TestAsParallelSamples:
    def test_returns_typed_view_for_parallel_sampling_step(self) -> None:
        r = _step_result(
            step_type=StepType.PARALLEL_SAMPLING,
            result_data={
                "n_samples": 5,
                "n_successes": 4,
                "aggregation": "majority_vote",
                "candidates": ["a", "a", "b", "a", "c"],
            },
        )
        view = r.as_parallel_samples()
        assert isinstance(view, ParallelSamples)
        assert view.n_samples == 5
        assert view.n_successes == 4
        assert view.aggregation == "majority_vote"
        assert view.candidates == ["a", "a", "b", "a", "c"]

    def test_returns_none_for_wrong_step_type(self) -> None:
        r = _step_result(step_type=StepType.LLM)
        assert r.as_parallel_samples() is None

    def test_aggregation_enum_value_or_name_works(self) -> None:
        # The executor stamps the enum value (string), not the enum
        # instance — but if someone passes the enum directly, pydantic
        # should still validate via str().
        r = _step_result(
            step_type=StepType.PARALLEL_SAMPLING,
            result_data={
                "n_samples": 3,
                "n_successes": 3,
                "aggregation": "llm_judge",
                "candidates": ["x", "y", "z"],
            },
        )
        view = r.as_parallel_samples()
        assert view is not None
        assert view.aggregation == "llm_judge"


# ---------------------------------------------------------------------------
# Cross-cutting — type mismatch + walrus pattern
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_walrus_pattern_works(self) -> None:
        """The walrus pattern is the documented use case — make sure
        it reads cleanly without an isinstance dance."""
        r = _step_result(
            step_type=StepType.AGENT_SKILL,
            result_data={"skill_name": "pdf"},
        )
        if (skill := r.as_skill_output()):
            assert skill.skill_name == "pdf"
        else:
            pytest.fail("walrus must bind the typed view on matching step")

    def test_every_accessor_returns_none_for_llm_step(self) -> None:
        r = _step_result(step_type=StepType.LLM, result_data={"x": 1})
        assert r.as_skill_output() is None
        assert r.as_debate_transcript() is None
        assert r.as_supervisor_decision() is None
        assert r.as_parallel_samples() is None

    def test_step_type_string_value_round_trip(self) -> None:
        """``step_type`` may be stored as a string (when a result was
        round-tripped via ``from_dict``). The accessors must handle
        both forms."""
        r = StepExecutionResult(
            step_number=1, step_title="t",
            step_type="agent_skill",  # string instead of enum
            result="ok",
            result_data={"skill_name": "pdf"},
            success=True,
        )
        view = r.as_skill_output()
        assert view is not None
        assert view.skill_name == "pdf"

    def test_top_level_exports(self) -> None:
        import mmar_carl
        for name in (
            "SkillOutput", "DebateTranscript", "DebateTurn",
            "SupervisorDecision", "ParallelSamples",
        ):
            assert hasattr(mmar_carl, name), f"missing top-level export: {name}"


# ---------------------------------------------------------------------------
# Validation failures return None, never raise
# ---------------------------------------------------------------------------


class TestPermissiveValidation:
    def test_malformed_transcript_returns_none(self) -> None:
        """A ``transcript`` entry missing required fields should make
        the whole accessor return None (rather than partially
        succeed)."""
        r = _step_result(
            step_type=StepType.DEBATE,
            result_data={
                "verdict": "x",
                "transcript": [
                    {"round": 1},  # missing role + argument
                ],
            },
        )
        # transcript turn validation should fail → accessor returns None
        # rather than raising.
        view = r.as_debate_transcript()
        assert view is None
