"""Tests for list-to-string coercion on ``LLMStepDescription`` text fields.

LLM planners (notably ``ChainBuilder.from_description``) frequently emit
``reasoning_questions`` / ``stage_action`` / ``example_reasoning`` as JSON
arrays. Before this fix, pydantic rejected the array and the whole chain
generation aborted with a ``ValidationError``. The validator now coerces
lists to a newline-joined ``"- item1\\n- item2"`` string so the chain
construction proceeds.
"""

from __future__ import annotations

import json

import pytest

from mmar_carl import LLMStepDescription, ReasoningChain


# ---------------------------------------------------------------------------
# Direct construction coercion
# ---------------------------------------------------------------------------


class TestStringInputUnchanged:
    """Plain string inputs continue to work as before."""

    def test_reasoning_questions_string_passes_through(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", reasoning_questions="What is the answer?"
        )
        assert s.reasoning_questions == "What is the answer?"

    def test_stage_action_string_passes_through(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", stage_action="Compute it."
        )
        assert s.stage_action == "Compute it."

    def test_example_reasoning_string_passes_through(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", example_reasoning="12 - 5 = 7"
        )
        assert s.example_reasoning == "12 - 5 = 7"

    def test_aim_string_passes_through(self) -> None:
        s = LLMStepDescription(number=1, title="A", aim="Analyze the problem.")
        assert s.aim == "Analyze the problem."

    def test_empty_string_still_accepted(self) -> None:
        s = LLMStepDescription(
            number=1,
            title="A",
            aim="x",
            reasoning_questions="",
            stage_action="",
            example_reasoning="",
        )
        assert s.reasoning_questions == ""
        assert s.stage_action == ""
        assert s.example_reasoning == ""


class TestListInputCoercedToBulletString:
    def test_reasoning_questions_list_becomes_bulleted_string(self) -> None:
        s = LLMStepDescription(
            number=1,
            title="A",
            aim="x",
            reasoning_questions=["What is the operation?", "What numbers are involved?"],
        )
        assert s.reasoning_questions == (
            "- What is the operation?\n- What numbers are involved?"
        )
        assert isinstance(s.reasoning_questions, str)

    def test_stage_action_list_becomes_bulleted_string(self) -> None:
        s = LLMStepDescription(
            number=1,
            title="A",
            aim="x",
            stage_action=["List the quantities.", "Apply the operation."],
        )
        assert s.stage_action == "- List the quantities.\n- Apply the operation."

    def test_example_reasoning_list_becomes_bulleted_string(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", example_reasoning=["a", "b", "c"]
        )
        assert s.example_reasoning == "- a\n- b\n- c"

    def test_aim_list_also_coerced(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim=["Goal A", "Goal B"]
        )
        assert s.aim == "- Goal A\n- Goal B"

    def test_single_item_list_still_bulleted(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", reasoning_questions=["only one"]
        )
        assert s.reasoning_questions == "- only one"

    def test_empty_list_becomes_empty_string(self) -> None:
        s = LLMStepDescription(
            number=1, title="A", aim="x", reasoning_questions=[]
        )
        assert s.reasoning_questions == ""

    def test_non_string_list_items_stringified(self) -> None:
        """LLMs sometimes emit mixed types in arrays; coerce them all."""
        s = LLMStepDescription(
            number=1, title="A", aim="x", reasoning_questions=[1, 2.5, "three"]
        )
        assert s.reasoning_questions == "- 1\n- 2.5\n- three"


# ---------------------------------------------------------------------------
# Round-trip via from_dict — the actual from_description path
# ---------------------------------------------------------------------------


class TestRoundTripFromDict:
    def test_from_dict_with_list_valued_planner_output(self) -> None:
        """Simulates the exact failure observed in the live benchmark:
        planner emits ``reasoning_questions: [...]`` in the JSON spec."""
        spec = {
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Plan",
                    "aim": "Identify the operation.",
                    "reasoning_questions": [
                        "What is the operation required?",
                        "What numbers are involved?",
                    ],
                    "stage_action": ["List numbers.", "Pick op."],
                    "example_reasoning": ["12 - 5 = 7"],
                    "dependencies": [],
                },
            ],
        }
        # Pre-fix: this raised ValidationError. Post-fix: it succeeds.
        chain = ReasoningChain.from_dict(spec, use_typed_steps=True)
        step = chain.steps[0]
        assert isinstance(step, LLMStepDescription)
        assert "What is the operation required?" in step.reasoning_questions
        assert "What numbers are involved?" in step.reasoning_questions
        # Bullets present
        assert step.reasoning_questions.startswith("- ")
        assert step.stage_action.startswith("- ")
        assert step.example_reasoning.startswith("- ")

    def test_from_dict_with_mixed_string_and_list_fields(self) -> None:
        spec = {
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Plan",
                    "aim": "x",  # string
                    "reasoning_questions": ["q1", "q2"],  # list
                    "stage_action": "single action",  # string
                    "example_reasoning": ["e1"],  # list
                },
            ],
        }
        chain = ReasoningChain.from_dict(spec, use_typed_steps=True)
        s = chain.steps[0]
        assert s.aim == "x"
        assert s.reasoning_questions == "- q1\n- q2"
        assert s.stage_action == "single action"
        assert s.example_reasoning == "- e1"

    def test_json_round_trip_preserves_coerced_string(self) -> None:
        """After coercion, to_dict / from_dict round-trips as a clean string."""
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1,
                    title="A",
                    aim="x",
                    reasoning_questions=["q1", "q2"],
                ),
            ],
        )
        dumped = json.loads(json.dumps(chain.to_dict()))
        restored = ReasoningChain.from_dict(dumped, use_typed_steps=True)
        assert restored.steps[0].reasoning_questions == "- q1\n- q2"


# ---------------------------------------------------------------------------
# Error handling still works for genuinely invalid inputs
# ---------------------------------------------------------------------------


class TestStillRejectsTrulyInvalidInputs:
    def test_dict_value_rejected(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            LLMStepDescription(
                number=1, title="A", aim="x", reasoning_questions={"not": "valid"}
            )

    def test_aim_required_validation_still_fires(self) -> None:
        with pytest.raises(ValueError, match="aim"):
            LLMStepDescription(number=1, title="A", aim="")
