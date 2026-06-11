"""Tests that the ``ChainBuilder.from_description`` planning prompt
declares field types explicitly so the LLM produces pydantic-valid output
on the first attempt.

Live benchmark observation: the planner naturally emits
``reasoning_questions`` as a JSON array; pydantic rejected it. Two
mitigations now in place — (a) list→bullet coercion in
``LLMStepDescription``, and (b) retry-with-feedback — but the
cleanest path is to *prevent* the mistake
by telling the LLM the types up front.
"""

from __future__ import annotations

import pytest

from mmar_carl.chain import ChainBuilder
from mmar_carl.models.llm_client_base import LLMClientBase


VALID_RESPONSE = (
    '{"steps": [{"number": 1, "step_type": "llm", "title": "t", "aim": "x"}]}'
)


class _SpyPlanner(LLMClientBase):
    def __init__(self, response: str = VALID_RESPONSE) -> None:
        self._response = response
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ---------------------------------------------------------------------------
# Prompt content checks
# ---------------------------------------------------------------------------


class TestPlanningPromptTypes:
    @pytest.mark.asyncio
    async def test_prompt_has_field_types_section(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        assert "Field types (strict" in prompt

    @pytest.mark.asyncio
    async def test_prompt_declares_number_is_int(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        assert "`number`: int" in prompt

    @pytest.mark.asyncio
    async def test_prompt_declares_title_is_string(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        assert "`title`: string" in prompt

    @pytest.mark.asyncio
    async def test_prompt_declares_step_type_enum(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        assert "`step_type`: string" in prompt
        assert "\"llm\"" in prompt
        assert "\"tool\"" in prompt
        assert "\"memory\"" in prompt
        assert "\"transform\"" in prompt

    @pytest.mark.asyncio
    async def test_prompt_declares_dependencies_as_list_of_int(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        assert "`dependencies`: list of int" in prompt

    @pytest.mark.asyncio
    async def test_prompt_warns_aim_is_string_not_array(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        # `aim`: string (NOT array)
        assert "`aim`: string" in prompt
        assert "NOT array" in prompt


class TestListProneFieldsExplicitlyMarkedAsStrings:
    """The four LLM-step fields LLMs habitually emit as JSON arrays should
    all carry the (NOT array) marker so the planner self-corrects."""

    @pytest.mark.asyncio
    async def test_reasoning_questions_marked_string_not_array(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        # Find the reasoning_questions line specifically
        rq_line = next(
            line for line in prompt.splitlines() if "`reasoning_questions`" in line
        )
        assert "string" in rq_line
        assert "NOT array" in rq_line

    @pytest.mark.asyncio
    async def test_stage_action_marked_string_not_array(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        sa_line = next(
            line for line in prompt.splitlines() if "`stage_action`" in line
        )
        assert "string" in sa_line
        assert "NOT array" in sa_line

    @pytest.mark.asyncio
    async def test_example_reasoning_marked_string_not_array(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        er_line = next(
            line for line in prompt.splitlines() if "`example_reasoning`" in line
        )
        assert "string" in er_line
        assert "NOT array" in er_line


class TestStepConfigMarkedAsObject:
    @pytest.mark.asyncio
    async def test_step_config_marked_object_not_array(self) -> None:
        planner = _SpyPlanner()
        await ChainBuilder.from_description("t", planner)
        prompt = planner.prompts[0]
        sc_line = next(
            line for line in prompt.splitlines() if "`step_config`" in line and "object" in line
        )
        assert "NOT array" in sc_line


# ---------------------------------------------------------------------------
# Existing prompt sections still present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_still_includes_task_and_rules() -> None:
    """The new section is additive — task, rules, schema all still present."""
    planner = _SpyPlanner()
    await ChainBuilder.from_description("My task here", planner)
    prompt = planner.prompts[0]
    assert "My task here" in prompt  # task
    assert "Rules:" in prompt  # rules section
    assert "Output schema:" in prompt  # schema section
    assert "do NOT use `triggered_by`" in prompt  # old rule still present


@pytest.mark.asyncio
async def test_retry_prompts_include_field_types() -> None:
    """The retry path inherits the same base prompt, so field types travel
    along with the remediation block."""
    planner = _SpyPlanner(response="bad json")
    with pytest.raises(ValueError):
        await ChainBuilder.from_description("t", planner, max_retries=2)
    # 3 attempts total; each one carries the Field types section
    for p in planner.prompts:
        assert "Field types (strict" in p


# ---------------------------------------------------------------------------
# End-to-end: prompt-with-types still produces a valid chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_with_canned_typed_response_still_succeeds() -> None:
    """A planner that produces a correctly-typed response yields a chain."""
    canned = '''
    {
      "steps": [
        {
          "number": 1,
          "title": "Plan",
          "step_type": "llm",
          "dependencies": [],
          "aim": "Identify the problem.",
          "reasoning_questions": "What are we solving?",
          "stage_action": "Enumerate quantities and operations.",
          "example_reasoning": "12 - 5 = 7"
        }
      ]
    }
    '''
    planner = _SpyPlanner(response=canned)
    chain = await ChainBuilder.from_description("solve a problem", planner)
    assert len(chain.steps) == 1
    assert chain.steps[0].aim == "Identify the problem."
    # Single-string field unchanged (no list-coercion bullet prefix)
    assert chain.steps[0].reasoning_questions == "What are we solving?"
