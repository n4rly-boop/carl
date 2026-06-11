"""Tests for ``ChainBuilder.from_description`` retry-with-feedback path.

When the planner emits invalid JSON or a malformed spec, the previous
implementation aborted on first failure. The retry loop now re-prompts
the LLM with the validation error embedded so it can self-correct,
typically recovering on attempt 2 without manual intervention.
"""

from __future__ import annotations

import pytest

from mmar_carl.chain import ChainBuilder
from mmar_carl.models.llm_client_base import LLMClientBase


VALID_RESPONSE = (
    '{"steps": [{"number": 1, "step_type": "llm", "title": "x", "aim": "do x"}]}'
)


class _ScriptedPlanner(LLMClientBase):
    """LLM client that returns canned responses from a queue."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            return "{}"  # empty - causes parse to fail with missing-steps error
        return self._responses.pop(0)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


# ---------------------------------------------------------------------------
# Happy path — no retries needed
# ---------------------------------------------------------------------------


class TestNoRetryNeeded:
    @pytest.mark.asyncio
    async def test_single_shot_success_no_retries_used(self) -> None:
        planner = _ScriptedPlanner([VALID_RESPONSE])
        chain = await ChainBuilder.from_description(
            "Test task", planner, max_retries=2
        )
        assert len(chain.steps) == 1
        assert len(planner.prompts) == 1
        # No "Your previous attempt failed" in the single prompt
        assert "previous attempt failed" not in planner.prompts[0]

    @pytest.mark.asyncio
    async def test_max_retries_zero_still_one_attempt(self) -> None:
        planner = _ScriptedPlanner([VALID_RESPONSE])
        chain = await ChainBuilder.from_description(
            "Test task", planner, max_retries=0
        )
        assert len(chain.steps) == 1
        assert len(planner.prompts) == 1


# ---------------------------------------------------------------------------
# Recovery via retry
# ---------------------------------------------------------------------------


class TestRecoveryByRetry:
    @pytest.mark.asyncio
    async def test_recovers_from_invalid_json_after_one_retry(self) -> None:
        planner = _ScriptedPlanner(["not json at all", VALID_RESPONSE])
        chain = await ChainBuilder.from_description(
            "Test task", planner, max_retries=2
        )
        assert len(chain.steps) == 1
        assert len(planner.prompts) == 2
        # Second prompt carries the error message
        assert "previous attempt failed" in planner.prompts[1]
        assert "JSON" in planner.prompts[1] or "json" in planner.prompts[1]

    @pytest.mark.asyncio
    async def test_recovers_from_missing_steps_key(self) -> None:
        planner = _ScriptedPlanner([
            '{"not_steps": []}',  # missing 'steps'
            VALID_RESPONSE,
        ])
        chain = await ChainBuilder.from_description("t", planner, max_retries=2)
        assert len(chain.steps) == 1
        assert "previous attempt failed" in planner.prompts[1]
        assert "steps" in planner.prompts[1]

    @pytest.mark.asyncio
    async def test_recovers_from_empty_steps_list(self) -> None:
        planner = _ScriptedPlanner([
            '{"steps": []}',  # empty list rejected
            VALID_RESPONSE,
        ])
        chain = await ChainBuilder.from_description("t", planner, max_retries=2)
        assert len(chain.steps) == 1

    @pytest.mark.asyncio
    async def test_recovers_from_too_many_steps(self) -> None:
        bad = (
            '{"steps": ['
            + ", ".join(
                f'{{"number": {i+1}, "step_type": "llm", "title": "s", "aim": "x"}}'
                for i in range(5)
            )
            + "]}"
        )
        planner = _ScriptedPlanner([bad, VALID_RESPONSE])
        chain = await ChainBuilder.from_description(
            "t", planner, max_steps=2, max_retries=2
        )
        assert len(chain.steps) == 1
        assert "max_steps" in planner.prompts[1] or "exceeding" in planner.prompts[1]

    @pytest.mark.asyncio
    async def test_recovers_after_two_failures(self) -> None:
        """First two attempts fail differently, third succeeds."""
        planner = _ScriptedPlanner([
            "not json",  # parse error
            '{"steps": "not a list"}',  # validation error
            VALID_RESPONSE,
        ])
        chain = await ChainBuilder.from_description("t", planner, max_retries=2)
        assert len(chain.steps) == 1
        assert len(planner.prompts) == 3
        # Both retry prompts include the error from THEIR preceding call
        assert "previous attempt failed" in planner.prompts[1]
        assert "previous attempt failed" in planner.prompts[2]


# ---------------------------------------------------------------------------
# Retries exhausted — raise with last error
# ---------------------------------------------------------------------------


class TestRetriesExhausted:
    @pytest.mark.asyncio
    async def test_persistent_bad_json_raises_after_retries(self) -> None:
        planner = _ScriptedPlanner(["bad", "bad", "bad"])
        with pytest.raises(ValueError) as exc_info:
            await ChainBuilder.from_description("t", planner, max_retries=2)
        assert "after 3 attempts" in str(exc_info.value)
        assert "from_description" in str(exc_info.value)
        # Used the full budget
        assert len(planner.prompts) == 3

    @pytest.mark.asyncio
    async def test_max_retries_zero_raises_after_single_attempt(self) -> None:
        planner = _ScriptedPlanner(["bad json"])
        with pytest.raises(ValueError) as exc_info:
            await ChainBuilder.from_description("t", planner, max_retries=0)
        assert "after 1 attempt" in str(exc_info.value)
        assert len(planner.prompts) == 1

    @pytest.mark.asyncio
    async def test_last_error_chained_via_from(self) -> None:
        planner = _ScriptedPlanner(["bad"])
        with pytest.raises(ValueError) as exc_info:
            await ChainBuilder.from_description("t", planner, max_retries=0)
        # Chained exception preserves the original ValueError
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# Prompt feedback shape
# ---------------------------------------------------------------------------


class TestFeedbackPromptShape:
    @pytest.mark.asyncio
    async def test_retry_prompt_includes_base_prompt_plus_error(self) -> None:
        planner = _ScriptedPlanner(["bad", VALID_RESPONSE])
        await ChainBuilder.from_description("My task here", planner, max_retries=1)
        # Both prompts include the task
        assert "My task here" in planner.prompts[0]
        assert "My task here" in planner.prompts[1]
        # Only the retry prompt has the feedback section
        assert "previous attempt failed" not in planner.prompts[0]
        assert "previous attempt failed" in planner.prompts[1]
        # And the instruction to NOT repeat the mistake
        assert "Do not repeat" in planner.prompts[1] or "do not repeat" in planner.prompts[1].lower()

    @pytest.mark.asyncio
    async def test_retry_prompts_compound_each_attempt(self) -> None:
        """Each retry rebuilds from the original prompt + the latest error
        (not a chain of all errors), so the LLM gets the most relevant
        feedback without context bloat."""
        planner = _ScriptedPlanner([
            "first bad",
            '{"steps": []}',  # second bad (different error)
            VALID_RESPONSE,
        ])
        await ChainBuilder.from_description("t", planner, max_retries=2)
        # 2nd prompt mentions first error (JSON)
        assert "JSON" in planner.prompts[1] or "json" in planner.prompts[1]
        # 3rd prompt mentions the SECOND error (empty list), not the first
        assert "non-empty list" in planner.prompts[2]


# ---------------------------------------------------------------------------
# Negative max_retries treated as 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_negative_max_retries_treated_as_single_attempt() -> None:
    planner = _ScriptedPlanner([VALID_RESPONSE])
    chain = await ChainBuilder.from_description("t", planner, max_retries=-5)
    assert len(chain.steps) == 1
    assert len(planner.prompts) == 1
