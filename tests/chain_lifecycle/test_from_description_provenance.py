"""Tests for planner-provenance capture.

``ChainBuilder.from_description`` records the full planner prompt, the
raw reply, and every retry attempt into ``chain.metadata`` so callers
can diagnose a failed run offline without re-hitting the LLM.
"""

from __future__ import annotations

import json

import pytest

from mmar_carl import ChainBuilder
from mmar_carl.models.llm_client_base import LLMClientBase


VALID_PLAN = json.dumps({
    "steps": [
        {"number": 1, "title": "Plan",
          "step_type": "llm", "aim": "Plan a solution."},
        {"number": 2, "title": "Synth",
          "step_type": "llm", "aim": "Synthesize.",
          "dependencies": [1]},
    ]
})

# A reply that fails validation — "steps" present but empty.
INVALID_PLAN = json.dumps({"steps": []})


class _ScriptedClient(LLMClientBase):
    """Replays a sequence of canned replies and records the prompts it received."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.received_prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.received_prompts.append(prompt)
        if not self._replies:
            raise AssertionError("scripted client exhausted")
        return self._replies.pop(0)


# ---------------------------------------------------------------------------
# Successful single-shot run
# ---------------------------------------------------------------------------


class TestSingleShot:
    async def test_provenance_keys_populated(self) -> None:
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("test task", client)
        md = chain.metadata
        assert "generated_from_description" in md
        assert "planner_prompt" in md
        assert "planner_reply" in md
        assert "planner_attempts" in md

    async def test_task_tag_preserved(self) -> None:
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("test task", client)
        assert chain.metadata["generated_from_description"] == "test task"

    async def test_planner_prompt_contains_planner_directive(self) -> None:
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("test task", client)
        prompt = chain.metadata["planner_prompt"]
        assert "CARL chain planner" in prompt

    async def test_planner_reply_matches_input(self) -> None:
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("test task", client)
        assert chain.metadata["planner_reply"] == VALID_PLAN

    async def test_attempts_list_has_one_entry_on_success(self) -> None:
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("test task", client)
        attempts = chain.metadata["planner_attempts"]
        assert len(attempts) == 1
        a = attempts[0]
        assert a["attempt"] == 1
        assert a["error"] is None
        assert "CARL chain planner" in a["prompt"]
        assert a["reply"] == VALID_PLAN


# ---------------------------------------------------------------------------
# Retry-loop provenance
# ---------------------------------------------------------------------------


class TestRetryLoop:
    async def test_attempts_record_each_retry(self) -> None:
        client = _ScriptedClient([INVALID_PLAN, VALID_PLAN])
        chain = await ChainBuilder.from_description(
            "retry task", client, max_retries=2,
        )
        attempts = chain.metadata["planner_attempts"]
        assert len(attempts) == 2
        # First attempt failed, second succeeded
        assert attempts[0]["error"] is not None
        assert "non-empty list" in attempts[0]["error"]
        assert attempts[1]["error"] is None
        # Attempts are numbered 1, 2
        assert [a["attempt"] for a in attempts] == [1, 2]

    async def test_remediation_prompt_recorded(self) -> None:
        """The second attempt's prompt should include the error from
        attempt 1 (the retry-loop appends a remediation turn)."""
        client = _ScriptedClient([INVALID_PLAN, VALID_PLAN])
        chain = await ChainBuilder.from_description(
            "retry task", client, max_retries=2,
        )
        attempts = chain.metadata["planner_attempts"]
        second_prompt = attempts[1]["prompt"]
        assert "Your previous attempt failed validation" in second_prompt
        assert "non-empty list" in second_prompt

    async def test_planner_reply_reflects_final_attempt(self) -> None:
        client = _ScriptedClient([INVALID_PLAN, VALID_PLAN])
        chain = await ChainBuilder.from_description(
            "retry task", client, max_retries=2,
        )
        # The final reply (which built the chain) is the second one
        assert chain.metadata["planner_reply"] == VALID_PLAN


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    async def test_huge_reply_truncated_in_metadata(self) -> None:
        """A pathologically large reply gets truncated with a clear
        marker so the chain spec stays serialisable."""
        # Embed a valid plan with 8K of junk in a step title.
        long_title = "X" * 8000
        plan = json.dumps({
            "steps": [
                {"number": 1, "title": long_title,
                  "step_type": "llm", "aim": "go"},
            ]
        })
        client = _ScriptedClient([plan])
        chain = await ChainBuilder.from_description("task", client)
        reply = chain.metadata["planner_reply"]
        # Below the raw size, plus the truncation marker
        assert "truncated" in reply
        assert len(reply) < len(plan)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_metadata_survives_to_dict(self) -> None:
        """The provenance must persist through ``chain.to_dict()`` so
        users can save / load a chain and still inspect how it was
        generated."""
        client = _ScriptedClient([VALID_PLAN])
        chain = await ChainBuilder.from_description("survive task", client)
        spec = chain.to_dict()
        meta = spec.get("metadata", {})
        for key in ("planner_prompt", "planner_reply", "planner_attempts"):
            assert key in meta, f"missing {key!r} after to_dict()"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


class TestFailurePath:
    async def test_repeated_failures_raise_with_all_attempts_recorded(self) -> None:
        """When every attempt fails, the error is raised — but the
        provenance attempts ARE still populated on the scripted client's
        prompt buffer (we don't get a chain back, but the test still
        proves the recording loop ran on every attempt)."""
        client = _ScriptedClient([INVALID_PLAN, INVALID_PLAN])
        with pytest.raises(ValueError, match="failed after 2 attempts"):
            await ChainBuilder.from_description(
                "doomed task", client, max_retries=1,
            )
        # Both prompts were sent
        assert len(client.received_prompts) == 2
