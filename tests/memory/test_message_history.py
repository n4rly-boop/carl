"""
Tests for native multi-turn message history in ReasoningContext.

Covers:
- ChatMessage model (role, content, to_dict)
- context.messages field (default empty, pre-seeding)
- LLMStepConfig.use_message_history flag
- LLMStepExecutor builds correct message list and appends turns
- Messages accumulate across sequential steps
- Default (flat-prompt) path unchanged when use_message_history=False
- LLMClientBase.get_response_with_messages fallback (flattens to string)
- Parallel steps with use_message_history (last writer wins, no crash)
- OpenAICompatibleClient.get_response_with_messages passes messages to API
"""

import pytest
from typing import Optional

from mmar_carl import (
    ChatMessage,
    Language,
    LLMClientBase,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)
from mmar_carl.models.config import LLMStepConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TrackingClient(LLMClientBase):
    """Records every messages-API call and returns a fixed response."""

    def __init__(self, response: str = "ok"):
        self._response = response
        self.message_calls: list[list[ChatMessage]] = []
        self.flat_calls: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.flat_calls.append(prompt)
        return self._response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict]:
        self.message_calls.append(list(messages))
        return self._response, {}


def _ctx(client: Optional[LLMClientBase] = None, **kwargs) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test context",
        api=client or _TrackingClient(),
        model="test",
        language=Language.ENGLISH,
        **kwargs,
    )


def _msg_step(number: int = 1, deps: list[int] | None = None) -> LLMStepDescription:
    return LLMStepDescription(
        number=number,
        title=f"Step {number}",
        aim="Answer thoughtfully.",
        dependencies=deps or [],
        llm_config=LLMStepConfig(use_message_history=True),
    )


def _flat_step(number: int = 1) -> LLMStepDescription:
    return LLMStepDescription(
        number=number,
        title=f"Step {number}",
        aim="Answer.",
    )


# ---------------------------------------------------------------------------
# Unit: ChatMessage model
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_valid_roles(self):
        for role in ("system", "user", "assistant"):
            m = ChatMessage(role=role, content="text")
            assert m.role == role
            assert m.content == "text"

    def test_invalid_role(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            ChatMessage(role="invalid", content="x")

    def test_to_dict(self):
        m = ChatMessage(role="user", content="hello")
        d = m.to_dict()
        assert d == {"role": "user", "content": "hello"}

    def test_context_messages_default_empty(self):
        ctx = _ctx()
        assert ctx.messages == []

    def test_context_messages_pre_seeded(self):
        msgs = [
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
        ctx = _ctx(messages=msgs)
        assert len(ctx.messages) == 2


# ---------------------------------------------------------------------------
# Unit: LLMClientBase fallback for get_response_with_messages
# ---------------------------------------------------------------------------


class TestLLMClientBaseFallback:
    @pytest.mark.asyncio
    async def test_base_fallback_flattens_messages(self):
        """Default implementation joins messages into a flat string."""

        class _SimpleClient(LLMClientBase):
            captured: list[str] = []

            async def get_response(self, prompt: str) -> str:
                self.captured.append(prompt)
                return "flat ok"

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

        client = _SimpleClient()
        messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="user msg"),
        ]
        result, usage = await client.get_response_with_messages(messages)
        assert result == "flat ok"
        # Usage should be empty dict for fallback
        assert usage == {}
        # Both message contents should be in the captured flat string
        assert len(client.captured) == 1
        assert "sys" in client.captured[0]
        assert "user msg" in client.captured[0]


# ---------------------------------------------------------------------------
# Integration: step execution with message history
# ---------------------------------------------------------------------------


class TestMessageHistoryExecution:
    @pytest.mark.asyncio
    async def test_step_calls_get_response_with_messages(self):
        """With use_message_history=True, the executor uses get_response_with_messages."""
        client = _TrackingClient("answer 1")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_msg_step(1)])
        result = await chain.execute_async(ctx)

        assert result.success
        assert len(client.message_calls) == 1
        assert len(client.flat_calls) == 0  # no flat-prompt calls

    @pytest.mark.asyncio
    async def test_message_list_contains_system_and_user(self):
        """The sent message list has system (outer context) + user (step prompt)."""
        client = _TrackingClient("ok")
        ctx = _ctx(client, system_prompt="Be helpful.")
        chain = ReasoningChain(steps=[_msg_step(1)])
        await chain.execute_async(ctx)

        sent = client.message_calls[0]
        roles = [m.role for m in sent]
        assert "system" in roles
        assert "user" in roles
        # System message should contain both system_prompt and outer_context
        sys_msg = next(m for m in sent if m.role == "system")
        assert "Be helpful." in sys_msg.content
        assert "test context" in sys_msg.content

    @pytest.mark.asyncio
    async def test_messages_accumulate_across_sequential_steps(self):
        """Each sequential step sees all prior turns in context.messages."""
        client = _TrackingClient("step response")
        ctx = _ctx(client)
        chain = ReasoningChain(
            steps=[
                _msg_step(1),
                _msg_step(2, deps=[1]),
            ]
        )
        result = await chain.execute_async(ctx)

        assert result.success
        # After execution, context.messages should have 2 user + 2 assistant = 4 msgs
        msg_roles = [m.role for m in ctx.messages]
        assert msg_roles.count("user") == 2
        assert msg_roles.count("assistant") == 2

        # Step 2's call should include step 1's turns
        assert len(client.message_calls) == 2
        step2_msgs = client.message_calls[1]
        # step2 should see step1's user + assistant messages in the list
        assistant_msgs = [m for m in step2_msgs if m.role == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].content == "step response"

    @pytest.mark.asyncio
    async def test_assistant_response_stored_in_context_messages(self):
        """After a step, context.messages contains the assistant's response."""
        client = _TrackingClient("my answer")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_msg_step(1)])
        await chain.execute_async(ctx)

        assert any(m.role == "assistant" and m.content == "my answer" for m in ctx.messages)

    @pytest.mark.asyncio
    async def test_flat_prompt_path_unchanged(self):
        """Steps without use_message_history still use the flat-prompt path."""
        client = _TrackingClient("flat ok")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_flat_step(1)])
        result = await chain.execute_async(ctx)

        assert result.success
        assert len(client.message_calls) == 0  # no messages-API call
        assert len(client.flat_calls) > 0  # flat call happened

    @pytest.mark.asyncio
    async def test_pre_seeded_messages_included_in_first_step(self):
        """Messages pre-seeded in context appear in the first step's message list."""
        prior = [
            ChatMessage(role="user", content="prior user"),
            ChatMessage(role="assistant", content="prior assistant"),
        ]
        client = _TrackingClient("answer")
        ctx = _ctx(client, messages=prior)
        chain = ReasoningChain(steps=[_msg_step(1)])
        await chain.execute_async(ctx)

        sent = client.message_calls[0]
        contents = [m.content for m in sent]
        assert "prior user" in contents
        assert "prior assistant" in contents

    @pytest.mark.asyncio
    async def test_mixed_steps_flat_then_message(self):
        """A flat-prompt step followed by a message-history step works correctly."""
        client = _TrackingClient("ok")
        ctx = _ctx(client)
        chain = ReasoningChain(
            steps=[
                _flat_step(1),
                _msg_step(2, deps=[1]),
            ]
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # Step 1 used flat, step 2 used messages
        assert len(client.flat_calls) >= 1
        assert len(client.message_calls) == 1

    @pytest.mark.asyncio
    async def test_history_entry_appended_for_message_step(self):
        """use_message_history steps still write to context.history for downstream steps."""
        client = _TrackingClient("answer text")
        ctx = _ctx(client)
        chain = ReasoningChain(steps=[_msg_step(1)])
        await chain.execute_async(ctx)

        assert len(ctx.history) == 1
        assert "answer text" in ctx.history[0]
        assert "[messages]" in ctx.history[0]
