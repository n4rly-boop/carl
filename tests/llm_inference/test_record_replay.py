"""Tests for ``RecordingLLMClient`` + ``PlayingLLMClient``.

A pytest-vcr-style cassette pattern: record once against a real (or
mocked) LLM, then replay deterministically without spending tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmar_carl import (
    CassetteMissError,
    PlayingLLMClient,
    RecordingLLMClient,
)
from mmar_carl.models.llm_client_base import ChatMessage, LLMClientBase
from mmar_carl.record_replay import _cassette_key


class _FakeClient(LLMClientBase):
    """Deterministic in-memory client used as the underlying real client."""

    def __init__(
        self,
        *,
        model: str = "fake-model",
        temperature: float = 0.5,
        usage: dict[str, int] | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._usage = usage or {"prompt": 10, "completion": 5, "total": 15}
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def temperature(self) -> float:
        return self._temperature

    async def get_response(self, prompt: str) -> str:
        self.call_count += 1
        return f"echo:{prompt}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.call_count += 1
        return f"echo:{prompt}"

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        self.call_count += 1
        return f"echo:{prompt}", dict(self._usage)

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        self.call_count += 1
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return f"reply-to:{last_user}", dict(self._usage)

    async def get_response_with_tools(self, system_prompt, user_prompt, tools, messages=None):
        self.call_count += 1
        return f"tools-reply:{user_prompt}", [
            {"id": "1", "name": "t", "arguments": {"x": 1}}
        ]


@pytest.fixture
def cassette(tmp_path: Path) -> Path:
    return tmp_path / "cassette.jsonl"


# ---------------------------------------------------------------------------
# Recording: writes file, forwards results, captures usage
# ---------------------------------------------------------------------------


class TestRecording:
    async def test_recording_returns_real_clients_response(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        assert await rec.get_response("hi") == "echo:hi"

    async def test_recording_writes_cassette_file(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        await rec.get_response_with_retries("foo")
        assert cassette.exists()
        with cassette.open() as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["method"] == "get_response_with_retries"
        assert entry["prompt"] == "foo"
        assert entry["response"] == "echo:foo"

    async def test_recording_appends_multiple_entries(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        await rec.get_response("a")
        await rec.get_response("b")
        await rec.get_response("c")
        assert rec.cassette_size == 3

    async def test_overwrite_resets_cassette(self, cassette: Path) -> None:
        cassette.write_text('{"key": "stale"}\n')
        RecordingLLMClient(_FakeClient(), cassette, overwrite=True)
        # File is wiped (recreated lazily)
        assert not cassette.exists() or cassette.read_text() == ""

    async def test_recording_captures_usage(self, cassette: Path) -> None:
        rec = RecordingLLMClient(
            _FakeClient(usage={"prompt": 7, "completion": 3, "total": 10}),
            cassette,
        )
        response, usage = await rec.get_response_with_usage("ping")
        assert response == "echo:ping"
        assert usage == {"prompt": 7, "completion": 3, "total": 10}
        # Same usage persisted
        with cassette.open() as f:
            entry = json.loads(next(line for line in f if line.strip()))
        assert entry["usage"] == {"prompt": 7, "completion": 3, "total": 10}

    async def test_recording_captures_messages(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ]
        response, _ = await rec.get_response_with_messages(msgs)
        assert response == "reply-to:hi"
        with cassette.open() as f:
            entry = json.loads(next(line for line in f if line.strip()))
        assert entry["method"] == "get_response_with_messages"
        assert entry["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]

    async def test_recording_proxies_introspection(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(model="qwen-1", temperature=0.7), cassette)
        assert rec.model_name == "qwen-1"
        assert rec.temperature == 0.7

    async def test_recording_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "cassette.jsonl"
        rec = RecordingLLMClient(_FakeClient(), deep)
        await rec.get_response("x")
        assert deep.exists()


# ---------------------------------------------------------------------------
# Playing: zero API calls, deterministic lookups
# ---------------------------------------------------------------------------


class TestPlaying:
    async def test_replay_round_trips_response(self, cassette: Path) -> None:
        real = _FakeClient()
        rec = RecordingLLMClient(real, cassette)
        await rec.get_response("hi")
        recorded_calls = real.call_count

        play = PlayingLLMClient(cassette)
        result = await play.get_response("hi")
        assert result == "echo:hi"
        # No additional calls to the real client during replay
        assert real.call_count == recorded_calls

    async def test_replay_preserves_usage(self, cassette: Path) -> None:
        rec = RecordingLLMClient(
            _FakeClient(usage={"prompt": 99, "completion": 11, "total": 110}),
            cassette,
        )
        await rec.get_response_with_usage("q")

        play = PlayingLLMClient(cassette)
        response, usage = await play.get_response_with_usage("q")
        assert response == "echo:q"
        assert usage == {"prompt": 99, "completion": 11, "total": 110}

    async def test_replay_preserves_messages(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        msgs = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="hi"),
        ]
        await rec.get_response_with_messages(msgs)

        play = PlayingLLMClient(cassette)
        result, _ = await play.get_response_with_messages(msgs)
        assert result == "reply-to:hi"

    async def test_replay_preserves_tool_calls(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        await rec.get_response_with_tools("sys", "user", [{"type": "function"}])

        play = PlayingLLMClient(cassette)
        response, tool_calls = await play.get_response_with_tools(
            "sys", "user", [{"type": "function"}]
        )
        assert response == "tools-reply:user"
        assert tool_calls == [{"id": "1", "name": "t", "arguments": {"x": 1}}]

    async def test_replay_defaults_model_from_cassette(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(model="qwen-3"), cassette)
        await rec.get_response("x")

        play = PlayingLLMClient(cassette)
        assert play.model_name == "qwen-3"

    async def test_replay_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PlayingLLMClient(tmp_path / "missing.jsonl")

    async def test_replay_miss_raises_with_preview(self, cassette: Path) -> None:
        rec = RecordingLLMClient(_FakeClient(), cassette)
        await rec.get_response("recorded")
        play = PlayingLLMClient(cassette)

        with pytest.raises(CassetteMissError) as exc:
            await play.get_response("not-in-cassette")
        # Error message names the method and shows the prompt
        assert "get_response" in str(exc.value)
        assert "not-in-cassette" in str(exc.value)


# ---------------------------------------------------------------------------
# Key stability
# ---------------------------------------------------------------------------


class TestKeyStability:
    def test_same_prompt_same_model_same_key(self) -> None:
        k1 = _cassette_key("get_response", prompt="hi", model="m", temperature=0.5)
        k2 = _cassette_key("get_response", prompt="hi", model="m", temperature=0.5)
        assert k1 == k2

    def test_different_model_different_key(self) -> None:
        k1 = _cassette_key("get_response", prompt="hi", model="A", temperature=0.5)
        k2 = _cassette_key("get_response", prompt="hi", model="B", temperature=0.5)
        assert k1 != k2

    def test_different_temperature_different_key(self) -> None:
        k1 = _cassette_key("get_response", prompt="hi", model="m", temperature=0.5)
        k2 = _cassette_key("get_response", prompt="hi", model="m", temperature=0.7)
        assert k1 != k2

    def test_different_methods_different_keys(self) -> None:
        k1 = _cassette_key("get_response", prompt="hi", model="m")
        k2 = _cassette_key("get_response_with_retries", prompt="hi", model="m")
        assert k1 != k2

    def test_none_fields_dropped_for_forward_compat(self) -> None:
        # Adding new optional params later (left None) shouldn't change the key
        k1 = _cassette_key("get_response", prompt="hi", model="m", temperature=None)
        k2 = _cassette_key("get_response", prompt="hi", model="m")
        assert k1 == k2


# ---------------------------------------------------------------------------
# End-to-end with a real ReasoningChain
# ---------------------------------------------------------------------------


class TestEndToEndChain:
    async def test_chain_records_and_replays(self, cassette: Path) -> None:
        from mmar_carl import (
            LLMStepDescription,
            ReasoningChain,
            ReasoningContext,
        )

        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="A", aim="Say hello."),
            LLMStepDescription(number=2, title="B", aim="Say goodbye.",
                                dependencies=[1]),
        ])

        # Phase 1: record
        real = _FakeClient()
        rec = RecordingLLMClient(real, cassette)
        ctx = ReasoningContext(outer_context="N/A", api=rec)
        result_rec = await chain.execute_async(ctx)
        assert result_rec.success
        recorded_calls = real.call_count
        assert recorded_calls >= 2  # one per LLM step

        # Phase 2: replay — no real client involved
        play = PlayingLLMClient(cassette)
        ctx2 = ReasoningContext(outer_context="N/A", api=play)
        result_play = await chain.execute_async(ctx2)
        assert result_play.success
        # Real client never touched again
        assert real.call_count == recorded_calls
        # Outputs identical
        assert result_play.get_final_output() == result_rec.get_final_output()
