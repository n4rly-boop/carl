"""Record-and-replay LLM client wrappers.

Two thin LLMClientBase subclasses that mirror the ergonomics of
``pytest-vcr``:

* :class:`RecordingLLMClient` wraps a real client, forwards every call,
  and appends a JSON-Lines cassette entry capturing the prompt + the
  client's response (plus token usage and tool-call structures when
  applicable). Cassettes are append-only by default — pass
  ``overwrite=True`` to start fresh.

* :class:`PlayingLLMClient` reads a cassette and replays responses by
  key lookup. No API calls. A miss raises :class:`CassetteMissError`
  with a prompt preview so users can re-record.

The cassette key is a stable SHA-256 of
``(method, prompt-or-args, model, temperature)`` — calls under
different models or temperatures don't collide. Each line of the
cassette is a single JSON object, so cassettes are human-readable
and ``git diff``-friendly.

Limitation: ``ReasoningContext.get_llm_client_for_step`` clones the
inner client when a per-step ``LLMStepConfig.model`` / temperature /
max_tokens override is set and the inner client is an
``OpenAICompatibleClient``. Since our wrappers are not
``OpenAICompatibleClient``, that override path is skipped and the
wrapped client is used as-is — fine for recording, but per-step model
overrides won't take effect when recording or replaying.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .models.llm_client_base import ChatMessage, LLMClientBase


class CassetteMissError(KeyError):
    """Raised when :class:`PlayingLLMClient` is asked for a key not in the cassette."""

    pass


def _stable_json(obj: Any) -> str:
    """JSON-serialise with stable key order for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _cassette_key(
    method: str,
    *,
    prompt: Optional[str] = None,
    messages: Optional[list[dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Build a stable SHA-256 key for one LLM interaction.

    None-valued fields are dropped so that adding new optional fields
    later doesn't invalidate previously-recorded cassette entries.
    """
    payload = {
        "method": method,
        "prompt": prompt,
        "messages": messages,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "tools": tools,
        "model": model,
        "temperature": temperature,
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


class RecordingLLMClient(LLMClientBase):
    """Wraps a real LLM client, forwarding calls and appending each
    (prompt, response) pair to a JSON-Lines cassette.

    Args:
        real_client: The underlying client (e.g.
            :class:`~mmar_carl.llm.OpenAICompatibleClient`).
        cassette_path: Path to the JSONL cassette file. Parent
            directories are created on first write.
        overwrite: If ``True`` and the cassette already exists, it is
            removed before the first write. Default ``False`` —
            cassettes are append-only, so a chain re-run can capture
            new interactions without losing old ones.
    """

    def __init__(
        self,
        real_client: LLMClientBase,
        cassette_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> None:
        self._real = real_client
        self._cassette_path = Path(cassette_path)
        if overwrite and self._cassette_path.exists():
            self._cassette_path.unlink()
        self._cassette_path.parent.mkdir(parents=True, exist_ok=True)

    # --- introspection proxies ---
    @property
    def model_name(self) -> Optional[str]:
        return self._real.model_name

    @property
    def temperature(self) -> Optional[float]:
        return self._real.temperature

    @property
    def max_tokens(self) -> Optional[int]:
        return self._real.max_tokens

    @property
    def supports_streaming(self) -> bool:
        # Streaming is not captured — users can fall back to non-streaming
        # methods, which are all supported.
        return False

    @property
    def cassette_path(self) -> Path:
        """Path to the on-disk cassette file."""
        return self._cassette_path

    @property
    def cassette_size(self) -> int:
        """Number of entries recorded so far (counts file lines)."""
        if not self._cassette_path.exists():
            return 0
        with self._cassette_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _write(self, entry: dict[str, Any]) -> None:
        with self._cassette_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def get_response(self, prompt: str) -> str:
        result = await self._real.get_response(prompt)
        key = _cassette_key(
            "get_response", prompt=prompt,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response",
            "prompt": prompt, "response": result,
            "model": self.model_name, "temperature": self.temperature,
        })
        return result

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        result = await self._real.get_response_with_retries(prompt, retries)
        key = _cassette_key(
            "get_response_with_retries", prompt=prompt,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response_with_retries",
            "prompt": prompt, "response": result,
            "model": self.model_name, "temperature": self.temperature,
        })
        return result

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        result = await self._real.get_response_with_system(system_prompt, user_prompt, retries)
        key = _cassette_key(
            "get_response_with_system",
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response_with_system",
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "response": result,
            "model": self.model_name, "temperature": self.temperature,
        })
        return result

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        response, usage = await self._real.get_response_with_usage(prompt, retries)
        key = _cassette_key(
            "get_response_with_usage", prompt=prompt,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response_with_usage",
            "prompt": prompt, "response": response, "usage": dict(usage),
            "model": self.model_name, "temperature": self.temperature,
        })
        return response, usage

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        response, usage = await self._real.get_response_with_messages(messages, retries)
        msg_dicts = [m.to_dict() for m in messages]
        key = _cassette_key(
            "get_response_with_messages", messages=msg_dicts,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response_with_messages",
            "messages": msg_dicts, "response": response, "usage": dict(usage),
            "model": self.model_name, "temperature": self.temperature,
        })
        return response, usage

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        content, tool_calls = await self._real.get_response_with_tools(
            system_prompt, user_prompt, tools, messages
        )
        key = _cassette_key(
            "get_response_with_tools",
            system_prompt=system_prompt, user_prompt=user_prompt,
            messages=messages, tools=tools,
            model=self.model_name, temperature=self.temperature,
        )
        self._write({
            "key": key, "method": "get_response_with_tools",
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "messages": messages, "tools": tools,
            "response": content, "tool_calls": tool_calls,
            "model": self.model_name, "temperature": self.temperature,
        })
        return content, tool_calls


class PlayingLLMClient(LLMClientBase):
    """Replays a cassette recorded by :class:`RecordingLLMClient`.

    Zero API calls are made. Construction loads the JSONL into an
    in-memory dict keyed by cassette key. A miss raises
    :class:`CassetteMissError` with the method name and a prompt
    preview so the caller can debug or re-record.

    Args:
        cassette_path: Path to the JSONL cassette.
        model: Optional override for the model name returned by
            :attr:`model_name`. Defaults to the model recorded on the
            first cassette entry.
        temperature: Optional override for :attr:`temperature`.
            Defaults to the value recorded on the first cassette entry.
    """

    def __init__(
        self,
        cassette_path: str | Path,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self._cassette_path = Path(cassette_path)
        self._model = model
        self._temperature = temperature
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._cassette_path.exists():
            raise FileNotFoundError(f"Cassette not found: {self._cassette_path}")
        with self._cassette_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self._entries[entry["key"]] = entry
                # Default model/temperature from first entry if user didn't set
                if self._model is None:
                    self._model = entry.get("model")
                if self._temperature is None:
                    self._temperature = entry.get("temperature")

    @property
    def cassette_path(self) -> Path:
        return self._cassette_path

    @property
    def cassette_size(self) -> int:
        return len(self._entries)

    @property
    def model_name(self) -> Optional[str]:
        return self._model

    @property
    def temperature(self) -> Optional[float]:
        return self._temperature

    @property
    def supports_streaming(self) -> bool:
        return False

    def _lookup(self, key: str, method: str, preview: str) -> dict[str, Any]:
        entry = self._entries.get(key)
        if entry is None:
            short = preview if len(preview) <= 80 else preview[:77] + "..."
            raise CassetteMissError(
                f"Cassette miss for method='{method}' "
                f"(key={key[:12]}…) preview={short!r}. "
                f"Re-record with RecordingLLMClient to capture this interaction."
            )
        return entry

    async def get_response(self, prompt: str) -> str:
        key = _cassette_key(
            "get_response", prompt=prompt,
            model=self._model, temperature=self._temperature,
        )
        return self._lookup(key, "get_response", prompt)["response"]

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        key = _cassette_key(
            "get_response_with_retries", prompt=prompt,
            model=self._model, temperature=self._temperature,
        )
        return self._lookup(key, "get_response_with_retries", prompt)["response"]

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        key = _cassette_key(
            "get_response_with_system",
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=self._model, temperature=self._temperature,
        )
        return self._lookup(key, "get_response_with_system", user_prompt)["response"]

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        key = _cassette_key(
            "get_response_with_usage", prompt=prompt,
            model=self._model, temperature=self._temperature,
        )
        entry = self._lookup(key, "get_response_with_usage", prompt)
        return entry["response"], dict(entry.get("usage", {}))

    async def get_response_with_messages(
        self, messages: list[ChatMessage], retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        msg_dicts = [m.to_dict() for m in messages]
        key = _cassette_key(
            "get_response_with_messages", messages=msg_dicts,
            model=self._model, temperature=self._temperature,
        )
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        entry = self._lookup(key, "get_response_with_messages", last_user)
        return entry["response"], dict(entry.get("usage", {}))

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        key = _cassette_key(
            "get_response_with_tools",
            system_prompt=system_prompt, user_prompt=user_prompt,
            messages=messages, tools=tools,
            model=self._model, temperature=self._temperature,
        )
        entry = self._lookup(key, "get_response_with_tools", user_prompt)
        return entry["response"], list(entry.get("tool_calls", []))
