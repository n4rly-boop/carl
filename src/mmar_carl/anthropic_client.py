"""Anthropic native LLM client.

CARL's existing :class:`mmar_carl.llm.OpenAICompatibleClient`
talks to Anthropic models via OpenRouter — that works but
loses Anthropic-specific features:

* **Native tool calling** — Anthropic emits ``tool_use``
  content blocks in a different shape from OpenAI's
  ``tool_calls`` list. Routing through OpenRouter loses the
  per-call ``tool_choice`` knob + the prompt-caching
  affordances.
* **Extended thinking** — Claude 3.7+ exposes a ``thinking``
  block when ``thinking={"type": "enabled", "budget_tokens":
  N}`` is supplied. OpenRouter forwards the response text but
  drops the structured thinking trace.
* **Vision** — Anthropic accepts ``image`` content blocks
  inline; OpenRouter shapes them as OpenAI's ``image_url``
  format.
* **Prompt caching** — Anthropic's ``cache_control``
  marker on a content block keeps the prefix in the model's
  cache for cheaper follow-ups. OpenRouter strips the marker.

This module ships :class:`AnthropicClient`, a thin async
wrapper over the official ``anthropic`` SDK that surfaces all
four. The class is a :class:`LLMClientBase` subclass so every
existing step executor + ``ReasoningContext.api`` slot accepts
it transparently — including the new structured-output
streaming path.

The ``anthropic`` SDK is lazy-imported on first use; install
via ``pip install anthropic`` (or the future
``mmar-carl[anthropic]`` extra). A missing install surfaces as
``AnthropicClientError`` rather than a raw ``ImportError`` so
callers handle one exception class.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field

from .models.llm_client_base import ChatMessage, LLMClientBase


class AnthropicClientError(RuntimeError):
    """Raised when the Anthropic client can't proceed —
    missing SDK install, missing API key, malformed config."""


class AnthropicClientConfig(BaseModel):
    """Configuration for :class:`AnthropicClient`.

    Mirrors the field set :class:`OpenAIClientConfig` uses plus
    Anthropic-specific knobs (``thinking_budget``,
    ``cache_system``).
    """

    api_key: str = Field(
        ...,
        description="Anthropic API key (env var ANTHROPIC_API_KEY is the SDK fallback).",
    )
    model: str = Field(
        default="claude-3-7-sonnet-latest",
        description="Anthropic model identifier (e.g. 'claude-3-7-sonnet-latest').",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description=(
            "Anthropic REQUIRES max_tokens — unlike OpenAI it has no "
            "implicit cap. Default 4096 matches the Anthropic SDK's typical "
            "ceiling."
        ),
    )
    timeout: float = Field(default=120.0, gt=0)
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Override the Anthropic API host (e.g. for an enterprise gateway "
            "or VPC endpoint). None uses the SDK default."
        ),
    )
    extra_headers: dict[str, str] = Field(default_factory=dict)
    thinking_budget: Optional[int] = Field(
        default=None,
        description=(
            "When set, enables Claude's extended-thinking mode with this "
            "token budget. Only models that support thinking honour it "
            "(Claude 3.7+). None disables the feature."
        ),
    )
    cache_system: bool = Field(
        default=False,
        description=(
            "When True, mark every system-prompt block with "
            "``cache_control={'type': 'ephemeral'}`` so the prefix lands "
            "in Anthropic's prompt cache. Saves cost on long system "
            "prompts that repeat across calls."
        ),
    )


class AnthropicClient(LLMClientBase):
    """Async client speaking Anthropic's native API directly.

    Construct with an :class:`AnthropicClientConfig`. The
    underlying :class:`anthropic.AsyncAnthropic` instance is
    lazy-built on first use so importing this module doesn't
    require the SDK to be installed.

    Use cases CARL gets here that the OpenAI-compatible path
    couldn't reach:

    * ``thinking_budget=N`` on the config enables Claude
      extended thinking — the structured ``thinking`` block
      is exposed via :meth:`get_response_with_thinking`.
    * Native tool-call format via
      :meth:`get_response_with_tools` (Anthropic returns
      ``tool_use`` content blocks that this method projects
      back into CARL's ``[{id, name, arguments}]`` shape).
    * Vision via :meth:`get_response_with_image` (single
      image-plus-text turn; multi-image follow-ups use the
      messages list directly).
    * ``cache_system=True`` flags the system prompt for
      Anthropic's prompt cache — cheaper repeated calls
      against a stable prefix.

    Testability: pass a ``sdk_client`` constructor kwarg with
    anything that quacks like :class:`anthropic.AsyncAnthropic`.
    The default builds the real SDK; tests inject a stub.
    """

    def __init__(
        self,
        config: AnthropicClientConfig,
        *,
        sdk_client: Any = None,
    ) -> None:
        self.config = config
        self._sdk_client = sdk_client

    # --- LLMClientBase introspection -----------------------------------

    @property
    def model_name(self) -> Optional[str]:
        return self.config.model

    @property
    def temperature(self) -> Optional[float]:
        return self.config.temperature

    @property
    def max_tokens(self) -> Optional[int]:
        return self.config.max_tokens

    @property
    def supports_streaming(self) -> bool:  # noqa: D401 — short overrides
        return True

    # --- Lazy SDK plumbing ---------------------------------------------

    @property
    def client(self) -> Any:
        """Lazy-initialise the :class:`anthropic.AsyncAnthropic`
        instance. Friendly error when the SDK isn't installed."""
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise AnthropicClientError(
                "anthropic SDK is not installed; install with "
                "`pip install anthropic` (or `pip install \"care[anthropic]\"` "
                "when using CARE) to use AnthropicClient"
            ) from exc
        if not self.config.api_key:
            raise AnthropicClientError(
                "AnthropicClientConfig.api_key must be set"
            )
        kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.extra_headers:
            kwargs["default_headers"] = self.config.extra_headers
        self._sdk_client = AsyncAnthropic(**kwargs)
        return self._sdk_client

    # --- Required LLMClientBase API ------------------------------------

    async def get_response(self, prompt: str) -> str:
        """Single-turn user-only call. Returns the assembled
        text content."""
        return await self._messages_create_text(
            messages=[{"role": "user", "content": prompt}],
            system=None,
        )

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Wraps :meth:`get_response` with exponential-backoff
        retries (``2 ** attempt`` seconds, max ``retries``
        attempts total). Mirrors the OpenAI client's behaviour
        so the same retry semantics apply across providers."""
        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                return await self.get_response(prompt)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or RuntimeError("Anthropic retries exhausted")

    async def get_response_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 3,
    ) -> str:
        """Two-block call: separate system + user. Lets
        Anthropic route the system prompt through its
        dedicated ``system=`` slot (and apply prompt caching
        when ``cache_system=True``)."""
        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                return await self._messages_create_text(
                    messages=[{"role": "user", "content": user_prompt}],
                    system=system_prompt or None,
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or RuntimeError("Anthropic retries exhausted")

    async def get_response_with_usage(
        self, prompt: str, retries: int = 3
    ) -> tuple[str, dict[str, int]]:
        """Returns ``(text, usage)`` with Anthropic's
        ``input_tokens`` / ``output_tokens`` projected into
        CARL's standard ``{prompt, completion, total}`` keys."""
        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                response = await self._messages_create(
                    messages=[{"role": "user", "content": prompt}],
                    system=None,
                )
                text = _extract_text(response)
                usage = _extract_usage(response)
                return text, usage
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or RuntimeError("Anthropic retries exhausted")

    async def get_response_with_messages(
        self,
        messages: list[ChatMessage],
        retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        """Send a multi-turn conversation as the Anthropic
        ``messages=`` list. System messages are concatenated
        and routed to the dedicated ``system=`` slot — Anthropic
        rejects system roles inside the messages array."""
        system_parts: list[str] = []
        chat_payload: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat_payload.append({"role": m.role, "content": m.content})
        if not chat_payload:
            return "", {}
        system = "\n\n".join(system_parts) if system_parts else None

        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                response = await self._messages_create(
                    messages=chat_payload,
                    system=system,
                )
                return _extract_text(response), _extract_usage(response)
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or RuntimeError("Anthropic retries exhausted")

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Native Anthropic tool calling.

        Args:
            system_prompt: Routed to Anthropic's ``system=``
                parameter.
            user_prompt: Used only when ``messages`` is
                ``None`` (single-turn).
            tools: Anthropic-style tool descriptors —
                ``[{"name": ..., "description": ...,
                "input_schema": {...}}]``. CARL adapters that
                already speak the OpenAI tool shape should
                project via :func:`openai_tools_to_anthropic`
                before calling this method.
            messages: Optional full conversation history. When
                supplied, ``user_prompt`` is ignored. System
                messages inside the array are concatenated
                into the ``system=`` slot per Anthropic's
                convention.

        Returns:
            ``(content, tool_calls)`` matching the
            :meth:`LLMClientBase.get_response_with_tools`
            contract — ``content`` is the assembled text
            (empty when the model only emitted tool_use
            blocks); ``tool_calls`` is a list of
            ``{"id", "name", "arguments"}`` dicts projected
            from Anthropic's ``tool_use`` content blocks.
        """
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if messages:
            chat: list[dict[str, Any]] = []
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if role == "system":
                    if isinstance(content, str):
                        system_parts.append(content)
                else:
                    chat.append({"role": role, "content": content})
        else:
            chat = [{"role": "user", "content": user_prompt}]
        system = "\n\n".join(system_parts) if system_parts else None

        response = await self._messages_create(
            messages=chat,
            system=system,
            tools=tools,
        )
        text = _extract_text(response)
        tool_calls = _extract_tool_calls(response)
        return text, tool_calls

    async def get_response_with_thinking(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Anthropic extended-thinking call.

        Requires ``config.thinking_budget`` to be set; raises
        :class:`AnthropicClientError` otherwise. Returns a
        dict ``{"text", "thinking", "usage"}`` so callers can
        log the structured reasoning trace alongside the final
        answer.

        Args:
            prompt: User-role text.
            system_prompt: Optional system block.
            retries: Same backoff convention as the other
                methods.

        Returns:
            ``{"text": str, "thinking": str, "usage": dict}``.
            ``thinking`` is empty when the model didn't emit a
            ``thinking`` block (e.g. on a model that ignored
            the budget).
        """
        if self.config.thinking_budget is None:
            raise AnthropicClientError(
                "AnthropicClient.get_response_with_thinking requires "
                "config.thinking_budget to be set"
            )

        last_error: BaseException | None = None
        for attempt in range(retries):
            try:
                response = await self._messages_create(
                    messages=[{"role": "user", "content": prompt}],
                    system=system_prompt,
                )
                return {
                    "text": _extract_text(response),
                    "thinking": _extract_thinking(response),
                    "usage": _extract_usage(response),
                }
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or RuntimeError("Anthropic retries exhausted")

    async def get_response_with_image(
        self,
        prompt: str,
        image_url: str,
        *,
        system_prompt: Optional[str] = None,
        media_type: str = "image/jpeg",
    ) -> str:
        """Single-turn vision call.

        Wraps the URL into an Anthropic ``image`` content block
        (or ``base64`` block if the URL looks like a data URI —
        the SDK can resolve URLs natively but the caller is
        responsible for ensuring the host is reachable).

        Args:
            prompt: User-role text describing what to do with
                the image.
            image_url: Source URL or ``data:image/...;base64,...``.
            system_prompt: Optional system block.
            media_type: MIME type when the URL is a remote
                fetch. Anthropic infers from the URL extension
                most of the time but accepts an explicit
                override.
        """
        if image_url.startswith("data:"):
            # Parse a data URI: data:image/png;base64,XXXX
            try:
                head, body = image_url.split(",", 1)
            except ValueError as exc:
                raise AnthropicClientError(
                    f"malformed data URI: {image_url[:32]!r}"
                ) from exc
            inferred_type = "image/jpeg"
            if ";" in head and head.startswith("data:"):
                inferred_type = head.split(";", 1)[0].removeprefix("data:")
            block_source = {
                "type": "base64",
                "media_type": inferred_type or media_type,
                "data": body,
            }
        else:
            block_source = {"type": "url", "url": image_url}

        content_blocks = [
            {"type": "image", "source": block_source},
            {"type": "text", "text": prompt},
        ]
        return await self._messages_create_text(
            messages=[{"role": "user", "content": content_blocks}],
            system=system_prompt or None,
        )

    # --- Streaming ------------------------------------------------------

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:  # type: ignore[override]
        """Async generator yielding text chunks from Anthropic's
        streaming endpoint. Used by every existing CARL
        executor that gates on :attr:`supports_streaming`.

        Anthropic exposes streaming via ``messages.stream(...)``
        which is an async context manager yielding events; we
        consume ``text_stream`` to flatten down to plain text
        chunks matching CARL's ``AsyncIterator[str]`` contract.
        """
        kwargs = self._build_create_kwargs(
            messages=[{"role": "user", "content": prompt}],
            system=None,
        )
        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

    # --- Internals ------------------------------------------------------

    async def _messages_create(
        self,
        *,
        messages: list[dict[str, Any]],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Any:
        kwargs = self._build_create_kwargs(
            messages=messages, system=system, tools=tools
        )
        return await self.client.messages.create(**kwargs)

    async def _messages_create_text(
        self,
        *,
        messages: list[dict[str, Any]],
        system: Optional[str],
    ) -> str:
        response = await self._messages_create(
            messages=messages, system=system
        )
        return _extract_text(response)

    def _build_create_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        system: Optional[str],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if system:
            if self.config.cache_system:
                # Anthropic accepts a list-of-blocks form on
                # `system=` for fine-grained cache control.
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if self.config.thinking_budget is not None:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.thinking_budget,
            }
        return kwargs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(response: Any) -> str:
    """Assemble plain text from an Anthropic ``Message``
    response — concatenates every ``text`` content block in
    order. Ignores ``tool_use`` / ``thinking`` blocks (those
    are surfaced via the dedicated methods)."""
    content = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in content:
        block_type = _block_attr(block, "type", "")
        if block_type == "text":
            text_val = _block_attr(block, "text", "")
            if text_val:
                parts.append(str(text_val))
    return "".join(parts)


def _extract_thinking(response: Any) -> str:
    """Concatenate every ``thinking`` content block — present
    only when the request enabled extended thinking."""
    content = getattr(response, "content", None) or []
    parts: list[str] = []
    for block in content:
        if _block_attr(block, "type", "") == "thinking":
            text_val = _block_attr(block, "thinking", "") or _block_attr(
                block, "text", ""
            )
            if text_val:
                parts.append(str(text_val))
    return "\n\n".join(parts)


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Project Anthropic ``tool_use`` blocks into CARL's
    ``[{id, name, arguments}]`` shape."""
    content = getattr(response, "content", None) or []
    calls: list[dict[str, Any]] = []
    for block in content:
        if _block_attr(block, "type", "") != "tool_use":
            continue
        calls.append(
            {
                "id": str(_block_attr(block, "id", "")),
                "name": str(_block_attr(block, "name", "")),
                "arguments": _block_attr(block, "input", {}) or {},
            }
        )
    return calls


def _extract_usage(response: Any) -> dict[str, int]:
    """Project Anthropic's ``Usage`` into
    ``{prompt, completion, total}`` so callers don't have to
    branch on the SDK type."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    prompt = _block_attr(usage, "input_tokens", 0) or 0
    completion = _block_attr(usage, "output_tokens", 0) or 0
    try:
        return {
            "prompt": int(prompt),
            "completion": int(completion),
            "total": int(prompt) + int(completion),
        }
    except (TypeError, ValueError):
        return {}


def _block_attr(block: Any, name: str, default: Any) -> Any:
    """Read ``name`` off a content block — handles both
    SDK-model attribute access and dict shapes (tests pass
    dicts; production gets Pydantic models)."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def openai_tools_to_anthropic(
    openai_tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate OpenAI-shaped tools into Anthropic's shape.

    OpenAI nests under ``"function": {"name", "description",
    "parameters"}``; Anthropic accepts ``{"name", "description",
    "input_schema"}`` at the top level. Lets CARL adapters that
    already speak OpenAI's surface call
    :meth:`AnthropicClient.get_response_with_tools` without
    re-authoring tool definitions.
    """
    out: list[dict[str, Any]] = []
    for tool in openai_tools or []:
        if "function" in tool and isinstance(tool["function"], dict):
            fn = tool["function"]
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                }
            )
        else:
            # Already Anthropic-shaped; pass through.
            out.append(dict(tool))
    return out


__all__ = [
    "AnthropicClient",
    "AnthropicClientConfig",
    "AnthropicClientError",
    "openai_tools_to_anthropic",
]
