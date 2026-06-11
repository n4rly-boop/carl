"""
Abstract base classes for CARL reasoning system.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """
    A single message in a multi-turn conversation.

    Used by :class:`~mmar_carl.models.context.ReasoningContext` to accumulate
    a structured conversation history across LLM steps when
    ``LLMStepConfig.use_message_history=True``.

    Example::

        ctx.messages.append(ChatMessage(role="user", content="What is 2+2?"))
        ctx.messages.append(ChatMessage(role="assistant", content="4"))
    """

    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Message role: 'system', 'user', or 'assistant'.",
    )
    content: str = Field(..., description="Message text content.")

    def to_dict(self) -> dict[str, str]:
        """Convert to OpenAI-style message dict."""
        return {"role": self.role, "content": self.content}


class LLMClientBase(ABC):
    """Abstract base class for LLM clients.

    Concrete subclasses (e.g. ``OpenAICompatibleClient``) carry their own
    configuration objects, but call sites that only need basic
    introspection (current model name, temperature, max_tokens, streaming
    support) should prefer the typed properties exposed here instead of
    digging into subclass-specific attributes via ``getattr`` or
    ``isinstance`` checks. Subclasses override the properties to return
    real values; the base returns ``None`` and ``False`` so callers can
    branch safely.
    """

    # ------------------------------------------------------------------
    # Introspection properties — override in subclasses where available
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> Optional[str]:
        """Return the model identifier this client is configured to use.

        Returns ``None`` when the client has no fixed model (e.g. clients
        that select per-call). Subclasses with a static model should
        override.
        """
        return None

    @property
    def temperature(self) -> Optional[float]:
        """Return the temperature this client uses by default, if any."""
        return None

    @property
    def max_tokens(self) -> Optional[int]:
        """Return the max-tokens cap this client applies, if any."""
        return None

    @property
    def supports_streaming(self) -> bool:
        """Whether this client implements :meth:`stream_response`.

        Default: detect by checking whether the subclass overrode the
        default abstract :meth:`stream_response` (i.e. did not raise
        ``NotImplementedError`` at class definition).  Subclasses that
        provide streaming should leave the default — it will detect them
        automatically. Subclasses that explicitly want to disable streaming
        despite providing the method may override and return ``False``.
        """
        try:
            base_method = LLMClientBase.stream_response  # type: ignore[attr-defined]
            return type(self).stream_response is not base_method  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover — defensive
            return False

    # ------------------------------------------------------------------
    # Optional streaming hook — declared here so call sites can call it
    # without ``hasattr`` / ``getattr`` plumbing.
    # ------------------------------------------------------------------

    def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream a response from the LLM, yielding text chunks.

        Default implementation raises ``NotImplementedError`` (eagerly, the
        moment the method is called — not on first iteration — since
        callers should check :attr:`supports_streaming` first when calling
        defensively).  Subclasses that support streaming (e.g.
        :class:`OpenAICompatibleClient`) override with an ``async def`` that
        yields chunks from the underlying provider's streaming API; their
        broader ``AsyncIterator[str]``-compatible return shape is accepted
        here.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            An async iterator yielding text chunks.

        Example::

            async for chunk in client.stream_response("Hello!"):
                print(chunk, end="", flush=True)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement stream_response(). "
            "Either subclass and override, or check supports_streaming before calling."
        )

    @abstractmethod
    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        pass

    @abstractmethod
    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            The LLM response as a string
        """
        pass

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        """
        Get a response using separate system and user messages.

        Default implementation falls back to concatenating system + user into
        a single user message (for backward-compatible mock clients). Concrete
        clients (e.g. OpenAICompatibleClient) should override this to send
        proper system/user message roles.

        Args:
            system_prompt: System-role instructions (e.g. skill instructions)
            user_prompt: User-role task message
            retries: Maximum retry attempts

        Returns:
            The LLM response as a string
        """
        combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        return await self.get_response_with_retries(combined, retries=retries)

    async def get_response_with_usage(self, prompt: str, retries: int = 3) -> tuple[str, dict[str, int]]:
        """
        Get a response plus token usage information.

        Returns a tuple ``(response_text, usage)`` where ``usage`` is a dict with
        keys ``"prompt"``, ``"completion"``, and ``"total"`` (all ints).

        Default implementation delegates to ``get_response_with_retries`` and returns
        empty usage. Concrete clients should override to populate usage from the API.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts

        Returns:
            Tuple of (response text, token usage dict)
        """
        result = await self.get_response_with_retries(prompt, retries=retries)
        return result, {}

    async def get_response_with_messages(
        self,
        messages: "list[ChatMessage]",
        retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        """
        Get a response from a structured multi-turn message list.

        Returns ``(response_text, token_usage)`` in the same style as
        :meth:`get_response_with_usage`.

        Default implementation flattens messages to a single string and
        delegates to :meth:`get_response_with_usage`.  Concrete clients
        (e.g. :class:`~mmar_carl.llm.OpenAICompatibleClient`) should
        override this to send the message list natively.

        Args:
            messages: Ordered list of :class:`ChatMessage` objects.
            retries:  Number of retry attempts.

        Returns:
            Tuple ``(response_text, token_usage_dict)``.
        """
        # Flatten: skip system role (prepend), join user/assistant turns
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.insert(0, m.content)
            else:
                parts.append(m.content)
        flat = "\n\n".join(parts)
        return await self.get_response_with_usage(flat, retries=retries)

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Get a response with OpenAI-style tool/function call support.

        Returns a tuple ``(content, tool_calls)`` where:
        - ``content``    — the LLM's text response (may be empty if tool calls returned)
        - ``tool_calls`` — list of ``{"id": str, "name": str, "arguments": dict}``

        Default implementation does NOT support tool calls — it calls
        ``get_response_with_system()`` and returns empty ``tool_calls``.

        Concrete clients that support function calling (e.g. OpenAICompatibleClient)
        should override this method to use the native tool call API.

        Args:
            system_prompt: System-role instructions
            user_prompt:   User-role message (ignored when ``messages`` is provided)
            tools:         OpenAI-style tool definitions
            messages:      Full conversation history (system_prompt/user_prompt ignored
                           when provided)
        """
        if messages:
            # Reconstruct a combined prompt from the last user message
            last_user = next(
                (m["content"] for m in reversed(messages) if m.get("role") == "user"),
                user_prompt,
            )
            content = await self.get_response_with_retries(last_user)
        else:
            content = await self.get_response_with_system(system_prompt, user_prompt)
        return content, []
