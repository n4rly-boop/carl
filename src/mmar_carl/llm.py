"""
LLM client implementations for CARL.

Provides integration with:
- OpenAI-compatible APIs (OpenRouter, Azure OpenAI, local LLMs, etc.)
"""

import asyncio
import os
import random as _random
from typing import Any, Callable, Literal, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from pydantic import BaseModel, Field

from mmar_carl.models import LLMClientBase


class RetryPolicy(BaseModel):
    """Per-client retry policy for LLM API calls.

    Default behaviour (when no policy is set) retries every exception up to
    3 times with `2 ** attempt` second backoff — which means even
    fast-failing errors like 401 (bad API key) waste 7+ seconds before
    surfacing. With a policy, callers can:

    1. Only retry **transient** HTTP errors (default: ``[429, 500, 502, 503,
       504]``). Permanent errors (401, 403, 404, 422, …) abort immediately.
    2. Pick a backoff strategy (constant vs exponential) with explicit
       initial / max delays.
    3. Add jitter (50% × random) so multiple concurrent retries don't
       thunder-herd the API on the same wall-clock instant.

    Example::

        client = create_openai_client(
            api_key="...",
            model="gpt-4o-mini",
            retry_policy=RetryPolicy(
                max_attempts=5,
                backoff="exponential",
                initial_delay_s=0.5,
                max_delay_s=10.0,
            ),
        )
    """

    max_attempts: int = Field(
        default=3, ge=1, le=20,
        description="Total attempts (initial + retries). Min 1 (no retries), max 20.",
    )
    retry_on_status: list[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description=(
            "HTTP status codes that warrant a retry. 4xx auth/validation "
            "errors (401, 403, 404, 422) are intentionally excluded so "
            "they abort immediately rather than wasting retry budget."
        ),
    )
    backoff: Literal["constant", "exponential"] = Field(
        default="exponential",
        description="``'exponential'`` doubles the delay each attempt; ``'constant'`` keeps it flat.",
    )
    initial_delay_s: float = Field(
        default=1.0, gt=0,
        description="Delay before the FIRST retry. Subsequent delays follow ``backoff``.",
    )
    max_delay_s: float = Field(
        default=30.0, gt=0,
        description="Cap on the per-retry delay regardless of `backoff` growth.",
    )
    jitter: bool = Field(
        default=True,
        description="If True, multiply each delay by ``0.5 + random() * 0.5`` to spread out retries.",
    )

    def is_retryable(self, exc: BaseException) -> bool:
        """Check whether *exc* should trigger a retry based on
        :attr:`retry_on_status`.

        Inspects ``exc.status_code`` (OpenAI/Anthropic-style) and falls
        back to ``exc.response.status_code`` (httpx-style). Errors with
        no recoverable status info are retried conservatively (typically
        network blips like ConnectionError, TimeoutError).
        """
        # OpenAI/Anthropic SDK exceptions expose .status_code directly.
        status = getattr(exc, "status_code", None)
        if status is None:
            # httpx-style: HTTPStatusError carries .response.
            response = getattr(exc, "response", None)
            if response is not None:
                status = getattr(response, "status_code", None)
        if status is None:
            # No status info — assume transient (network error, timeout)
            # and let it retry.
            return True
        try:
            return int(status) in self.retry_on_status
        except (TypeError, ValueError):
            return True

    def compute_delay(self, attempt: int, *, rng: Optional[_random.Random] = None) -> float:
        """Return the delay BEFORE attempt index *attempt* (0-based).

        ``attempt=0`` means "delay before the second call" (first retry).
        Optional ``rng`` for deterministic tests; defaults to ``random.random``.
        """
        if self.backoff == "constant":
            base = self.initial_delay_s
        else:  # exponential
            base = self.initial_delay_s * (2**attempt)
        delay = min(base, self.max_delay_s)
        if self.jitter:
            r = (rng.random() if rng is not None else _random.random())
            # Multiply by [0.5, 1.0) to add up-to-50% downward jitter.
            delay = delay * (0.5 + r * 0.5)
        return delay


async def _apply_retry_policy(
    coro_factory: Callable[[], Any],
    policy: RetryPolicy,
) -> Any:
    """Run ``coro_factory()`` with retry semantics from *policy*.

    ``coro_factory`` must produce a fresh coroutine on each call — bare
    coroutines aren't reusable across retries. Returns the awaited result
    on success; raises the last exception when retries are exhausted or
    a non-retryable error fires.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(policy.max_attempts):
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            if not policy.is_retryable(e):
                # Non-retryable (e.g. 401 bad API key) — abort immediately.
                raise
            if attempt < policy.max_attempts - 1:
                await asyncio.sleep(policy.compute_delay(attempt))
    raise last_error or RuntimeError("All retries exhausted")

# Environment variable for OpenAI-compatible API base URL
# Defaults to OpenRouter if not set
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


class OpenAIClientConfig(BaseModel):
    """Configuration for OpenAI-compatible LLM clients."""

    base_url: str = Field(default_factory=lambda: OPENAI_BASE_URL, description="Base URL for the OpenAI-compatible API")
    api_key: str = Field(..., description="API key for authentication")
    model: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response (None for model default)")
    timeout: float = Field(default=120.0, gt=0, description="Request timeout in seconds")
    verify_ssl: bool = Field(default=True, description="Whether to verify TLS certificates for HTTPS connections")
    extra_headers: dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")
    extra_body: dict[str, Any] = Field(default_factory=dict, description="Additional request body parameters")
    retry_policy: Optional[RetryPolicy] = Field(
        default=None,
        description=(
            "Optional :class:`RetryPolicy` controlling how the retry methods "
            "(``get_response_with_retries``, ``get_response_with_usage``, …) "
            "handle failures. When ``None`` (default), preserves the original "
            "behaviour: retry every exception up to ``retries=N`` times with "
            "``2 ** attempt`` seconds backoff. Set to a ``RetryPolicy`` "
            "instance to enable status-aware retries (don't waste retries on "
            "401/403/404), bounded backoff, and jitter."
        ),
    )


class OpenAICompatibleClient(LLMClientBase):
    """
    LLM client for OpenAI-compatible APIs (OpenRouter, Azure OpenAI, local LLMs, etc.).

    This client uses the official OpenAI Python library to communicate with any
    OpenAI-compatible API. It supports:
    - OpenRouter (default)
    - Azure OpenAI
    - Local LLMs with OpenAI-compatible APIs (LM Studio, Ollama, vLLM, etc.)
    - Any other OpenAI-compatible service

    Example usage with OpenRouter:
        ```python
        config = OpenAIClientConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-...",
            model="anthropic/claude-3.5-sonnet",
            extra_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "Your App Name"
            }
        )
        client = OpenAICompatibleClient(config)
        response = await client.get_response("Hello!")
        ```

    Example usage with local LLM:
        ```python
        config = OpenAIClientConfig(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            model="local-model",
        )
        client = OpenAICompatibleClient(config)
        ```
    """

    def __init__(self, config: OpenAIClientConfig):
        """
        Initialize the OpenAI-compatible client.

        Args:
            config: Configuration for the client

        Raises:
            ImportError: If openai package is not installed
        """
        self.config = config
        self._client: Optional[AsyncOpenAI] = None

    # --- LLMClientBase typed-introspection overrides ---

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
    def client(self) -> "AsyncOpenAI":
        """Lazy initialization of the AsyncOpenAI client."""
        if self._client is None:
            client_kwargs: dict[str, Any] = {
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
                "default_headers": self.config.extra_headers if self.config.extra_headers else None,
            }

            # Some enterprise/self-hosted gateways use custom certificates.
            # Allow opting out of TLS verification for compatibility.
            if not self.config.verify_ssl:
                if DefaultAsyncHttpxClient is not None:
                    client_kwargs["http_client"] = DefaultAsyncHttpxClient(
                        verify=False,
                        timeout=self.config.timeout,
                    )
                else:
                    client_kwargs["http_client"] = httpx.AsyncClient(
                        verify=False,
                        timeout=self.config.timeout,
                    )

            self._client = AsyncOpenAI(
                **client_kwargs,
            )
        return self._client

    async def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            The LLM response as a string
        """
        return await self._make_request(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """
        Get a response from the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM
            retries: Maximum number of retry attempts (used only when
                ``self.config.retry_policy`` is ``None``).

        Returns:
            The LLM response as a string

        Raises:
            Exception: If all retries fail, or immediately on non-retryable
            errors when a :class:`RetryPolicy` is configured.
        """
        if self.config.retry_policy is not None:
            return await _apply_retry_policy(
                lambda: self._make_request(prompt), self.config.retry_policy
            )
        # Legacy path: retry every exception, fixed exponential backoff.
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return await self._make_request(prompt)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or Exception("All retries failed")

    async def get_response_with_usage(self, prompt: str, retries: int = 3) -> tuple[str, dict[str, int]]:
        """Get a response plus token usage from the OpenAI-compatible API.

        Honours ``self.config.retry_policy`` when set (same semantics as
        :meth:`get_response_with_retries`). When unset, falls back to the
        legacy ``retries`` + exponential-backoff behaviour.
        """
        if self.config.retry_policy is not None:
            return await _apply_retry_policy(
                lambda: self._make_request_with_usage(prompt),
                self.config.retry_policy,
            )
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return await self._make_request_with_usage(prompt)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)
        raise last_error or Exception("All retries failed")

    async def get_response_with_messages(
        self,
        messages: list,
        retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        """
        Get a response from a structured multi-turn message list.

        Sends the messages list directly to the OpenAI-compatible API without
        any flattening, preserving system/user/assistant roles natively.

        Args:
            messages: List of :class:`~mmar_carl.models.llm_client_base.ChatMessage`
                      objects. Both ``ChatMessage`` instances and raw dicts with
                      ``"role"``/``"content"`` keys are accepted.
            retries: Number of retry attempts.

        Returns:
            Tuple ``(response_text, token_usage_dict)``.
        """
        last_error: Optional[Exception] = None

        for attempt in range(retries):
            try:
                raw = [
                    m.to_dict() if hasattr(m, "to_dict") else m
                    for m in messages
                ]
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=raw,
                    temperature=self.config.temperature,
                    **({"max_tokens": self.config.max_tokens} if self.config.max_tokens else {}),
                    **({"extra_body": self.config.extra_body} if self.config.extra_body else {}),
                )
                content = ""
                if response.choices and response.choices[0].message.content:
                    content = response.choices[0].message.content

                usage: dict[str, int] = {}
                if response.usage is not None:
                    p = response.usage.prompt_tokens or 0
                    c = response.usage.completion_tokens or 0
                    usage = {"prompt": p, "completion": c, "total": p + c}

                return content, usage
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt)

        raise last_error or Exception("All retries failed")

    async def get_response_with_system(
        self, system_prompt: str, user_prompt: str, retries: int = 3
    ) -> str:
        """
        Send a request with separate system and user messages.

        Uses proper OpenAI message roles so the skill instructions land in the
        system role rather than being concatenated into the user message.
        """
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return await self._make_request_with_system(system_prompt, user_prompt)
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_error or Exception("All retries failed")

    async def get_response_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict[str, Any]],
        messages: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Send a request with OpenAI-style tool definitions.

        Returns ``(content, tool_calls)`` where ``tool_calls`` is a list of::

            {"id": str, "name": str, "arguments": dict}

        If ``messages`` is provided it is used as-is (for multi-turn conversations).
        Otherwise a fresh ``[system, user]`` message list is built.
        """
        if messages:
            all_messages: list[dict[str, Any]] = list(messages)
        else:
            all_messages = []
            if system_prompt:
                all_messages.append({"role": "system", "content": system_prompt})
            all_messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": all_messages,
            "temperature": self.config.temperature,
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        import json as _json

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        if not choice:
            return "", []

        content = choice.message.content or ""

        tool_calls: list[dict[str, Any]] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return content, tool_calls

    async def _make_request_with_system(self, system_prompt: str, user_prompt: str) -> str:
        """Make a request with separate system/user messages."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        response = await self.client.chat.completions.create(**kwargs)
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    async def _make_request(self, prompt: str) -> str:
        """
        Make a single request to the LLM.

        Args:
            prompt: The prompt to send

        Returns:
            The response content as a string
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        response = await self.client.chat.completions.create(**kwargs)

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""

    async def _make_request_with_usage(self, prompt: str) -> tuple[str, dict[str, int]]:
        """Make a single request and return (response_text, token_usage)."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        response = await self.client.chat.completions.create(**kwargs)

        content = ""
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content

        usage: dict[str, int] = {}
        if response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            usage = {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            }

        return content, usage

    async def stream_response(self, prompt: str) -> Any:
        """
        Stream a response from the LLM.

        Args:
            prompt: The prompt to send to the LLM

        Yields:
            Chunks of the response as they arrive

        Example:
            ```python
            async for chunk in client.stream_response("Hello!"):
                print(chunk, end="", flush=True)
            ```
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "stream": True,
        }

        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens

        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        async with await self.client.chat.completions.create(**kwargs) as stream:
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def get_response_streaming(self, prompt: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        """
        Get a response from the LLM with optional streaming callback.

        Args:
            prompt: The prompt to send to the LLM
            on_chunk: Optional callback called with each chunk

        Returns:
            The complete LLM response as a string
        """
        full_response = ""
        async for chunk in self.stream_response(prompt):
            full_response += chunk
            if on_chunk:
                on_chunk(chunk)
        return full_response

    async def close(self) -> None:
        """
        Close the underlying OpenAI client and release resources.

        This should be called when the client is no longer needed to ensure
        proper cleanup of httpx connections and avoid event loop issues.
        """
        if self._client is not None:
            await self._client.close()
            self._client = None


def create_openai_client(
    api_key: str,
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: float = 120.0,
    verify_ssl: bool = True,
    extra_headers: Optional[dict[str, str]] = None,
    extra_body: Optional[dict[str, Any]] = None,
) -> OpenAICompatibleClient:
    """
    Factory function to create an OpenAI-compatible LLM client.

    This is a convenience function for creating OpenAICompatibleClient instances.

    Args:
        api_key: API key for authentication
        model: Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')
        base_url: Base URL for the API (default: from OPENAI_BASE_URL env var,
            or OpenRouter if not set)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (None for model default)
        timeout: Request timeout in seconds (default: 120.0)
        verify_ssl: Whether to verify TLS certificates (default: True)
        extra_headers: Additional HTTP headers
        extra_body: Additional request body parameters

    Returns:
        Configured OpenAICompatibleClient instance

    Example:
        ```python
        # Using env var (OPENAI_BASE_URL) or default (OpenRouter)
        client = create_openai_client(
            api_key="sk-or-v1-...",
            model="anthropic/claude-3.5-sonnet"
        )

        # Explicit base_url overrides env var
        client = create_openai_client(
            api_key="not-needed",
            model="llama3",
            base_url="http://localhost:11434/v1"
        )
        ```
    """
    config = OpenAIClientConfig(
        base_url=base_url or OPENAI_BASE_URL,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        verify_ssl=verify_ssl,
        extra_headers=extra_headers or {},
        extra_body=extra_body or {},
    )
    return OpenAICompatibleClient(config)
