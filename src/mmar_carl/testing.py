"""
ChainTestHarness — deterministic unit testing for CARL reasoning chains.

Provides a harness that injects mock responses per step number or tool name,
so chains can be tested without any real LLM calls.

Example usage::

    chain = ReasoningChain(steps=[...])
    harness = ChainTestHarness(chain)
    harness.set_step_response(1, "extracted content")
    harness.set_tool_response("web_search", {"results": ["a", "b"]})

    result = await harness.run(outer_context="test input")

    harness.assert_step_called(1)
    harness.assert_history_contains("extracted content")
    harness.assert_memory_contains("output", "key", contains="value")
"""

from __future__ import annotations

from typing import Any, Optional

from .models.context import ReasoningContext
from .models.llm_client_base import LLMClientBase
from .models.results import ReasoningResult, StepExecutionResult
from .models.enums import Language


class _HarnessLLMClient(LLMClientBase):
    """Mock LLM client that returns pre-set responses keyed by step number."""

    def __init__(self, harness: "ChainTestHarness") -> None:
        self._harness = harness

    async def get_response(self, prompt: str) -> str:
        step_num = self._harness._current_step
        if step_num is not None and step_num in self._harness._step_responses:
            return self._harness._step_responses[step_num]
        return self._harness._default_response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class ChainTestHarness:
    """
    Deterministic test harness for CARL reasoning chains.

    Injects mock LLM responses per step number and mock tool responses by
    tool name, enabling full chain execution without real LLM calls.

    Example::

        harness = ChainTestHarness(chain)
        harness.set_step_response(2, "summary text")
        harness.set_tool_response("fetch_data", {"rows": [1, 2, 3]})

        result = await harness.run("input context")

        harness.assert_step_called(2)
        harness.assert_history_contains("summary text")
    """

    def __init__(
        self,
        chain: Any,
        *,
        default_response: str = "",
        language: Language = Language.ENGLISH,
    ) -> None:
        """
        Args:
            chain: A ``ReasoningChain`` instance to test.
            default_response: LLM response returned when no per-step response is set.
            language: Language to use in the ``ReasoningContext``.
        """
        self._chain = chain
        self._default_response = default_response
        self._language = language

        self._step_responses: dict[int, str] = {}
        self._tool_responses: dict[str, Any] = {}

        # Runtime state, populated during ``run()``
        self._current_step: Optional[int] = None
        self._steps_called: list[int] = []
        self._context: Optional[ReasoningContext] = None
        self._result: Optional[ReasoningResult] = None

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def set_step_response(self, step_number: int, response: str) -> "ChainTestHarness":
        """Pre-set the LLM response for a specific step.

        Args:
            step_number: The step number whose LLM call will be intercepted.
            response: The text the mock LLM will return for that step.

        Returns:
            ``self`` (for chaining).
        """
        self._step_responses[step_number] = response
        return self

    def set_tool_response(self, tool_name: str, response: Any) -> "ChainTestHarness":
        """Pre-set the return value for a registered tool.

        The tool is registered on the ``ReasoningContext`` before the chain
        runs, so any ToolStep that calls ``tool_name`` will receive
        ``response``.

        Args:
            tool_name: Name the tool is registered under.
            response: Value returned when the tool is called.

        Returns:
            ``self`` (for chaining).
        """
        self._tool_responses[tool_name] = response
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        outer_context: str = "",
        *,
        memory: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        model: str = "test",
    ) -> ReasoningResult:
        """Execute the chain with mocked LLM and tool responses.

        Args:
            outer_context: The input data string passed to ``ReasoningContext``.
            memory: Optional pre-populated memory dict (``{namespace: {key: value}}``).
            metadata: Optional pre-populated metadata dict.
            model: Model identifier (doesn't affect mock execution).

        Returns:
            The ``ReasoningResult`` from chain execution.
        """
        self._current_step = None
        self._steps_called = []

        harness = self

        ctx = ReasoningContext(
            outer_context=outer_context,
            api=_HarnessLLMClient(harness),
            model=model,
            language=self._language,
        )

        if memory:
            for ns, kv in memory.items():
                if isinstance(kv, dict):
                    ctx.memory[ns] = dict(kv)

        if metadata:
            ctx.metadata.update(metadata)

        def _on_step_start(step_num: int, step_title: str) -> None:
            harness._current_step = step_num
            harness._steps_called.append(step_num)

        ctx.on_step_start = _on_step_start

        for tool_name, response in self._tool_responses.items():
            _resp = response  # capture in closure

            def _make_tool(resp: Any):
                def _tool(**kwargs: Any) -> Any:
                    return resp
                _tool.__name__ = tool_name
                return _tool

            ctx.register_tool(tool_name, _make_tool(_resp))

        self._context = ctx
        self._result = await self._chain.execute_async(ctx)
        return self._result

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    def assert_step_called(self, step_number: int) -> None:
        """Assert that the given step was executed during the last ``run()`` call."""
        assert step_number in self._steps_called, (
            f"Step {step_number} was not called. "
            f"Steps that ran: {self._steps_called}"
        )

    def assert_step_not_called(self, step_number: int) -> None:
        """Assert that the given step was NOT executed (e.g., skipped branch)."""
        assert step_number not in self._steps_called, (
            f"Step {step_number} was expected to be skipped, but it ran."
        )

    def assert_history_contains(self, text: str) -> None:
        """Assert that ``text`` appears in at least one history entry."""
        assert self._result is not None, "Call run() before asserting."
        history = self._result.history
        assert any(text in entry for entry in history), (
            f"Expected {text!r} in history but got: {history}"
        )

    def assert_history_not_contains(self, text: str) -> None:
        """Assert that ``text`` does NOT appear in any history entry."""
        assert self._result is not None, "Call run() before asserting."
        history = self._result.history
        assert not any(text in entry for entry in history), (
            f"Did not expect {text!r} in history, but found it."
        )

    def assert_memory_contains(
        self,
        namespace: str,
        key: str,
        *,
        contains: str,
    ) -> None:
        """Assert that ``context.memory[namespace][key]`` contains ``contains``.

        Args:
            namespace: Memory namespace.
            key: Key within the namespace.
            contains: Substring that must appear in the string representation of the value.
        """
        assert self._context is not None, "Call run() before asserting."
        ns_dict = self._context.memory.get(namespace, {})
        value = ns_dict.get(key)
        assert value is not None, (
            f"memory[{namespace!r}][{key!r}] is None or missing. "
            f"Available keys in namespace: {list(ns_dict.keys())}"
        )
        assert contains in str(value), (
            f"Expected {contains!r} in memory[{namespace!r}][{key!r}]={value!r}"
        )

    def assert_memory_equals(
        self,
        namespace: str,
        key: str,
        *,
        value: Any,
    ) -> None:
        """Assert that ``context.memory[namespace][key] == value``."""
        assert self._context is not None, "Call run() before asserting."
        actual = self._context.memory.get(namespace, {}).get(key)
        assert actual == value, (
            f"memory[{namespace!r}][{key!r}]: expected {value!r}, got {actual!r}"
        )

    def assert_succeeded(self) -> None:
        """Assert that the chain completed successfully."""
        assert self._result is not None, "Call run() before asserting."
        assert self._result.success, (
            f"Chain failed. Failed steps: "
            f"{[(r.step_number, r.error_message) for r in self._result.get_failed_steps()]}"
        )

    def assert_failed(self) -> None:
        """Assert that the chain did NOT complete successfully."""
        assert self._result is not None, "Call run() before asserting."
        assert not self._result.success, "Expected chain to fail, but it succeeded."

    def get_step_result(self, step_number: int) -> Optional[StepExecutionResult]:
        """Return the ``StepExecutionResult`` for a specific step, or None."""
        assert self._result is not None, "Call run() before querying step results."
        return self._result.get_step_result(step_number)

    @property
    def result(self) -> Optional[ReasoningResult]:
        """The ``ReasoningResult`` from the last ``run()`` call, or None."""
        return self._result

    @property
    def context(self) -> Optional[ReasoningContext]:
        """The ``ReasoningContext`` used in the last ``run()`` call, or None."""
        return self._context
