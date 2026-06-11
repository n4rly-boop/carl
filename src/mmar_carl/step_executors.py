"""
Step executors for different step types in CARL reasoning chains.

Each step type has a dedicated executor that handles its specific execution logic.
"""

import asyncio
import fnmatch
import inspect
import json
import os
import re
import shutil
import tempfile
import functools
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

from simpleeval import EvalWithCompoundTypes

from .models import (
    ConditionalStepConfig,
    ExecutionMode,
    LLMClientBase,
    LLMStepConfig,
    Language,
    MCPStepConfig,
    MemoryOperation,
    MemoryStepConfig,
    PromptTemplate,
    ReasoningContext,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    StepDescription,
    StepExecutionResult,
    StepType,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
)
from .models.agent_skill import AgentSkillExecutionMode, AgentSkillStepConfig
from .skill_loader import SkillLoader, SkillNotFoundError, SkillParseError

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .models.agent_skill import SkillManifest


# internal sentinel raised when an executor observes a
# mid-step cancellation. The outer ``execute`` method catches it and
# converts it to a ``skipped=True, error_message="cancelled by user"``
# StepExecutionResult — distinct from a step *failure*, which gets a
# traceback. Kept private to this module; downstream code should rely
# on ``StepExecutionResult.skipped`` instead.
class _StepCancelled(Exception):
    __slots__ = ("step_number",)

    def __init__(self, step_number: int) -> None:
        super().__init__(f"step {step_number} cancelled by user")
        self.step_number = step_number


# dispatch a chunk to ``on_llm_chunk`` while supporting both
# the legacy single-arg shape and the extended ``(chunk, *, step_number,
# stage)`` shape. Introspection caches the signature on the callback so we
# don't pay the inspect cost per chunk.
def _callback_accepts_chunk_metadata(callback: Callable[..., Any]) -> bool:
    cached = getattr(callback, "_carl_chunk_meta_kw_cached", None)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        # C-extension callables / mocks without a discoverable sig fall back to
        # the single-arg shape.
        result = False
    else:
        params = sig.parameters
        has_step_kw = (
            "step_number" in params
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        )
        result = has_step_kw
    try:
        setattr(callback, "_carl_chunk_meta_kw_cached", result)
    except (AttributeError, TypeError):
        # Bound builtins / immutable callables — re-check next time; cost is small.
        pass
    return result


def _dispatch_llm_chunk(
    callback: Optional[Callable[..., Any]],
    chunk: str,
    *,
    step_number: Optional[int] = None,
    stage: Optional[str] = None,
) -> None:
    """Fire ``on_llm_chunk`` with the right argument shape.

    If the callback's signature accepts ``step_number`` or has ``**kwargs``,
    we pass the metadata; otherwise we fall back to ``callback(chunk)``.
    Exceptions are swallowed so a buggy consumer can't crash the step.
    """
    if callback is None:
        return
    try:
        if _callback_accepts_chunk_metadata(callback):
            callback(chunk, step_number=step_number, stage=stage)
        else:
            callback(chunk)
    except Exception:
        # Streaming consumer raised — log? In practice we silently drop so
        # the LLM call keeps progressing; mirrors the existing on_chunk
        # try/except in _execute_with_streaming.
        pass


def _cancelled_step_result(
    step: Any, step_type: StepType, context: ReasoningContext, start_time: float,
) -> StepExecutionResult:
    """Build a uniform ``skipped=True`` result for cancelled steps."""
    return StepExecutionResult(
        step_number=step.number,
        step_title=step.title,
        step_type=step_type,
        result="",
        success=False,
        skipped=True,
        error_message="cancelled by user",
        execution_time=time.time() - start_time,
        updated_history=context.history.copy(),
    )


# ---------------------------------------------------------------------------
# AsyncToolWrapper — safe sync-to-async adapter for registered tools
# ---------------------------------------------------------------------------


class AsyncToolWrapper:
    """
    Wraps a synchronous callable for safe use in async CARL tool steps.

    CARL's tool steps run inside an asyncio event loop. Calling a blocking
    synchronous function directly will freeze the loop and prevent parallel
    steps from making progress. ``AsyncToolWrapper`` offloads the callable
    to a thread pool via :func:`asyncio.to_thread`, keeping the event loop
    responsive during parallel batch execution.

    An optional per-call ``timeout`` (seconds) is enforced via
    :func:`asyncio.wait_for`. The timeout cancels the *awaitable* and raises
    :class:`asyncio.TimeoutError`; the underlying thread is not forcibly
    stopped (Python limitation) but will eventually complete on its own.

    Usage::

        # Wrap at registration time
        context.register_tool("slow_fn", AsyncToolWrapper(slow_fn, timeout=10.0))

        # Or use the class decorator:
        @AsyncToolWrapper.wrap(timeout=5.0)
        def my_tool(query: str) -> str:
            return requests.get(f"https://api.example.com?q={query}").text
        context.register_tool("search", my_tool)

    Notes
    -----
    - Async callables (``asyncio.iscoroutinefunction`` returns ``True``) are
      passed through unchanged — wrapping them would add unnecessary overhead.
    - Wrapped tools appear as coroutine functions to :class:`ToolStepExecutor`,
      so they are awaited directly without an extra ``run_in_executor`` call.
    """

    def __init__(self, fn: Callable, *, timeout: Optional[float] = None):
        if asyncio.iscoroutinefunction(fn):
            raise TypeError(
                f"AsyncToolWrapper: '{getattr(fn, '__name__', fn)}' is already an async function. "
                "Only synchronous callables should be wrapped."
            )
        self._fn = fn
        self._timeout = timeout
        functools.update_wrapper(self, fn)

    async def __call__(self, **kwargs: Any) -> Any:  # type: ignore[override]
        coro = asyncio.to_thread(self._fn, **kwargs)
        if self._timeout is not None:
            return await asyncio.wait_for(coro, timeout=self._timeout)
        return await coro

    @classmethod
    def wrap(cls, timeout: Optional[float] = None) -> Callable:
        """
        Decorator factory for wrapping a sync function.

        Example::

            @AsyncToolWrapper.wrap(timeout=10.0)
            def fetch_data(url: str) -> str:
                ...
        """
        def decorator(fn: Callable) -> "AsyncToolWrapper":
            return cls(fn, timeout=timeout)
        return decorator

    def __repr__(self) -> str:
        name = getattr(self._fn, "__name__", repr(self._fn))
        timeout_part = f", timeout={self._timeout}" if self._timeout is not None else ""
        return f"AsyncToolWrapper({name!r}{timeout_part})"


def _get_nested_value(data: Any, path: str) -> Any:
    """Get nested value from dict/list by dotted path."""
    if not path:
        return data

    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        elif isinstance(current, list):
            if part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                return None
        else:
            return None
    return current


def resolve_context_reference(source: str, context: ReasoningContext) -> Any:
    """Resolve a dynamic value from history, memory, metadata, step results, or outer context."""

    # Check for string literals first (quoted strings)
    if (source.startswith('"') and source.endswith('"')) or \
       (source.startswith("'") and source.endswith("'")):
        return source[1:-1]  # Remove quotes and return as-is

    if source.startswith("$history"):
        match = re.match(r"\$history\[(-?\d+)\]", source)
        if match:
            index = int(match.group(1))
            if context.history:
                return context.history[index]
            return None
        return context.get_current_history()

    if source.startswith("$memory."):
        parts = source[8:].split(".", 1)
        if len(parts) == 2:
            return context.memory_read(parts[1], namespace=parts[0])
        return context.memory_read(parts[0])

    if source.startswith("$ltm."):
        return context.ltm_retrieve(source[5:])

    if source.startswith("$event."):
        return context.get_event_payload(source[7:])

    if source.startswith("$metadata."):
        return _get_nested_value(context.metadata, source[10:])

    if source.startswith("$steps."):
        return _get_nested_value(context.metadata.get("step_results", {}), source[7:])

    if source == "$outer_context":
        outer_value = context.outer_context
        # If outer context is a string that looks like JSON, try to parse it
        if isinstance(outer_value, str):
            outer_value = outer_value.strip()
            if outer_value.startswith(("{", "[")) or \
               (outer_value.startswith('"') and outer_value.endswith('"')):
                try:
                    return json.loads(outer_value)
                except json.JSONDecodeError:
                    return outer_value  # Not valid JSON, return as-is
        return outer_value

    return context.metadata.get(source)


def _looks_complete_json(text: str) -> bool:
    """Cheap brace/bracket-balance check for the structured-
    output streaming path.

    Returns ``True`` when ``text`` contains at least one
    balanced top-level ``{...}`` or ``[...]`` — used by
    :meth:`StructuredOutputStepExecutor._execute_structured_streaming`
    as an early-exit signal. False positives are harmless
    (the caller still tries :func:`_extract_json_payload` and
    keeps streaming on failure); false negatives just delay
    the early-exit until later in the stream.

    Skips over string contents so braces inside JSON strings
    don't unbalance the counter, and tracks brackets in case
    the schema is a top-level list.
    """
    if not text:
        return False
    in_string = False
    escape = False
    curly = 0
    square = 0
    saw_open = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            curly += 1
            saw_open = True
        elif ch == "}":
            curly -= 1
        elif ch == "[":
            square += 1
            saw_open = True
        elif ch == "]":
            square -= 1
    return saw_open and curly == 0 and square == 0


def _extract_json_payload(text: str) -> Any:
    """
    Extract and parse JSON payload from model response.

    Handles multiple formats:
    - Raw JSON string
    - JSON wrapped in markdown code blocks (```json...```)
    - JSON embedded in text (extracts first {...} or [...])

    Args:
        text: Raw text response from LLM

    Returns:
        Parsed JSON object (dict or list)

    Raises:
        json.JSONDecodeError: If no valid JSON found
    """
    stripped = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _extract_model_name(llm_client: Any) -> str | None:
    """Best-effort read of ``model_name`` from an LLM client.

    Returns ``None`` when the attribute is missing or not a ``str``
    (e.g. ``MagicMock`` clients in tests would otherwise pollute the
    field with a mock object that fails pydantic validation).
    """
    name = getattr(llm_client, "model_name", None)
    return name if isinstance(name, str) else None


class StepExecutorBase(ABC):
    """Abstract base class for step executors."""

    @abstractmethod
    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """
        Execute a step and return the result.

        Args:
            step: The step description to execute
            context: The reasoning context
            prompt_template: Optional prompt template for LLM steps

        Returns:
            StepExecutionResult with execution outcome
        """
        pass


class LLMSelfCriticEvaluator(SelfCriticEvaluatorBase):
    """Default self-critic evaluator that uses the step LLM."""

    @staticmethod
    def _build_prompt(base_prompt: str, candidate: str, custom_instruction: str = "") -> str:
        extra_instruction = custom_instruction.strip()
        extra_block = (
            f"\nAdditional reviewer instruction:\n{extra_instruction}\n"
            if extra_instruction
            else ""
        )
        return (
            "You are a strict reviewer of an LLM answer.\n"
            "Evaluate whether the answer fully satisfies the task.\n"
            "Return only a valid JSON object with this exact schema:\n"
            '{"verdict":"APPROVE|DISAPPROVE","review":"short text review (1-3 lines)"}\n'
            "Do not return markdown, explanations, or any text outside this JSON object."
            f"{extra_block}\n"
            f"Task:\n{base_prompt}\n\n"
            f"Candidate answer:\n{candidate}"
        )

    @staticmethod
    def _parse_decision(text: str) -> SelfCriticDecision:
        try:
            payload = _extract_json_payload(text)
        except Exception:
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="Evaluator response is not valid JSON.",
                metadata={"llm_calls": 1},
            )

        if not isinstance(payload, dict):
            return SelfCriticDecision(
                verdict="DISAPPROVE",
                review_text="Evaluator JSON response must be an object.",
                metadata={"llm_calls": 1},
            )

        verdict_raw = str(payload.get("verdict", "")).strip().upper()
        verdict = verdict_raw if verdict_raw in {"APPROVE", "DISAPPROVE"} else "DISAPPROVE"

        review_value = payload.get("review", "")
        review_text = str(review_value).strip() if review_value is not None else ""

        if not review_text:
            review_text = "Evaluator JSON response has empty 'review' field."
            verdict = "DISAPPROVE"

        if verdict_raw and verdict_raw not in {"APPROVE", "DISAPPROVE"}:
            review_text = f"Invalid verdict '{verdict_raw}'. {review_text}"
            verdict = "DISAPPROVE"

        return SelfCriticDecision(
            verdict=verdict,
            review_text=review_text,
            metadata={"llm_calls": 1},
        )

    async def evaluate(
        self,
        step: Any,
        candidate: str,
        base_prompt: str,
        context: Any,
        llm_client: Any,
        retries: int,
    ) -> SelfCriticDecision:
        _ = context
        llm_config = getattr(step, "llm_config", None)
        custom_instruction = ""
        if llm_config is not None:
            custom_instruction = getattr(llm_config, "self_critic_instruction", "") or ""

        critique_prompt = self._build_prompt(base_prompt, candidate, custom_instruction=custom_instruction)
        critique = await llm_client.get_response_with_retries(critique_prompt, retries=retries)
        return self._parse_decision(critique)


class LLMStepExecutor(StepExecutorBase):
    """Executor for standard LLM reasoning steps."""

    _DEFAULT_SELF_CRITIC_EVALUATOR = "llm"
    _DEFAULT_SELF_CRITIC_REVISIONS = 1

    async def _execute_with_streaming(
        self,
        llm_client: Any,
        prompt: str,
        retries: int,
        on_chunk: Callable[..., None],
        *,
        step_number: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> str:
        """
        Execute LLM call with streaming and retry logic.

        Args:
            llm_client: The LLM client with stream_response method
            prompt: The prompt to send
            retries: Maximum retry attempts
            on_chunk: Callback for each chunk. May accept either the legacy
                ``(chunk)`` shape or the extended ``(chunk, *, step_number,
                stage)`` shape; routes through
                ``_dispatch_llm_chunk`` so both work.
            step_number: Step number forwarded to extended-signature callbacks.
            stage: Stage label (e.g. ``"draft"`` / ``"critic"``) forwarded
                to extended-signature callbacks so CARE can route chunks to
                the right pane.

        Returns:
            Complete response as string
        """
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                full_response = ""
                async for chunk in llm_client.stream_response(prompt):
                    full_response += chunk
                    _dispatch_llm_chunk(
                        on_chunk, chunk,
                        step_number=step_number, stage=stage,
                    )
                return full_response
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("All streaming retries failed")

    @staticmethod
    def _resolve_execution_mode(step: StepDescription) -> ExecutionMode:
        """Resolve execution mode from per-step LLM config."""
        llm_config = getattr(step, "llm_config", None)
        if llm_config is None:
            return ExecutionMode.FAST

        mode = getattr(llm_config, "execution_mode", ExecutionMode.FAST)
        if isinstance(mode, ExecutionMode):
            return mode
        if isinstance(mode, str):
            try:
                return ExecutionMode(mode)
            except ValueError:
                return ExecutionMode.FAST
        return ExecutionMode.FAST

    @staticmethod
    def _resolve_model_name(llm_client: Any) -> str | None:
        """Resolve model name for tracing."""
        if hasattr(llm_client, "config") and hasattr(llm_client.config, "model"):
            return llm_client.config.model
        return None

    async def _execute_llm_call(
        self,
        llm_client: Any,
        prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
        generation_name: str,
        allow_streaming: bool,
        *,
        step_number: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> tuple[str, dict[str, int]]:
        """Execute one LLM call with optional tracing and streaming.

        Returns (result_text, token_usage) where token_usage is a dict with keys
        "prompt", "completion", "total" (or empty dict if usage unavailable).

        ``step_number`` and ``stage`` are forwarded to the streaming chunk
        callback when its signature accepts them.
        """
        parent_span = context.metadata.get("__langfuse_span")
        generation = None
        if parent_span is not None:
            gen_kwargs: dict[str, Any] = {"name": generation_name, "input": prompt}
            if model_name:
                gen_kwargs["model"] = model_name
            generation = parent_span.start_observation(**gen_kwargs, as_type="generation")

        try:
            # Streaming gate: must (1) be enabled for this step, (2) have a
            # chunk callback configured, (3) have a client that exposes
            # streaming. For LLMClientBase subclasses we check the typed
            # ``supports_streaming`` property (the base always declares
            # ``stream_response`` but raises if not overridden, so naive
            # ``hasattr`` is a false positive). For non-base mock clients,
            # ``hasattr`` is still the right fallback.
            can_stream = (
                llm_client.supports_streaming
                if isinstance(llm_client, LLMClientBase)
                else hasattr(llm_client, "stream_response")
            )
            if allow_streaming and context.on_llm_chunk and can_stream:
                result = await self._execute_with_streaming(
                    llm_client, prompt, retries, context.on_llm_chunk,
                    step_number=step_number, stage=stage,
                )
                usage: dict[str, int] = {}
            elif isinstance(llm_client, LLMClientBase):
                result, usage = await llm_client.get_response_with_usage(prompt, retries=retries)
            else:
                result = await llm_client.get_response_with_retries(prompt, retries=retries)
                usage = {}
        except Exception as exc:
            if generation is not None:
                generation.update(output=f"ERROR: {exc}")
                generation.end()
            raise

        if generation is not None:
            generation.update(output=result)
            generation.end()

        return result, usage

    @staticmethod
    def _build_regeneration_prompt(base_prompt: str, candidate: str, review_text: str) -> str:
        """Build prompt to regenerate the candidate after self-critic disapproval."""
        return (
            "One or more evaluators DISAPPROVED the candidate answer.\n"
            "Regenerate the same task output with higher quality.\n"
            "Use the review notes below, and return only the improved final answer.\n\n"
            f"Original task:\n{base_prompt}\n\n"
            f"Previous answer:\n{candidate}\n\n"
            f"Review notes:\n{review_text}"
        )

    @staticmethod
    def _append_replan_feedback(full_prompt: str, context: ReasoningContext, step_number: int) -> tuple[str, bool]:
        """
        Append RE-PLAN feedback to the prompt when present.

        The executor injects per-step feedback into context metadata under
        '__replan_feedback'. The step executor consumes it as an additional
        instruction block for regenerated runs.
        """
        _ = step_number
        raw_feedback = context.metadata.get("__replan_feedback")
        if isinstance(raw_feedback, str):
            feedback_items = [raw_feedback]
        elif isinstance(raw_feedback, list):
            feedback_items = [str(item).strip() for item in raw_feedback if str(item).strip()]
        else:
            feedback_items = []

        if not feedback_items:
            return full_prompt, False

        feedback_block = "\n".join(f"- {item}" for item in feedback_items)
        updated_prompt = (
            f"{full_prompt}\n\n"
            "RE-PLAN feedback for this retry:\n"
            "Use these notes to improve your answer quality and direction.\n"
            f"{feedback_block}"
        )
        return updated_prompt, True

    @staticmethod
    def _normalize_llm_calls(value: Any) -> int:
        """Normalize optional llm_calls metadata value."""
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resolve_disapprove_feedback(llm_config: LLMStepConfig, evaluator_name: str) -> str:
        """Resolve static regeneration feedback for disapproved evaluator."""
        feedback_map = llm_config.self_critic_disapprove_feedback
        if not isinstance(feedback_map, dict):
            return ""

        specific = feedback_map.get(evaluator_name)
        wildcard = feedback_map.get("*")
        resolved = specific if specific is not None else wildcard
        return str(resolved).strip() if resolved is not None else ""

    def _ensure_default_self_critic_evaluator(self, context: ReasoningContext) -> None:
        """Ensure built-in 'llm' self-critic evaluator is available.
        
        Note: This method is kept for backward compatibility. The default evaluator
        is now registered during context initialization to avoid race conditions.
        """
        if context.get_self_critic_evaluator(self._DEFAULT_SELF_CRITIC_EVALUATOR) is None:
            context.register_self_critic_evaluator(
                self._DEFAULT_SELF_CRITIC_EVALUATOR, LLMSelfCriticEvaluator()
            )

    async def _execute_fast_mode(
        self,
        llm_client: Any,
        full_prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
        *,
        step_number: Optional[int] = None,
    ) -> tuple[str, dict[str, Any], dict[str, int]]:
        result, usage = await self._execute_llm_call(
            llm_client=llm_client,
            prompt=full_prompt,
            retries=retries,
            context=context,
            model_name=model_name,
            generation_name="llm_generation",
            allow_streaming=True,
            step_number=step_number,
            stage="fast",
        )
        return result, {
            "execution_mode": ExecutionMode.FAST.value,
            "llm_calls": 1,
            "rounds": 1,
            "evaluator_decisions": [],
        }, usage

    async def _execute_self_critic_mode(
        self,
        step: StepDescription,
        llm_config: LLMStepConfig,
        llm_client: Any,
        full_prompt: str,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
    ) -> tuple[str, dict[str, Any], dict[str, int]]:
        draft, usage = await self._execute_llm_call(
            llm_client=llm_client,
            prompt=full_prompt,
            retries=retries,
            context=context,
            model_name=model_name,
            generation_name="llm_generation_draft",
            allow_streaming=False,
        )
        total_usage: dict[str, int] = dict(usage)
        llm_calls = 1
        self._ensure_default_self_critic_evaluator(context)

        evaluator_names = llm_config.self_critic_evaluators or [self._DEFAULT_SELF_CRITIC_EVALUATOR]
        max_revisions = max(0, llm_config.self_critic_max_revisions)

        candidate = draft
        round_summaries: list[dict[str, Any]] = []
        max_revisions_reached = False

        for revision_round in range(max_revisions + 1):
            round_approved = True
            disapprove_reviews: list[str] = []
            evaluator_decisions: list[dict[str, Any]] = []

            for evaluator_name in evaluator_names:
                evaluator = context.get_self_critic_evaluator(evaluator_name)
                if evaluator is None:
                    raise ValueError(
                        f"Self-critic evaluator '{evaluator_name}' is not registered. "
                        f"Available evaluators: {context.list_self_critic_evaluators()}"
                    )

                decision = await evaluator.evaluate(
                    step=step,
                    candidate=candidate,
                    base_prompt=full_prompt,
                    context=context,
                    llm_client=llm_client,
                    retries=retries,
                )
                if not isinstance(decision, SelfCriticDecision):
                    raise TypeError(
                        f"Self-critic evaluator '{evaluator_name}' returned invalid result type: "
                        f"{type(decision).__name__}. Expected SelfCriticDecision."
                    )

                decision_meta = decision.metadata if isinstance(decision.metadata, dict) else {}
                decision_llm_calls = self._normalize_llm_calls(decision_meta.get("llm_calls", 0))
                llm_calls += decision_llm_calls

                verdict = decision.normalized_verdict()
                review_text = (decision.review_text or "").strip()
                if verdict == "DISAPPROVE":
                    static_feedback = self._resolve_disapprove_feedback(llm_config, evaluator_name)
                    if static_feedback:
                        review_text = f"{review_text}\n{static_feedback}".strip() if review_text else static_feedback
                if not review_text:
                    review_text = f"Evaluator '{evaluator_name}' returned empty review text."
                    verdict = "DISAPPROVE"

                evaluator_decisions.append(
                    {
                        "evaluator": evaluator_name,
                        "verdict": verdict,
                        "has_review": bool(review_text),
                        "llm_calls": decision_llm_calls,
                    }
                )
                if verdict == "DISAPPROVE":
                    round_approved = False
                    disapprove_reviews.append(f"[{evaluator_name}] {review_text}")

            round_summaries.append(
                {
                    "round": revision_round + 1,
                    "approved": round_approved,
                    "evaluators": evaluator_decisions,
                }
            )

            if round_approved:
                break

            if revision_round >= max_revisions:
                max_revisions_reached = True
                break

            # Keep current round review text local; only used for this regeneration pass.
            current_review_text = "\n\n".join(disapprove_reviews).strip()
            candidate, regen_usage = await self._execute_llm_call(
                llm_client=llm_client,
                prompt=self._build_regeneration_prompt(full_prompt, candidate, current_review_text),
                retries=retries,
                context=context,
                model_name=model_name,
                generation_name="llm_generation_regenerate",
                allow_streaming=False,
            )
            llm_calls += 1
            for k, v in regen_usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

        mode_details = {
            "execution_mode": ExecutionMode.SELF_CRITIC.value,
            "llm_calls": llm_calls,
            "rounds": len(round_summaries),
            "max_revisions": max_revisions,
            "evaluator_policy": "all_must_approve",
            "evaluator_decisions": round_summaries,
        }
        if max_revisions_reached:
            mode_details["quality_warning"] = (
                f"Reached self_critic_max_revisions={max_revisions} without full evaluator approval."
            )
        return candidate, mode_details, total_usage

    async def _execute_message_history_mode(
        self,
        step: StepDescription,
        step_prompt: str,
        llm_client: Any,
        retries: int,
        context: ReasoningContext,
        model_name: str | None,
    ) -> tuple[str, dict[str, int], list]:
        """
        Execute an LLM step using the structured messages API.

        Builds a message list:
          1. System message with system_prompt + outer_context
          2. Any prior messages from ``context.messages``
          3. Current step prompt as a 'user' message

        After the LLM responds, appends the user message and the assistant
        response to ``context.messages`` so future steps can continue the
        conversation.

        Returns (result_text, token_usage, updated_messages_list).
        """
        from .models.llm_client_base import ChatMessage

        # Build system message
        system_parts: list[str] = []
        if context.system_prompt:
            system_parts.append(context.system_prompt)
        if context.outer_context:
            if context.language == Language.ENGLISH:
                system_parts.append(f"Context:\n{context.outer_context}")
            else:
                system_parts.append(f"Контекст:\n{context.outer_context}")
        system_content = "\n\n".join(system_parts) if system_parts else ""

        # Assemble the full message list
        messages: list[ChatMessage] = []
        if system_content:
            messages.append(ChatMessage(role="system", content=system_content))
        messages.extend(context.messages)
        user_msg = ChatMessage(role="user", content=step_prompt)
        messages.append(user_msg)

        # Call the LLM
        if isinstance(llm_client, LLMClientBase):
            result, usage = await llm_client.get_response_with_messages(messages, retries=retries)
        else:
            # Fallback: flatten to a single string for non-CARL clients
            flat = "\n\n".join(m.content for m in messages)
            result = await llm_client.get_response_with_retries(flat, retries=retries)
            usage = {}

        # Persist the new turns into context.messages
        updated_messages = list(context.messages)
        updated_messages.append(user_msg)
        updated_messages.append(ChatMessage(role="assistant", content=result))

        return result, usage, updated_messages

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute an LLM reasoning step."""
        start_time = time.time()
        template = prompt_template or PromptTemplate()

        try:
            # Generate prompt for this step with RAG-like context extraction
            step_prompt = template.format_step_prompt(step, context.outer_context, context.language)

            # Get LLM client and config for this step (may have per-step overrides)
            llm_config = getattr(step, "llm_config", None)
            llm_client = context.get_llm_client_for_step(llm_config)

            # Get retry count (per-step override or context default)
            step_retries = getattr(step, "retry_max", None)
            retries = step_retries if step_retries is not None else context.retry_max

            model_name = self._resolve_model_name(llm_client)
            execution_mode = self._resolve_execution_mode(step)

            # Choose execution path: message-history vs flat-prompt
            use_msg_history = getattr(llm_config, "use_message_history", False)
            updated_messages = None  # will be set only in message-history mode
            used_replan_feedback = False

            if use_msg_history:
                result, token_usage, updated_messages = await self._execute_message_history_mode(
                    step=step,
                    step_prompt=step_prompt,
                    llm_client=llm_client,
                    retries=retries,
                    context=context,
                    model_name=model_name,
                )
                mode_details: dict[str, Any] = {"execution_mode": "message_history"}
            else:
                full_prompt = template.format_chain_prompt(
                    outer_context=context.outer_context,
                    current_task=step_prompt,
                    history=context.get_current_history(),
                    language=context.language,
                    system_prompt=context.system_prompt,
                )
                full_prompt, used_replan_feedback = self._append_replan_feedback(full_prompt, context, step.number)

                if execution_mode == ExecutionMode.SELF_CRITIC:
                    if llm_config is None:
                        llm_config = LLMStepConfig(
                            execution_mode=ExecutionMode.SELF_CRITIC,
                            self_critic_max_revisions=self._DEFAULT_SELF_CRITIC_REVISIONS,
                        )
                    result, mode_details, token_usage = await self._execute_self_critic_mode(
                        step=step,
                        llm_config=llm_config,
                        llm_client=llm_client,
                        full_prompt=full_prompt,
                        retries=retries,
                        context=context,
                        model_name=model_name,
                    )
                else:
                    result, mode_details, token_usage = await self._execute_fast_mode(
                        llm_client=llm_client,
                        full_prompt=full_prompt,
                        retries=retries,
                        context=context,
                        model_name=model_name,
                        step_number=step.number,
                    )

            context.metadata.setdefault("execution_mode_details", {})
            if used_replan_feedback:
                mode_details["replan_feedback_used"] = True
            context.metadata["execution_mode_details"][str(step.number)] = mode_details

            # Emit a warning when token budget is exceeded
            budget = getattr(llm_config, "token_budget_warning", None) if llm_config else None
            if budget is not None and token_usage:
                total_tokens = token_usage.get("total", 0)
                if total_tokens >= budget:
                    warnings.warn(
                        f"Step {step.number} '{step.title}' used {total_tokens} tokens "
                        f"(budget warning threshold: {budget}).",
                        stacklevel=2,
                    )

            # Update context history
            if use_msg_history:
                mode_suffix = " [messages]"
            elif execution_mode != ExecutionMode.FAST:
                mode_suffix = f" [{execution_mode.value}]"
            else:
                mode_suffix = ""

            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title}{mode_suffix}\nResult: {result}\n"
            else:  # Russian
                step_result = f"Шаг {step.number}. {step.title}{mode_suffix}\nРезультат: {result}\n"
            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.LLM,
                result=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
                updated_messages=updated_messages or [],
                token_usage=token_usage,
                model=_extract_model_name(llm_client),
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.LLM,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class ToolStepExecutor(StepExecutorBase):
    """Executor for tool/function call steps."""

    def _resolve_input_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve an input value from context."""
        return resolve_context_reference(source, context)

    def _coerce_value(self, value: Any, param_name: str, config: ToolStepConfig) -> Any:
        """Coerce value to the expected type based on parameter definition."""
        # Find parameter definition
        param_def = None
        for param in config.parameters:
            if param.name == param_name:
                param_def = param
                break

        if param_def is None:
            return value  # No type info, return as-is

        # Coerce based on parameter type
        if param_def.type == "int":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        elif param_def.type == "float":
            try:
                return float(value)
            except (ValueError, TypeError):
                return value
        elif param_def.type == "bool":
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        elif param_def.type == "list":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        elif param_def.type == "dict":
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        return value

    async def _call_tool(
        self,
        tool_callable: Any,
        kwargs: dict[str, Any],
        timeout: float,
    ) -> Any:
        """Invoke a tool callable (sync or async) with a timeout."""
        _is_async = asyncio.iscoroutinefunction(tool_callable) or asyncio.iscoroutinefunction(
            getattr(type(tool_callable), "__call__", None)
        )
        if _is_async:
            return await asyncio.wait_for(tool_callable(**kwargs), timeout=timeout)
        return await asyncio.wait_for(asyncio.to_thread(tool_callable, **kwargs), timeout=timeout)

    async def _call_with_recovery(
        self,
        config: "ToolStepConfig",
        kwargs: dict[str, Any],
        context: ReasoningContext,
    ) -> tuple[Any, str]:
        """
        Call the primary tool with optional retry / fallback logic.

        Returns ``(raw_result, effective_tool_name)`` where ``effective_tool_name``
        is ``config.tool_name`` on success or the fallback tool name when a fallback
        was triggered.
        """
        from .models.config import ToolErrorRecovery  # local import avoids circular

        recovery: Optional[ToolErrorRecovery] = config.error_recovery
        retry_max = recovery.retry_max if recovery else 0
        retry_delay = recovery.retry_delay if recovery else 0.0

        tool_callable = context.get_tool(config.tool_name)
        if tool_callable is None:
            raise ValueError(f"Tool '{config.tool_name}' not registered in context")

        # Enforce allowed_tool_tags whitelist (informational vs destructive separation).
        # A step that pins allowed_tool_tags=["information"] cannot call a tool
        # registered only with tags={"math"}. Untagged tools never match a non-empty
        # whitelist — that's the point: opt in explicitly.
        if config.allowed_tool_tags:
            tool_tags = context.get_tool_tags(config.tool_name)
            wanted = set(config.allowed_tool_tags)
            if not (wanted & tool_tags):
                raise PermissionError(
                    f"Tool '{config.tool_name}' tags {sorted(tool_tags) or '[]'} do not "
                    f"intersect step's allowed_tool_tags {sorted(wanted)}"
                )

        last_exc: Optional[Exception] = None
        timed_out = False

        for attempt in range(retry_max + 1):
            try:
                result = await self._call_tool(tool_callable, kwargs, config.timeout)
                return result, config.tool_name
            except asyncio.TimeoutError:
                timed_out = True
                last_exc = asyncio.TimeoutError(f"Tool timed out after {config.timeout}s")
                # Timeouts are not retriable — fall through to fallback immediately
                break
            except Exception as exc:
                last_exc = exc
                timed_out = False
                if attempt < retry_max:
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay)

        # Primary tool exhausted — try fallback
        if recovery:
            fallback_name = recovery.on_timeout if timed_out else recovery.on_exception
            if fallback_name:
                fallback = context.get_tool(fallback_name)
                if fallback is None:
                    raise ValueError(
                        f"Fallback tool '{fallback_name}' not registered in context"
                    )
                result = await self._call_tool(fallback, kwargs, config.timeout)
                return result, fallback_name

        raise last_exc or RuntimeError("Tool call failed")

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a tool call step."""
        start_time = time.time()
        config: ToolStepConfig = step.step_config  # type: ignore

        try:
            # Build arguments from input mapping
            kwargs: dict[str, Any] = {}
            for param_name, source in config.input_mapping.items():
                raw_value = self._resolve_input_value(source, context)
                kwargs[param_name] = self._coerce_value(raw_value, param_name, config)

            # Execute the tool with retry / fallback support
            result, effective_tool = await self._call_with_recovery(config, kwargs, context)

            # Convert result to string for history
            result_str = json.dumps(result) if not isinstance(result, str) else result

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [TOOL: {effective_tool}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ИНСТРУМЕНТ: {effective_tool}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result=result_str,
                result_data=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except asyncio.TimeoutError:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result="",
                success=False,
                error_message=f"Tool execution timed out after {config.timeout}s",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


def _stringify_mcp_result(result: Any) -> str:
    """Serialise an MCP ``call_tool`` result for the step history/result.

    A real server returns ``CallToolResult.content`` — a list of content
    blocks (``TextContent`` etc.) that are pydantic models, not JSON-native,
    so a bare ``json.dumps`` crashes the step. Extract each block's ``.text``
    (joining when there are several). Plain strings pass through; anything
    else (dict / list-of-dict — e.g. a mocked session or structured content)
    falls back to ``json.dumps`` with a ``default=str`` safety net, which is
    a no-op for already-serialisable values, so existing behaviour is
    unchanged.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list) and any(hasattr(item, "text") for item in result):
        return "\n".join(
            (
                getattr(item, "text", None)
                if getattr(item, "text", None) is not None
                else json.dumps(item, default=str)
            )
            for item in result
        )
    return json.dumps(result, default=str)


class MCPStepExecutor(StepExecutorBase):
    """
    Executor for MCP (Model Context Protocol) steps.

    Note: This is an EXPERIMENTAL feature. Full MCP support requires:
      - mcp-sdk library installed (pip install mcp)
      - Proper server connection handling
      - Stdio transport is partially implemented

    For production use, consider registering MCP tools as regular Python tools
    via context.register_tool() instead.
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute an MCP protocol step."""
        start_time = time.time()
        config: MCPStepConfig = step.step_config  # type: ignore

        try:
            # Build arguments
            arguments = config.arguments.copy()
            for arg_name, source in config.argument_mapping.items():
                arguments[arg_name] = self._resolve_input_value(source, context)

            # Try to use MCP SDK if available
            try:
                from mcp import ClientSession  # noqa: F401
                from mcp.client.stdio import stdio_client  # noqa: F401

                result = await self._execute_mcp_call(config, arguments)
            except ImportError:
                # MCP SDK not available - return error
                raise ImportError(
                    "MCP SDK not installed. Install with: pip install mcp"
                    "\nAlternatively, register the MCP tool as a regular tool."
                )

            # Convert result to string (a real server returns a list of
            # MCP content blocks — see `_stringify_mcp_result`).
            result_str = _stringify_mcp_result(result)

            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [MCP: {config.server.server_name}/{config.tool_name}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [MCP: {config.server.server_name}/{config.tool_name}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP,
                result=result_str,
                result_data=result,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    def _resolve_input_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve an input value from context."""
        return resolve_context_reference(source, context)

    async def _execute_mcp_call(self, config: MCPStepConfig, arguments: dict) -> Any:
        """Execute the actual MCP call. Requires mcp-sdk."""
        from mcp import ClientSession

        transport = config.server.transport

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            server_params = StdioServerParameters(
                command=config.server.command or "",
                args=config.server.args,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(config.tool_name, arguments=arguments),
                        timeout=config.timeout,
                    )
                    return result.content

        elif transport == "sse":
            # Legacy HTTP + SSE transport (MCP pre-2025-03-26 spec)
            from mcp.client.sse import sse_client

            url = config.server.url
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='sse'. "
                    "Set url to the SSE endpoint of the MCP server."
                )
            async with sse_client(
                url=url,
                headers=dict(config.server.headers) if config.server.headers else None,
                timeout=config.timeout or 30.0,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(config.tool_name, arguments=arguments),
                        timeout=config.timeout,
                    )
                    return result.content

        elif transport == "http":
            # Streamable HTTP transport (MCP 2025-03-26 spec)
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            url = config.server.url
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='http'. "
                    "Set url to the MCP server endpoint."
                )
            http_client = httpx.AsyncClient(
                headers=dict(config.server.headers) if config.server.headers else {},
                timeout=config.timeout or 30.0,
            )
            async with streamable_http_client(url=url, http_client=http_client) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(config.tool_name, arguments=arguments),
                        timeout=config.timeout,
                    )
                    return result.content

        else:
            raise NotImplementedError(
                f"MCP transport '{transport}' is not supported. "
                "Use one of: 'stdio', 'http', 'sse'."
            )


class MCPResourceStepExecutor(StepExecutorBase):
    """
    Executor for MCP resource-reading steps (``MCPResourceStepDescription``).

    Fetches a named resource from an MCP server via ``session.read_resource(uri)``
    — the read-only counterpart to ``MCPStepExecutor`` which calls tools.
    Content is written to ``memory[output_namespace][output_memory_key]`` (when
    set) and surfaced in history. Uses the same transport infrastructure
    (stdio / sse / streamable HTTP) as :class:`MCPStepExecutor`.
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import MCPResourceStepConfig

        start_time = time.time()
        config: MCPResourceStepConfig = step.step_config  # type: ignore

        try:
            try:
                from mcp import ClientSession  # noqa: F401
            except ImportError:
                raise ImportError(
                    "MCP SDK not installed. Install with: pip install mcp"
                )

            content = await self._read_resource(config)

            # Serialise the content for history / result
            if isinstance(content, str):
                content_str = content
            else:
                content_str = json.dumps(content, ensure_ascii=False, default=str)

            # Persist to memory if requested
            if config.output_memory_key:
                context.memory_write(
                    config.output_memory_key, content,
                    namespace=config.output_namespace,
                )

            if context.language == Language.ENGLISH:
                history_entry = (
                    f"Step {step.number}. {step.title} "
                    f"[MCP RESOURCE: {config.server.server_name}/{config.resource_uri}]\n"
                    f"Result: {content_str[:1000]}\n"
                )
            else:
                history_entry = (
                    f"Шаг {step.number}. {step.title} "
                    f"[MCP РЕСУРС: {config.server.server_name}/{config.resource_uri}]\n"
                    f"Результат: {content_str[:1000]}\n"
                )
            updated_history = context.history.copy()
            updated_history.append(history_entry)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP_RESOURCE,
                result=content_str,
                result_data=content,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as exc:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MCP_RESOURCE,
                result="",
                success=False,
                error_message=str(exc),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    async def _read_resource(self, config: Any) -> Any:
        """Read the resource. Mirrors ``MCPStepExecutor._execute_mcp_call``'s
        transport dispatch but calls ``session.read_resource(uri)`` instead of
        ``session.call_tool(...)``."""
        from mcp import ClientSession

        transport = config.server.transport

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            server_params = StdioServerParameters(
                command=config.server.command or "",
                args=config.server.args,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.read_resource(config.resource_uri),
                        timeout=config.timeout,
                    )
                    return self._extract_content(result)

        elif transport == "sse":
            from mcp.client.sse import sse_client

            url = config.server.url
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='sse'."
                )
            async with sse_client(
                url=url,
                headers=dict(config.server.headers) if config.server.headers else None,
                timeout=config.timeout or 30.0,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.read_resource(config.resource_uri),
                        timeout=config.timeout,
                    )
                    return self._extract_content(result)

        elif transport == "http":
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            url = config.server.url
            if not url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='http'."
                )
            http_client = httpx.AsyncClient(
                headers=dict(config.server.headers) if config.server.headers else {},
                timeout=config.timeout or 30.0,
            )
            async with streamable_http_client(url=url, http_client=http_client) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.read_resource(config.resource_uri),
                        timeout=config.timeout,
                    )
                    return self._extract_content(result)

        else:
            raise NotImplementedError(
                f"MCP transport '{transport}' is not supported. "
                "Use one of: 'stdio', 'http', 'sse'."
            )

    @staticmethod
    def _extract_content(result: Any) -> Any:
        """Pull the actual content out of ``ReadResourceResult``.

        MCP's ``read_resource`` returns a ``ReadResourceResult`` with a
        ``contents: list[ResourceContent]`` field. Each content item carries
        either ``text`` or ``blob`` plus metadata (uri, mimeType). For the
        common single-resource case we return just the inner text/blob; for
        multi-content responses we return the full list as-is so callers can
        inspect mime types etc.
        """
        contents = getattr(result, "contents", None)
        if contents is None:
            # Fallback: caller mocked content directly, return as-is
            return getattr(result, "content", result)
        if not contents:
            return ""
        # Single-content: return just the text/blob
        if len(contents) == 1:
            item = contents[0]
            text = getattr(item, "text", None)
            if text is not None:
                return text
            blob = getattr(item, "blob", None)
            if blob is not None:
                return blob
            return item
        # Multi-content: return the whole list
        return contents


class MemoryStepExecutor(StepExecutorBase):
    """Executor for memory read/write steps."""

    def _resolve_value(self, source: str, context: ReasoningContext) -> Any:
        """Resolve a value from context for write operations."""
        resolved = resolve_context_reference(source, context)
        return source if resolved is None and not source.startswith("$") else resolved

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a memory operation step."""
        start_time = time.time()
        config: MemoryStepConfig = step.step_config  # type: ignore

        try:
            result_data: Any = None
            result_str = ""

            if config.operation == MemoryOperation.READ:
                result_data = context.memory_read(config.memory_key, config.namespace, config.default_value)
                result_str = json.dumps(result_data) if not isinstance(result_data, str) else result_data

            elif config.operation == MemoryOperation.WRITE:
                if config.value_source:
                    value = self._resolve_value(config.value_source, context)
                else:
                    value = config.default_value
                context.memory_write(config.memory_key, value, config.namespace)
                result_str = f"Written to {config.namespace}.{config.memory_key}"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "value": value}

            elif config.operation == MemoryOperation.APPEND:
                if config.value_source:
                    value = self._resolve_value(config.value_source, context)
                else:
                    value = config.default_value
                context.memory_append(config.memory_key, value, config.namespace)
                result_str = f"Appended to {config.namespace}.{config.memory_key}"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "appended": value}

            elif config.operation == MemoryOperation.DELETE:
                existed = context.memory_delete(config.memory_key, config.namespace)
                result_str = f"Deleted {config.namespace}.{config.memory_key}" if existed else "Key not found"
                result_data = {"key": config.memory_key, "namespace": config.namespace, "existed": existed}

            elif config.operation == MemoryOperation.LIST:
                keys = context.memory_list(config.namespace)
                result_str = ", ".join(keys) if keys else "(empty)"
                result_data = keys

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [MEMORY: {config.operation}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ПАМЯТЬ: {config.operation}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MEMORY,
                result=result_str,
                result_data=result_data,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.MEMORY,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class TransformStepExecutor(StepExecutorBase):
    """Executor for data transformation steps (no LLM call)."""

    def _get_input(self, input_key: str, context: ReasoningContext) -> Any:
        """Get input data based on input_key."""
        resolved = resolve_context_reference(input_key, context)
        return "" if resolved is None else resolved

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute a data transformation step."""
        start_time = time.time()
        config: TransformStepConfig = step.step_config  # type: ignore

        try:
            input_data = self._get_input(config.input_key, context)
            result_data: Any = None

            if config.transform_type == "extract":
                # Extract using regex expression
                if config.expression:
                    matches = re.findall(config.expression, str(input_data))
                    result_data = matches
                else:
                    result_data = input_data

            elif config.transform_type == "format":
                # Format using output_format template
                if config.output_format:
                    result_data = config.output_format.format(input=input_data)
                else:
                    result_data = str(input_data)

            elif config.transform_type == "aggregate":
                # Simple aggregation - join if list
                if isinstance(input_data, list):
                    result_data = "\n".join(str(item) for item in input_data)
                else:
                    result_data = str(input_data)

            elif config.transform_type == "filter":
                # Filter using expression as regex
                if isinstance(input_data, list) and config.expression:
                    result_data = [item for item in input_data if re.search(config.expression, str(item))]
                elif isinstance(input_data, str) and config.expression:
                    lines = input_data.split("\n")
                    result_data = [line for line in lines if re.search(config.expression, line)]
                else:
                    result_data = input_data

            elif config.transform_type == "map":
                # Apply template to each item
                if isinstance(input_data, list) and config.map_template:
                    result_data = [config.map_template.format(item=item) for item in input_data]
                else:
                    result_data = input_data

            result_str = json.dumps(result_data) if not isinstance(result_data, str) else result_data

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [TRANSFORM: {config.transform_type}]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [ТРАНСФОРМАЦИЯ: {config.transform_type}]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TRANSFORM,
                result=result_str,
                result_data=result_data,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TRANSFORM,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


class ConditionalStepExecutor(StepExecutorBase):
    """
    Executor for conditional branching steps.

    Evaluates conditions and returns the next_step number in result_data. The
    DAGExecutor reads this value after each batch and calls
    ``_skip_conditional_branches()`` to mark non-chosen branch steps as skipped.

    For routing to take effect, branch steps MUST declare a dependency on the
    conditional step (``dependencies=[<conditional_step_number>]``), so that they
    only become ready AFTER the routing decision has been made.

    The executor evaluates conditions using simpleeval for safety.
    """

    def __init__(self):
        """Initialize with safe expression evaluator."""
        self._evaluator = EvalWithCompoundTypes()
        # Only allow safe functions - no code execution
        self._evaluator.functions = {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'bool': bool,
        }

    def _evaluate_condition(self, condition: str, value: Any) -> bool:
        """
        Evaluate a condition against a value safely.

        Supports simple conditions like:
        - "contains:keyword" - checks if value contains keyword
        - "equals:value" - checks equality
        - "startswith:prefix" - checks prefix
        - "endswith:suffix" - checks suffix
        - "matches:regex" - regex match
        - "empty" - checks if empty
        - "nonempty" - checks if not empty

        For complex expressions (requires simpleeval):
        - "len(value) > 5"
        - "value != 'skip'"
        - "int(value) >= 10"
        """
        value_str = str(value)

        # Built-in condition patterns (always supported, no eval)
        if condition.startswith("contains:"):
            return condition[9:] in value_str
        elif condition.startswith("equals:"):
            return value_str == condition[7:]
        elif condition.startswith("startswith:"):
            return value_str.startswith(condition[11:])
        elif condition.startswith("endswith:"):
            return value_str.endswith(condition[9:])
        elif condition.startswith("matches:"):
            return bool(re.search(condition[8:], value_str))
        elif condition == "empty":
            return not value_str.strip()
        elif condition == "nonempty":
            return bool(value_str.strip())

        # Complex expressions evaluated safely via simpleeval
        try:
            self._evaluator.names = {"value": value, "v": value}
            result = self._evaluator.eval(condition)
            return bool(result)
        except Exception:
            return False

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """
        Execute a conditional step.

        Returns the next step number in result_data.
        """
        start_time = time.time()
        config: ConditionalStepConfig = step.step_config  # type: ignore

        try:
            # Get the value to evaluate
            value = self._get_condition_value(config.condition_context_key, context)

            # Evaluate branches
            next_step: Optional[int] = None
            matched_condition = ""

            for branch in config.branches:
                if self._evaluate_condition(branch.condition, value):
                    next_step = branch.next_step
                    matched_condition = branch.condition
                    break

            if next_step is None:
                next_step = config.default_step
                matched_condition = "(default)"

            result_str = f"Condition matched: {matched_condition}, next step: {next_step}"

            # Format history entry
            if context.language == Language.ENGLISH:
                step_result = f"Step {step.number}. {step.title} [CONDITIONAL]\nResult: {result_str}\n"
            else:
                step_result = f"Шаг {step.number}. {step.title} [УСЛОВИЕ]\nРезультат: {result_str}\n"

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.CONDITIONAL,
                result=result_str,
                result_data={"next_step": next_step, "matched_condition": matched_condition},
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.CONDITIONAL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    def _get_condition_value(self, key: str, context: ReasoningContext) -> Any:
        """Get the value to evaluate conditions against."""
        resolved = resolve_context_reference(key, context)
        return "" if resolved is None else resolved


class StructuredOutputStepExecutor(StepExecutorBase):
    """Executor for structured output generation with schema validation."""

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        start_time = time.time()
        config: StructuredOutputStepConfig = step.step_config  # type: ignore

        try:
            input_data = resolve_context_reference(config.input_source, context)
            serialized_input = input_data if isinstance(input_data, str) else json.dumps(input_data, ensure_ascii=False)

            schema_json = json.dumps(config.output_schema, indent=2, ensure_ascii=False)
            strict_instruction = "Return ONLY valid JSON with no markdown or additional text." if config.strict_json else "Return valid JSON."
            full_prompt = (
                f"You are a structured output generator.\n"
                f"Task: {config.instruction or step.title}\n"
                f"Schema name: {config.schema_name}\n"
                f"JSON Schema:\n{schema_json}\n\n"
                f"Input:\n{serialized_input}\n\n"
                f"{strict_instruction}"
            )

            llm_config = getattr(step, "llm_config", None)
            llm_client = context.get_llm_client_for_step(llm_config)

            # Get retry count (per-step override or context default)
            step_retries = getattr(step, "retry_max", None)
            retries = step_retries if step_retries is not None else context.retry_max

            # Resolve model name for LangFuse generation tracing
            model_name = None
            if hasattr(llm_client, "config") and hasattr(llm_client.config, "model"):
                model_name = llm_client.config.model

            # Create a LangFuse generation observation inside the step span
            parent_span = context.metadata.get("__langfuse_span")
            generation = None
            if parent_span is not None:
                gen_kwargs: dict[str, Any] = {"name": "llm_generation", "input": full_prompt}
                if model_name:
                    gen_kwargs["model"] = model_name
                generation = parent_span.start_observation(**gen_kwargs, as_type="generation")

            # Streaming gate: when the
            # client exposes ``stream_response`` and the
            # context has an ``on_llm_chunk`` callback wired,
            # stream tokens through to the UI + watch for an
            # early-complete balanced JSON object so we can
            # parse + return as soon as the structured payload
            # arrives — without waiting for trailing tokens
            # (which the model occasionally hallucinates after
            # the closing ``}``).
            can_stream = (
                llm_client.supports_streaming
                if isinstance(llm_client, LLMClientBase)
                else hasattr(llm_client, "stream_response")
            )
            if context.on_llm_chunk and can_stream:
                raw_result = await self._execute_structured_streaming(
                    llm_client,
                    full_prompt,
                    retries=retries,
                    on_chunk=context.on_llm_chunk,
                    step_number=step.number,
                )
            else:
                raw_result = await llm_client.get_response_with_retries(
                    full_prompt, retries=retries
                )

            if generation is not None:
                generation.update(output=raw_result)
                generation.end()

            parsed = _extract_json_payload(raw_result)

            if context.language == Language.ENGLISH:
                step_result = (
                    f"Step {step.number}. {step.title} [STRUCTURED_OUTPUT]\n"
                    f"Result: {json.dumps(parsed, ensure_ascii=False)}\n"
                )
            else:
                step_result = (
                    f"Шаг {step.number}. {step.title} [СТРУКТУРИРОВАННЫЙ ВЫВОД]\n"
                    f"Результат: {json.dumps(parsed, ensure_ascii=False)}\n"
                )

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.STRUCTURED_OUTPUT,
                result=json.dumps(parsed, ensure_ascii=False),
                result_data=parsed,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.STRUCTURED_OUTPUT,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    async def _execute_structured_streaming(
        self,
        llm_client: Any,
        prompt: str,
        *,
        retries: int,
        on_chunk: Callable[..., None],
        step_number: int | None = None,
    ) -> str:
        """Streaming variant of ``get_response_with_retries`` for
        structured output.

        Forwards every chunk to ``on_chunk`` via
        :func:`_dispatch_llm_chunk` (same callback shape the
        LLM step executor uses) AND scans the running buffer
        for an early-complete balanced JSON object. The moment
        a balanced ``{...}`` parses cleanly via
        :func:`_extract_json_payload`, the helper closes the
        stream and returns — saves the latency of any trailing
        hallucinated tokens the model emits past the closing
        brace. Falls back to the full accumulated buffer when
        no balanced object materialises (the final-parse path
        in the caller still runs).
        """
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                buffer = ""
                early_result: str | None = None
                async for chunk in llm_client.stream_response(prompt):
                    buffer += chunk
                    _dispatch_llm_chunk(
                        on_chunk, chunk,
                        step_number=step_number, stage="structured_output",
                    )
                    if early_result is None and _looks_complete_json(buffer):
                        try:
                            _extract_json_payload(buffer)
                            early_result = buffer
                        except Exception:  # noqa: BLE001
                            # Brace-balance was a false signal —
                            # keep accumulating.
                            pass
                return early_result or buffer
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)

        raise last_error or Exception("All streaming retries failed")


class AgentSkillStepExecutor(StepExecutorBase):
    """
    Executor for AgentSkill steps.

    Supports three execution modes:
    - LLM: Loads SKILL.md instructions as system prompt, calls LLM for the task.
    - SCRIPT: Runs a bundled script directly without an LLM call.
    - HYBRID: Runs script first, uses LLM as fallback or for post-processing.
    """

    def _get_loader(self, config: AgentSkillStepConfig) -> SkillLoader:
        """Create a SkillLoader from config settings."""
        from .models.agent_skill import AgentSkillSource

        extra_paths: list[str] = []
        if isinstance(config.skill, AgentSkillSource):
            extra_paths = list(config.skill.search_paths)
        return SkillLoader(
            extra_search_paths=extra_paths,
            cache_dir=config.skill_cache_dir,
            enable_cache=config.cache_skills,
        )

    def _resolve_inputs(
        self, config: AgentSkillStepConfig, context: ReasoningContext
    ) -> dict[str, str]:
        """Resolve all input_mapping values from context."""
        resolved: dict[str, str] = {}
        for key, ref in config.input_mapping.items():
            value = resolve_context_reference(ref, context)
            resolved[key] = str(value) if value is not None else ""
        return resolved

    def _interpolate_task(self, task: str, resolved_inputs: dict[str, str]) -> str:
        """Substitute {key} placeholders in the task string with resolved values."""
        try:
            return task.format_map(resolved_inputs)
        except KeyError:
            # If some keys are missing, return as-is
            return task

    # Regex matching security-sensitive terms that trigger provider ToS filters
    _SECURITY_TERMS_RE = re.compile(
        r'\b(password|encrypt(?:ion|ing|ed)?|decrypt(?:ion|ing|ed)?|crypt(?:ograph(?:y|ic))?)\b',
        re.IGNORECASE,
    )

    @staticmethod
    def _filter_security_sections(text: str) -> str:
        """
        Remove markdown sections whose headings contain security-sensitive terms.

        Scans for headings (# / ## / ###…) that mention password/encrypt/decrypt
        and skips all lines until the next heading at the same or higher level.
        Also strips inline list items (comma-separated) containing those terms.
        """
        lines = text.split('\n')
        result: list[str] = []
        skip_section = False
        skip_level = 0

        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.+)', line)
            if m:
                level = len(m.group(1))
                heading_text = m.group(2)
                if AgentSkillStepExecutor._SECURITY_TERMS_RE.search(heading_text):
                    # Start skipping this section
                    skip_section = True
                    skip_level = level
                    continue
                elif skip_section and level <= skip_level:
                    # Returned to same or higher level — stop skipping
                    skip_section = False

            if not skip_section:
                result.append(line)

        return '\n'.join(result)

    @staticmethod
    def _sanitize_description(description: str) -> str:
        """
        Remove comma-list items or inline phrases mentioning security terms from
        the skill description so they don't appear in the system prompt sent to
        the LLM.
        """
        # Split on commas, filter out items with security terms, rejoin
        parts = [p for p in description.split(',')
                 if not AgentSkillStepExecutor._SECURITY_TERMS_RE.search(p)]
        return ','.join(parts)

    @staticmethod
    def _clean_instructions(
        instructions: str,
        strip_code_blocks: bool,
        max_chars: Optional[int],
        filter_security_terms: bool = True,
    ) -> str:
        """
        Optionally strip fenced code blocks, filter security-sensitive sections,
        and truncate skill instructions.

        Skill SKILL.md files can be 8-9KB+ with many code examples. Injecting
        them verbatim can trigger provider ToS filters (e.g. password-decryption
        examples in the PDF skill). Stripping code blocks and truncating keeps
        the instructional prose while reducing prompt size.
        """
        if strip_code_blocks:
            # Remove fenced code blocks (``` ... ```) including language tags
            instructions = re.sub(r'```[^\n]*\n.*?```', '', instructions, flags=re.DOTALL)
            # Collapse multiple blank lines left by removal
            instructions = re.sub(r'\n{3,}', '\n\n', instructions)
            instructions = instructions.strip()

        if filter_security_terms:
            instructions = AgentSkillStepExecutor._filter_security_sections(instructions)
            instructions = re.sub(r'\n{3,}', '\n\n', instructions).strip()

        if max_chars is not None and len(instructions) > max_chars:
            instructions = instructions[:max_chars].rstrip() + "\n\n[...instructions truncated...]"

        return instructions

    def _build_llm_system_prompt(
        self,
        manifest: "SkillManifest",
        config: AgentSkillStepConfig,
    ) -> str:
        """Build the system prompt for LLM mode."""
        parts: list[str] = []

        if config.system_prompt_prefix:
            parts.append(config.system_prompt_prefix.strip())
            parts.append("")

        parts.append(f"## Skill: {manifest.name}")

        if config.include_skill_description and manifest.description:
            description = manifest.description
            if config.filter_security_terms:
                description = self._sanitize_description(description)
            parts.append(description)
            parts.append("")

        # Surface allowed-tools constraint so the LLM respects it
        if manifest.restricts_tools():
            tools = manifest.get_allowed_tools()
            parts.append(f"**Allowed tools**: {', '.join(tools)}")
            parts.append("")

        if manifest.instructions:
            cleaned = self._clean_instructions(
                manifest.instructions,
                strip_code_blocks=config.strip_code_blocks,
                max_chars=config.max_instructions_chars,
                filter_security_terms=config.filter_security_terms,
            )
            parts.append(cleaned)

        return "\n".join(parts).strip()

    async def _execute_llm_mode(
        self,
        step: Any,
        config: AgentSkillStepConfig,
        context: ReasoningContext,
        manifest: "SkillManifest",
        resolved_inputs: dict[str, str],
        user_message_override: Optional[str] = None,
    ) -> str:
        """Execute in LLM mode: inject SKILL.md as system prompt, call LLM.

        Uses get_response_with_system() so the skill instructions go into the
        system role (not concatenated into the user message). This avoids
        triggering provider ToS filters caused by large user-role messages
        containing code examples.

        Args:
            user_message_override: If provided, replaces the default interpolated task
                string. Used by SUBAGENT mode to inject script outputs into the user turn.
        """
        system_prompt = self._build_llm_system_prompt(manifest, config)
        user_message = user_message_override or self._interpolate_task(config.task, resolved_inputs)

        llm_client = context.get_llm_client_for_step(config.llm_config)
        retries = context.retry_max

        result = await llm_client.get_response_with_system(
            system_prompt=system_prompt,
            user_prompt=user_message,
            retries=retries,
        )
        return result

    async def _execute_subagent_mode(
        self,
        step: Any,
        config: AgentSkillStepConfig,
        context: ReasoningContext,
        manifest: "SkillManifest",
        resolved_inputs: dict[str, str],
    ) -> tuple[str, dict]:
        """Execute in SUBAGENT mode: script provides raw data, LLM synthesizes.

        Unlike HYBRID (which is script-OR-LLM), SUBAGENT always does both:
          1. Run the skill script to get concrete data (e.g. extracted PDF text)
          2. Pass that data to the LLM as context so it can reason about it

        If the script fails or doesn't exist, falls back to LLM-only with
        an error note in the user message. This mirrors the AgentSkills spec
        "subagent delegation" pattern where concrete tool execution feeds a
        higher-level reasoning step.
        """
        metadata: dict[str, Any] = {"execution_mode": "subagent"}
        script_output = ""
        script_error = ""

        # Step 1: try to get concrete data from a script
        try:
            stdout, stderr, returncode = await self._execute_script_mode(
                config, manifest, resolved_inputs
            )
            if stdout.strip():
                script_output = stdout.strip()
                metadata["script_used"] = True
                metadata["script_chars"] = len(script_output)
            if stderr.strip():
                script_error = stderr.strip()
            if returncode != 0:
                metadata["script_returncode"] = returncode
        except Exception as err:
            script_error = str(err)
            metadata["script_error"] = script_error

        # Step 2: build enriched user message
        base_task = self._interpolate_task(config.task, resolved_inputs)

        if script_output:
            user_message = (
                f"{base_task}\n\n"
                f"## Data from skill script\n\n"
                f"{script_output}"
            )
        elif script_error:
            user_message = (
                f"{base_task}\n\n"
                f"## Note\n\nThe skill script could not run ({script_error}). "
                f"Please complete the task based on the skill instructions alone."
            )
            metadata["llm_fallback"] = True
        else:
            user_message = base_task

        # Step 3: LLM synthesizes using skill knowledge + script data
        result = await self._execute_llm_mode(
            step, config, context, manifest, resolved_inputs,
            user_message_override=user_message,
        )
        return result, metadata

    # ------------------------------------------------------------------ #
    #  Workspace helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _create_workspace() -> tuple[Path, Path, Path]:
        """
        Create a per-execution workspace with in/ and out/ subdirs.

        Returns ``(workspace_root, workspace_in, workspace_out)``.
        The caller is responsible for cleanup (``shutil.rmtree``).
        """
        workspace_root = Path(tempfile.mkdtemp(prefix="carl_skill_"))
        workspace_in = workspace_root / "in"
        workspace_out = workspace_root / "out"
        workspace_in.mkdir()
        workspace_out.mkdir()
        return workspace_root, workspace_in, workspace_out

    @staticmethod
    def _stage_input_files(
        resolved_inputs: dict[str, str],
        workspace_in: Path,
    ) -> dict[str, str]:
        """
        Copy input files from resolved_inputs into workspace_in.

        For each value that resolves to an existing file path, copies the file
        into workspace_in/<key> and returns the updated mapping with workspace paths.
        Non-file values are passed through unchanged.

        Returns updated resolved_inputs with workspace paths substituted.
        """
        updated: dict[str, str] = {}
        for key, value in resolved_inputs.items():
            if value and os.path.isfile(value):
                dest = workspace_in / os.path.basename(value)
                shutil.copy2(value, dest)
                updated[key] = str(dest)
            else:
                updated[key] = value
        return updated

    @staticmethod
    def _collect_output_files(
        workspace_out: Path,
        glob_patterns: list[str],
    ) -> list[dict[str, str]]:
        """
        Collect files from workspace_out matching the given glob patterns.

        Returns a list of ``{"path": str, "name": str, "size": int}`` dicts.
        """
        results: list[dict[str, str]] = []
        if not workspace_out.is_dir():
            return results
        for f in workspace_out.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(workspace_out)
            rel_str = str(rel)
            if any(fnmatch.fnmatch(rel_str, pat) for pat in glob_patterns):
                results.append({
                    "path": str(f),
                    "name": rel_str,
                    "size": str(f.stat().st_size),
                })
        return results

    # ------------------------------------------------------------------ #
    #  LLM_AGENT tool definitions and execution                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _llm_agent_tool_definitions() -> list[dict[str, Any]]:
        """Return the OpenAI-style tool definitions for LLM_AGENT mode."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_script",
                    "description": (
                        "Run a Python (.py) or shell (.sh) script from the skill's "
                        "scripts/ directory. Use this to execute skill functionality."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script_path": {
                                "type": "string",
                                "description": (
                                    "Relative path within the skill directory, "
                                    "e.g. 'scripts/extract_text.py'"
                                ),
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Positional arguments for the script. "
                                    "Use workspace paths like '/workspace/in/<file>' "
                                    "and '/workspace/out/<file>' for I/O."
                                ),
                            },
                        },
                        "required": ["script_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Read a text file from the workspace or skill directory. "
                        "Returns the file contents as a string."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "File path. Use workspace-relative paths like "
                                    "'/workspace/in/file.txt' or '/workspace/out/result.json', "
                                    "or skill-relative like 'references/REFERENCE.md'."
                                ),
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": (
                        "Write content to a file in the output workspace (/workspace/out/). "
                        "Use this to save results, reports, or generated files."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": (
                                    "Filename to create in /workspace/out/ "
                                    "(basename only, no directory traversal)."
                                ),
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file.",
                            },
                        },
                        "required": ["filename", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_resources",
                    "description": (
                        "List all available scripts, reference files, and assets in the skill. "
                        "Call this first to discover what tools the skill provides."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_resource",
                    "description": (
                        "Read a reference or asset file from the skill directory "
                        "(e.g. 'references/FORMS.md', 'assets/template.pptx'). "
                        "Use this to load skill documentation or templates."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resource_path": {
                                "type": "string",
                                "description": "Relative path within the skill directory.",
                            }
                        },
                        "required": ["resource_path"],
                    },
                },
            },
        ]

    async def _execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        manifest: "SkillManifest",
        workspace_in: Path,
        workspace_out: Path,
        config: AgentSkillStepConfig,
        pip_overlay: Optional[Path] = None,
    ) -> str:
        """Execute a single tool call from the LLM_AGENT loop."""

        if tool_name == "list_resources":
            lines = ["**Skill resources:**"]
            if manifest.scripts:
                lines.append("Scripts: " + ", ".join(manifest.scripts))
            if manifest.references:
                lines.append("References: " + ", ".join(manifest.references))
            if manifest.assets:
                lines.append("Assets: " + ", ".join(manifest.assets))
            if not (manifest.scripts or manifest.references or manifest.assets):
                lines.append("(no bundled resources)")
            return "\n".join(lines)

        if tool_name == "read_resource":
            resource_path = arguments.get("resource_path", "")
            resource_path = resource_path.lstrip("/")
            abs_path = os.path.join(manifest.skill_dir, resource_path)
            if not os.path.isfile(abs_path):
                return f"[Error: resource not found: {resource_path}]"
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                if len(content) > 8000:
                    content = content[:8000] + "\n[...truncated...]"
                return content
            except Exception as exc:
                return f"[Error reading resource: {exc}]"

        if tool_name == "read_file":
            path_str = arguments.get("path", "")
            # Resolve /workspace/in and /workspace/out to actual temp paths
            path_str = path_str.replace("/workspace/in", str(workspace_in))
            path_str = path_str.replace("/workspace/out", str(workspace_out))
            # Skill-relative paths
            if not os.path.isabs(path_str):
                path_str = os.path.join(manifest.skill_dir, path_str)
            if not os.path.isfile(path_str):
                return f"[Error: file not found: {path_str}]"
            try:
                with open(path_str, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                if len(content) > 10000:
                    content = content[:10000] + "\n[...truncated...]"
                return content
            except Exception as exc:
                return f"[Error reading file: {exc}]"

        if tool_name == "write_file":
            filename = arguments.get("filename", "output.txt")
            # Prevent directory traversal
            filename = os.path.basename(filename)
            content = arguments.get("content", "")
            dest = workspace_out / filename
            try:
                dest.write_text(content, encoding="utf-8")
                return f"Written {len(content)} bytes to /workspace/out/{filename}"
            except Exception as exc:
                return f"[Error writing file: {exc}]"

        if tool_name == "run_script":
            script_path = arguments.get("script_path", "")
            args_list = arguments.get("args", [])

            # Resolve /workspace paths in args
            resolved_args = [
                a.replace("/workspace/in", str(workspace_in))
                 .replace("/workspace/out", str(workspace_out))
                for a in args_list
            ]

            # Build absolute script path
            if not os.path.isabs(script_path):
                abs_script = os.path.join(manifest.skill_dir, script_path)
            else:
                abs_script = script_path

            if not os.path.isfile(abs_script):
                return f"[Error: script not found: {script_path}]"

            if abs_script.endswith(".py"):
                cmd = [config.python_executable, abs_script] + resolved_args
            else:
                cmd = ["/bin/sh", abs_script] + resolved_args

            # route through the resolved skill runtime
            # so the LLM_AGENT's tool-call execution is sandboxed when
            # the chain uses `runtime="docker"` etc.
            stdout, stderr, rc = await self._runtime_run(
                cmd,
                cwd=config.working_dir or manifest.skill_dir,
                env=self._build_subprocess_env(pip_overlay),
                timeout=config.timeout,
            )
            if rc == 124 and stderr.startswith("[timeout after "):
                return f"[Error: script timed out after {config.timeout}s]"

            output = stdout
            if stderr.strip():
                output += f"\n[stderr]: {stderr.strip()}"
            if rc != 0:
                output = f"[exit code {rc}] " + output
            return output or "(empty output)"

        return f"[Unknown tool: {tool_name}]"

    async def _execute_llm_agent_mode(
        self,
        step: Any,
        config: AgentSkillStepConfig,
        context: ReasoningContext,
        manifest: "SkillManifest",
        resolved_inputs: dict[str, str],
        pip_overlay: Optional[Path] = None,
    ) -> tuple[str, list[dict[str, str]], dict[str, Any], Optional[Path]]:
        """
        Execute in LLM_AGENT mode: iterative tool-calling loop.

        The LLM receives SKILL.md as system prompt and a constrained tool surface.
        It can call ``run_script``, ``read_file``, ``write_file``, ``list_resources``,
        and ``read_resource`` until it produces a final response.

        Returns:
            (result_text, output_files, metadata, persisted_workspace_root)
            ``persisted_workspace_root`` is non-None only when
            ``config.persist_workspace=True``; the caller is responsible for
            communicating the path to downstream steps via memory.
        """
        metadata: dict[str, Any] = {"execution_mode": "llm_agent"}

        workspace_root, workspace_in, workspace_out = self._create_workspace()
        persisted_workspace: Optional[Path] = None

        try:
            # Stage input files into workspace/in
            staged_inputs = self._stage_input_files(resolved_inputs, workspace_in)

            # Build system prompt
            system_prompt = self._build_llm_system_prompt(manifest, config)
            system_prompt += (
                f"\n\n## Workspace\n"
                f"Input files are in `/workspace/in/`. "
                f"Write output files to `/workspace/out/`.\n"
                f"Actual paths: in={workspace_in}, out={workspace_out}"
            )

            # Build initial user message
            base_task = self._interpolate_task(config.task, staged_inputs)
            input_summary = "\n".join(
                f"  {k}: {v}" for k, v in staged_inputs.items() if v
            )
            if input_summary:
                user_message = f"{base_task}\n\n**Inputs:**\n{input_summary}"
            else:
                user_message = base_task

            # Append structured-output schema instruction if configured
            if config.output_schema is not None:
                schema_json = json.dumps(config.output_schema, ensure_ascii=False, indent=2)
                user_message += (
                    "\n\n**Output requirement:** "
                    "When you are done, your final reply (the one *without* tool calls) "
                    "MUST be a single valid JSON value matching this schema. "
                    "Do not wrap it in Markdown code fences:\n\n"
                    f"{schema_json}"
                )

            tools = self._llm_agent_tool_definitions()
            llm_client = context.get_llm_client_for_step(config.llm_config)

            # Multi-turn conversation history (list of message dicts)
            history: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            final_response = ""
            iterations = 0
            tool_calls_made = 0
            content: str = ""
            tool_calls: list[dict[str, Any]] = []

            for iteration in range(config.llm_max_iterations):
                iterations = iteration + 1

                # cancellation poll at the top of every
                # tool-call iteration. Each iteration kicks off an LLM
                # call plus tool dispatch; without this check a long
                # LLM_AGENT run could ignore a user cancel for many
                # seconds. ``llm_max_iterations`` was the only bound.
                if context.is_cancelled():
                    raise _StepCancelled(step.number)

                content, tool_calls = await llm_client.get_response_with_tools(
                    system_prompt=system_prompt,
                    user_prompt=user_message,
                    tools=tools,
                    messages=history,
                )

                if not tool_calls:
                    # LLM produced a final answer — done
                    final_response = content
                    break

                # Append assistant message with tool calls
                history.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # emit one "tool_call" event per
                # LLM-issued tool call before dispatch, so CARE's TUI
                # can render the call (e.g. "read_file(in/a.pdf)")
                # before the result arrives.
                for tc in tool_calls:
                    context.emit_step_event(
                        step.number, "llm_agent.tool_call",
                        {"tool": tc.get("name"), "args": tc.get("arguments")},
                    )

                # Execute all tool calls concurrently, preserve order for history
                tool_calls_made += len(tool_calls)
                tool_results = await asyncio.gather(*[
                    self._execute_tool_call(
                        tc["name"],
                        tc["arguments"],
                        manifest,
                        workspace_in,
                        workspace_out,
                        config,
                        pip_overlay=pip_overlay,
                    )
                    for tc in tool_calls
                ])
                # emit "tool_result" per call. Paired
                # by index with the tool_call events above so consumers
                # can correlate.
                for tc, tool_result in zip(tool_calls, tool_results):
                    context.emit_step_event(
                        step.number, "llm_agent.tool_result",
                        {"tool": tc.get("name"), "result": tool_result},
                    )
                for tc, tool_result in zip(tool_calls, tool_results):
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            else:
                # Hit max iterations — use last content as result
                final_response = content or (
                    f"[LLM_AGENT: reached max iterations ({config.llm_max_iterations}). "
                    f"Last tool calls: {[tc['name'] for tc in tool_calls]}]"
                )

            metadata["iterations"] = iterations
            metadata["tool_calls_made"] = tool_calls_made

            # Structured output validation
            if config.output_schema is not None:
                from .skill_output_schema import (
                    SkillOutputSchemaError,
                    parse_and_validate_skill_output,
                )
                try:
                    parsed = parse_and_validate_skill_output(
                        final_response, config.output_schema
                    )
                    metadata["parsed_output"] = parsed
                    metadata["schema_validated"] = True
                except SkillOutputSchemaError as exc:
                    metadata["schema_validated"] = False
                    metadata["schema_warnings"] = [str(exc)]
                    if config.output_schema_strict:
                        # Re-raise so the caller marks the step failed
                        raise

            # Collect output files
            output_files = self._collect_output_files(workspace_out, config.output_files_glob)
            if output_files:
                metadata["output_files"] = output_files

            # Persist workspace if requested (caller writes path to memory)
            if config.persist_workspace:
                persisted_workspace = workspace_root
                metadata["workspace_root"] = str(workspace_root)

            return final_response, output_files, metadata, persisted_workspace

        finally:
            if not config.persist_workspace:
                shutil.rmtree(workspace_root, ignore_errors=True)

    # ------------------------------------------------------------------ #
    #  Script mode (enhanced with workspace)                              #
    # ------------------------------------------------------------------ #

    async def _install_extra_pip(
        self,
        packages: list[str],
        target_dir: Path,
    ) -> None:
        """
        Install ``packages`` into ``target_dir`` using pip.

        Runs ``pip install --target <target_dir> <packages...>`` via the
        resolved skill runtime so a Docker / E2B backend can run it
        inside the sandbox instead of on the host.
        The target directory can be prepended to ``PYTHONPATH`` when
        launching skill scripts so installed packages are importable
        without modifying the host environment.

        Raises:
            RuntimeError: if pip exits with a non-zero return code.
        """
        import sys

        cmd = [sys.executable, "-m", "pip", "install", "--target", str(target_dir)] + packages
        _, stderr_text, returncode = await self._runtime_run(cmd)
        if returncode:
            raise RuntimeError(
                f"pip install failed (exit {returncode}) for {packages}:\n"
                f"{stderr_text.strip()}"
            )

    @staticmethod
    def _build_subprocess_env(pip_overlay: Optional[Path]) -> Optional[dict[str, str]]:
        """
        Return a subprocess env dict with pip overlay prepended to PYTHONPATH.

        Returns ``None`` when ``pip_overlay`` is None (subprocess inherits env).
        """
        if pip_overlay is None:
            return None
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(pip_overlay) + (":" + existing if existing else "")
        return env

    async def _runtime_run(
        self,
        cmd: list[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> tuple[str, str, int]:
        """route a subprocess call through the resolved
        skill runtime.

        Replaces the prior direct ``asyncio.create_subprocess_exec``
        sites in :meth:`_install_extra_pip`, :meth:`_execute_script_mode`,
        and the LLM_AGENT ``run_script`` tool. Each call now goes
        through ``self._skill_runtime.run`` so a Docker / E2B backend
        can intercept all skill subprocess execution by simply being
        registered as the ``runtime`` — no further executor changes
        needed.

        The runtime contract takes a :class:`SkillRuntimeHandle`. For
        the SCRIPT-mode path (and ``run_script`` from LLM_AGENT) we
        construct an ephemeral handle whose workspace fields point at
        ``cwd``; ``LocalSkillRuntime`` ignores those fields, while
        future container backends use them to mount the skill bundle.

        Returns ``(stdout_str, stderr_str, exit_code)`` for parity
        with the prior helper signatures. Bytes are decoded as UTF-8
        with replacement (matches the pre-refactor behaviour).
        """
        from .skill_runtime import LocalSkillRuntime, SkillRuntimeHandle  # noqa: PLC0415

        # Lazy init: tests that call _execute_script_mode directly (without
        # going through execute()) won't have set _skill_runtime — default
        # to LocalSkillRuntime to preserve the legacy host-subprocess
        # behaviour in that path.
        runtime = getattr(self, "_skill_runtime", None) or LocalSkillRuntime()
        self._skill_runtime = runtime

        # The handle carries cwd as workspace_root so backends that
        # need a working directory have one. Local backend ignores it.
        ws_path = Path(cwd) if cwd is not None else Path(tempfile.gettempdir())
        handle = SkillRuntimeHandle(
            workspace_root=ws_path,
            workspace_in=ws_path,
            workspace_out=ws_path,
            backend={
                "isolation": "ephemeral",
                "network_policy": getattr(self, "_network_policy", "none"),
                "network_allowlist": getattr(self, "_network_allowlist", []),
            },
        )
        result = await runtime.run(
            handle, cmd,
            env=env, timeout=timeout, cwd=cwd,
        )
        return (
            result.stdout.decode("utf-8", errors="replace"),
            result.stderr.decode("utf-8", errors="replace"),
            result.exit_code,
        )

    async def _execute_script_mode(
        self,
        config: AgentSkillStepConfig,
        manifest: "SkillManifest",
        resolved_inputs: dict[str, str],
        pip_overlay: Optional[Path] = None,
    ) -> tuple[str, str, int]:
        """
        Execute script mode. Returns (stdout, stderr, return_code).
        """
        # Determine script path
        if config.script_name:
            script_rel = config.script_name
        else:
            # Auto-detect: find exactly one .py or .sh in manifest.scripts
            py_scripts = [s for s in manifest.scripts if s.endswith(".py")]
            sh_scripts = [s for s in manifest.scripts if s.endswith(".sh")]
            all_scripts = py_scripts + sh_scripts
            if not all_scripts:
                raise ValueError(
                    f"Skill '{manifest.name}' has no scripts, and script_name was not specified"
                )
            if len(all_scripts) > 1:
                raise ValueError(
                    f"Skill '{manifest.name}' has multiple scripts ({all_scripts}). "
                    f"Specify script_name in AgentSkillStepConfig to disambiguate."
                )
            script_rel = all_scripts[0]

        script_path = os.path.join(manifest.skill_dir, script_rel)
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        working_dir = config.working_dir or manifest.skill_dir

        # Build CLI arguments: --key value for each resolved input + script_args
        cli_args: list[str] = []
        all_args = {**resolved_inputs, **config.script_args}
        for k, v in all_args.items():
            cli_args.extend([f"--{k}", v])

        if script_rel.endswith(".py"):
            cmd = [config.python_executable, script_path] + cli_args
        else:
            cmd = ["/bin/sh", script_path] + cli_args

        # route through the resolved skill runtime so
        # Docker / E2B backends can intercept all skill subprocess
        # execution. ``LocalSkillRuntime.run`` already surfaces a
        # timeout as exit code 124 + a stderr marker, so we map that
        # back to the legacy TimeoutError contract for callers that
        # rely on it.
        stdout_text, stderr_text, exit_code = await self._runtime_run(
            cmd,
            cwd=working_dir,
            env=self._build_subprocess_env(pip_overlay),
            timeout=config.timeout,
        )
        if exit_code == 124 and stderr_text.startswith("[timeout after "):
            raise asyncio.TimeoutError(
                f"Script '{script_rel}' timed out after {config.timeout}s"
            )

        return (stdout_text, stderr_text, exit_code)

    async def execute(
        self,
        step: Any,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        """Execute an AgentSkill step."""
        start_time = time.time()
        config: AgentSkillStepConfig = step.step_config  # type: ignore

        try:
            # Load the skill manifest
            loader = self._get_loader(config)
            skill_source = config.skill  # already coerced to AgentSkillSource

            try:
                manifest = await loader.load(skill_source)  # type: ignore[arg-type]
            except (SkillNotFoundError, SkillParseError) as e:
                return StepExecutionResult(
                    step_number=step.number,
                    step_title=step.title,
                    step_type=StepType.AGENT_SKILL,
                    result="",
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    updated_history=context.history.copy(),
                )

            # Resolve skill runtime.
            # Unknown / unavailable backends raise SkillRuntimeError, which
            # surfaces as a step failure with a clear "install extras"
            # message — much friendlier than a runtime AttributeError later.
            from .skill_runtime import (  # noqa: PLC0415
                SkillRuntimeError, get_skill_runtime, resolve_network_policy,
            )
            try:
                skill_runtime = get_skill_runtime(config.runtime)
            except SkillRuntimeError as e:
                return StepExecutionResult(
                    step_number=step.number,
                    step_title=step.title,
                    step_type=StepType.AGENT_SKILL,
                    result="",
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    updated_history=context.history.copy(),
                )
            # Stash on self so internal helpers can reach it without
            # changing every signature in this large class. The executor
            # is constructed fresh per chain run (via get_executor), so
            # cross-step bleed isn't a concern.
            self._skill_runtime = skill_runtime

            # resolve the network policy upfront and
            # stash the manifest's allowed-tools tokens on
            # ``self._network_*`` so backends can pick them up via
            # ``handle.backend["network_policy"]`` after ``prepare``.
            # Validation here surfaces a typo'd policy as a step
            # failure (with the canonical "Valid options: …" message)
            # instead of crashing later inside the backend.
            try:
                self._network_policy, self._network_allowlist = resolve_network_policy(
                    config.runtime_config,
                    manifest_allowed_tools=manifest.get_allowed_tools()
                    if manifest.allowed_tools else None,
                )
            except SkillRuntimeError as e:
                return StepExecutionResult(
                    step_number=step.number,
                    step_title=step.title,
                    step_type=StepType.AGENT_SKILL,
                    result="",
                    success=False,
                    error_message=str(e),
                    execution_time=time.time() - start_time,
                    updated_history=context.history.copy(),
                )

            # Resolve input mapping
            resolved_inputs = self._resolve_inputs(config, context)

            # Install extra_pip packages into an isolated overlay dir if requested.
            # Modes that run subprocesses (SCRIPT, HYBRID, SUBAGENT, LLM_AGENT) will
            # receive pip_overlay so scripts can import the installed packages via
            # PYTHONPATH.  LLM mode never runs scripts, so pip_overlay stays None.
            pip_overlay: Optional[Path] = None
            pip_overlay_dir: Optional[Path] = None
            if config.extra_pip:
                pip_overlay_dir = Path(tempfile.mkdtemp(prefix="carl_pip_"))
                pip_overlay = pip_overlay_dir
                await self._install_extra_pip(config.extra_pip, pip_overlay_dir)

            try:
                # Execute based on mode
                mode = config.execution_mode
                result_str = ""
                output_files: list[dict[str, str]] = []
                metadata: dict[str, Any] = {"skill_name": manifest.name, "execution_mode": mode.value}

                if mode == AgentSkillExecutionMode.LLM:
                    result_str = await self._execute_llm_mode(
                        step, config, context, manifest, resolved_inputs
                    )

                elif mode == AgentSkillExecutionMode.SCRIPT:
                    stdout, stderr, returncode = await self._execute_script_mode(
                        config, manifest, resolved_inputs, pip_overlay=pip_overlay
                    )
                    if returncode != 0:
                        error_msg = (
                            f"Script exited with code {returncode}.\n"
                            f"stderr: {stderr.strip()}"
                        )
                        return StepExecutionResult(
                            step_number=step.number,
                            step_title=step.title,
                            step_type=StepType.AGENT_SKILL,
                            result="",
                            success=False,
                            error_message=error_msg,
                            execution_time=time.time() - start_time,
                            updated_history=context.history.copy(),
                        )
                    result_str = stdout
                    metadata["stderr"] = stderr
                    metadata["returncode"] = returncode

                elif mode == AgentSkillExecutionMode.HYBRID:
                    # Try script first; fall back to LLM if script fails or returns empty
                    script_succeeded = False
                    stdout = ""
                    stderr = ""
                    try:
                        stdout, stderr, returncode = await self._execute_script_mode(
                            config, manifest, resolved_inputs, pip_overlay=pip_overlay
                        )
                        if returncode == 0 and stdout.strip():
                            result_str = stdout
                            script_succeeded = True
                            metadata["script_used"] = True
                            metadata["stderr"] = stderr
                    except Exception as script_err:
                        metadata["script_error"] = str(script_err)

                    if not script_succeeded:
                        # Fall back to LLM
                        metadata["llm_fallback"] = True
                        result_str = await self._execute_llm_mode(
                            step, config, context, manifest, resolved_inputs
                        )

                elif mode == AgentSkillExecutionMode.SUBAGENT:
                    result_str, sub_metadata = await self._execute_subagent_mode(
                        step, config, context, manifest, resolved_inputs
                    )
                    metadata.update(sub_metadata)

                elif mode == AgentSkillExecutionMode.LLM_AGENT:
                    result_str, output_files, agent_metadata, persisted_ws = (
                        await self._execute_llm_agent_mode(
                            step, config, context, manifest, resolved_inputs,
                            pip_overlay=pip_overlay,
                        )
                    )
                    metadata.update(agent_metadata)

                    # Write persisted workspace path to memory when persist_workspace=True
                    if persisted_ws is not None:
                        ws_key = f"{config.output_memory_key or step.title}_workspace"
                        context.memory_write(ws_key, str(persisted_ws), namespace="agent_skill")
                        metadata["persisted_workspace"] = str(persisted_ws)

                    # Store output file paths in memory if output_file_key set
                    if config.output_file_key and output_files:
                        first_file = output_files[0].get("path", "")
                        if first_file:
                            context.memory_write(
                                config.output_file_key,
                                first_file,
                                namespace="agent_skill",
                            )

                    # Build result combining text + file list based on output_capture
                    if config.output_capture == "files":
                        result_str = json.dumps(output_files) if output_files else result_str
                    elif config.output_capture == "both" and output_files:
                        file_list = ", ".join(f["name"] for f in output_files)
                        result_str = f"{result_str}\n\nOutput files: {file_list}".strip()

                else:
                    raise ValueError(f"Unknown AgentSkillExecutionMode: {mode}")

            finally:
                # Clean up pip overlay directory (separate from skill workspace)
                if pip_overlay_dir is not None:
                    shutil.rmtree(pip_overlay_dir, ignore_errors=True)

            # Optionally store output file path in memory
            if config.output_file_key and result_str.strip():
                # Heuristic: if result looks like a file path, store it
                stripped = result_str.strip()
                if os.path.exists(stripped) or stripped.startswith("/") or stripped.startswith("./"):
                    context.memory_write(
                        config.output_file_key,
                        stripped,
                        namespace="agent_skill",
                    )

            # Write text result to memory if output_memory_key is set
            if config.output_memory_key and result_str.strip():
                context.memory_write(
                    config.output_memory_key,
                    result_str.strip(),
                    namespace=config.output_memory_namespace,
                )

            # Build history entry
            skill_name = manifest.name
            step_result = (
                f"Step {step.number}. {step.title} [skill:{skill_name}]\n"
                f"Result: {result_str}\n"
            )

            updated_history = context.history.copy()
            updated_history.append(step_result)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_SKILL,
                result=result_str,
                result_data=metadata,
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except _StepCancelled:
            # mid-step cancellation surfaces as a
            # skipped result, not a failure.
            return _cancelled_step_result(
                step, StepType.AGENT_SKILL, context, start_time,
            )
        except asyncio.TimeoutError as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_SKILL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_SKILL,
                result="",
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )


# =============================================================================
# AgentHandoffStepExecutor — delegate to a complete sub-chain
# =============================================================================


class AgentHandoffStepExecutor(StepExecutorBase):
    """
    Executor for agent handoff steps.

    Creates an isolated ``ReasoningContext`` derived from the parent,
    applies input_mapping to pre-populate sub-chain memory, runs the
    sub-chain, and merges the result back into parent context.

    Input mapping keys use dotted ``"namespace.key"`` notation for sub-chain
    memory; bare keys without a dot use the ``"input"`` namespace.

    On success the sub-chain's final history entry is written to
    ``context.memory_write(output_memory_key, ..., namespace=output_namespace)``
    and the full ``ReasoningResult`` is stored in ``result_data["sub_result"]``.
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import AgentHandoffStepConfig

        start_time = time.time()
        config: AgentHandoffStepConfig = step.step_config  # type: ignore
        sub_chain = getattr(step, "sub_chain", None)

        if sub_chain is None:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_HANDOFF,
                result="",
                success=False,
                error_message="AgentHandoffStepDescription.sub_chain is None — assign a ReasoningChain instance.",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

        # cancellation poll before dispatching to the
        # sub-chain. The sub-chain inherits the cancel token via the
        # parent (see _build_sub_context), but skipping the dispatch
        # entirely is cheaper and surfaces a uniform skipped=True
        # result.
        if context.is_cancelled():
            return _cancelled_step_result(
                step, StepType.AGENT_HANDOFF, context, start_time,
            )

        try:
            sub_ctx = self._build_sub_context(config, context, step)

            # Execute sub-chain with optional timeout
            if config.timeout is not None:
                sub_result = await asyncio.wait_for(
                    sub_chain.execute_async(sub_ctx),
                    timeout=config.timeout,
                )
            else:
                sub_result = await sub_chain.execute_async(sub_ctx)

            # Determine result text (last sub-chain history entry or final output)
            result_text = sub_result.get_final_output() if sub_result.success else ""
            if not result_text and sub_result.history:
                result_text = sub_result.history[-1]

            # Write result back to parent memory
            if config.output_memory_key and sub_result.success:
                context.memory_write(
                    config.output_memory_key, result_text, namespace=config.output_namespace
                )

            step_success = sub_result.success or not config.propagate_failure

            if context.language == Language.ENGLISH:
                status = "completed" if sub_result.success else "failed"
                history_entry = (
                    f"Step {step.number}. {step.title} [HANDOFF]\n"
                    f"Result: Sub-chain {status}. "
                    f"{len(sub_result.step_results)} steps executed.\n"
                )
            else:
                status = "завершена" if sub_result.success else "завершилась с ошибкой"
                history_entry = (
                    f"Шаг {step.number}. {step.title} [ПЕРЕДАЧА]\n"
                    f"Результат: Подцепочка {status}. "
                    f"Выполнено шагов: {len(sub_result.step_results)}.\n"
                )

            updated_history = context.history.copy()
            updated_history.append(history_entry)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_HANDOFF,
                result=result_text,
                result_data={
                    "sub_result": sub_result,
                    "sub_chain_success": sub_result.success,
                    "steps_executed": len(sub_result.step_results),
                },
                success=step_success,
                error_message=None if step_success else f"Sub-chain failed: {sub_result.error or 'unknown error'}",
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except asyncio.TimeoutError:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_HANDOFF,
                result="",
                success=False,
                error_message=f"Sub-chain timed out after {config.timeout}s",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )
        except Exception as exc:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.AGENT_HANDOFF,
                result="",
                success=False,
                error_message=str(exc),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_sub_context(
        self,
        config,
        parent_ctx: ReasoningContext,
        step: StepDescription,
    ) -> ReasoningContext:
        """Create an isolated ReasoningContext for the sub-chain."""
        import copy as _copy

        sub_ctx = ReasoningContext(
            outer_context=parent_ctx.outer_context,
            api=parent_ctx.api,
            model=parent_ctx.model,
            retry_max=parent_ctx.retry_max,
            history=[],  # fresh history for sub-chain
            language=parent_ctx.language,
            system_prompt=parent_ctx.system_prompt,
            memory=_copy.deepcopy(parent_ctx.memory),
            max_history_entries=parent_ctx.max_history_entries,
            trim_strategy=parent_ctx.trim_strategy,
        )

        # Inherit tools from parent
        if config.inherit_tools:
            sub_ctx._tool_registry = parent_ctx._tool_registry.copy()

        # Apply input mapping
        for dest_key, source_ref in config.input_mapping.items():
            value = resolve_context_reference(source_ref, parent_ctx)
            if "." in dest_key:
                namespace, key = dest_key.split(".", 1)
            else:
                namespace, key = "input", dest_key
            sub_ctx.memory_write(key, value, namespace=namespace)

        return sub_ctx


# =============================================================================
# SupervisorStepExecutor — LLM routes a task to one of N sub-chains
# =============================================================================


class SupervisorStepExecutor(StepExecutorBase):
    """
    Executor for supervisor / hierarchical routing steps.

    Asks an LLM to pick one of the registered ``agents`` for the current task.
    The reply is matched against agent names (case-insensitive, whitespace-
    trimmed, prefix-tolerant); on no match, ``config.fallback_agent`` is used
    if set, otherwise the step fails.

    The chosen sub-chain runs with an isolated context derived from the parent
    (same input_mapping / output_memory_key semantics as
    :class:`AgentHandoffStepExecutor`).
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import SupervisorStepConfig

        start_time = time.time()
        config: SupervisorStepConfig = step.step_config  # type: ignore
        agents: dict[str, Any] = getattr(step, "agents", {}) or {}

        if not agents:
            return self._fail(step, start_time, context, "SupervisorStepDescription.agents is empty")

        # cancellation poll before the routing LLM call.
        # The sub-chain itself inherits the cancel token via the parent
        # context, but skipping the routing call entirely surfaces a
        # uniform skipped=True result and avoids a wasted LLM round-trip.
        if context.is_cancelled():
            return _cancelled_step_result(
                step, StepType.SUPERVISOR, context, start_time,
            )

        try:
            # 1. Resolve task string
            task_text = resolve_context_reference(config.task_source, context)
            if task_text is None:
                task_text = ""
            task_str = task_text if isinstance(task_text, str) else json.dumps(task_text, default=str)

            # 2. Build routing prompt and ask the LLM
            agent_names = list(agents.keys())
            routing_prompt = config.routing_prompt.format(
                task=task_str,
                agents=", ".join(agent_names),
            )
            llm_client = context.get_llm_client_for_step(config.llm_config)
            routing_reply = await llm_client.get_response_with_retries(
                routing_prompt, retries=context.retry_max
            )

            # 3. Resolve agent
            selected = self._match_agent(routing_reply, agent_names)
            if selected is None:
                if config.fallback_agent and config.fallback_agent in agents:
                    selected = config.fallback_agent
                else:
                    return self._fail(
                        step,
                        start_time,
                        context,
                        (
                            f"Supervisor LLM reply '{routing_reply.strip()[:80]}' did not "
                            f"match any agent in {agent_names}"
                            + (" and no fallback_agent is set" if not config.fallback_agent else "")
                        ),
                    )

            sub_chain = agents[selected]

            # surface the routing decision as an
            # intra-step event so CARE's TUI can render "supervisor
            # picked agent X" as a nested log line under the parent
            # step before sub-chain execution begins.
            context.emit_step_event(
                step.number, "supervisor.route_selected", {"agent_name": selected},
            )

            # 4. Build sub-context (same pattern as AgentHandoffStepExecutor)
            sub_ctx = self._build_sub_context(config, context, selected, task_str)

            if config.timeout is not None:
                sub_result = await asyncio.wait_for(
                    sub_chain.execute_async(sub_ctx),
                    timeout=config.timeout,
                )
            else:
                sub_result = await sub_chain.execute_async(sub_ctx)

            result_text = sub_result.get_final_output() if sub_result.success else ""
            if not result_text and sub_result.history:
                result_text = sub_result.history[-1]

            if config.output_memory_key and sub_result.success:
                context.memory_write(
                    config.output_memory_key,
                    result_text,
                    namespace=config.output_namespace,
                )

            step_success = sub_result.success or not config.propagate_failure

            if context.language == Language.ENGLISH:
                status = "completed" if sub_result.success else "failed"
                history_entry = (
                    f"Step {step.number}. {step.title} [SUPERVISOR → {selected}]\n"
                    f"Result: Specialist '{selected}' {status}. "
                    f"{len(sub_result.step_results)} steps executed.\n"
                )
            else:
                status = "завершён" if sub_result.success else "завершился с ошибкой"
                history_entry = (
                    f"Шаг {step.number}. {step.title} [СУПЕРВИЗОР → {selected}]\n"
                    f"Результат: Специалист '{selected}' {status}. "
                    f"Выполнено шагов: {len(sub_result.step_results)}.\n"
                )

            updated_history = context.history.copy()
            updated_history.append(history_entry)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.SUPERVISOR,
                result=result_text,
                result_data={
                    "agent_selected": selected,
                    "routing_reply": routing_reply,
                    "sub_result": sub_result,
                    "sub_chain_success": sub_result.success,
                    "steps_executed": len(sub_result.step_results),
                },
                success=step_success,
                error_message=None if step_success else f"Sub-chain failed: {sub_result.error or 'unknown error'}",
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except asyncio.TimeoutError:
            return self._fail(
                step, start_time, context,
                f"Sub-chain timed out after {config.timeout}s",
            )
        except Exception as exc:
            return self._fail(step, start_time, context, str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_agent(reply: str, agent_names: list[str]) -> Optional[str]:
        """Match the LLM's reply to an agent name.

        Strategy (most-strict → least-strict):
        1. Exact match against trimmed/lowered reply.
        2. Reply contains an agent name as a whole word.
        3. Any agent name appears as a substring of the reply.
        """
        if not reply:
            return None
        normalised = reply.strip().lower()
        lower_names = {name.lower(): name for name in agent_names}

        # 1. Exact match
        if normalised in lower_names:
            return lower_names[normalised]

        # 2. Whole-word match (avoids matching 'code' inside 'codebase')
        words = re.findall(r"\b[\w\-]+\b", normalised)
        for w in words:
            if w in lower_names:
                return lower_names[w]

        # 3. Substring fallback (last resort)
        for low, original in lower_names.items():
            if low in normalised:
                return original
        return None

    def _build_sub_context(
        self,
        config,
        parent_ctx: ReasoningContext,
        selected_agent: str,
        task_str: str,
    ) -> ReasoningContext:
        """Create an isolated ReasoningContext for the selected sub-chain.

        Mirrors :py:meth:`AgentHandoffStepExecutor._build_sub_context` but also
        seeds ``memory.input.task`` and ``memory.input.supervisor_agent`` so
        sub-chains can read the routing decision and routed task directly.
        """
        import copy as _copy

        sub_ctx = ReasoningContext(
            outer_context=parent_ctx.outer_context,
            api=parent_ctx.api,
            model=parent_ctx.model,
            retry_max=parent_ctx.retry_max,
            history=[],
            language=parent_ctx.language,
            system_prompt=parent_ctx.system_prompt,
            memory=_copy.deepcopy(parent_ctx.memory),
            max_history_entries=parent_ctx.max_history_entries,
            trim_strategy=parent_ctx.trim_strategy,
        )

        if config.inherit_tools:
            sub_ctx._tool_registry = parent_ctx._tool_registry.copy()
            sub_ctx._tool_tags = {n: set(t) for n, t in parent_ctx._tool_tags.items()}

        # Seed the routed task + selection for sub-chain consumption
        sub_ctx.memory_write("task", task_str, namespace="input")
        sub_ctx.memory_write("supervisor_agent", selected_agent, namespace="input")

        # Apply user-supplied input mapping (same semantics as AgentHandoff)
        for dest_key, source_ref in config.input_mapping.items():
            value = resolve_context_reference(source_ref, parent_ctx)
            if "." in dest_key:
                namespace, key = dest_key.split(".", 1)
            else:
                namespace, key = "input", dest_key
            sub_ctx.memory_write(key, value, namespace=namespace)

        return sub_ctx

    def _fail(
        self,
        step: StepDescription,
        start_time: float,
        context: ReasoningContext,
        message: str,
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.SUPERVISOR,
            result="",
            success=False,
            error_message=message,
            execution_time=time.time() - start_time,
            updated_history=context.history.copy(),
        )


# =============================================================================
# DebateStepExecutor — round-robin multi-agent debate + judge synthesis
# =============================================================================


class DebateStepExecutor(StepExecutorBase):
    """
    Executor for round-robin debate steps.

    Each round, every role's LLM call sees the topic plus the running
    transcript and produces its next argument. After all rounds finish, a
    single judge call synthesises the transcript into a verdict.

    Total LLM calls: ``len(roles) * rounds + 1``.
    """

    _DEFAULT_ROLE_INSTRUCTION = (
        "You are the '{role}' in a structured debate.\n"
        "Round {round} of the debate.\n"
        "Topic: {task}\n\n"
        "Debate so far:\n{transcript}\n\n"
        "Make your next argument. Keep it concise (2-4 sentences). "
        "Do NOT begin with your role name — the transcript will label you."
    )

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import DebateStepConfig

        start_time = time.time()
        config: DebateStepConfig = step.step_config  # type: ignore

        try:
            task_value = resolve_context_reference(config.task_source, context)
            if task_value is None:
                task_value = ""
            task_str = (
                task_value if isinstance(task_value, str) else json.dumps(task_value, default=str)
            )

            transcript: list[dict[str, Any]] = []
            transcript_text_so_far = ""

            for round_idx in range(1, config.rounds + 1):
                # cancellation poll between rounds.
                # A multi-round debate against an LLM is one of the
                # longest-running step types; users need their cancel to
                # take effect mid-step, not just at the next batch
                # boundary.
                if context.is_cancelled():
                    return StepExecutionResult(
                        step_number=step.number,
                        step_title=step.title,
                        step_type=StepType.DEBATE,
                        result="",
                        success=False,
                        skipped=True,
                        error_message="cancelled by user",
                        execution_time=time.time() - start_time,
                        updated_history=context.history.copy(),
                    )
                # one event per round so CARE's TUI can
                # render a "Round N" header before listing each role's
                # argument as a child.
                context.emit_step_event(
                    step.number, "debate.round_started",
                    {"round": round_idx, "role": None},
                )
                for role in config.roles:
                    role_instruction = config.role_prompts.get(role) or self._DEFAULT_ROLE_INSTRUCTION
                    prompt = role_instruction.format(
                        task=task_str,
                        role=role,
                        round=round_idx,
                        transcript=transcript_text_so_far or "(no prior arguments yet)",
                    )
                    role_llm_config = config.role_llm_configs.get(role) or config.llm_config
                    client = context.get_llm_client_for_step(role_llm_config)
                    argument = await client.get_response_with_retries(
                        prompt, retries=context.retry_max
                    )
                    argument = (argument or "").strip()
                    transcript.append({"round": round_idx, "role": role, "argument": argument})
                    transcript_text_so_far = self._format_transcript(transcript)
                    # fire per-turn argument event so
                    # CARE can stream each role's contribution as it
                    # arrives. The argument text may be long; consumers
                    # can truncate for display.
                    context.emit_step_event(
                        step.number, "debate.turn_argument",
                        {"round": round_idx, "role": role, "argument": argument},
                    )

            # Judge synthesis
            judge_prompt = config.judge_prompt.format(
                task=task_str,
                transcript=transcript_text_so_far or "(no debate occurred)",
            )
            judge_client = context.get_llm_client_for_step(config.llm_config)
            verdict = await judge_client.get_response_with_retries(
                judge_prompt, retries=context.retry_max
            )
            verdict = (verdict or "").strip()

            if config.output_memory_key:
                context.memory_write(
                    config.output_memory_key, verdict, namespace=config.output_namespace
                )

            if context.language == Language.ENGLISH:
                history_entry = (
                    f"Step {step.number}. {step.title} [DEBATE: "
                    f"{', '.join(config.roles)} × {config.rounds} rounds]\n"
                    f"Result: {verdict}\n"
                )
            else:
                history_entry = (
                    f"Шаг {step.number}. {step.title} [ДЕБАТЫ: "
                    f"{', '.join(config.roles)} × {config.rounds} раундов]\n"
                    f"Результат: {verdict}\n"
                )

            updated_history = context.history.copy()
            updated_history.append(history_entry)

            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.DEBATE,
                result=verdict,
                result_data={
                    "verdict": verdict,
                    "transcript": transcript,
                    "rounds_executed": config.rounds,
                    "role_call_count": len(config.roles) * config.rounds,
                    "topic": task_str,
                },
                success=True,
                execution_time=time.time() - start_time,
                updated_history=updated_history,
            )

        except Exception as exc:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.DEBATE,
                result="",
                success=False,
                error_message=str(exc),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    @staticmethod
    def _format_transcript(transcript: list[dict[str, Any]]) -> str:
        """Render the running transcript as plain text for inclusion in prompts."""
        lines: list[str] = []
        for turn in transcript:
            lines.append(
                f"[Round {turn['round']} · {turn['role']}] {turn['argument']}"
            )
        return "\n".join(lines)


# =============================================================================
# EvaluationStepExecutor — inline quality gate
# =============================================================================


class EvaluationStepExecutor(StepExecutorBase):
    """
    Executor for inline quality-gate (evaluation) steps.

    Evaluates the output of a previously completed step against a list of
    criteria using either rule-based pattern matching (``"rule"``) or an LLM
    judge (``"llm"``).  Reacts to failures according to ``config.on_fail``:

    - ``CONTINUE``: logs the failure but lets the chain proceed.
    - ``ABORT``: marks the step as failed, halting the chain.
    - ``RETRY_WITH_FEEDBACK``: calls the LLM with the original output + critique
      to generate an improved response, then re-evaluates; repeats up to
      ``config.max_retries`` times, then falls through to CONTINUE.
    """

    # Simple pattern conditions (no eval needed)
    _PATTERN_RE = re.compile(r"^(min_words|contains|startswith|endswith):(.+)$")

    def __init__(self):
        self._evaluator = EvalWithCompoundTypes()
        self._evaluator.functions = {
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "bool": bool,
            "any": any,
            "all": all,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import EvalFailAction, EvaluationStepConfig

        start_time = time.time()
        config: EvaluationStepConfig = step.step_config  # type: ignore

        try:
            evaluated_text = str(resolve_context_reference(config.input_source, context) or "")
            passed, critique = await self._evaluate(config, evaluated_text, context)

            if passed:
                return self._make_result(
                    step, start_time, context.history, "PASS", True, True, {}
                )

            # --- Failed ---
            if config.on_fail == EvalFailAction.ABORT:
                msg = f"EVALUATION FAILED: {critique}"
                return self._make_result(step, start_time, context.history, msg, False, False, {
                    "critique": critique,
                    "verdict": "FAIL",
                })

            if config.on_fail == EvalFailAction.RETRY_WITH_FEEDBACK:
                verdict, final_critique, improved_text = await self._retry_with_feedback(
                    config, evaluated_text, critique, context
                )
                summary = f"EVALUATION {verdict}: {final_critique}" if verdict == "FAIL" else f"EVALUATION {verdict} (after retry)"
                rd: dict = {"critique": final_critique, "verdict": verdict}
                if improved_text is not None:
                    rd["improved_response"] = improved_text
                return self._make_result(step, start_time, context.history, summary, True, True, rd)

            # on_fail == CONTINUE
            msg = f"EVALUATION FAILED (continuing): {critique}"
            return self._make_result(step, start_time, context.history, msg, True, True, {
                "critique": critique,
                "verdict": "FAIL",
            })

        except Exception as exc:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.EVALUATION,
                result="",
                success=False,
                error_message=str(exc),
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_result(
        self,
        step: StepDescription,
        start_time: float,
        history: list[str],
        summary: str,
        success: bool,
        chain_continues: bool,
        result_data: dict,
    ) -> StepExecutionResult:
        if step.number and hasattr(step, "title"):
            entry = f"Step {step.number}. {step.title} [EVALUATION]\nResult: {summary}\n"
        else:
            entry = f"[EVALUATION]\nResult: {summary}\n"
        updated_history = history.copy()
        updated_history.append(entry)
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.EVALUATION,
            result=summary,
            result_data=result_data,
            success=success,
            error_message=None if chain_continues else summary,
            execution_time=time.time() - start_time,
            updated_history=updated_history,
        )

    async def _evaluate(
        self, config, evaluated_text: str, context: ReasoningContext
    ) -> tuple[bool, str]:
        """Return (passed, critique)."""
        if config.evaluation_method == "llm":
            return await self._evaluate_llm(config, evaluated_text, context)
        return self._evaluate_rules(config, evaluated_text)

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(self, config, evaluated_text: str) -> tuple[bool, str]:
        """Evaluate all criteria as rule patterns or simpleeval expressions."""
        failed: list[str] = []
        for criterion in config.criteria:
            if not self._check_criterion(criterion, evaluated_text):
                failed.append(criterion)
        if failed:
            return False, "Failed criteria: " + "; ".join(failed)
        return True, ""

    def _check_criterion(self, criterion: str, value: str) -> bool:
        m = self._PATTERN_RE.match(criterion)
        if m:
            kind, arg = m.group(1), m.group(2)
            if kind == "min_words":
                try:
                    return len(value.split()) >= int(arg)
                except ValueError:
                    return False
            if kind == "contains":
                return arg.lower() in value.lower()
            if kind == "startswith":
                return value.startswith(arg)
            if kind == "endswith":
                return value.endswith(arg)

        # simpleeval expression over `value`
        try:
            self._evaluator.names = {"value": value, "v": value}
            return bool(self._evaluator.eval(criterion))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # LLM evaluation
    # ------------------------------------------------------------------

    async def _evaluate_llm(
        self, config, evaluated_text: str, context: ReasoningContext
    ) -> tuple[bool, str]:
        """Call an LLM judge and parse its verdict."""
        llm_client = context.api
        if llm_client is None:
            return False, "No LLM client available for evaluation"

        criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(config.criteria))
        prompt = (
            "You are a quality evaluator. Assess the following text against the criteria below.\n"
            "For each criterion, state PASS or FAIL with a brief reason on the same line.\n"
            "Then give an overall verdict on a line starting with 'Overall: PASS' or 'Overall: FAIL'.\n"
            "If the overall verdict is FAIL, add a line starting with 'Critique:' with specific improvement feedback.\n\n"
            f"TEXT TO EVALUATE:\n{evaluated_text}\n\n"
            f"CRITERIA:\n{criteria_block}\n\n"
            "Respond with:\n"
            "1: PASS/FAIL — reason\n"
            "...\n"
            "Overall: PASS or FAIL\n"
            "Critique: <feedback if FAIL>"
        )

        try:
            if isinstance(llm_client, LLMClientBase):
                result, _ = await llm_client.get_response_with_usage(prompt, retries=1)
            else:
                result = await llm_client.get_response_with_retries(prompt, retries=1)
        except Exception as exc:
            return False, f"LLM evaluation error: {exc}"

        passed = self._parse_verdict(result)
        critique = self._parse_critique(result)
        return passed, critique

    @staticmethod
    def _parse_verdict(response: str) -> bool:
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("overall:"):
                return "pass" in stripped.lower() and "fail" not in stripped.lower()
        # Fallback: if response contains FAIL anywhere, treat as fail
        return "FAIL" not in response.upper()

    @staticmethod
    def _parse_critique(response: str) -> str:
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("critique:"):
                return stripped[len("critique:"):].strip()
        # Extract overall line if no explicit critique
        for line in response.splitlines():
            if line.strip().lower().startswith("overall:"):
                return line.strip()
        return response.strip()[:300]

    # ------------------------------------------------------------------
    # RETRY_WITH_FEEDBACK
    # ------------------------------------------------------------------

    async def _retry_with_feedback(
        self,
        config,
        initial_text: str,
        initial_critique: str,
        context: ReasoningContext,
    ) -> tuple[str, str, Optional[str]]:
        """
        Attempt up to ``config.max_retries`` LLM-driven improvements.

        Returns (final_verdict, final_critique, best_improved_text_or_None).

        The improved text is stored in result_data["improved_response"] so that
        downstream steps can access it.  It is NOT injected back into history
        because the DAGExecutor architecture only propagates the last history
        entry produced by a step — previous entries cannot be replaced.
        """
        llm_client = context.api
        criteria_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(config.criteria))
        current_text = initial_text
        current_critique = initial_critique
        best_improved: Optional[str] = None

        for _ in range(max(config.max_retries, 0)):
            if llm_client is None:
                break
            improve_prompt = (
                "The following response failed a quality evaluation. "
                "Please provide an improved version that satisfies ALL criteria.\n\n"
                f"ORIGINAL RESPONSE:\n{current_text}\n\n"
                f"EVALUATION CRITIQUE:\n{current_critique}\n\n"
                f"CRITERIA TO SATISFY:\n{criteria_block}\n\n"
                "Provide only the improved response text."
            )
            try:
                if isinstance(llm_client, LLMClientBase):
                    improved, _ = await llm_client.get_response_with_usage(improve_prompt, retries=1)
                else:
                    improved = await llm_client.get_response_with_retries(improve_prompt, retries=1)
            except Exception:
                break

            current_text = improved
            best_improved = improved

            # Re-evaluate the improved text
            passed, critique = await self._evaluate(config, current_text, context)
            if passed:
                return "PASS", "", improved
            current_critique = critique

        return "FAIL", current_critique, best_improved


# =============================================================================
# ParallelSamplingStepExecutor
# =============================================================================


class ParallelSamplingStepExecutor(StepExecutorBase):
    """
    Executes a base LLM step N times in parallel and aggregates the results.

    Aggregation strategies
    ----------------------
    majority_vote
        Normalises each response (lower-case, stripped) and returns the
        response text whose normalised form appears most often.  Ties are
        broken by the first occurrence.
    best_of_n / llm_judge
        Presents all candidates to an LLM judge which picks the best one.
        The raw text of the winning candidate is returned (not a summary).
    """

    _DEFAULT_JUDGE_PROMPT = (
        "You are an expert evaluator. Below are {n} candidate responses to the same task.\n"
        "Select the single best response. Reply with ONLY the number of the best candidate "
        "(e.g. '3'), nothing else.\n\n"
        "{candidates}"
    )

    async def execute(
        self,
        step: "StepDescription",
        context: "ReasoningContext",
        prompt_template: Optional["PromptTemplate"] = None,
    ) -> "StepExecutionResult":
        from .models.config import ParallelSamplingAggregation
        from .models.steps import LLMStepDescription

        start_time = time.time()
        cfg = step.config  # ParallelSamplingStepConfig
        base_step: LLMStepDescription = step.base_step

        # --- run n_samples copies in parallel ---
        llm_executor = LLMStepExecutor()

        async def _sample() -> "StepExecutionResult":
            sample_step = base_step.model_copy(update={"number": step.number})
            # Each sample sees the same context (read-only); only the final winner goes in history.
            return await llm_executor.execute(sample_step, context, prompt_template)


        # cancellation poll before kicking off N samples.
        # A cancel issued mid-batch should stop us from scheduling more
        # LLM calls; in-flight samples may still complete (asyncio
        # cooperative — we don't forcibly tear them down here).
        if context.is_cancelled():
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.PARALLEL_SAMPLING,
                result="",
                success=False,
                skipped=True,
                error_message="cancelled by user",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

        try:
            sample_results = await asyncio.gather(*[_sample() for _ in range(cfg.n_samples)])
        except Exception as e:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.PARALLEL_SAMPLING,
                result="",
                success=False,
                error_message=f"Parallel sampling failed: {e}",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

        # emit one event per sample so CARE's TUI can
        # render nested log lines under the parent step ("sample 0:
        # …", "sample 1: …"). Fired after gather so the order is
        # deterministic; live progress streaming for in-flight samples
        # is out of scope here (the underlying LLMStepExecutor doesn't
        # yet emit per-sample chunks back through this layer).
        for idx, sr in enumerate(sample_results):
            context.emit_step_event(
                step.number, "parallel_sampling.sample",
                {
                    "sample_idx": idx,
                    "output": sr.result if sr.success else "",
                    "success": sr.success,
                    "error_message": sr.error_message,
                },
            )

        # Separate successes from failures
        successes = [r for r in sample_results if r.success]
        if not successes:
            errors = "; ".join(r.error_message or "" for r in sample_results)
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.PARALLEL_SAMPLING,
                result="",
                success=False,
                error_message=f"All {cfg.n_samples} samples failed. Errors: {errors}",
                execution_time=time.time() - start_time,
                updated_history=context.history.copy(),
            )

        candidates = [r.result for r in successes]

        # Aggregate token usage across all samples
        total_usage: dict[str, int] = {}
        for r in sample_results:
            for k, v in r.token_usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

        # --- aggregation ---
        if cfg.aggregation == ParallelSamplingAggregation.MAJORITY_VOTE:
            winner = self._majority_vote(candidates, cfg.normalize_for_vote)
        else:
            # BEST_OF_N or LLM_JUDGE: use LLM to pick
            winner = await self._llm_judge(
                candidates=candidates,
                context=context,
                judge_prompt_template=cfg.judge_prompt,
            )

        # Build result in the same format as LLMStepExecutor
        if context.language == Language.ENGLISH:
            step_result = f"Step {step.number}. {step.title}\nResult: {winner}\n"
        else:
            step_result = f"Шаг {step.number}. {step.title}\nРезультат: {winner}\n"
        updated_history = context.history.copy()
        updated_history.append(step_result)

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.PARALLEL_SAMPLING,
            result=winner,
            result_data={
                "n_samples": cfg.n_samples,
                "n_successes": len(successes),
                "aggregation": cfg.aggregation,
                "candidates": candidates,
            },
            success=True,
            execution_time=time.time() - start_time,
            updated_history=updated_history,
            token_usage=total_usage,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _majority_vote(candidates: list[str], normalize: bool) -> str:
        """Return the candidate whose normalised form appears most often."""
        from collections import Counter

        def _norm(s: str) -> str:
            return s.strip().lower() if normalize else s

        counts = Counter(_norm(c) for c in candidates)
        winning_norm = counts.most_common(1)[0][0]
        # Return the raw (un-normalised) text of the first matching candidate
        for c in candidates:
            if _norm(c) == winning_norm:
                return c
        return candidates[0]

    async def _llm_judge(
        self,
        candidates: list[str],
        context: "ReasoningContext",
        judge_prompt_template: str = "",
    ) -> str:
        """Ask an LLM to pick the best candidate; return its raw text."""
        numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(candidates))
        template = judge_prompt_template.strip() or self._DEFAULT_JUDGE_PROMPT
        prompt = template.format(n=len(candidates), candidates=numbered)

        llm_client = context.api
        try:
            if isinstance(llm_client, LLMClientBase):
                raw, _ = await llm_client.get_response_with_usage(prompt, retries=2)
            else:
                raw = await llm_client.get_response_with_retries(prompt, retries=2)
        except Exception:
            # Fallback to majority vote if judge call fails
            return self._majority_vote(candidates, normalize=True)

        # Parse the judge's answer — expect a digit
        match = re.search(r"\b(\d+)\b", raw.strip())
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return candidates[idx]
        # If we can't parse, return the first candidate
        return candidates[0]


# =============================================================================
# Tool Discovery Utilities
# =============================================================================


def carl_tool(fn: Optional[Callable] = None, *, tags: list[str] | None = None) -> Callable:
    """
    Decorator that marks a function as discoverable by :class:`ToolDiscoveryStepExecutor`.

    Usage::

        @carl_tool
        def search(query: str) -> str:
            ...

        @carl_tool(tags=["information", "external"])
        def fetch_data(url: str) -> dict:
            ...

    The decorator attaches ``__carl_tool__ = True`` and (optionally)
    ``__carl_tool_tags__ = [...]`` to the function so that
    :class:`ModuleToolSource` can filter by tag.
    """
    def _decorator(func: Callable) -> Callable:
        func.__carl_tool__ = True  # type: ignore[attr-defined]
        if tags is not None:
            func.__carl_tool_tags__ = list(tags)  # type: ignore[attr-defined]
        return func

    if fn is not None:
        # Called without arguments: @carl_tool
        return _decorator(fn)
    # Called with arguments: @carl_tool(tags=[...])
    return _decorator


class ToolDiscoveryStepExecutor(StepExecutorBase):
    """
    Executor for :class:`~mmar_carl.models.steps.ToolDiscoveryStepDescription`.

    Resolves the configured :class:`~mmar_carl.models.config.ToolSource`, discovers
    callables, and registers them in the context's tool registry so that subsequent
    :class:`~mmar_carl.models.steps.ToolStepDescription` steps can use them.

    Tool sources:

    * :class:`~mmar_carl.models.config.ModuleToolSource` — imports a Python module
      and registers all public callables (optionally filtered by name prefix or tag).
    * :class:`~mmar_carl.models.config.CallableToolSource` — calls a factory function
      that returns a ``dict[str, Callable]``.
    * :class:`~mmar_carl.models.config.DictToolSource` — registers a static dict.
    """

    async def execute(
        self,
        step: Any,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        import importlib
        from .models.config import (
            CallableToolSource,
            DictToolSource,
            ModuleToolSource,
            ToolDiscoveryStepConfig,
        )

        config: ToolDiscoveryStepConfig = step.config
        source = config.source
        discovered: dict[str, Callable] = {}

        try:
            if isinstance(source, ModuleToolSource):
                module = importlib.import_module(source.module)
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue
                    attr = getattr(module, attr_name, None)
                    if not callable(attr):
                        continue
                    # Name-prefix filter
                    if source.name_prefix and not attr_name.startswith(source.name_prefix):
                        continue
                    # Tag filter
                    if source.tag:
                        tool_tags = getattr(attr, "__carl_tool_tags__", [])
                        if source.tag not in tool_tags:
                            continue
                    # Determine registration name
                    tool_name = attr_name
                    if source.strip_prefix and source.name_prefix and attr_name.startswith(source.name_prefix):
                        tool_name = attr_name[len(source.name_prefix):]
                    discovered[tool_name] = attr

            elif isinstance(source, CallableToolSource):
                result = source.factory()
                if not isinstance(result, dict):
                    raise ValueError(
                        f"CallableToolSource factory must return a dict[str, Callable], "
                        f"got {type(result).__name__}"
                    )
                discovered = {str(k): v for k, v in result.items()}

            elif isinstance(source, DictToolSource):
                discovered = dict(source.tools)

            else:
                raise ValueError(f"Unknown ToolSource type: {type(source).__name__}")

        except Exception as exc:
            return StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=StepType.TOOL_DISCOVERY,
                result="",
                success=False,
                error_message=f"Tool discovery failed: {exc}",
            )

        # Register discovered tools in context
        for tool_name, tool_callable in discovered.items():
            context.register_tool(tool_name, tool_callable, timeout=config.tool_timeout)

        tool_names = sorted(discovered.keys())
        summary = f"Discovered {len(tool_names)} tool(s): {', '.join(tool_names)}"

        # Optionally write tool list to memory
        if config.output_memory_key:
            context.memory_write(config.output_memory_key, tool_names, namespace="tools")

        history_entry = f"Step {step.number}. {step.title}\nResult: {summary}\n"

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.TOOL_DISCOVERY,
            result=summary,
            result_data={"tool_names": tool_names},
            success=True,
            updated_history=context.history + [history_entry],
        )


# =============================================================================
# HumanInputStepExecutor
# =============================================================================


class HumanInputStepExecutor(StepExecutorBase):
    """
    Executor for ``HumanInputStepDescription``.

    Suspends the coroutine and waits for a human to call
    ``context.provide_human_input(value)``.  An optional timeout causes the
    step to resume automatically with ``config.fallback_value``.

    If ``context.on_human_input_requested`` is set, it is called with
    ``(prompt: str, future: asyncio.Future)`` so the application can notify
    its UI layer that input is needed.  If the callback is *not* set, the
    step falls back to ``fallback_value`` immediately (handy for tests).
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        from .models.config import HumanInputStepConfig

        start_time = time.time()
        config: HumanInputStepConfig = step.step_config  # type: ignore

        # Create a future that can be resolved by provide_human_input()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        context._human_input_future = future

        value = config.fallback_value
        timed_out = False

        try:
            if context.on_human_input_requested is not None:
                # Fire callback — the app/UI should eventually call provide_human_input()
                cb_result = context.on_human_input_requested(config.prompt, future)
                if asyncio.iscoroutine(cb_result):
                    asyncio.ensure_future(cb_result)

                # Now await the future (with optional timeout)
                try:
                    if config.timeout is not None:
                        value = await asyncio.wait_for(
                            asyncio.shield(future), timeout=config.timeout
                        )
                    else:
                        value = await future
                except asyncio.TimeoutError:
                    timed_out = True
                    value = config.fallback_value
                    if not future.done():
                        future.cancel()
            else:
                # No callback registered — use fallback immediately (test / non-interactive mode)
                value = config.fallback_value
        finally:
            context._human_input_future = None

        # Build history entry
        if context.language == Language.ENGLISH:
            status = "timed out" if timed_out else "received"
            history_entry = (
                f"Step {step.number}. {step.title}\n"
                f"Human input ({status}): {value}\n"
            )
        else:
            status = "тайм-аут" if timed_out else "получен"
            history_entry = (
                f"Шаг {step.number}. {step.title}\n"
                f"Ввод пользователя ({status}): {value}\n"
            )

        # Optionally write to memory
        if config.output_memory_key:
            context.memory_write(config.output_memory_key, value, namespace="human_input")

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.HUMAN_INPUT,
            result=value,
            result_data={"timed_out": timed_out, "prompt": config.prompt},
            success=True,
            execution_time=time.time() - start_time,
            updated_history=context.history + [history_entry],
        )


# =============================================================================
# Step Executor Factory
# =============================================================================


_EXECUTORS: dict[StepType, StepExecutorBase] = {
    StepType.LLM: LLMStepExecutor(),
    StepType.TOOL: ToolStepExecutor(),
    StepType.MCP: MCPStepExecutor(),
    StepType.MEMORY: MemoryStepExecutor(),
    StepType.TRANSFORM: TransformStepExecutor(),
    StepType.CONDITIONAL: ConditionalStepExecutor(),
    StepType.STRUCTURED_OUTPUT: StructuredOutputStepExecutor(),
    StepType.AGENT_SKILL: AgentSkillStepExecutor(),
    StepType.EVALUATION: EvaluationStepExecutor(),
    StepType.AGENT_HANDOFF: AgentHandoffStepExecutor(),
    StepType.PARALLEL_SAMPLING: ParallelSamplingStepExecutor(),
    StepType.TOOL_DISCOVERY: ToolDiscoveryStepExecutor(),
    StepType.HUMAN_INPUT: HumanInputStepExecutor(),
    StepType.SUPERVISOR: SupervisorStepExecutor(),
    StepType.DEBATE: DebateStepExecutor(),
    StepType.MCP_RESOURCE: MCPResourceStepExecutor(),
}


def get_executor(step_type: StepType) -> StepExecutorBase:
    """Get the executor for a given step type."""
    executor = _EXECUTORS.get(step_type)
    if executor is None:
        raise ValueError(f"No executor registered for step type: {step_type}")
    return executor


def register_executor(step_type: StepType, executor: StepExecutorBase) -> None:
    """Register a custom executor for a step type."""
    _EXECUTORS[step_type] = executor
