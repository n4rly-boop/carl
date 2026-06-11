"""
Reasoning context for CARL reasoning system.
"""

import asyncio
import re
import warnings
from typing import Any, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_serializer

from mmar_carl.llm import OpenAIClientConfig, OpenAICompatibleClient
from mmar_carl.models.base import SelfCriticEvaluatorBase
from mmar_carl.models.config import LLMStepConfig
from mmar_carl.models.enums import Language
from mmar_carl.models.llm_client_base import ChatMessage, LLMClientBase
from mmar_carl.models.replan import ReplanCheckerBase


class _CancelToken:
    """Tiny mutable container for the cancel-requested flag.

    Sharing a token across the parent context and its parallel snapshots
    means a single :meth:`ReasoningContext.cancel` call reaches every
    in-flight snapshot on its next poll — no second-batch lag.
    """

    __slots__ = ("requested",)

    def __init__(self) -> None:
        self.requested: bool = False


class _PauseToken:
    """Mutable container for the pause-requested flag plus an asyncio
    resume event.

    Shared by reference between parent and parallel snapshots — same
    contract as :class:`_CancelToken` — so a single
    :meth:`ReasoningContext.request_pause` reaches every in-flight
    participant. The DAG executor checks ``requested`` between batches
    and awaits ``resume_event`` until the caller clears the flag.

    The ``asyncio.Event`` is created lazily so the token can be
    constructed in a non-running loop (e.g. at import time). Callers
    invoking :meth:`ReasoningContext.wait_for_resume` materialise the
    event on demand.
    """

    __slots__ = ("requested", "_event")

    def __init__(self) -> None:
        self.requested: bool = False
        self._event: Optional[asyncio.Event] = None

    @property
    def event(self) -> asyncio.Event:
        """Lazy asyncio.Event — created on first access from inside a
        running event loop. Starts in the *set* state so an un-paused
        ``await wait_for_resume()`` returns immediately.
        """
        if self._event is None:
            self._event = asyncio.Event()
            self._event.set()
        return self._event


class ContextSnapshot(BaseModel):
    """Pause-time snapshot of a :class:`ReasoningContext`'s mutable state.

    Captures everything the DAG executor would need to resume a paused
    run on a different process: memory, history, metadata, accumulated
    chat messages, and the cancel-token state. Runtime-only fields
    (LLM client, tool registry, callbacks, MCP session caches, etc.)
    are *not* captured — CARE rebuilds those from its configuration
    files before calling :meth:`ReasoningContext.restore`.
    """

    model_config = {"arbitrary_types_allowed": True}

    outer_context: str = Field(default="", description="The original input/task description.")
    history: list[str] = Field(default_factory=list, description="Step output history.")
    memory: dict[str, Any] = Field(default_factory=dict, description="Namespaced memory state.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form metadata bag. Framework-internal `__`-prefixed "
            "keys are stripped at snapshot time, matching the "
            "`RunRecord.from_run` convention — keeps Langfuse trace "
            "handles and replan buffers out of the durable snapshot."
        ),
    )
    messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ChatMessage history serialised via model_dump.",
    )
    cancelled: bool = Field(
        default=False,
        description="Whether the run was already cancelled at snapshot time.",
    )
    completed_step_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Step numbers that already completed at snapshot time. "
            "Used by ``chain.execute_async(resume_from=...)`` to skip "
            "already-executed steps when resuming a paused run."
        ),
    )


def _resolve_pipeline_ref(raw: Any, kwargs: dict[str, Any], prev_output: Any) -> Any:
    """
    Resolve a single value from a pipeline step's arg template.

    Recognised string forms:

    - ``"$input"``               → ``kwargs`` (full dict)
    - ``"$input.a.b"``           → ``kwargs["a"]["b"]``  (dotted dict path)
    - ``"$prev_output"``         → ``prev_output``
    - ``"$prev_output.a.b"``     → ``prev_output["a"]["b"]``

    Anything else (non-string, or string without one of the recognised prefixes)
    is returned unchanged so callers can pass literal values, ints, lists, etc.
    Missing keys along a dotted path yield ``None`` rather than raising — this
    matches the lenient resolution behaviour used elsewhere in CARL.
    """
    if not isinstance(raw, str):
        return raw
    if raw == "$input":
        return kwargs
    if raw == "$prev_output":
        return prev_output
    if raw.startswith("$input."):
        path = raw[len("$input."):].split(".")
        cur: Any = kwargs
    elif raw.startswith("$prev_output."):
        path = raw[len("$prev_output."):].split(".")
        cur = prev_output
    else:
        return raw
    for segment in path:
        if isinstance(cur, dict) and segment in cur:
            cur = cur[segment]
        else:
            return None
    return cur


class ReasoningContext(BaseModel):
    """
    Context object that maintains state during reasoning execution.

    Contains the input data, API object for LLM calls, execution history, and configuration.
    Supports tool registry and memory storage for extended step types.

    Individual LLM steps can override the model using llm_config:
        ```python
        LLMStepDescription(
            number=1,
            title="Complex Task",
            aim="...",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```

    Callbacks for monitoring execution:
        ```python
        context = ReasoningContext(
            outer_context=data,
            api=client,
            on_step_start=lambda num, title: print(f"Starting step {num}"),
            on_step_complete=lambda result: print(f"Step {result.step_number} done"),
            on_progress=lambda completed, total: print(f"Progress: {completed}/{total}"),
        )
        ```

    For streaming LLM responses:
        ```python
        def handle_chunk(chunk: str):
            print(chunk, end="", flush=True)

        context = ReasoningContext(
            outer_context=data,
            api=client,
            on_llm_chunk=handle_chunk,
        )
        ```
    """

    outer_context: str = Field(..., description="Input data as string (it can be CSV or other text information)")
    api: Any = Field(..., description="API object for LLM execution (LLMClientBase or compatible)")
    model: str = Field(default="default", description="Specific model to use")
    retry_max: int = Field(default=3, description="Maximum retry attempts")
    history: list[str] = Field(default_factory=list, description="Accumulated reasoning history")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata and state")
    language: Language = Field(default=Language.RUSSIAN, description="Language for reasoning prompts")
    system_prompt: str = Field(default="", description="System prompt to include in each reasoning step")

    # Deprecated: Use per-step llm_config instead
    default_model: Optional[str] = Field(
        default=None,
        description="Deprecated. Use per-step llm_config instead.",
    )

    # Memory storage for MEMORY step types
    memory: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Memory storage organized by namespace")

    # Optional schema validation for memory writes (see mmar_carl.memory_schema)
    memory_schema: Optional[dict[str, dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional schema declaring expected types for memory keys. "
            "Format: ``{namespace: {key: type_spec}}``. Writes to declared "
            "(namespace, key) pairs that violate the spec raise "
            ":class:`mmar_carl.memory_schema.MemorySchemaError`. Pairs not in "
            "the schema are silently allowed (schemas are additive)."
        ),
    )

    # === Session & Long-term Memory ===
    session_id: str = Field(
        default="",
        description=(
            "Optional session identifier used to scope long-term memory reads/writes. "
            "Two contexts with the same session_id share the same LTM scope."
        ),
    )
    long_term_memory: Optional[Any] = Field(
        default=None,
        description=(
            "Optional long-term memory store (LTMBase subclass). "
            "When set, use context.remember(key, value) and context.recall(query) "
            "to persist and search facts across chain runs. "
            "Also enables the '$ltm.key' reference syntax in input_mapping."
        ),
    )

    # Structured multi-turn conversation history (opt-in via LLMStepConfig.use_message_history)
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Structured multi-turn conversation history. Populated automatically by "
            "LLM steps that have ``llm_config.use_message_history=True``. "
            "Can also be pre-populated to seed the conversation with prior context."
        ),
    )

    # === History Management ===
    max_history_entries: int = Field(
        default=0,
        ge=0,
        description="Maximum history entries to keep (0 = unlimited). Prevents context overflow in long chains.",
    )
    trim_strategy: Literal["oldest", "compress"] = Field(
        default="oldest",
        description=(
            "Strategy for history trimming when max_history_entries is exceeded. "
            "'oldest': drop the oldest entries (FIFO, default). "
            "'compress': strip verbose step headers from history entries, keeping only "
            "the result content. Entries are compressed as they are added, reducing the "
            "per-entry token cost while preserving all information."
        ),
    )

    # === Callbacks for monitoring execution ===
    # Note: Callbacks are excluded from JSON serialization via @field_serializer
    on_step_start: Optional[Callable[[int, str], None]] = Field(
        default=None,
        description="Callback called when a step starts: on_step_start(step_number, step_title)",
    )
    on_step_complete: Optional[Callable[[Any], None]] = Field(
        default=None,
        description="Callback called when a step completes: on_step_complete(StepExecutionResult)",
    )
    on_chain_complete: Optional[Callable[[Any], None]] = Field(
        default=None,
        description="Callback called when the entire chain completes: on_chain_complete(ReasoningResult)",
    )
    on_progress: Optional[Callable[[int, int], None]] = Field(
        default=None,
        description="Progress callback: on_progress(completed_steps, total_steps)",
    )
    on_llm_chunk: Optional[Callable[..., None]] = Field(
        default=None,
        description=(
            "Streaming callback for LLM responses. Two supported signatures: "
            "``on_llm_chunk(chunk_text)`` (legacy) and "
            "``on_llm_chunk(chunk_text, *, step_number, stage)``. "
            "Step executors introspect the signature and route accordingly so "
            "both shapes work transparently. CARE uses the extended shape to "
            "route chunks to the right step pane and stage label "
            "(``\"fast\"`` / ``\"critic\"`` / ``\"regenerate\"`` etc.)."
        ),
    )
    on_human_input_requested: Optional[Callable] = Field(
        default=None,
        description=(
            "Callback invoked when a HumanInputStep needs input: "
            "on_human_input_requested(prompt: str, future: asyncio.Future). "
            "The callback should eventually call context.provide_human_input(value) "
            "to resolve the future and resume execution."
        ),
    )
    on_step_event: Optional[Callable[[int, str, dict[str, Any]], None]] = Field(
        default=None,
        description=(
            "Generic intra-step progress callback: "
            "``on_step_event(step_number, event_type, payload)``. Fires for "
            "fine-grained events that don't correspond to step start / end:\n"
            "  * ``'llm_agent.tool_call'`` — ``{tool, args}`` from each "
            "AgentSkill LLM_AGENT iteration.\n"
            "  * ``'llm_agent.tool_result'`` — ``{tool, result}``.\n"
            "  * ``'debate.round_started'`` — ``{round, role}``.\n"
            "  * ``'debate.turn_argument'`` — ``{round, role, argument}``.\n"
            "  * ``'parallel_sampling.sample'`` — ``{sample_idx, output}``.\n"
            "  * ``'supervisor.route_selected'`` — ``{agent_name}``.\n"
            "Exceptions raised inside the callback are swallowed (logged via "
            "``log_warning``) so a misbehaving consumer can't crash the run."
        ),
    )

    # === Cancellation support ===
    #
    # Cancellation is backed by a tiny mutable token rather than a bare bool
    # so that parallel-execution snapshots can *share* the parent's state by
    # reference. A user-requested cancel from outside the chain now reaches
    # every in-flight snapshot immediately, not just the next batch.
    _cancel_token: "_CancelToken" = PrivateAttr(default_factory=lambda: _CancelToken())

    # === Pause / resume support ===
    #
    # Same shared-token pattern as cancellation. The DAG executor checks the
    # flag at batch boundaries and awaits ``token.event`` until the caller
    # clears the pause.
    _pause_token: "_PauseToken" = PrivateAttr(default_factory=lambda: _PauseToken())

    # === MCP session pooling ===
    #
    # When non-None, every MCP call (list_tools / call_tool /
    # list_resources) reuses the pooled session instead of opening a
    # fresh transport per call. Lifecycle is managed by the caller via
    # `async with context.mcp_pool() as pool:` (or by manually flipping
    # this attribute).
    _mcp_pool: Optional[Any] = PrivateAttr(default=None)

    # === Completed-step tracking ===
    #
    # The DAG executor stamps step numbers here after each successful
    # completion so ``ctx.snapshot()`` can record them for resume.
    _executed_step_numbers: list[int] = PrivateAttr(default_factory=list)

    # === Human-input support ===
    _human_input_future: Optional[Any] = PrivateAttr(default=None)

    # === Private instance attributes (NOT class attributes!) ===
    # FIX: Tool registry must be instance-level, not class-level
    _tool_registry: dict[str, Callable] = PrivateAttr(default_factory=dict)

    # Tags associated with each registered tool (parallel to _tool_registry)
    _tool_tags: dict[str, set[str]] = PrivateAttr(default_factory=dict)

    # In-chain event bus: name → last emitted payload. Used by steps that
    # declare ``triggered_by=[...]`` instead of (or in addition to) numeric
    # dependencies, and by the ``$event.<name>`` reference syntax.
    _emitted_events: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal LLM client cache (keyed by model for per-step clients)
    _llm_client: LLMClientBase | None = PrivateAttr(default=None)
    _llm_client_cache: dict[str, LLMClientBase] = PrivateAttr(default_factory=dict)

    # Self-critic evaluator registry
    _self_critic_evaluator_registry: dict[str, SelfCriticEvaluatorBase] = PrivateAttr(default_factory=dict)

    # RE-PLAN checker registry
    _replan_checker_registry: dict[str, ReplanCheckerBase] = PrivateAttr(default_factory=dict)

    # === Serialization: Exclude callbacks from JSON/dict output ===
    @field_serializer(
        "on_step_start",
        "on_step_complete",
        "on_chain_complete",
        "on_progress",
        "on_llm_chunk",
        "on_human_input_requested",
        when_used="json-unless-none",
    )
    def serialize_callback(self, value: Optional[Callable]) -> Optional[str]:
        """Serialize callbacks as '<callback>' to avoid JSON serialization errors."""
        return "<callback>" if value is not None else None

    def model_post_init(self, __context: Any) -> None:
        """Create LLM client after model initialization."""

        # Initialize instance-level tool registry (fix for class attribute bug)
        if not hasattr(self, "_tool_registry") or self._tool_registry is None:
            self._tool_registry = {}

        # Initialize parallel tag registry
        if not hasattr(self, "_tool_tags") or self._tool_tags is None:
            self._tool_tags = {}

        # Initialize event-bus registry
        if not hasattr(self, "_emitted_events") or self._emitted_events is None:
            self._emitted_events = {}

        # Initialize LLM client cache
        if not hasattr(self, "_llm_client_cache") or self._llm_client_cache is None:
            self._llm_client_cache = {}

        # Initialize self-critic evaluator registry
        if not hasattr(self, "_self_critic_evaluator_registry") or self._self_critic_evaluator_registry is None:
            self._self_critic_evaluator_registry = {}

        # Initialize RE-PLAN checker registry
        if not hasattr(self, "_replan_checker_registry") or self._replan_checker_registry is None:
            self._replan_checker_registry = {}

        # Warn about deprecated default_model
        if self.default_model is not None:
            warnings.warn(
                "ReasoningContext.default_model is deprecated. Use per-step llm_config instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Register default self-critic evaluator to avoid race conditions
        self._register_default_self_critic_evaluator()

        # Check if api is already an LLMClientBase (e.g., OpenAICompatibleClient)
        self._llm_client = self.api

        # Initialize default namespace for memory
        if "default" not in self.memory:
            self.memory["default"] = {}

    @property
    def llm_client(self) -> LLMClientBase:
        """Get the default LLM client (creates it if not already created)."""
        return self.api

    def get_llm_client_for_step(self, llm_config: Optional[LLMStepConfig] = None) -> LLMClientBase:
        """
        Get an LLM client for a specific step, potentially with overrides.

        Resolution order (highest → lowest priority):
        1. Per-step ``llm_config`` (merged on top of chain default)
        2. Chain-level ``default_llm_config`` (set via ``ReasoningChain(default_llm_config=...)``
           and injected into ``context.metadata["__default_llm_config"]``)
        3. Context default client (``context.api``)

        Args:
            llm_config: Optional per-step LLM configuration

        Returns:
            LLM client with appropriate configuration
        """
        # Merge per-step config on top of chain-level default when both present
        chain_default: Optional[LLMStepConfig] = self.metadata.get("__default_llm_config")
        if llm_config is None:
            llm_config = chain_default
        elif chain_default is not None:
            # Per-step config wins; fill unset fields from chain default
            llm_config = LLMStepConfig(
                model=llm_config.model if llm_config.model is not None else chain_default.model,
                temperature=llm_config.temperature if llm_config.temperature is not None else chain_default.temperature,
                max_tokens=llm_config.max_tokens if llm_config.max_tokens is not None else chain_default.max_tokens,
                execution_mode=llm_config.execution_mode,
                self_critic_evaluators=llm_config.self_critic_evaluators,
                self_critic_max_revisions=llm_config.self_critic_max_revisions,
                self_critic_instruction=llm_config.self_critic_instruction,
                self_critic_disapprove_feedback=llm_config.self_critic_disapprove_feedback,
            )

        # If no override, return default client
        if llm_config is None:
            return self.llm_client

        # Check if we need to create an overridden client
        has_override = (
            llm_config.model is not None or llm_config.temperature is not None or llm_config.max_tokens is not None
        )

        if not has_override:
            return self.llm_client

        # For OpenAI-compatible clients, handle model/temperature/max_tokens overrides
        if isinstance(self._llm_client, OpenAICompatibleClient):
            # Create a cache key based on the override parameters
            cache_key = f"openai:{llm_config.model or ''}:{llm_config.temperature or ''}:{llm_config.max_tokens or ''}"

            if cache_key not in self._llm_client_cache:
                # Create a new client with overridden config
                base_config = self._llm_client.config

                new_config = OpenAIClientConfig(
                    base_url=base_config.base_url,
                    api_key=base_config.api_key,
                    model=llm_config.model or base_config.model,
                    temperature=llm_config.temperature
                    if llm_config.temperature is not None
                    else base_config.temperature,
                    max_tokens=llm_config.max_tokens if llm_config.max_tokens is not None else base_config.max_tokens,
                    timeout=base_config.timeout,
                    verify_ssl=base_config.verify_ssl,
                    extra_headers=base_config.extra_headers,
                    extra_body=base_config.extra_body,
                )
                self._llm_client_cache[cache_key] = OpenAICompatibleClient(new_config)

            return self._llm_client_cache[cache_key]

        return self.llm_client

    # Regex to extract result content from verbose step history entries.
    # Matches "Result: " (English) or "Результат: " (Russian) and captures everything after.
    _RESULT_PREFIX_RE: ClassVar[re.Pattern] = re.compile(
        r"(?:Result|Результат): (.*)",
        re.DOTALL,
    )

    @staticmethod
    def _compress_entry(entry: str) -> str:
        """
        Compress a history entry by stripping its verbose step header.

        Entries are formatted as::

            Step N. Title [optional mode]\nResult: <content>\n

        Compression extracts just ``<content>``, removing the step header.
        If the entry doesn't match the expected format it is returned unchanged.
        """
        m = ReasoningContext._RESULT_PREFIX_RE.search(entry)
        if m:
            return m.group(1).rstrip("\n")
        return entry

    def add_to_history(self, entry: str) -> None:
        """
        Add a new entry to the reasoning history.

        Behaviour depends on ``trim_strategy``:

        - ``"oldest"`` (default): when ``max_history_entries`` is exceeded, the oldest
          entries are dropped (FIFO).
        - ``"compress"``: before appending, compress ALL existing entries that are still
          in verbose format (strip their ``"Step N. Title\\nResult: "`` header, keeping
          only the result content). This reduces per-entry token cost without losing
          information. Oldest entries are still dropped when the limit is exceeded.
        """
        if self.trim_strategy == "compress":
            # Compress existing entries in place (idempotent — already-compressed
            # entries without the header pattern are returned unchanged).
            self.history = [self._compress_entry(e) for e in self.history]

        self.history.append(entry)

        # Trim history if limit is set
        if self.max_history_entries > 0 and len(self.history) > self.max_history_entries:
            # Keep the most recent entries
            self.history = self.history[-self.max_history_entries :]

    def get_current_history(self) -> str:
        """Get the current reasoning history as a single string."""
        return "\n".join(self.history)

    # === Cancellation Methods ===

    def cancel(self) -> None:
        """Request cancellation of the running chain.

        When called on a parent context that has already spawned parallel
        snapshots, the cancel flips through the shared token so every
        in-flight snapshot also observes the cancellation on its next
        ``is_cancelled()`` poll.
        """
        self._cancel_token.requested = True

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_token.requested

    def is_cancellation_requested(self) -> bool:
        """Spec-mirror alias of :meth:`is_cancelled` — kept so CARE
        (and code lifted from the design doc) reads naturally.
        """
        return self._cancel_token.requested

    def reset_cancellation(self) -> None:
        """Reset cancellation flag for a new execution."""
        self._cancel_token.requested = False

    def request_cancellation(self) -> None:
        """Spec-mirror alias of :meth:`cancel`."""
        self._cancel_token.requested = True

    # === Pause / Resume Methods ===

    def request_pause(self) -> None:
        """Request a pause of the running chain.

        The DAG executor checks the flag between batches; in-flight
        steps complete normally, then execution awaits
        :meth:`wait_for_resume` until :meth:`clear_pause` is called.
        Snapshots created during a parallel batch share the same
        pause token so a single call reaches every participant.
        """
        self._pause_token.requested = True
        # Block any subsequent ``wait_for_resume`` callers.
        self._pause_token.event.clear()

    def is_pause_requested(self) -> bool:
        """Return whether a pause has been requested but not yet cleared."""
        return self._pause_token.requested

    def clear_pause(self) -> None:
        """Resume a paused run. Releases every coroutine awaiting
        :meth:`wait_for_resume`. Safe to call when no pause is active.
        """
        self._pause_token.requested = False
        # Wake everyone awaiting resume.
        self._pause_token.event.set()

    async def wait_for_resume(self) -> None:
        """Block until the pause flag is cleared.

        When no pause is active, returns immediately (the event starts
        in the *set* state). When :meth:`request_pause` flips the
        flag, this method awaits the event until :meth:`clear_pause`
        sets it again.
        """
        await self._pause_token.event.wait()

    # === Step completion tracking ===

    def record_step_executed(self, step_number: int) -> None:
        """Record that a step finished successfully.

        Called by :class:`DAGExecutor` after every successful step
        completion. Captured into :class:`ContextSnapshot` so a
        :meth:`ReasoningChain.execute_async(resume_from=...)` call can
        skip already-completed work.
        """
        if step_number not in self._executed_step_numbers:
            self._executed_step_numbers.append(int(step_number))

    def get_executed_step_numbers(self) -> list[int]:
        """Return the list of step numbers that have completed in this
        context's execution so far.
        """
        return list(self._executed_step_numbers)

    # === Snapshot / Restore ===

    def snapshot(self) -> "ContextSnapshot":
        """Capture the mutable state of this context as a
        :class:`ContextSnapshot`.

        Framework-internal ``__``-prefixed metadata keys (Langfuse trace
        handles, replan buffers, etc.) are stripped — matches the
        `RunRecord.from_run` convention so the snapshot is safe to
        persist across processes.
        """
        clean_metadata = {
            k: v for k, v in self.metadata.items()
            if not (isinstance(k, str) and k.startswith("__"))
        }
        return ContextSnapshot(
            outer_context=self.outer_context,
            history=list(self.history),
            memory={
                ns: dict(values) if isinstance(values, dict) else values
                for ns, values in self.memory.items()
            },
            metadata=clean_metadata,
            messages=[
                m.model_dump() if hasattr(m, "model_dump") else dict(m)
                for m in self.messages
            ],
            cancelled=self._cancel_token.requested,
            completed_step_numbers=list(self._executed_step_numbers),
        )

    def restore(self, snapshot: "ContextSnapshot") -> None:
        """Restore mutable state from a :class:`ContextSnapshot`.

        Replaces ``outer_context`` / ``history`` / ``memory`` /
        ``metadata`` / ``messages`` on this context. Runtime-only state
        (LLM client, tool registry, callbacks) is left untouched — the
        caller has typically just constructed the context with those
        wired up before calling :meth:`restore`.
        """
        self.outer_context = snapshot.outer_context
        self.history = list(snapshot.history)
        self.memory = {
            ns: dict(values) if isinstance(values, dict) else values
            for ns, values in snapshot.memory.items()
        }
        self.metadata = dict(snapshot.metadata)
        self.messages = [
            ChatMessage(**m) if isinstance(m, dict) else m
            for m in snapshot.messages
        ]
        if snapshot.cancelled:
            self._cancel_token.requested = True
        self._executed_step_numbers = list(snapshot.completed_step_numbers)

    # === Human-Input Methods ===

    def provide_human_input(self, value: str) -> bool:
        """
        Resolve a pending ``HumanInputStep`` with the given value.

        Call this from outside the chain (e.g. from a UI event handler or CLI
        prompt) after ``on_human_input_requested`` fires.  If no step is
        currently waiting, returns ``False`` and the value is silently ignored.

        Returns
        -------
        bool
            ``True`` if a pending future was resolved; ``False`` otherwise.
        """
        future = self._human_input_future
        if future is not None and not future.done():
            future.set_result(value)
            return True
        return False

    # === Tool Registry Methods ===

    def register_tool(
        self,
        tool_name: str,
        tool_callable: Callable,
        *,
        timeout: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Register a tool/function for use in TOOL steps.

        Args:
            tool_name: Name of the tool (must match ToolStepConfig.tool_name)
            tool_callable: The callable to execute (sync or async)
            timeout: Optional per-call timeout in seconds. When set on a
                *synchronous* callable, the tool is automatically wrapped with
                :class:`~mmar_carl.step_executors.AsyncToolWrapper` so it runs
                in a thread pool (via ``asyncio.to_thread``) with the given
                timeout enforced by ``asyncio.wait_for``. Has no effect on async
                callables since those should manage their own timeouts.
            tags: Optional list of tags for grouping / restricting tool exposure.
                A step configured with ``allowed_tool_tags=["math"]`` will
                refuse to call a tool that doesn't have at least one matching
                tag. Useful for keeping destructive or sensitive tools out of
                steps that only need read-only / informational access.

        Note:
            Tools MUST be stateless for safe parallel execution.
            If a tool has instance state, it may cause race conditions
            when multiple steps execute in parallel.
        """
        # Warn if tool appears to have instance state
        if hasattr(tool_callable, "__self__") and hasattr(tool_callable.__self__, "__dict__"):
            if tool_callable.__self__.__dict__:
                warnings.warn(
                    f"Tool '{tool_name}' has instance state and may not be safe for parallel execution. "
                    "Ensure tools are stateless or use appropriate locking.",
                    UserWarning,
                    stacklevel=2,
                )

        # Auto-wrap sync callables when a timeout is requested
        if timeout is not None and not asyncio.iscoroutinefunction(tool_callable):
            from ..step_executors import AsyncToolWrapper  # type: ignore[attr-defined]
            tool_callable = AsyncToolWrapper(tool_callable, timeout=timeout)

        self._tool_registry[tool_name] = tool_callable
        # Always set tags (overwrites prior tags on re-registration)
        self._tool_tags[tool_name] = set(tags) if tags else set()

    def register_tools_from_path(
        self,
        glob_pattern: str,
        *,
        tag_filter: Optional[list[str]] = None,
        name_prefix: str = "",
    ) -> list[str]:
        """Discover and register ``@carl_tool``-decorated callables from disk.

        lets users keep their tool implementations in a
        directory tree (e.g. ``~/.config/care/tools/*.py``) and register
        them all at startup with one call. Each Python file matching
        ``glob_pattern`` is loaded as a standalone module; every public
        callable carrying ``__carl_tool__ = True`` (set by the
        :func:`~mmar_carl.step_executors.carl_tool` decorator) is
        registered via :meth:`register_tool` under its function name.

        Args:
            glob_pattern: Filesystem glob expanded via :func:`glob.glob`
                with ``recursive=True`` so ``"**/*.py"`` matches nested
                directories. ``~`` is expanded to the user's home dir.
            tag_filter: Optional whitelist — only tools whose
                ``__carl_tool_tags__`` set intersects the filter are
                registered. When ``None`` (default), all decorated tools
                are registered regardless of tags.
            name_prefix: Optional prefix prepended to each registered
                tool name (useful for namespacing third-party tool
                directories so they don't collide with built-ins).

        Returns:
            The list of registered tool names, in discovery order.

        Notes
        -----
        - Files that fail to import are skipped silently (defensive —
          one broken plugin shouldn't take down the whole catalog). Per-
          file failures can be surfaced by setting log level to DEBUG.
        - Duplicate tool names overwrite prior registrations (same
          semantics as :meth:`register_tool`). Use ``name_prefix`` to
          avoid collisions between directories.
        - Tags from ``@carl_tool(tags=[...])`` are preserved in the
          registry, so :meth:`get_tool_tags` works on tools discovered
          this way.
        """
        import glob as _glob
        import importlib.util as _importlib_util
        import logging as _logging
        import os as _os

        log = _logging.getLogger("mmar_carl")
        expanded = _os.path.expanduser(glob_pattern)
        registered: list[str] = []
        filter_set = set(tag_filter) if tag_filter else None

        for file_path in _glob.glob(expanded, recursive=True):
            if not file_path.endswith(".py"):
                continue
            module_name = (
                f"_carl_tools_{abs(hash(file_path))}_"
                f"{_os.path.basename(file_path).removesuffix('.py')}"
            )
            try:
                spec = _importlib_util.spec_from_file_location(
                    module_name, file_path,
                )
                if spec is None or spec.loader is None:
                    continue
                module = _importlib_util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except Exception as exc:
                log.debug(
                    "register_tools_from_path: skipping %s — %s",
                    file_path, exc,
                )
                continue

            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name, None)
                if not callable(attr):
                    continue
                if not getattr(attr, "__carl_tool__", False):
                    continue
                tool_tags = list(getattr(attr, "__carl_tool_tags__", []) or [])
                if filter_set is not None and not (filter_set & set(tool_tags)):
                    continue
                tool_name = f"{name_prefix}{attr_name}"
                self.register_tool(tool_name, attr, tags=tool_tags or None)
                registered.append(tool_name)
        return registered

    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a registered tool by name."""
        return self._tool_registry.get(tool_name)

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tool_registry

    def get_tool_tags(self, tool_name: str) -> set[str]:
        """Return the tags associated with *tool_name* (empty set if untagged or unknown)."""
        return set(self._tool_tags.get(tool_name, set()))

    def register_tool_pipeline(
        self,
        pipeline_name: str,
        steps: list[tuple[str, dict[str, Any]]],
        *,
        tags: Optional[list[str]] = None,
    ) -> None:
        """
        Register a sequential pipeline of registered tools as a single new tool.

        Each step is a ``(tool_name, arg_template)`` pair. ``arg_template`` values
        are interpreted as follows:

        - ``"$input"`` — replaced with the dict of kwargs passed to the pipeline.
        - ``"$input.<key>"`` — replaced with ``kwargs[<key>]`` (dotted path supported
          for nested dicts: ``"$input.user.id"``).
        - ``"$prev_output"`` — replaced with the previous step's raw return value.
        - ``"$prev_output.<key>"`` — dotted-path lookup into the previous output.
        - any other value — passed through unchanged (literals, ints, dicts, …).

        The first step's ``$prev_output`` references resolve to ``None``.

        Pipelines call each registered tool through :py:meth:`get_tool` so they
        respect ``AsyncToolWrapper`` timeouts and accept both sync and async
        callables. A pipeline raises ``ValueError`` if any referenced tool is
        unregistered when the pipeline runs, and propagates any underlying
        tool exception.

        The returned pipeline is registered under ``pipeline_name`` and is callable
        from :class:`ToolStepConfig` just like any other tool.

        Args:
            pipeline_name: Name to register the pipeline under.
            steps: List of ``(tool_name, arg_template_dict)`` pairs to run in order.
            tags: Optional tags for the pipeline tool itself (inherits the same
                tag-whitelist semantics as :py:meth:`register_tool`).

        Raises:
            ValueError: If ``steps`` is empty or a step entry is malformed.
        """
        if not steps:
            raise ValueError("register_tool_pipeline requires at least one step")
        for i, entry in enumerate(steps):
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(entry[1], dict)
            ):
                raise ValueError(
                    f"Pipeline '{pipeline_name}' step {i} must be a (tool_name: str, args: dict) tuple"
                )

        # Snapshot the spec so later mutation of the input list cannot reshape
        # an already-registered pipeline.
        spec: list[tuple[str, dict[str, Any]]] = [(name, dict(args)) for name, args in steps]

        async def _pipeline_runner(**kwargs: Any) -> Any:
            prev_output: Any = None
            for tool_name, arg_template in spec:
                resolved = {
                    pname: _resolve_pipeline_ref(raw, kwargs, prev_output)
                    for pname, raw in arg_template.items()
                }
                callable_ = self.get_tool(tool_name)
                if callable_ is None:
                    raise ValueError(
                        f"Pipeline '{pipeline_name}' references unregistered tool '{tool_name}'"
                    )
                if asyncio.iscoroutinefunction(callable_) or asyncio.iscoroutinefunction(
                    getattr(type(callable_), "__call__", None)
                ):
                    prev_output = await callable_(**resolved)
                else:
                    prev_output = await asyncio.to_thread(callable_, **resolved)
            return prev_output

        # Mark the pipeline so callers / debuggers can identify it
        _pipeline_runner.__name__ = f"pipeline:{pipeline_name}"  # type: ignore[attr-defined]
        _pipeline_runner.is_pipeline = True  # type: ignore[attr-defined]
        _pipeline_runner.pipeline_steps = tuple(  # type: ignore[attr-defined]
            (n, dict(a)) for n, a in spec
        )
        self.register_tool(pipeline_name, _pipeline_runner, tags=tags)

    def list_tools(
        self,
        *,
        tags: Optional[list[str]] = None,
        match_all: bool = False,
    ) -> list[str]:
        """
        List registered tool names, optionally filtered by tag.

        Args:
            tags: If provided, only return tools whose tag set intersects this
                list (``match_all=False``, default) or contains every entry
                (``match_all=True``). When ``None``, return every registered tool.
            match_all: When ``True``, require *every* tag in ``tags`` to be
                present on the tool. When ``False``, any single tag match is
                enough.
        """
        if not tags:
            return list(self._tool_registry.keys())

        wanted = set(tags)
        if match_all:
            return [
                name
                for name in self._tool_registry
                if wanted.issubset(self._tool_tags.get(name, set()))
            ]
        return [
            name
            for name in self._tool_registry
            if wanted & self._tool_tags.get(name, set())
        ]

    # === Event Bus Methods ===

    def emit_event(self, name: str, payload: Any = None) -> None:
        """
        Emit a named event with an optional payload.

        Subsequent steps that declare ``triggered_by=[name]`` become eligible
        for execution once their numeric dependencies are also satisfied.
        The payload (last value wins if the same name is re-emitted) is
        readable via :py:meth:`get_event_payload` and through the
        ``$event.<name>`` reference syntax.

        Args:
            name: Event name. Empty strings are rejected.
            payload: Optional value associated with the event. Defaults to
                ``None`` for fire-and-forget signals.

        Raises:
            ValueError: If *name* is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise ValueError("Event name cannot be empty")
        self._emitted_events[name] = payload

    def get_event_payload(self, name: str, default: Any = None) -> Any:
        """Return the last payload emitted for *name*, or *default* if not seen."""
        return self._emitted_events.get(name, default)

    def emit_step_event(
        self,
        step_number: int,
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Fire ``on_step_event`` if a callback is registered.

        Generic dispatch for **intra-step** progress events — distinct
        from the chain-level event bus (``emit_event``). Used by
        executors that perform multiple sub-operations inside one step
        (LLM_AGENT tool calls, Debate rounds, ParallelSampling
        samples, Supervisor route selection).

        Args:
            step_number: The chain step that produced the event.
            event_type: Dotted-namespace identifier (e.g.
                ``'llm_agent.tool_call'``). See the
                :attr:`on_step_event` docstring for the canonical
                list.
            payload: Free-form dict carrying event details.

        Exceptions raised by the consumer callback are swallowed (and
        logged at WARNING level) so a misbehaving instrumentation
        consumer can't crash the run. Mirrors the
        ``on_step_complete`` contract.
        """
        if self.on_step_event is None:
            return
        data = payload or {}
        try:
            self.on_step_event(step_number, event_type, data)
        except Exception as exc:
            try:
                from ..logging_utils import log_warning  # noqa: PLC0415
                log_warning(
                    f"on_step_event callback raised on "
                    f"(step={step_number}, event_type={event_type!r}): {exc}"
                )
            except Exception:
                pass

    def has_event(self, name: str) -> bool:
        """Check whether *name* has been emitted (at least once)."""
        return name in self._emitted_events

    def event_names(self) -> list[str]:
        """List of all event names that have been emitted so far."""
        return list(self._emitted_events.keys())

    # === MCP Tool Auto-Discovery ===

    async def register_mcp_tools(
        self,
        server: Any,
        *,
        prefix: Optional[str] = None,
        timeout: float = 30.0,
        tags: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Discover all tools exposed by an MCP server and register them as CARL tools.

        Bridges the gap between MCP and the Tool step type: any MCP tool becomes
        callable via ``ToolStepConfig(tool_name="mcp:server/tool")`` (or whatever
        naming you choose with ``prefix``).

        Connects to the server (via the same transport infrastructure as
        :class:`MCPStepExecutor`), calls ``session.list_tools()``, and for
        each returned tool registers a wrapper through
        :py:meth:`register_tool` that — when invoked — opens a fresh MCP
        connection and dispatches the call. Connections are NOT pooled: each
        tool invocation is independent. This trades some per-call overhead
        for simplicity and reliability (no shared-state surprises across
        parallel chain steps).

        Args:
            server: An ``MCPServerConfig`` describing the server (transport,
                command, url, headers, …). Use the type from
                ``mmar_carl.models.config``.
            prefix: Tool-name prefix template. The default
                ``f"mcp:{server.server_name}/"`` produces names like
                ``"mcp:docs/search"``. Pass an empty string to use the raw
                tool names from the server (collisions with existing tools
                are then the caller's responsibility — :py:meth:`register_tool`
                will silently overwrite). Pass a custom string to use it as a
                literal prefix.
            timeout: Per-call timeout (seconds) for both the initial
                ``list_tools`` and every subsequent ``call_tool`` invocation.
            tags: Optional list of tags applied to every discovered tool.
                Lets callers gate the MCP tools behind ``allowed_tool_tags``
                (e.g. ``tags=["mcp", server.server_name, "external"]``).

        Returns:
            List of the CARL tool names that were registered (e.g.
            ``["mcp:docs/search", "mcp:docs/fetch"]``).

        Raises:
            ImportError: If the ``mcp`` SDK isn't installed.
            RuntimeError: If the server returns a malformed ``list_tools``
                response (no ``tools`` attribute / non-iterable).
        """
        try:
            from mcp import ClientSession  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "MCP SDK not installed. Install with: pip install mcp"
            ) from exc

        if prefix is None:
            prefix = f"mcp:{server.server_name}/"

        # Open one short-lived session to discover the tool list.
        tools_list = await self._mcp_list_tools(server, timeout=timeout)

        registered: list[str] = []
        for tool_meta in tools_list:
            raw_name = getattr(tool_meta, "name", None)
            if not raw_name:
                continue
            full_name = f"{prefix}{raw_name}"

            # Each wrapper captures the raw tool name and the server config so
            # later invocations know which tool to dispatch. We bind via
            # default-arg trick to avoid late-binding gotchas in the loop.
            async def _mcp_tool_wrapper(
                _server: Any = server,
                _raw_name: str = raw_name,
                _timeout: float = timeout,
                **kwargs: Any,
            ) -> Any:
                return await self._mcp_call_tool(
                    _server, _raw_name, kwargs, timeout=_timeout
                )

            _mcp_tool_wrapper.__name__ = f"mcp_tool:{full_name}"  # type: ignore[attr-defined]
            _mcp_tool_wrapper.is_mcp_tool = True  # type: ignore[attr-defined]
            _mcp_tool_wrapper.mcp_server_name = server.server_name  # type: ignore[attr-defined]
            _mcp_tool_wrapper.mcp_raw_tool_name = raw_name  # type: ignore[attr-defined]

            self.register_tool(full_name, _mcp_tool_wrapper, tags=tags)
            registered.append(full_name)

        return registered

    def mcp_pool(self) -> Any:
        """Return an MCP session pool bound to this context.

        Use as an ``async with`` block. While the pool is open, every
        MCP call (``register_mcp_tools``, ``_mcp_call_tool``,
        ``_mcp_list_tools``, ``list_mcp_resources``) reuses one
        :class:`mcp.ClientSession` per server — keyed by
        ``(server_name, transport, command-or-url)`` — instead of
        opening a fresh transport per call.

        Cleanup happens automatically on exit: every session opened
        through the pool is exited in the same task that opened it.

        Example::

            async with context.mcp_pool() as pool:
                result = await chain.execute_async(context)
                # `pool.stats()` reports how many sessions were opened.
        """
        from ..mcp_pool import MCPSessionPool  # noqa: PLC0415

        # Wrap the pool in a class-level async-CM adapter — `async with`
        # looks up `__aenter__` / `__aexit__` on the TYPE, not the
        # instance, so per-instance method assignment doesn't fire.
        ctx = self

        class _ContextBoundPool:
            __slots__ = ("_pool",)

            def __init__(self) -> None:
                self._pool = MCPSessionPool()

            async def __aenter__(self) -> "MCPSessionPool":
                entered = await self._pool.__aenter__()
                ctx._mcp_pool = entered
                return entered

            async def __aexit__(self, exc_type, exc, tb) -> Any:
                try:
                    return await self._pool.__aexit__(exc_type, exc, tb)
                finally:
                    ctx._mcp_pool = None

        return _ContextBoundPool()

    async def _mcp_list_tools(self, server: Any, *, timeout: float) -> list[Any]:
        """Open a short-lived MCP session and call ``list_tools()``.

        when ``self._mcp_pool`` is active, the pooled
        session is reused; otherwise a per-call session is opened
        (legacy behaviour).
        """
        import asyncio as _asyncio

        from mcp import ClientSession

        # Pooled path: reuse a long-lived session.
        if self._mcp_pool is not None:
            session = await self._mcp_pool.acquire(server, timeout=timeout)
            result = await _asyncio.wait_for(
                session.list_tools(), timeout=timeout,
            )
            tools = getattr(result, "tools", None)
            if tools is None or not hasattr(tools, "__iter__"):
                raise RuntimeError(
                    f"MCP server '{server.server_name}' returned a malformed "
                    f"list_tools response (no 'tools' attribute)."
                )
            return list(tools)

        transport = server.transport

        async def _with_session(read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await _asyncio.wait_for(session.list_tools(), timeout=timeout)
                tools = getattr(result, "tools", None)
                if tools is None or not hasattr(tools, "__iter__"):
                    raise RuntimeError(
                        f"MCP server '{server.server_name}' returned a malformed "
                        f"list_tools response (no 'tools' attribute)."
                    )
                return list(tools)

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=server.command or "", args=server.args
            )
            async with stdio_client(params) as (read, write):
                return await _with_session(read, write)

        if transport == "sse":
            from mcp.client.sse import sse_client

            if not server.url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='sse'."
                )
            async with sse_client(
                url=server.url,
                headers=dict(server.headers) if server.headers else None,
                timeout=timeout,
            ) as (read, write):
                return await _with_session(read, write)

        if transport == "http":
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            if not server.url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='http'."
                )
            http_client = httpx.AsyncClient(
                headers=dict(server.headers) if server.headers else {},
                timeout=timeout,
            )
            async with streamable_http_client(
                url=server.url, http_client=http_client
            ) as (read, write, _get_session_id):
                return await _with_session(read, write)

        raise NotImplementedError(
            f"MCP transport '{transport}' is not supported. "
            "Use one of: 'stdio', 'http', 'sse'."
        )

    async def list_mcp_resources(
        self, server: Any, *, timeout: float = 30.0,
    ) -> list[Any]:
        """List every resource exposed by an MCP server.

        used by CARE's catalog screen to render what
        each configured MCP server offers (read-only data sources like
        ``docs://`` URIs, file lists, etc.). Distinct from
        :py:meth:`register_mcp_tools` which discovers callable *tools*.

        Args:
            server: An ``MCPServerConfig`` describing the server
                (transport, command, url, headers, …).
            timeout: Per-call timeout (seconds) for the ``list_resources``
                call.

        Returns:
            List of resource descriptors as returned by the server.
            Each entry typically carries ``uri``, ``name``, ``description``,
            and ``mimeType`` fields per the MCP spec.

        Raises:
            ImportError: If the ``mcp`` SDK isn't installed.
            RuntimeError: If the server returns a malformed
                ``list_resources`` response.
        """
        import asyncio as _asyncio

        from mcp import ClientSession

        # reuse pooled session if an MCP pool is active.
        if self._mcp_pool is not None:
            session = await self._mcp_pool.acquire(server, timeout=timeout)
            result = await _asyncio.wait_for(
                session.list_resources(), timeout=timeout,
            )
            resources = getattr(result, "resources", None)
            if resources is None or not hasattr(resources, "__iter__"):
                raise RuntimeError(
                    f"MCP server '{server.server_name}' returned a "
                    f"malformed list_resources response (no "
                    f"'resources' attribute)."
                )
            return list(resources)

        transport = server.transport

        async def _with_session(read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await _asyncio.wait_for(
                    session.list_resources(), timeout=timeout,
                )
                resources = getattr(result, "resources", None)
                if resources is None or not hasattr(resources, "__iter__"):
                    raise RuntimeError(
                        f"MCP server '{server.server_name}' returned a "
                        f"malformed list_resources response (no "
                        f"'resources' attribute)."
                    )
                return list(resources)

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=server.command or "", args=server.args,
            )
            async with stdio_client(params) as (read, write):
                return await _with_session(read, write)

        if transport == "sse":
            from mcp.client.sse import sse_client

            if not server.url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='sse'."
                )
            async with sse_client(
                url=server.url,
                headers=dict(server.headers) if server.headers else None,
                timeout=timeout,
            ) as (read, write):
                return await _with_session(read, write)

        if transport == "http":
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            if not server.url:
                raise ValueError(
                    "MCPServerConfig.url is required for transport='http'."
                )
            http_client = httpx.AsyncClient(
                headers=dict(server.headers) if server.headers else {},
                timeout=timeout,
            )
            async with streamable_http_client(
                url=server.url, http_client=http_client,
            ) as (read, write, _get_session_id):
                return await _with_session(read, write)

        raise NotImplementedError(
            f"MCP transport '{transport}' is not supported. "
            "Use one of: 'stdio', 'http', 'sse'."
        )

    async def _mcp_call_tool(
        self, server: Any, tool_name: str, arguments: dict, *, timeout: float
    ) -> Any:
        """Open a short-lived MCP session and call ``tool_name`` with arguments.

        reuses pooled session when ``self._mcp_pool`` is
        active; otherwise falls back to the legacy per-call session.
        """
        import asyncio as _asyncio

        from mcp import ClientSession

        # Pooled path.
        if self._mcp_pool is not None:
            session = await self._mcp_pool.acquire(server, timeout=timeout)
            result = await _asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout,
            )
            return result.content

        transport = server.transport

        async def _call_via(read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await _asyncio.wait_for(
                    session.call_tool(tool_name, arguments=arguments),
                    timeout=timeout,
                )
                return result.content

        if transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(
                command=server.command or "", args=server.args
            )
            async with stdio_client(params) as (read, write):
                return await _call_via(read, write)

        if transport == "sse":
            from mcp.client.sse import sse_client

            if not server.url:
                raise ValueError("MCPServerConfig.url is required for transport='sse'.")
            async with sse_client(
                url=server.url,
                headers=dict(server.headers) if server.headers else None,
                timeout=timeout,
            ) as (read, write):
                return await _call_via(read, write)

        if transport == "http":
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            if not server.url:
                raise ValueError("MCPServerConfig.url is required for transport='http'.")
            http_client = httpx.AsyncClient(
                headers=dict(server.headers) if server.headers else {},
                timeout=timeout,
            )
            async with streamable_http_client(
                url=server.url, http_client=http_client
            ) as (read, write, _get_session_id):
                return await _call_via(read, write)

        raise NotImplementedError(
            f"MCP transport '{transport}' is not supported. "
            "Use one of: 'stdio', 'http', 'sse'."
        )

    # === Self-Critic Evaluator Registry Methods ===

    def register_self_critic_evaluator(self, name: str, evaluator: SelfCriticEvaluatorBase) -> None:
        """
        Register a self-critic evaluator strategy by name.

        Args:
            name: Evaluator name referenced by LLMStepConfig.self_critic_evaluators
            evaluator: Evaluator strategy implementation
        """
        if not name or not name.strip():
            raise ValueError("Self-critic evaluator name cannot be empty")
        self._self_critic_evaluator_registry[name.strip()] = evaluator

    def get_self_critic_evaluator(self, name: str) -> Optional[SelfCriticEvaluatorBase]:
        """Get a registered self-critic evaluator by name."""
        return self._self_critic_evaluator_registry.get(name)

    def list_self_critic_evaluators(self) -> list[str]:
        """List all registered self-critic evaluator names."""
        return list(self._self_critic_evaluator_registry.keys())

    # === RE-PLAN Checker Registry Methods ===

    def register_replan_checker(self, name: str, checker: ReplanCheckerBase) -> None:
        """
        Register a RE-PLAN checker strategy by name.

        Args:
            name: Checker name referenced by RegisteredReplanCheckerConfig.name
            checker: Checker strategy implementation
        """
        if not name or not name.strip():
            raise ValueError("RE-PLAN checker name cannot be empty")
        self._replan_checker_registry[name.strip()] = checker

    def get_replan_checker(self, name: str) -> Optional[ReplanCheckerBase]:
        """Get a registered RE-PLAN checker by name."""
        return self._replan_checker_registry.get(name)

    def list_replan_checkers(self) -> list[str]:
        """List all registered RE-PLAN checker names."""
        return list(self._replan_checker_registry.keys())

    # === Default Self-Critic Evaluator Registration ===

    def _register_default_self_critic_evaluator(self) -> None:
        """Register the built-in 'llm' self-critic evaluator if not already present."""
        # Lazy import to avoid circular dependency
        from mmar_carl.step_executors import LLMSelfCriticEvaluator

        default_name = "llm"
        if self.get_self_critic_evaluator(default_name) is None:
            self.register_self_critic_evaluator(default_name, LLMSelfCriticEvaluator())

    # === Memory Methods ===

    def memory_read(self, key: str, namespace: str = "default", default: Any = None) -> Any:
        """Read a value from memory."""
        ns = self.memory.get(namespace, {})
        return ns.get(key, default)

    def memory_write(self, key: str, value: Any, namespace: str = "default") -> None:
        """Write a value to memory.

        When ``self.memory_schema`` declares this ``(namespace, key)`` pair,
        the value is type-checked first and :class:`MemorySchemaError` is
        raised on a mismatch (no write occurs).
        """
        if self.memory_schema is not None:
            from ..memory_schema import validate_memory_write
            validate_memory_write(self.memory_schema, namespace, key, value)
        if namespace not in self.memory:
            self.memory[namespace] = {}
        self.memory[namespace][key] = value

    def memory_append(self, key: str, value: Any, namespace: str = "default") -> None:
        """Append a value to a list in memory (creates list if not exists).

        When ``self.memory_schema`` declares this ``(namespace, key)`` pair as
        a ``list`` (or a list-of-X), the new value is checked: if the declared
        type is a plain ``list``, the element is allowed unconditionally; if
        the declared type is a parameterised ``list[T]``, the element is
        type-checked against ``T``.
        """
        if self.memory_schema is not None:
            self._validate_append(namespace, key, value)
        if namespace not in self.memory:
            self.memory[namespace] = {}
        if key not in self.memory[namespace]:
            self.memory[namespace][key] = []
        if isinstance(self.memory[namespace][key], list):
            self.memory[namespace][key].append(value)
        else:
            raise ValueError(f"Memory key '{key}' is not a list")

    def _validate_append(self, namespace: str, key: str, value: Any) -> None:
        """Schema check for append: the existing/future container must be list,
        and (when declared as ``list[T]``) the appended element must match T."""
        import typing
        from ..memory_schema import MemorySchemaError

        if self.memory_schema is None:
            return
        ns_spec = self.memory_schema.get(namespace)
        if ns_spec is None or key not in ns_spec:
            return
        spec = ns_spec[key]
        origin = typing.get_origin(spec)
        if origin is list or spec is list:
            args = typing.get_args(spec)
            if args:
                element_type = args[0]
                # Recursively normalize element type for Union/Optional support
                from ..memory_schema import _normalize_type_spec
                expected = _normalize_type_spec(element_type)
                if not isinstance(value, expected):
                    raise MemorySchemaError(
                        namespace, f"{key}[]", expected, value
                    )
            return
        # Schema declares a non-list type for this key — appending is wrong.
        raise MemorySchemaError(namespace, key, (list,), value)

    def memory_delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a value from memory. Returns True if key existed."""
        if namespace in self.memory and key in self.memory[namespace]:
            del self.memory[namespace][key]
            return True
        return False

    def memory_list(self, namespace: str = "default") -> list[str]:
        """List all keys in a memory namespace."""
        return list(self.memory.get(namespace, {}).keys())

    # === Long-term Memory Methods ===

    def remember(self, key: str, value: Any, *, session_id: str | None = None) -> None:
        """
        Persist *value* under *key* in the long-term memory store.

        Uses ``self.session_id`` by default; pass *session_id* explicitly to
        write into a different scope.

        Raises:
            RuntimeError: If ``long_term_memory`` is not set on this context.
        """
        if self.long_term_memory is None:
            raise RuntimeError(
                "context.remember() requires a long_term_memory store. "
                "Pass long_term_memory=InMemoryLTM() (or JsonFileLTM(...)) "
                "when constructing ReasoningContext."
            )
        sid = session_id if session_id is not None else self.session_id
        self.long_term_memory.store(key, value, session_id=sid)

    def recall(
        self,
        query: str,
        *,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search the long-term memory store for entries matching *query*.

        Uses substring search by default; the underlying LTM implementation may
        use vector search if configured.

        Args:
            query: Search term (case-insensitive substring by default).
            session_id: Override the session scope (default: ``self.session_id``).
            top_k: Maximum number of results to return.

        Returns:
            List of ``{"key": ..., "value": ..., "score": ...}`` dicts,
            ordered by descending relevance score.

        Raises:
            RuntimeError: If ``long_term_memory`` is not set on this context.
        """
        if self.long_term_memory is None:
            raise RuntimeError(
                "context.recall() requires a long_term_memory store. "
                "Pass long_term_memory=InMemoryLTM() (or JsonFileLTM(...)) "
                "when constructing ReasoningContext."
            )
        sid = session_id if session_id is not None else self.session_id
        return self.long_term_memory.search(query, session_id=sid, top_k=top_k)

    def ltm_retrieve(self, key: str, *, session_id: str | None = None) -> Any:
        """
        Retrieve a single value from the long-term memory store by exact key.

        Returns ``None`` if the key does not exist or if no LTM store is configured.
        """
        if self.long_term_memory is None:
            return None
        sid = session_id if session_id is not None else self.session_id
        return self.long_term_memory.retrieve(key, session_id=sid)

    async def close(self) -> None:
        """
        Close any open LLM clients and release resources.

        Should be called when the context is no longer needed, especially for
        OpenAI-compatible clients that maintain HTTP connections.

        Note: For proper cleanup when using asyncio.run(), the caller should
        wait for all background tasks after calling this method.

        Note: After calling close(), the context can still be used for a new
        execution - LLM clients will be recreated on demand.

        Example:
            ```python
            context = ReasoningContext(...)
            try:
                result = chain.execute(context)
            finally:
                await context.close()
            ```
        """

        # Close the main client
        if isinstance(self._llm_client, OpenAICompatibleClient):
            await self._llm_client.close()
        # FIX: Reset the main client so it can be recreated if context is reused
        self._llm_client = None

        # Close any cached clients
        for client in self._llm_client_cache.values():
            if isinstance(client, OpenAICompatibleClient):
                await client.close()

        self._llm_client_cache.clear()

    # ------------------------------------------------------------------
    # CARE library: prime a context from a saved chain's metadata
    #
    # ------------------------------------------------------------------

    @classmethod
    def from_chain_inputs(
        cls,
        chain: Any,
        *,
        api: Any,
        outer_context: str | None = None,
        files: dict[str, str] | None = None,
        load_files_from_metadata: bool = True,
        **kwargs: Any,
    ) -> "ReasoningContext":
        """Build a fresh :class:`ReasoningContext` from a saved chain.

        Pulls inputs from ``chain.get_care_metadata()`` when present:

        * ``outer_context`` ← explicit arg, else ``task_description`` from
          metadata, else an empty string.
        * For every entry in ``CareChainMetadata.context_files`` whose
          ``path`` still resolves on disk, the file's UTF-8 contents are
          loaded into ``context.memory["input"][<basename>]``. Files
          that don't exist anymore are silently skipped — the CARE TUI
          can re-render the missing list via :meth:`get_care_metadata`
          and prompt the user. Disable this auto-load with
          ``load_files_from_metadata=False``.
        * ``files=`` (an explicit ``{name: text}`` dict) augments /
          overrides the metadata-loaded files. Wins on key clash so
          callers can patch a single attachment without redoing the
          whole replay flow.

        Any extra ``**kwargs`` (e.g. ``language=``, ``system_prompt=``)
        are forwarded straight to the :class:`ReasoningContext`
        constructor.
        """
        meta = None
        if hasattr(chain, "get_care_metadata"):
            try:
                meta = chain.get_care_metadata()
            except Exception:
                meta = None

        if outer_context is None:
            outer_context = (
                meta.task_description if meta and meta.task_description else ""
            )

        memory: dict[str, dict[str, Any]] = {"input": {}}

        # Hydrate from metadata's recorded file paths first.
        if load_files_from_metadata and meta is not None:
            from pathlib import Path as _Path  # noqa: PLC0415
            for cf in meta.context_files:
                try:
                    p = _Path(cf.path)
                    if p.is_file():
                        memory["input"][p.name] = p.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    # Path moved, deleted, or binary — leave a placeholder
                    # so downstream steps know the slot was *expected* to
                    # be populated but the file is currently unreachable.
                    memory["input"].setdefault(
                        _Path(cf.path).name if cf.path else "missing",
                        f"[missing context file: {cf.path}]",
                    )

        # Caller-supplied files override / extend.
        if files:
            memory["input"].update(files)

        # If the caller passed memory= explicitly, merge rather than
        # clobber — their keys win.
        if "memory" in kwargs:
            caller_memory = kwargs.pop("memory") or {}
            for ns, kv in caller_memory.items():
                memory.setdefault(ns, {}).update(kv)

        return cls(
            outer_context=outer_context, api=api,
            memory=memory, **kwargs,
        )

    model_config = {"arbitrary_types_allowed": True}
