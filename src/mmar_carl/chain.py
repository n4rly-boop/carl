"""
Main chain definition and execution API for CARL.

Provides the primary interface for defining and executing reasoning chains.
Supports JSON serialization/deserialization for chain persistence.
"""

import asyncio
import json
import re
import warnings
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    from .cost import CostEstimate
    from .models.care_metadata import CareChainMetadata
    from .models.preflight import PreflightReport

from pydantic import BaseModel, Field

from .executor import DAGExecutor
from .models.dataset import DatasetEvaluationReport
from .models import (
    # Enums
    Language,
    # Step Configurations
    ConditionalBranch,
    ConditionalStepConfig,
    ContextQuery,
    ContextSearchConfig,
    ExecutionMode,
    LLMStepConfig,
    LoopConfig,
    MCPServerConfig,
    MCPStepConfig,
    MemoryStepConfig,
    PromptTemplate,
    ReasoningContext,
    ReasoningResult,
    # Step Type Enum
    StepType,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
    # New Typed Step Classes
    StepDescriptionBase,
    LLMStepDescription,
    ToolStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    TransformStepDescription,
    ConditionalStepDescription,
    StructuredOutputStepDescription,
    AgentSkillStepDescription,
    AgentHandoffStepDescription,
    SupervisorStepDescription,
    DebateStepDescription,
    EvaluationStepDescription,
    ParallelSamplingStepDescription,
    ToolDiscoveryStepDescription,
    HumanInputStepDescription,
    MCPResourceStepDescription,
    AnyStepDescription,
    StepDescription,
    ReplanPolicy,
    StepGroup,
)
from .models.agent_skill import AgentSkillStepConfig, AgentSkillSource
from .models.config import (
    AgentHandoffStepConfig,
    DebateStepConfig,
    EvaluationStepConfig,
    HumanInputStepConfig,
    MCPResourceStepConfig,
    ParallelSamplingStepConfig,
    SupervisorStepConfig,
    ToolDiscoveryStepConfig,
)


class ChainFormatNewerError(Exception):
    """Raised when :meth:`ReasoningChain.from_dict` encounters a chain
    serialised with a ``format_version`` newer than the running
    library understands.

    Callers (e.g. CARE) catch this and prompt the
    user to upgrade ``mmar-carl`` rather than silently losing data.

    Attributes
    ----------
    required_version:
        The ``format_version`` value found in the serialised data.
    this_version:
        The ``format_version`` that this library understands.
    """

    def __init__(self, required_version: int, this_version: int) -> None:
        super().__init__(
            f"Chain was serialised with format_version={required_version} but "
            f"this version of mmar-carl only understands format_version="
            f"{this_version}. Upgrade mmar-carl to load this chain."
        )
        self.required_version = required_version
        self.this_version = this_version


class ReflectionOptions(BaseModel):
    """
    Configuration options for chain reflection.

    Controls what information is included in the reflection prompt
    and how verbose the analysis should be.

    Example:
        ```python
        # Minimal reflection (faster, cheaper)
        options = ReflectionOptions(
            include_chain_structure=False,
            include_dependency_analysis=False,
            max_output_preview_chars=200,
        )
        reflection = chain.reflect_async("Analyze data", options=options)

        # Detailed reflection (more context)
        options = ReflectionOptions(
            include_step_definitions=True,
            include_execution_metrics=True,
            include_dependency_analysis=True,
        )
        ```
    """

    include_chain_structure: bool = Field(
        default=True,
        description="Include chain configuration and step type distribution",
    )
    include_step_definitions: bool = Field(
        default=True,
        description="Include original step definitions (aim, queries, etc.)",
    )
    include_execution_metrics: bool = Field(
        default=True,
        description="Include timing stats and parallel efficiency",
    )
    include_dependency_analysis: bool = Field(
        default=True,
        description="Include dependency graph analysis and parallelization opportunities",
    )
    include_metric_scores: bool = Field(
        default=True,
        description=(
            "Include MetricBase evaluation scores (step-level and chain-level) in the "
            "reflection prompt so the LLM can reference concrete quality signals"
        ),
    )
    extra_feedback: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Optional user-provided data or context to append to the reflection prompt. "
            "Pass a dict for labelled entries or a plain string for freeform notes."
        ),
    )
    max_output_preview_chars: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Maximum characters to show in output previews",
    )
    max_result_preview_chars: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Maximum characters to show in step result previews",
    )
    language: Optional[Language] = Field(
        default=None,
        description="Language for reflection prompt (None = use context language)",
    )
    dataset_report: DatasetEvaluationReport | None = Field(
        default=None,
        description=(
            "Optional dataset evaluation report produced by DatasetEvaluator. "
            "When provided, a dedicated section listing problem cases is added to "
            "the reflection prompt so the LLM can focus improvements on patterns "
            "observed across failing cases."
        ),
    )


class ReasoningChain:
    """
    Main interface for defining and executing reasoning chains.

    Provides a high-level API that combines chain definition with DAG execution.

    Accepts both legacy StepDescription and new typed step classes:
    - StepDescription (legacy, backward compatible)
    - LLMStepDescription, ToolStepDescription, MCPStepDescription, etc. (new API)

    Note on parallel execution:
        - Steps in the same batch execute in parallel with isolated memory
        - Tool registry is shared - tools MUST be stateless for thread safety
        - Memory writes are only visible to subsequent batches, not parallel siblings

    Note on conditional steps:
        - CONDITIONAL steps are currently informational only
        - They return next_step in result_data but execution proceeds topologically
    """

    # Bump this integer whenever the serialized dict layout changes in a
    # backward-incompatible way.  from_dict() warns when it encounters a
    # format_version it doesn't know how to handle.
    FORMAT_VERSION: int = 1

    def __init__(
        self,
        steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription],
        max_workers: int | str = 3,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
        metadata: dict[str, Any] | None = None,
        search_config: ContextSearchConfig | None = None,
        timeout: float | None = None,
        trace_name: str | None = None,
        session_id: str | None = None,
        replan_policy: ReplanPolicy | None = None,
        metrics: Optional[list] = None,
        default_llm_config: LLMStepConfig | None = None,
        step_groups: Sequence["StepGroup"] | None = None,
        memory_schema: Optional[dict[str, dict[str, Any]]] = None,
        max_injections: int = 50,
    ):
        # Normalize steps to support both legacy and new types
        self.steps: list[StepDescription | StepDescriptionBase | AnyStepDescription] = list(steps)
        self.metrics: list = metrics or []
        self.max_workers = max_workers
        self.enable_progress = enable_progress
        self.metadata = metadata or {}
        self.timeout = timeout
        self.trace_name = trace_name
        self.session_id = session_id
        self.replan_policy = replan_policy
        self.default_llm_config = default_llm_config
        self.step_groups: list["StepGroup"] = list(step_groups or [])
        self.memory_schema: Optional[dict[str, dict[str, Any]]] = memory_schema
        self.max_injections: int = max_injections

        # Bake group-level llm_config into each member step's llm_config before
        # validation runs. Group-set fields fill in only those slots the step
        # itself didn't explicitly set, so per-step config still wins.
        self._apply_step_groups()

        # Set up prompt template with search configuration
        if prompt_template:
            self.prompt_template = prompt_template
            if search_config:
                self.prompt_template.search_config = search_config
        else:
            self.prompt_template = PromptTemplate(search_config=search_config or ContextSearchConfig())

        self._validate_steps()
        self.executor = DAGExecutor(
            max_workers=max_workers,
            prompt_template=self.prompt_template,
            enable_progress=enable_progress,
            timeout=timeout,
            replan_policy=replan_policy,
            max_injections=max_injections,
        )

        # Store last execution result for reflection
        self._last_result: ReasoningResult | None = None
        self._last_context: ReasoningContext | None = None

    def _apply_step_groups(self) -> None:
        """
        Merge each :class:`StepGroup` ``llm_config`` into its member steps' configs.

        Only fields *explicitly set* on the group (per pydantic's
        ``model_fields_set``) are propagated, and only to step fields that the
        caller *did not* explicitly set. This preserves the documented
        precedence: per-step > group > chain default.

        Validation:

        - Every step number referenced by a group must exist in the chain.
        - A step may appear in at most one group.
        """
        if not self.step_groups:
            return

        step_by_number: dict[int, Any] = {s.number: s for s in self.steps}
        seen_steps: dict[int, str] = {}  # step_number → group_name

        for group in self.step_groups:
            group_set = group.llm_config.model_fields_set
            for step_num in group.steps:
                if step_num not in step_by_number:
                    raise ValueError(
                        f"StepGroup '{group.name}' references non-existent step {step_num}"
                    )
                if step_num in seen_steps:
                    raise ValueError(
                        f"Step {step_num} listed in multiple groups: "
                        f"'{seen_steps[step_num]}' and '{group.name}'"
                    )
                seen_steps[step_num] = group.name

                step = step_by_number[step_num]
                # Only LLM-style steps carry an ``llm_config`` attribute; skip
                # silently for tool / memory / transform / etc. steps so groups
                # can list mixed step numbers without exploding.
                if not hasattr(step, "llm_config"):
                    continue

                step_cfg: LLMStepConfig | None = getattr(step, "llm_config", None)
                if step_cfg is None:
                    # No per-step config: clone the group's *set* fields into a fresh
                    # LLMStepConfig (so unset group fields don't pollute defaults).
                    new_cfg = LLMStepConfig(
                        **{
                            name: getattr(group.llm_config, name)
                            for name in group_set
                        }
                    )
                    step.llm_config = new_cfg
                else:
                    step_set = step_cfg.model_fields_set
                    # For each field the group set but the step did *not* set,
                    # copy group's value onto the step config in place.
                    for field_name in group_set - step_set:
                        setattr(step_cfg, field_name, getattr(group.llm_config, field_name))
                        # Pydantic v2 keeps ``model_fields_set`` immutable after init;
                        # update it via __pydantic_fields_set__ so downstream merge
                        # logic sees this field as "set".
                        step_cfg.__pydantic_fields_set__.add(field_name)

    def _validate_steps(self) -> None:
        if not self.steps:
            raise ValueError("Reasoning chain must have at least one step")
        step_numbers = [step.number for step in self.steps]

        # Check for duplicate step numbers
        if len(step_numbers) != len(set(step_numbers)):
            duplicates = [num for num in step_numbers if step_numbers.count(num) > 1]
            raise ValueError(f"Duplicate step numbers found: {duplicates}")

        # Check for missing dependencies
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_numbers:
                    raise ValueError(f"Step {step.number} depends on non-existent step {dep}")

        # Check for cycles (basic validation)
        self._check_for_cycles()

        # Warn about likely-broken $memory.* and $history[N] references
        self._validate_references()

    def _check_for_cycles(self) -> None:
        """
        Basic cycle detection using dependency graph.

        Raises:
            ValueError: If cycles are detected
        """
        visited = set()
        rec_stack = set()

        def visit(step_num: int) -> bool:
            if step_num in rec_stack:
                return True  # Cycle detected
            if step_num in visited:
                return False

            visited.add(step_num)
            rec_stack.add(step_num)

            # Visit dependencies
            step = next(s for s in self.steps if s.number == step_num)
            for dep in step.dependencies:
                if visit(dep):
                    return True

            rec_stack.remove(step_num)
            return False

        for step in self.steps:
            if step.number not in visited:
                if visit(step.number):
                    raise ValueError(f"Cycle detected involving step {step.number}")

    def _validate_references(self) -> None:
        """
        Warn about ``$memory.*`` and ``$history[N]`` references that are
        likely to be unresolvable at runtime.

        Issues ``UserWarning`` for:
        - ``$memory.ns.key`` read before any step in the chain writes it
          (the key may still be pre-populated via ``context.memory`` — this
          is a warning, not an error)
        - ``$history[N]`` where N >= 0 and the expected history depth at
          that step's batch position is <= N (i.e. not enough prior steps)

        Scanned fields by step type:
        - TOOL / AGENT_SKILL / MCP: ``input_mapping`` / ``argument_mapping`` values
        - MEMORY: ``value_source``
        - TRANSFORM: ``input_key``
        - CONDITIONAL: ``condition_context_key``
        - STRUCTURED_OUTPUT: ``input_source``
        """
        from .models.enums import MemoryOperation

        # Build topological batches (same logic as DAGExecutor)
        completed: set[int] = set()
        batches: list[list[Any]] = []
        remaining = list(self.steps)
        while remaining:
            batch = [s for s in remaining if all(d in completed for d in s.dependencies)]
            if not batch:
                break  # cycle already caught; stop gracefully
            batches.append(batch)
            for s in batch:
                completed.add(s.number)
            remaining = [s for s in remaining if s.number not in completed]

        # Map every memory write in the chain: "ns.key" -> frozenset of step numbers
        memory_writers: dict[str, set[int]] = {}
        for step in self.steps:
            if step.step_type == StepType.MEMORY:
                cfg = getattr(step, "config", None) or getattr(step, "step_config", None)
                if hasattr(cfg, "operation") and cfg.operation in (
                    MemoryOperation.WRITE,
                    MemoryOperation.APPEND,
                ):
                    ns = getattr(cfg, "namespace", None) or "default"
                    full_key = f"{ns}.{cfg.memory_key}"
                    memory_writers.setdefault(full_key, set()).add(step.number)

        def _refs_from_step(step: Any) -> list[str]:
            """Return all reference strings (starting with '$') in a step's config."""
            # Typed classes use `config`; legacy StepDescription uses `step_config`
            cfg = getattr(step, "config", None) or getattr(step, "step_config", None)
            if cfg is None:
                return []
            st = step.step_type
            if st in (StepType.TOOL, StepType.AGENT_SKILL):
                mapping = getattr(cfg, "input_mapping", {}) or {}
                return [v for v in mapping.values() if isinstance(v, str) and v.startswith("$")]
            if st == StepType.MCP:
                mapping = getattr(cfg, "argument_mapping", {}) or {}
                return [v for v in mapping.values() if isinstance(v, str) and v.startswith("$")]
            if st == StepType.MEMORY:
                vs = getattr(cfg, "value_source", None)
                return [vs] if vs and isinstance(vs, str) and vs.startswith("$") else []
            if st == StepType.TRANSFORM:
                ik = getattr(cfg, "input_key", None)
                return [ik] if ik and isinstance(ik, str) and ik.startswith("$") else []
            if st == StepType.CONDITIONAL:
                ck = getattr(cfg, "condition_context_key", None)
                return [ck] if ck and isinstance(ck, str) and ck.startswith("$") else []
            if st == StepType.STRUCTURED_OUTPUT:
                src = getattr(cfg, "input_source", None)
                return [src] if src and isinstance(src, str) and src.startswith("$") else []
            return []

        history_depth = 0  # history entries produced by all prior batches
        prior_step_nums: set[int] = set()

        for batch in batches:
            for step in batch:
                for ref in _refs_from_step(step):
                    # --- $history[N] check ---
                    if ref.startswith("$history["):
                        m = re.match(r"\$history\[(-?\d+)\]", ref)
                        if m:
                            idx = int(m.group(1))
                            if history_depth == 0:
                                warnings.warn(
                                    f"Step {step.number} '{step.title}': reference '{ref}' "
                                    f"is used but no prior steps have executed yet — "
                                    f"history will be empty and the reference will resolve to None.",
                                    UserWarning,
                                    stacklevel=5,
                                )
                            elif idx >= 0 and history_depth <= idx:
                                warnings.warn(
                                    f"Step {step.number} '{step.title}': reference '{ref}' "
                                    f"may be out of range — expected history depth at this "
                                    f"point is {history_depth} (index {idx} >= depth). "
                                    f"Ensure enough prior steps have executed.",
                                    UserWarning,
                                    stacklevel=5,
                                )
                    # --- $memory.ns.key check ---
                    elif ref.startswith("$memory."):
                        tail = ref[8:]
                        parts = tail.split(".", 1)
                        if len(parts) == 2:
                            ns, mem_key = parts[0], parts[1]
                        else:
                            ns, mem_key = "default", tail
                        full_key = f"{ns}.{mem_key}"
                        writers = memory_writers.get(full_key, set())
                        # A prior writer is one that appears in a completed batch
                        if not (writers & prior_step_nums):
                            warnings.warn(
                                f"Step {step.number} '{step.title}': reference '{ref}' "
                                f"reads a memory key that no prior step writes. "
                                f"Pre-populate context.memory['{ns}']['{mem_key}'] "
                                f"before execution if this is intentional.",
                                UserWarning,
                                stacklevel=5,
                            )

            # After processing this batch, advance state for next batch
            history_depth += len(batch)
            for step in batch:
                prior_step_nums.add(step.number)

    async def execute_async(
        self,
        context: ReasoningContext,
        *,
        resume_from: Optional[Any] = None,
    ) -> ReasoningResult:
        """
        Execute the reasoning chain asynchronously.

        Args:
            context: Reasoning context with input data and LLM client.
            resume_from: Optional :class:`ContextSnapshot` from a prior
                paused / cancelled run. When supplied, the context is
                first restored from the snapshot (history / memory /
                metadata / messages / cancel state) and every step
                whose number appears in ``snapshot.completed_step_numbers``
                is treated as already executed — the DAG scheduler
                skips it and the chain resumes at the first un-done
                step. Combined with :meth:`ReasoningContext.snapshot`,
                this is CARE's cross-process resume primitive.

        Returns:
            Complete reasoning result
        """
        # resume from a prior snapshot.
        resume_step_numbers: set[int] = set()
        if resume_from is not None:
            context.restore(resume_from)
            completed = getattr(resume_from, "completed_step_numbers", None) or []
            resume_step_numbers = {int(n) for n in completed}

        # Add chain metadata to context (include trace_name and session_id)
        chain_meta = dict(self.metadata)
        if self.trace_name:
            chain_meta["trace_name"] = self.trace_name
        if self.session_id:
            chain_meta["session_id"] = self.session_id
        updates: dict[str, Any] = {
            "chain_steps": len(self.steps),
            "chain_metadata": chain_meta,
            "replan_policy_enabled": bool(self.replan_policy and self.replan_policy.enabled),
        }
        if self.default_llm_config is not None:
            updates["__default_llm_config"] = self.default_llm_config
        context.metadata.update(
            updates
        )

        # Push the chain-level memory schema onto the context if the user
        # hasn't already set one on the context directly. Context-level setting
        # wins so callers can override per-execution if needed.
        if self.memory_schema is not None and context.memory_schema is None:
            context.memory_schema = self.memory_schema

        result = await self.executor.execute(
            self.steps, context,
            resume_from_step_numbers=resume_step_numbers or None,
        )

        # Run chain-level metrics on the chain result
        if self.metrics and result.success:
            chain_metric_scores: dict[str, float] = {}
            for metric in self.metrics:
                try:
                    score = await metric.compute_async(result)
                    chain_metric_scores[metric.name] = float(score)
                except Exception as e:
                    from .logging_utils import log_warning

                    log_warning(f"Chain metric '{getattr(metric, 'name', repr(metric))}' failed: {e}")
            if chain_metric_scores:
                result = result.model_copy(update={"metrics": chain_metric_scores})
                # Send to LangFuse as scores (idiomatic way to attach evaluation
                # metrics to a trace; works even after trace.end())
                trace = context.metadata.get("__langfuse_trace")
                if trace is not None:
                    for metric_name, score_value in chain_metric_scores.items():
                        try:
                            trace.score(name=metric_name, value=score_value)
                        except Exception:
                            pass

        # Store for reflection
        self._last_result = result
        self._last_context = context

        return result

    async def stream_async(
        self,
        context: ReasoningContext,
    ):
        """Streaming-first execution mode.

        Async generator that yields each :class:`StepExecutionResult`
        as soon as its step finishes, then yields the final
        :class:`ReasoningResult` last. Lets UIs render partial
        progress instead of blocking on the whole chain.

        For a 3-step OpenRouter chain that completes in ~24 s end-to-
        end, step 1's output becomes inspectable around t≈7 s instead
        of at t≈24 s::

            async for item in chain.stream_async(ctx):
                if isinstance(item, StepExecutionResult):
                    print(f"step {item.step_number} done: {item.success}")
                else:  # ReasoningResult — terminal
                    print(f"chain success={item.success}")

        Notes:
            * The user's existing ``context.on_step_complete`` callback
              is preserved — both the user callback and the streaming
              wrapper fire on every step.
            * Trace events still fire in real time via the executor's
              internal hooks; this method just exposes them as an
              async iterator surface.
            * Yielding order is **completion order** (not chain
              definition order) — parallel batches surface their
              steps as each one finishes.

        Yields:
            ``StepExecutionResult`` per completed step, then a single
            terminal ``ReasoningResult``.
        """
        from .models.results import StepExecutionResult  # noqa: PLC0415

        # Sentinel that signals end-of-stream on the queue.
        _SENTINEL: Any = object()
        queue: asyncio.Queue[Any] = asyncio.Queue()

        # Stash + chain the user's pre-existing on_step_complete so we
        # don't clobber their wiring (logging, tracing, UI updates).
        prior_callback = context.on_step_complete

        def _push(step_result: StepExecutionResult) -> None:
            # Forward to the user's callback first (failures swallowed
            # to match the executor's existing "callbacks shouldn't
            # break the chain" contract).
            if prior_callback is not None:
                try:
                    prior_callback(step_result)
                except Exception:
                    pass
            queue.put_nowait(step_result)

        context.on_step_complete = _push

        # Drive ``execute_async`` in a background task so its
        # callbacks can publish to the queue while we're iterating.
        async def _run() -> "ReasoningResult":
            try:
                return await self.execute_async(context)
            finally:
                # Always restore the user's callback so the context is
                # safe to reuse for a second run.
                context.on_step_complete = prior_callback

        task = asyncio.create_task(_run())

        try:
            while True:
                # If execute_async finishes before more step events
                # arrive, drain the queue and break — we'll yield the
                # ReasoningResult below.
                get_item = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {get_item, task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if get_item in done:
                    item = get_item.result()
                    yield item
                    if task.done() and queue.empty():
                        break
                    continue
                # ``execute_async`` finished first. Drain any
                # remaining step results that the callback enqueued
                # before tearing down, then break.
                get_item.cancel()
                while not queue.empty():
                    yield queue.get_nowait()
                break
            # Surface any execute_async exception cleanly.
            result = await task
        except BaseException:
            # Cancel the background task on cancellation / GC.
            if not task.done():
                task.cancel()
            raise

        # Terminal item: the full ReasoningResult.
        yield result

    @staticmethod
    def _build_async_context_diagnostic() -> str:
        """Compose the actionable error/warning message for sync ``execute()``
        called from within a running event loop.

        Includes (1) the exact replacement snippet, (2) a notebook-specific
        hint when IPython/Jupyter is detected, and (3) why the thread-pool
        fallback is dangerous.
        """
        import sys

        in_ipython = "IPython" in sys.modules
        notebook_hint = (
            "\n  Jupyter / IPython detected: enable autoawait with "
            "'%autoawait' (or use 'await chain.execute_async(context)' in "
            "an async cell). Sync execute() in a notebook can deadlock "
            "long-lived httpx clients."
            if in_ipython
            else ""
        )
        return (
            "ReasoningChain.execute() called from a running event loop. "
            "Use the async API instead:\n"
            "\n"
            "    result = await chain.execute_async(context)\n"
            "\n"
            "The sync fallback spins up a thread-pool executor with a "
            "*nested* event loop, which can deadlock httpx-backed LLM "
            "clients (OpenAI, Anthropic, OpenRouter) and leaks task "
            "cleanup across loops. Pass strict_async=True to "
            "execute() to make this raise instead of warn."
            + notebook_hint
        )

    def execute(
        self,
        context: ReasoningContext,
        *,
        strict_async: bool = False,
    ) -> ReasoningResult:
        """
        Execute the reasoning chain synchronously.

        Args:
            context: Reasoning context with input data and LLM client.
            strict_async: When True and called from inside a running event
                loop (e.g. an async test, a Jupyter notebook with autoawait,
                a FastAPI request handler), raise ``RuntimeError`` instead of
                warning-and-falling-back-to-a-thread-pool. The thread-pool
                fallback spins up a *nested* event loop which can deadlock
                long-lived ``httpx`` / Anthropic / OpenAI clients and is
                generally a footgun.  Defaults to False for backward
                compatibility; pass True in new code.

        Returns:
            Complete reasoning result

        Raises:
            RuntimeError: If ``strict_async=True`` and called from an async
                context. The error message includes the exact replacement
                snippet (``await chain.execute_async(context)``).

        Note:
            This method creates a new event loop and may have limitations when
            called from async contexts (e.g., Jupyter notebooks, FastAPI).

            For async applications, prefer execute_async() instead:

            ```python
            result = await chain.execute_async(context)
            ```

            If you must use execute() in an async context, be aware of potential
            issues with signal handlers and context variables.
        """
        import warnings

        # Detect async context and emit an actionable diagnostic.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to use asyncio.run path below.
            pass
        else:
            diagnostic = self._build_async_context_diagnostic()
            if strict_async:
                raise RuntimeError(diagnostic)
            warnings.warn(diagnostic, UserWarning, stacklevel=2)

        async def _execute_and_cleanup() -> ReasoningResult:
            """Run the chain and close LLM clients before the event loop shuts down."""
            try:
                return await self.execute_async(context)
            finally:
                await context.close()
                # Wait for background tasks (e.g., httpx connection cleanup) to complete.
                # httpx spawns tasks during aclose() that need time to finish before
                # asyncio.run() closes the event loop.
                current_task = asyncio.current_task()
                # Give tasks time to spawn and complete
                for _ in range(10):
                    await asyncio.sleep(0.01)
                    other_tasks = [t for t in asyncio.all_tasks() if t != current_task]
                    if not other_tasks:
                        break
                    # Wait for any remaining tasks with timeout
                    try:
                        await asyncio.wait_for(asyncio.gather(*other_tasks, return_exceptions=True), timeout=0.1)
                    except asyncio.TimeoutError:
                        # Some tasks are still running, continue waiting
                        pass

        try:
            # Check if we're already in an event loop
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_execute_and_cleanup())
        else:
            # We're in an event loop, create a task and run it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _execute_and_cleanup())
                return future.result()

    async def execute_from_trace(
        self,
        trace: Any,
        from_step: int,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Resume execution from step *from_step* using a prior trace to short-circuit
        all earlier LLM-style steps.

        Behaves like :py:meth:`replay` but only for step numbers ``< from_step``.
        Step numbers ``>= from_step`` use the *real* LLM client on ``context.api``
        — so this is the right tool when you want to iterate on the tail of a
        long chain without paying for the prefix every time.

        Non-LLM steps (Tool / Memory / Transform / Conditional / …) execute
        normally in *both* halves: their behaviour depends only on context state,
        which is reconstructed by re-running them. If a tool has expensive side
        effects (network calls, file writes), consider memoising it for the
        first N steps before invoking this method.

        Args:
            trace: An :class:`~mmar_carl.execution_trace.ExecutionTrace`
                from a prior run of this chain.
            from_step: The step number to start using the real LLM at. Earlier
                steps use the trace's recorded ``result`` values. Must be in
                ``[min_step_number, max_step_number + 1]``; ``max + 1`` is
                allowed and trivially returns the trace-replay result.
            context: A :class:`ReasoningContext` carrying the live API client
                used for ``step_number >= from_step``.

        Returns:
            A new :class:`ReasoningResult` for the full chain.

        Raises:
            ValueError: ``from_step`` is below the chain's smallest step number.
        """
        from .models.llm_client_base import LLMClientBase

        step_numbers = [s.number for s in self.steps]
        if from_step < min(step_numbers):
            raise ValueError(
                f"from_step={from_step} is below the chain's smallest step "
                f"number ({min(step_numbers)})"
            )

        # Trace step_number → recorded successful result
        traced_responses: dict[int, str] = {
            event.step_number: event.result
            for event in trace.events
            if not event.skipped and event.success
        }

        # Tracks which step is currently in flight via on_step_start.
        current_step: list[Optional[int]] = [None]
        real_client = context.api

        class _HybridClient(LLMClientBase):
            """Dispatches by step number: trace for prefix, real client for suffix."""

            async def get_response(self, prompt: str) -> str:
                snum = current_step[0]
                if snum is not None and snum < from_step:
                    # Step in the trace-replayed prefix
                    return traced_responses.get(snum, "")
                # Real call into the user's client
                if hasattr(real_client, "get_response_with_retries"):
                    return await real_client.get_response_with_retries(  # type: ignore[no-any-return]
                        prompt, retries=context.retry_max
                    )
                return await real_client.get_response(prompt)  # type: ignore[no-any-return]

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

            async def get_response_with_usage(
                self, prompt: str, retries: int = 3
            ) -> tuple[str, dict[str, int]]:
                snum = current_step[0]
                if snum is not None and snum < from_step:
                    return traced_responses.get(snum, ""), {}
                # Delegate when the real client supports it
                if hasattr(real_client, "get_response_with_usage"):
                    return await real_client.get_response_with_usage(  # type: ignore[no-any-return]
                        prompt, retries=retries
                    )
                text = await self.get_response_with_retries(prompt, retries=retries)
                return text, {}

        # Fresh execution context, identical state but hybrid API
        resumed_ctx = ReasoningContext(
            outer_context=context.outer_context,
            api=_HybridClient(),
            model=context.model,
            language=context.language,
            retry_max=context.retry_max,
            history=list(context.history),
            memory={ns: dict(kv) for ns, kv in context.memory.items()},
            metadata=dict(context.metadata),
            system_prompt=context.system_prompt,
            max_history_entries=context.max_history_entries,
            trim_strategy=context.trim_strategy,
            messages=list(context.messages),
        )

        original_on_step_start = context.on_step_start

        def _on_step_start(step_num: int, step_title: str) -> None:
            current_step[0] = step_num
            if original_on_step_start:
                original_on_step_start(step_num, step_title)

        resumed_ctx.on_step_start = _on_step_start
        resumed_ctx.on_step_complete = context.on_step_complete
        resumed_ctx.on_progress = context.on_progress
        resumed_ctx.on_llm_chunk = context.on_llm_chunk

        # Copy tools + tags so tool steps in either half see the same registry
        resumed_ctx._tool_registry = context._tool_registry.copy()
        resumed_ctx._tool_tags = {n: set(t) for n, t in context._tool_tags.items()}

        return await self.execute_async(resumed_ctx)

    async def replay(
        self,
        trace: Any,
        context: ReasoningContext,
    ) -> ReasoningResult:
        """
        Re-execute the chain feeding saved LLM responses from a prior trace.

        Uses the per-step ``result`` values recorded in *trace* to drive the
        chain without making real LLM calls.  All other step types (Tool,
        Memory, Transform, etc.) still run normally so side-effects are
        replayed faithfully.

        Args:
            trace: An :class:`~mmar_carl.execution_trace.ExecutionTrace`
                obtained from a prior ``result.trace``.
            context: A fresh :class:`ReasoningContext` — the API client is
                replaced with a deterministic replay mock, but
                ``outer_context``, ``memory``, and ``metadata`` are forwarded
                as-is.

        Returns:
            A new :class:`ReasoningResult` produced by the replayed execution.
        """
        from .models.llm_client_base import LLMClientBase

        # Build step-number → recorded response mapping (skip skipped/failed)
        step_responses: dict[int, str] = {
            event.step_number: event.result
            for event in trace.events
            if not event.skipped and event.success
        }

        # Track current step via on_step_start callback
        current_step: list[Optional[int]] = [None]

        class _ReplayClient(LLMClientBase):
            async def get_response(self, prompt: str) -> str:
                snum = current_step[0]
                return step_responses.get(snum, "") if snum is not None else ""  # type: ignore[arg-type]

            async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
                return await self.get_response(prompt)

        # Build a replay context that reuses all state but has the mock API
        replay_ctx = ReasoningContext(
            outer_context=context.outer_context,
            api=_ReplayClient(),
            model=context.model,
            language=context.language,
            retry_max=context.retry_max,
            history=list(context.history),
            memory={ns: dict(kv) for ns, kv in context.memory.items()},
            metadata=dict(context.metadata),
            system_prompt=context.system_prompt,
            max_history_entries=context.max_history_entries,
            trim_strategy=context.trim_strategy,
            messages=list(context.messages),
        )

        original_on_step_start = context.on_step_start

        def _on_step_start(step_num: int, step_title: str) -> None:
            current_step[0] = step_num
            if original_on_step_start:
                original_on_step_start(step_num, step_title)

        replay_ctx.on_step_start = _on_step_start
        # Propagate other callbacks if set
        replay_ctx.on_step_complete = context.on_step_complete
        replay_ctx.on_progress = context.on_progress
        replay_ctx.on_llm_chunk = context.on_llm_chunk

        return await self.execute_async(replay_ctx)

    async def batch_execute_async(
        self,
        contexts: list[ReasoningContext],
        max_concurrent: int = 1,
    ) -> list[ReasoningResult]:
        """
        Execute the chain on multiple contexts asynchronously.

        Args:
            contexts: List of :class:`ReasoningContext` objects to run.
            max_concurrent: Maximum number of chains running in parallel.
                Defaults to ``1`` (sequential) to respect LLM API rate limits.
                Increase carefully — each concurrent execution issues its own
                LLM requests.

        Returns:
            List of :class:`ReasoningResult` in the same order as ``contexts``.
        """
        if max_concurrent == 1:
            results = []
            for ctx in contexts:
                results.append(await self.execute_async(ctx))
            return results

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _run(ctx: ReasoningContext) -> ReasoningResult:
            async with semaphore:
                return await self.execute_async(ctx)

        return list(await asyncio.gather(*[_run(ctx) for ctx in contexts]))

    def batch_execute(
        self,
        contexts: list[ReasoningContext],
        max_concurrent: int = 1,
    ) -> list[ReasoningResult]:
        """
        Synchronous wrapper around :meth:`batch_execute_async`.

        Args:
            contexts: List of :class:`ReasoningContext` objects to run.
            max_concurrent: Maximum number of chains running in parallel.

        Returns:
            List of :class:`ReasoningResult` in the same order as ``contexts``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.batch_execute_async(contexts, max_concurrent=max_concurrent))
        else:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.batch_execute_async(contexts, max_concurrent=max_concurrent),
                )
                return future.result()

    # =========================================================================
    # Reflection Methods
    # =========================================================================

    async def reflect_async(
        self,
        task_description: str,
        context: ReasoningContext | None = None,
        language: Language | None = None,
        options: ReflectionOptions | None = None,
    ) -> str:
        """
        Generate a reflection on the chain execution results.

        Analyzes how well the chain accomplished the given task based on
        step results and overall execution.

        Args:
            task_description: Description of the original task/goal
            context: Optional context to use (uses last execution context if not provided)
            language: Language for reflection (uses context language if not provided)
            options: Optional ReflectionOptions to control what's included in the prompt

        Returns:
            Reflection text analyzing the execution results

        Raises:
            RuntimeError: If no execution has been performed yet

        Example:
            ```python
            # Basic reflection
            reflection = await chain.reflect_async("Analyze sentiment")

            # Minimal reflection (faster, cheaper)
            options = ReflectionOptions(
                include_chain_structure=False,
                include_dependency_analysis=False,
                max_output_preview_chars=200,
            )
            reflection = await chain.reflect_async("Analyze sentiment", options=options)
            ```
        """
        if self._last_result is None:
            raise RuntimeError("No execution result available. Call execute() before reflect().")

        ctx = context or self._last_context
        if ctx is None:
            raise RuntimeError("No context available. Provide a context or execute the chain first.")

        # Determine language (options > parameter > context)
        opts = options or ReflectionOptions()
        lang = opts.language or language or ctx.language

        # Build reflection prompt with options
        reflection_prompt = self._build_reflection_prompt(task_description, lang, opts)

        # Get LLM response
        llm_client = ctx.llm_client
        reflection = await llm_client.get_response_with_retries(reflection_prompt, retries=2)

        return reflection

    def reflect(
        self,
        task_description: str,
        context: ReasoningContext | None = None,
        language: Language | None = None,
        options: ReflectionOptions | None = None,
    ) -> str:
        """
        Generate a reflection on the chain execution results (synchronous).

        Analyzes how well the chain accomplished the given task based on
        step results and overall execution.

        Args:
            task_description: Description of the original task/goal
            context: Optional context to use (uses last execution context if not provided)
            language: Language for reflection (uses context language if not provided)
            options: Optional ReflectionOptions to control what's included in the prompt

        Returns:
            Reflection text analyzing the execution results

        Raises:
            RuntimeError: If no execution has been performed yet

        Example:
            ```python
            chain = ReasoningChain(steps=[...])
            context = ReasoningContext(outer_context="data", api=client)
            result = chain.execute(context)

            # Reflect on how well the task was accomplished
            reflection = chain.reflect(
                task_description="Analyze customer sentiment and extract key themes"
            )
            print(reflection)

            # With options for minimal reflection
            from mmar_carl import ReflectionOptions
            options = ReflectionOptions(include_dependency_analysis=False)
            reflection = chain.reflect("Analyze sentiment", options=options)
            ```
        """

        async def _reflect_and_cleanup() -> str:
            """Run reflection and close LLM clients before the event loop shuts down."""
            try:
                return await self.reflect_async(task_description, context, language, options)
            finally:
                ctx = context or self._last_context
                if ctx is not None:
                    await ctx.close()
                    # Wait for background tasks (e.g., httpx connection cleanup) to complete.
                    current_task = asyncio.current_task()
                    for _ in range(10):
                        await asyncio.sleep(0.01)
                        other_tasks = [t for t in asyncio.all_tasks() if t != current_task]
                        if not other_tasks:
                            break
                        try:
                            await asyncio.wait_for(asyncio.gather(*other_tasks, return_exceptions=True), timeout=0.1)
                        except asyncio.TimeoutError:
                            pass

        try:
            # Check if we're already in an event loop
            _ = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_reflect_and_cleanup())
        else:
            # We're in an event loop, create a task and run it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _reflect_and_cleanup())
                return future.result()

    def _build_reflection_prompt(self, task_description: str, language: Language, options: ReflectionOptions) -> str:
        """
        Build the reflection prompt based on execution results.

        This method now provides concrete, actionable advice for improving the chain.
        The prompt instructs the LLM to provide specific code/prompt changes that
        a developer can directly implement.

        Args:
            task_description: The original task/goal
            language: Language for the prompt
            options: ReflectionOptions controlling what's included

        Returns:
            Formatted reflection prompt with enhanced context and actionable guidance
        """
        result = self._last_result

        # Gather enhanced context using helper methods (respecting options)
        context_sections = []

        if options.include_chain_structure:
            context_sections.append(self._build_chain_structure_summary())

        if options.include_step_definitions:
            context_sections.append(self._build_step_definitions_summary())

        context_sections.append(self._build_step_details_summary(max_preview_chars=options.max_result_preview_chars))

        if options.include_execution_metrics:
            context_sections.append(self._build_execution_metrics_summary())

        if options.include_dependency_analysis:
            context_sections.append(self._build_dependency_analysis())

        if options.include_metric_scores:
            scores_section = self._build_metric_scores_summary()
            if scores_section:
                context_sections.append(scores_section)

        if options.dataset_report is not None:
            context_sections.append(self._build_dataset_report_section(options.dataset_report, language))

        if options.extra_feedback is not None:
            context_sections.append(self._build_extra_feedback_summary(options.extra_feedback, language))

        full_context = "\n\n".join(context_sections)

        # Final output
        if result is None:
            final_output = "(No execution result available)"
        else:
            final_output = result.get_final_output() or result.get_full_output()
        if len(final_output) > options.max_output_preview_chars:
            final_output = final_output[: options.max_output_preview_chars] + "..."

        if language == Language.ENGLISH:
            return f"""You are an expert chain architecture analyst. Your task is to provide **CONCRETE, ACTIONABLE ADVICE** for rewriting this reasoning chain to better accomplish the given task.

## Original Task
{task_description}

{full_context}

## Final Output
```
{final_output}
```

## Instructions: Provide Actionable Improvements

Analyze the chain execution and provide **specific, implementable changes**. Each recommendation must include exact code/prompt modifications.

### Required Output Format

Provide your analysis in these sections:

#### 1. SPECIFIC CHANGES NEEDED
For each step that needs improvement, provide:
- **Step number and title**
- **Problem**: What specifically is wrong (vague aim, missing context query, wrong dependency, etc.)
- **Fix**: EXACT code to replace the current step

Example format:
```
STEP 2 (Competitive Analysis)
Problem: Queries for "CompetitorData" which doesn't exist in outer_context
Fix: Remove the step OR update step_context_queries to use existing fields:
  step_context_queries=["Comment", "Product", "Rating"]
```

#### 2. STEP-BY-STEP REWRITE INSTRUCTIONS
For steps that need modification, show "before → after":
```
BEFORE:
aim = "Analyze the data"
AFTER:
aim = "Categorize each comment into themes: Product Quality, Shipping, Value for Money, or Customer Service. Count occurrences per product."
```

#### 3. NEW STEPS TO ADD
Identify missing functionality and provide FULL step definitions:
```
ADD NEW STEP 2.5: Count Themes by Product
LLMStepDescription(
    number=2.5,
    title="Aggregate Theme Counts",
    aim="Count theme occurrences per product",
    reasoning_questions="How many comments mention each theme for each product?",
    dependencies=[1, 2]
)
```

#### 4. STEPS TO REMOVE/COMBINE
Identify redundant steps:
```
REMOVE: Step 3 (Redundant Summary)
- This step duplicates what step 4 already does
- Update step 4 dependencies to [1, 2] instead of [1, 2, 3]
```

#### 5. DEPENDENCY FIXES
Identify bottlenecks and provide exact changes:
```
STEP 2: Remove dependency on step 1
CHANGE: step_2.dependencies = []
REASON: Competitive analysis doesn't need theme extraction to run first
BENEFIT: Steps 1 and 2 can run in parallel
```

#### 6. STRUCTURE REORGANIZATION
If chain structure should change:
```
CURRENT BOTTLENECK: Step 3 waits for [1, 2] but only needs step 1
RECOMMENDED: Make step 2 independent, restructure as:
  - Level 1: Steps 1, 2 (parallel)
  - Level 2: Step 3 (depends on 1 only)
  - Level 3: Step 4 (depends on 1, 2, 3)
```

## Critical Rules
1. **Be Specific**: Show exact code changes, not general suggestions
2. **Reference Step Numbers**: Always mention which step you're modifying
3. **Provide Complete Code**: For new steps, include ALL required fields
4. **Explain WHY**: Each change should have a clear rationale
5. **Prioritize Impact**: Focus on changes that will have the biggest effect on output quality
6. **Check Context Queries**: Verify all step_context_queries exist in the provided data
7. **Validate Dependencies**: Ensure dependencies are actually necessary
8. **Use Metric Scores**: If evaluation metric scores are present above, reference them to justify quality issues and prioritize improvements (low scores = higher priority fixes)

Begin your analysis now. Focus on ACTIONABLE advice that can be immediately implemented."""

        else:  # Russian
            return f"""Ты — эксперт по анализу архитектуры цепочек рассуждений. Твоя задача — предоставить **КОНКРЕТНЫЕ, ПРИКЛАДНЫЕ РЕКОМЕНДАЦИИ** по переписыванию этой цепочки для лучшего выполнения поставленной задачи.

## Исходная задача
{task_description}

{full_context}

## Итоговый результат
```
{final_output}
```

## Инструкции: Предоставь конкретные улучшения

Проанализируй выполнение цепочки и предоставь **конкретные, реализуемые изменения**. Каждая рекомендация должна включать точные изменения кода/промптов.

### Требуемый формат вывода

Предоставь анализ в следующих разделах:

#### 1. КОНКРЕТНЫЕ НЕОБХОДИМЫЕ ИЗМЕНЕНИЯ
Для каждого шага, требующего улучшения, укажи:
- **Номер шага и название**
- **Проблема**: Что именно неправильно (неясная цель, отсутствующий контекстный запрос, неправильная зависимость и т.д.)
- **Решение**: ТОЧНЫЙ код для замены текущего шага

Формат примера:
```
ШАГ 2 (Анализ конкурентов)
Проблема: Запрашивает "CompetitorData", которого нет в outer_context
Решение: Удали шаг ИЛИ обнови step_context_queries для использования существующих полей:
  step_context_queries=["Comment", "Product", "Rating"]
```

#### 2. ПОШАГОВЫЕ ИНСТРУКЦИИ ПО ПЕРЕПИСЫВАНИЮ
Для шагов, требующих изменений, покажи "до → после":
```
ДО:
aim = "Проанализировать данные"
ПОСЛЕ:
aim = "Категоризировать каждый комментарий по темам: Качество продукта, Доставка, Соотношение цены и качества или Сервис обслуживания. Подсчитать количество по каждому продукту."
```

#### 3. НОВЫЕ ШАГИ ДЛЯ ДОБАВЛЕНИЯ
Определи отсутствующую функциональность и предоставь ПОЛНЫЕ определения шагов:
```
ДОБАВИТЬ НОВЫЙ ШАГ 2.5: Подсчитать темы по продуктам
LLMStepDescription(
    number=2.5,
    title="Агрегировать количество тем",
    aim="Подсчитать упоминания каждой темы по каждому продукту",
    reasoning_questions="Сколько комментариев упоминают каждую тему для каждого продукта?",
    dependencies=[1, 2]
)
```

#### 4. ШАГИ ДЛЯ УДАЛЕНИЯ/ОБЪЕДИНЕНИЯ
Определи избыточные шаги:
```
УДАЛИТЬ: Шаг 3 (Избыточная сводка)
- Этот шаг дублирует то, что уже делает шаг 4
- Обнови зависимости шага 4 на [1, 2] вместо [1, 2, 3]
```

#### 5. ИСПРАВЛЕНИЕ ЗАВИСИМОСТЕЙ
Определи узкие места и предоставь точные изменения:
```
ШАГ 2: Удали зависимость от шага 1
ИЗМЕНИ: step_2.dependencies = []
ПРИЧИНА: Анализ конкурентов не требует предварительного извлечения тем
ПРЕИМУЩЕСТВО: Шаги 1 и 2 могут выполняться параллельно
```

#### 6. РЕОРГАНИЗАЦИЯ СТРУКТУРЫ
Если структура цепочки должна измениться:
```
ТЕКУЩЕЕ УЗКОЕ МЕСТО: Шаг 3 ожидает [1, 2], но нужен только шаг 1
РЕКОМЕНДАЦИЯ: Сделай шаг 2 независимым, реструктурируй как:
  - Уровень 1: Шаги 1, 2 (параллельно)
  - Уровень 2: Шаг 3 (зависит только от 1)
  - Уровень 3: Шаг 4 (зависит от 1, 2, 3)
```

## Критические правила
1. **Будь конкретным**: Показывай точные изменения кода, а не общие рекомендации
2. **Указывай номера шагов**: Всегда упоминай, какой шаг ты изменяешь
3. **Предоставляй полный код**: Для новых шагов включай ВСЕ необходимые поля
4. **Объясняй ПОЧЕМУ**: Каждое изменение должно иметь чёткое обоснование
5. **Приоритет влияния**: Сосредоточься на изменениях с наибольшим эффектом на качество вывода
6. **Проверяй контекстные запросы**: Убедись, что все step_context_queries существуют в предоставленных данных
7. **Проверяй зависимости**: Убедись, что зависимости действительно необходимы
8. **Используй метрики**: Если выше присутствуют оценочные метрики, ссылайся на них для обоснования проблем качества и расстановки приоритетов улучшений (низкие баллы = более высокий приоритет исправления)

Начни анализ сейчас. Сосредоточься на ПРИКЛАДНЫХ рекомендациях, которые можно немедленно реализовать."""

    # =========================================================================
    # Reflection Helper Methods
    # =========================================================================

    def _build_chain_structure_summary(self) -> str:
        """
        Build a summary of the chain structure and configuration.

        Returns:
            Formatted chain structure summary
        """
        # Count step types
        step_type_counts: dict[str, int] = {}
        for step in self.steps:
            st = step.step_type
            step_type_counts[st] = step_type_counts.get(st, 0) + 1

        # Calculate parallelization potential
        exec_plan = self.get_execution_plan()
        parallelizable = sum(1 for level in exec_plan["execution_levels"] if level["parallelizable"])
        total_levels = len(exec_plan["execution_levels"])

        return f"""## Chain Structure

**Configuration:**
- Total Steps: {len(self.steps)}
- Max Workers: {self.max_workers}
- Parallelization Potential: {parallelizable}/{total_levels} levels can run in parallel
- Parallelization Ratio: {exec_plan.get("parallelization_ratio", 0):.1%}

**Step Type Distribution:**
{chr(10).join(f"- {step_type}: {count}" for step_type, count in sorted(step_type_counts.items()))}

**Execution Plan (Batches):**
{chr(10).join(f"Level {l['level']}: Steps {l['steps']} ({'parallel' if l['parallelizable'] else 'sequential'})" for l in exec_plan["execution_levels"])}"""  # noqa: E741

    def _build_step_definitions_summary(self) -> str:
        """
        Build a summary of the original step definitions for analysis.

        Returns:
            Formatted step definitions summary
        """
        lines = ["## Original Step Definitions", ""]
        for step in self.steps:
            lines.append(f"### Step {step.number}: {step.title}")
            lines.append(f"Type: {step.step_type}")
            lines.append(f"Dependencies: {step.dependencies if step.dependencies else 'None'}")

            # LLM-specific fields
            if step.is_llm_step():
                aim = getattr(step, "aim", "")
                reasoning_questions = getattr(step, "reasoning_questions", "")
                step_context_queries = getattr(step, "step_context_queries", None)
                lines.append(f"Aim: {aim}")
                lines.append(f"Reasoning Questions: {reasoning_questions}")
                lines.append(f"Context Queries: {step_context_queries if step_context_queries else 'None'}")

            # Non-LLM step config
            if step.step_config is not None:
                lines.append(f"Config: {step.step_config}")

            lines.append("")

        return "\n".join(lines)

    def _build_step_details_summary(self, max_preview_chars: int = 500) -> str:
        """
        Build a summary of step execution results with detailed information.

        Args:
            max_preview_chars: Maximum characters to show in result previews

        Returns:
            Formatted step details summary
        """
        result = self._last_result
        if not result:
            return "## Step Execution Results\n\nNo execution results available."

        lines = ["## Step Execution Results", ""]
        for step_result in result.step_results:
            status = "✓ SUCCESS" if step_result.success else "✗ FAILED"
            lines.append(f"### Step {step_result.step_number} ({step_result.step_type}): {step_result.step_title}")
            lines.append(f"Status: {status}")

            if step_result.execution_time:
                lines.append(f"Execution Time: {step_result.execution_time:.3f}s")

            if not step_result.success and step_result.error_message:
                lines.append(f"Error: {step_result.error_message}")

            # Show result preview
            result_preview = step_result.result if step_result.result else "(no result)"
            if len(result_preview) > max_preview_chars:
                result_preview = result_preview[:max_preview_chars] + "..."
            lines.append(f"Result Preview:\n{result_preview}")

            lines.append("")

        return "\n".join(lines)

    def _build_metric_scores_summary(self) -> str:
        """
        Build a summary of MetricBase evaluation scores from step and chain results.

        Step scores and chain scores are both included when available.
        Returns an empty string if no metrics were computed.

        This section is fed to the LLM during reflection so it can reference
        concrete quality signals (e.g. low keyword_coverage on a step) when
        producing improvement recommendations.
        """
        result = self._last_result
        if not result:
            return ""

        lines: list[str] = []

        # Step-level metric scores
        steps_with_metrics = [sr for sr in result.step_results if sr.metrics]
        if steps_with_metrics:
            lines.append("### Step Metric Scores")
            for sr in steps_with_metrics:
                for name, value in sr.metrics.items():
                    lines.append(f"- Step {sr.step_number} ({sr.step_title}): {name} = {value:.4g}")

        # Chain-level metric scores
        if result.metrics:
            lines.append("### Chain Metric Scores (on final output)")
            for name, value in result.metrics.items():
                lines.append(f"- {name} = {value:.4g}")

        if not lines:
            return ""

        return "## Evaluation Metric Scores\n\n" + "\n".join(lines)

    def _build_extra_feedback_summary(self, extra_feedback: dict[str, Any] | str, language: Language) -> str:
        """
        Format optional user-provided feedback/context for the reflection prompt.

        Args:
            extra_feedback: Dict of labelled entries or a plain string.
            language: Language for the section header.

        Returns:
            Formatted section string ready to be appended to the prompt.
        """
        if isinstance(extra_feedback, str):
            content = extra_feedback
        else:
            content = "\n".join(f"- **{k}**: {v}" for k, v in extra_feedback.items())

        if language == Language.ENGLISH:
            header = "## Additional Feedback"
        else:
            header = "## Дополнительный контекст"

        return f"{header}\n\n{content}"

    def _build_dataset_report_section(self, report: DatasetEvaluationReport, language: Language) -> str:
        """
        Build a prompt section from a DatasetEvaluationReport.

        Summarises dataset-level statistics and lists each selected problem case
        with input/output previews and its metric score.

        Args:
            report: The evaluation report produced by DatasetEvaluator.
            language: Language for section headers and labels.

        Returns:
            Formatted section string ready to be appended to the prompt.
        """
        total = len(report.all_results)
        k = len(report.selected_cases)

        strategy = report.strategy
        if strategy.mode == "threshold":
            direction = "below" if strategy.higher_is_better else "above"
            strategy_desc = f"threshold {strategy.threshold} ({direction})"
        else:  # top_k_worst
            strategy_desc = f"top-{strategy.k} worst"

        lines: list[str] = []

        if language == Language.ENGLISH:
            lines += [
                "## Dataset Evaluation",
                "",
                f"**Metric:** {report.metric_name}",
                f"**Cases evaluated:** {total}  |  "
                f"**Mean score:** {report.mean_score:.3f}  "
                f"(min: {report.min_score:.3f}, max: {report.max_score:.3f})",
                f"**Problem cases selected:** {k}/{total}  (strategy: {strategy_desc})",
                "",
                "> **Overfitting warning:** Optimize for patterns visible across "
                "problem cases, not individual quirks. Verify that improvements do "
                "not degrade the mean score.",
            ]
            if not report.selected_cases:
                lines += ["", "_No problem cases found — all cases pass the selection criterion._"]
            else:
                lines += ["", "### Problem Cases", ""]
                for i, r in enumerate(report.selected_cases, 1):
                    label = r.case.label or f"case_{i}"
                    input_preview = r.case.input[:300].replace("\n", " ")
                    output_preview = r.chain_output[:300].replace("\n", " ")
                    lines += [
                        f"**Case {i}** `{label}`  —  score: **{r.score:.3f}**",
                        f"- Input:  `{input_preview}`",
                        f"- Output: `{output_preview}`",
                        "",
                    ]
        else:  # Russian
            lines += [
                "## Оценка датасета",
                "",
                f"**Метрика:** {report.metric_name}",
                f"**Кейсов оценено:** {total}  |  "
                f"**Средний балл:** {report.mean_score:.3f}  "
                f"(мин: {report.min_score:.3f}, макс: {report.max_score:.3f})",
                f"**Проблемных кейсов отобрано:** {k}/{total}  (стратегия: {strategy_desc})",
                "",
                "> **Предупреждение о переоптимизации:** Оптимизируй под паттерны, "
                "видимые по нескольким проблемным кейсам, а не под отдельные случаи. "
                "Убедись, что улучшения не ухудшают средний балл.",
            ]
            if not report.selected_cases:
                lines += ["", "_Проблемных кейсов не найдено — все кейсы проходят критерий отбора._"]
            else:
                lines += ["", "### Проблемные кейсы", ""]
                for i, r in enumerate(report.selected_cases, 1):
                    label = r.case.label or f"case_{i}"
                    input_preview = r.case.input[:300].replace("\n", " ")
                    output_preview = r.chain_output[:300].replace("\n", " ")
                    lines += [
                        f"**Кейс {i}** `{label}`  —  балл: **{r.score:.3f}**",
                        f"- Входные данные:  `{input_preview}`",
                        f"- Результат цепочки: `{output_preview}`",
                        "",
                    ]

        return "\n".join(lines)

    def _build_execution_metrics_summary(self) -> str:
        """
        Build a summary of execution metrics and timing.

        Returns:
            Formatted execution metrics summary
        """
        result = self._last_result
        if not result:
            return "## Execution Metrics\n\nNo execution metrics available."

        successful = result.get_successful_steps()
        failed = result.get_failed_steps()

        # Calculate timing stats
        individual_times = [sr.execution_time for sr in result.step_results if sr.execution_time]
        total_individual_time = sum(individual_times) if individual_times else 0
        avg_time = total_individual_time / len(individual_times) if individual_times else 0
        wall_time = result.total_execution_time or 0

        # Calculate parallel efficiency (skip when max_workers is "auto")
        efficiency = 0.0
        if wall_time > 0 and total_individual_time > 0 and isinstance(self.max_workers, int):
            efficiency = min(1.0, total_individual_time / (wall_time * self.max_workers))

        lines = [
            "## Execution Metrics",
            "",
            f"**Overall Status:** {'SUCCESS' if result.success else 'FAILED'}",
            f"**Total Steps:** {len(result.step_results)}",
            f"**Successful:** {len(successful)}",
            f"**Failed:** {len(failed)}",
            "",
            "**Timing:**",
            f"Wall Clock Time: {wall_time:.3f}s",
            f"Total Individual Step Time: {total_individual_time:.3f}s",
            f"Average Step Time: {avg_time:.3f}s",
            f"Parallel Efficiency: {efficiency:.1%}",
            "",
        ]

        # Show slowest steps
        if individual_times:
            sorted_by_time = sorted(
                [(sr.step_number, sr.step_title, sr.execution_time) for sr in result.step_results if sr.execution_time],
                key=lambda x: x[2],
                reverse=True,
            )
            lines.append("**Slowest Steps:**")
            for step_num, title, time_val in sorted_by_time[:3]:
                lines.append(f"  Step {step_num} ({title}): {time_val:.3f}s")

        return "\n".join(lines)

    def _build_dependency_analysis(self) -> str:
        """
        Build an analysis of the dependency structure.

        Returns:
            Formatted dependency analysis
        """
        lines = [
            "## Dependency Analysis",
            "",
            "**Step Dependencies:**",
        ]

        # Find entry points (steps with no dependencies)
        entry_points = [s for s in self.steps if not s.dependencies]
        lines.append(f"Entry Points (no dependencies): {[s.number for s in entry_points]}")

        # Find leaves (steps that nothing depends on)
        depended_on = set()
        for step in self.steps:
            depended_on.update(step.dependencies)
        leaves = [s.number for s in self.steps if s.number not in depended_on]
        lines.append(f"Leaf Steps (nothing depends on them): {leaves}")

        # Calculate max depth
        def get_depth(step_num: int, memo: dict[int, int]) -> int:
            if step_num in memo:
                return memo[step_num]
            step = next(s for s in self.steps if s.number == step_num)
            if not step.dependencies:
                memo[step_num] = 1
                return 1
            max_dep_depth = max(get_depth(dep, memo) for dep in step.dependencies)
            memo[step_num] = max_dep_depth + 1
            return memo[step_num]

        depths = {s.number: get_depth(s.number, {}) for s in self.steps}
        max_depth = max(depths.values()) if depths else 0
        lines.append(f"Maximum Dependency Depth: {max_depth}")

        # Find potential parallelization opportunities
        lines.append("")
        lines.append("**Parallelization Opportunities:**")

        # Group steps by depth
        by_depth: dict[int, list[int]] = {}
        for step_num, depth in depths.items():
            by_depth.setdefault(depth, []).append(step_num)

        for depth in sorted(by_depth.keys()):
            steps_at_depth = by_depth[depth]
            if len(steps_at_depth) > 1:
                lines.append(f"Depth {depth}: Steps {steps_at_depth} can run in parallel")
            else:
                lines.append(f"Depth {depth}: Step {steps_at_depth[0]} (sequential)")

        # Check for potential bottlenecks
        lines.append("")
        lines.append("**Potential Bottlenecks:**")
        bottleneck_steps = []
        for step in self.steps:
            # Count how many steps depend on this one
            dependents = sum(1 for s in self.steps if step.number in s.dependencies)
            if dependents > 2:
                bottleneck_steps.append((step.number, step.title, dependents))

        if bottleneck_steps:
            for step_num, title, count in sorted(bottleneck_steps, key=lambda x: x[2], reverse=True):
                lines.append(f"Step {step_num} ({title}): {count} steps depend on this")
        else:
            lines.append("No significant bottlenecks detected")

        return "\n".join(lines)

    def get_last_result(self) -> ReasoningResult | None:
        """Get the last execution result (if any)."""
        return self._last_result

    def get_last_context(self) -> ReasoningContext | None:
        """Get the last execution context (if any)."""
        return self._last_context

    def get_execution_plan(self) -> dict[str, Any]:
        """
        Get the execution plan showing parallelization opportunities.

        Returns:
            Dictionary describing the execution plan
        """
        # Build dependency levels
        levels = []
        remaining_steps = self.steps.copy()
        # Track which step numbers have been "completed" (added to a level)
        completed_step_numbers = set()

        while remaining_steps:
            current_level = []
            for step in remaining_steps[:]:
                # A step can be added if all its dependencies are in completed_step_numbers
                deps_satisfied = all(dep in completed_step_numbers for dep in step.dependencies)
                if deps_satisfied:
                    current_level.append(step)
                    remaining_steps.remove(step)

            # Mark all steps in this level as completed
            for step in current_level:
                completed_step_numbers.add(step.number)

            if current_level:
                levels.append(
                    {
                        "level": len(levels) + 1,
                        "steps": [step.number for step in current_level],
                        "parallelizable": len(current_level) > 1,
                        "step_titles": [step.title for step in current_level],
                    }
                )

        return {
            "total_steps": len(self.steps),
            "max_workers": self.max_workers,
            "execution_levels": levels,
            "estimated_parallel_batches": len(levels),
            "parallelization_ratio": len([s for level in levels for s in level["steps"] if level["parallelizable"]])
            / len(self.steps)
            if self.steps
            else 0,
        }

    def get_step_dependencies(self) -> dict[int, list[int]]:
        """
        Get a mapping of step dependencies.

        Returns:
            Dictionary mapping step numbers to their dependencies
        """
        return {step.number: step.dependencies.copy() for step in self.steps}

    def get_steps_summary(self) -> list[dict[str, Any]]:
        """
        Get a summary of all steps in the chain.

        Returns:
            List of step summaries
        """
        return [
            {
                "number": step.number,
                "title": step.title,
                "step_type": step.step_type,
                "aim": getattr(step, "aim", ""),
                "dependencies": step.dependencies,
                "step_context_queries": getattr(step, "step_context_queries", None),
                "has_dependencies": step.has_dependencies(),
            }
            for step in self.steps
        ]

    # =========================================================================
    # CARE-namespace metadata accessors
    # =========================================================================

    def set_care_metadata(
        self,
        meta: "CareChainMetadata | None" = None,
        *,
        task_description: str | None = None,
        context_files: "list | None" = None,
        generated_by: str | None = None,
        mage_metadata: dict | None = None,
        display_name: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> "ReasoningChain":
        """Write a typed CARE metadata block under ``chain.metadata['care']``.

        Two call styles:

        * Pass a ready ``CareChainMetadata`` via ``meta=...`` (CARE's
          usual path — it has its own model and just hands it over).
        * Pass individual kwargs for ad-hoc construction (handy in
          examples and tests).

        Mixing the two raises ``ValueError``. Returns ``self`` so the
        call can chain.
        """
        from .models.care_metadata import (  # noqa: PLC0415
            CARE_METADATA_NAMESPACE, CareChainMetadata,
        )

        any_kwarg = any(
            v is not None for v in (
                task_description, context_files, generated_by, mage_metadata,
                display_name, description, tags,
            )
        )
        if meta is not None and any_kwarg:
            raise ValueError(
                "set_care_metadata: pass either `meta=` or the individual "
                "kwargs, not both."
            )
        if meta is None:
            meta = CareChainMetadata(
                task_description=task_description,
                context_files=context_files or [],
                generated_by=generated_by,
                mage_metadata=mage_metadata or {},
                display_name=display_name,
                description=description,
                tags=tags or [],
            )
        self.metadata[CARE_METADATA_NAMESPACE] = meta.to_metadata_dict()
        return self

    def get_care_metadata(self) -> "CareChainMetadata | None":
        """Read the CARE metadata block back as a typed model.

        Returns ``None`` when the chain has no ``care`` namespace
        attached — callers should treat that as "this chain wasn't
        created by CARE / a CARE-aware tool" and fall back to the raw
        ``chain.metadata`` if they need older keys.
        """
        from .models.care_metadata import (  # noqa: PLC0415
            CARE_METADATA_NAMESPACE, CareChainMetadata,
        )

        raw = self.metadata.get(CARE_METADATA_NAMESPACE)
        if not raw or not isinstance(raw, dict):
            return None
        return CareChainMetadata.model_validate(raw)

    # =========================================================================
    # Pre-flight introspection
    # =========================================================================

    def required_tools(self) -> list[str]:
        """De-duplicated tool names this chain will try to invoke.

        Walks every step's config looking for a ``tool_name`` field
        (``ToolStepDescription``). Does NOT walk
        ``ToolDiscoveryStepDescription`` — discovered tools are only
        known at runtime, so they can't be pre-flighted.

        Returns names in first-seen order so the list reads top-to-
        bottom in chain order, which CARE's TUI prefers over alpha
        sort.
        """
        seen: list[str] = []
        for step in self.steps:
            tool_name = self._extract_tool_name(step)
            if tool_name and tool_name not in seen:
                seen.append(tool_name)
        return seen

    def required_mcp_servers(self) -> list[str]:
        """De-duplicated MCP server names referenced by MCP steps.

        Pulls ``server.server_name`` from ``MCPStepDescription`` and
        ``MCPResourceStepDescription`` configs. Servers are
        self-described by their step config (transport + URL /
        command), so this is a diagnostic list — CARE renders it next
        to the run button so the user can sanity-check what the chain
        will dial out to.
        """
        seen: list[str] = []
        for step in self.steps:
            server_name = self._extract_mcp_server_name(step)
            if server_name and server_name not in seen:
                seen.append(server_name)
        return seen

    def required_skills(self) -> list[str]:
        """De-duplicated AgentSkill identifiers this chain references.

        For ``AgentSkillStepDescription`` steps:
        - When ``config.skill`` is a URI string (``github://...``,
          ``module://...``, etc.), it's returned verbatim.
        - When ``config.skill`` is an ``AgentSkillSource``, the first
          set option among ``path`` / ``name`` / ``git_url`` /
          ``package`` is returned as a string (prefixed with the
          appropriate scheme so callers can `resolve_skill` it).
        """
        seen: list[str] = []
        for step in self.steps:
            skill_id = self._extract_skill_id(step)
            if skill_id and skill_id not in seen:
                seen.append(skill_id)
        return seen

    def preflight(self, context: Any) -> "PreflightReport":
        """Build a :class:`PreflightReport` for ``context``.

        Compares the chain's static needs against the supplied
        :class:`ReasoningContext`. Missing-tool detection is the only
        comparison that's currently implemented — MCP servers and
        skills are reported as references only because they're
        resolved out-of-process / out-of-context.
        """
        from .models.preflight import PreflightReport  # noqa: PLC0415

        req_tools = self.required_tools()
        req_mcp = self.required_mcp_servers()
        req_skills = self.required_skills()

        registered = set()
        if context is not None and hasattr(context, "list_tools"):
            try:
                registered = set(context.list_tools())
            except Exception:
                registered = set()

        missing_tools = [t for t in req_tools if t not in registered]

        return PreflightReport(
            required_tools=req_tools,
            required_mcp_servers=req_mcp,
            required_skills=req_skills,
            missing_tools=missing_tools,
            missing_mcp_servers=[],
            missing_skills=[],
        )

    # ------------------------------------------------------------------
    # Helpers — keep introspection robust across typed + legacy steps.
    # ------------------------------------------------------------------

    @staticmethod
    def _step_type_value(step: Any) -> str:
        """Resolve a step's type to its string-enum value across both
        the typed-step ``step_type`` property and the legacy dict form."""
        st = getattr(step, "step_type", None)
        if st is None and isinstance(step, dict):
            st = step.get("step_type")
        if hasattr(st, "value"):
            return str(st.value)
        return str(st) if st else ""

    @staticmethod
    def _step_config(step: Any) -> Any:
        return getattr(step, "config", None) or getattr(step, "step_config", None)

    @classmethod
    def _extract_tool_name(cls, step: Any) -> Optional[str]:
        # Only Tool steps contribute tool_name — MCP steps also expose a
        # tool_name but it belongs to the remote MCP server, not the local
        # registry.
        if cls._step_type_value(step) != "tool":
            return None
        cfg = cls._step_config(step)
        if cfg is None:
            return None
        name = getattr(cfg, "tool_name", None)
        if isinstance(cfg, dict):
            name = cfg.get("tool_name")
        if isinstance(name, str) and name:
            return name
        return None

    @classmethod
    def _extract_mcp_server_name(cls, step: Any) -> Optional[str]:
        if cls._step_type_value(step) not in ("mcp", "mcp_resource"):
            return None
        cfg = cls._step_config(step)
        if cfg is None:
            return None
        server = getattr(cfg, "server", None)
        if server is not None:
            name = getattr(server, "server_name", None)
            if isinstance(name, str) and name:
                return name
        if isinstance(cfg, dict):
            server_dict = cfg.get("server")
            if isinstance(server_dict, dict):
                name = server_dict.get("server_name")
                if isinstance(name, str) and name:
                    return name
        return None

    @classmethod
    def _extract_skill_id(cls, step: Any) -> Optional[str]:
        if cls._step_type_value(step) != "agent_skill":
            return None
        cfg = cls._step_config(step)
        if cfg is None:
            return None
        skill = getattr(cfg, "skill", None)
        if isinstance(cfg, dict):
            skill = cfg.get("skill")
        # URI / plain string form
        if isinstance(skill, str) and skill:
            return skill
        # AgentSkillSource form — pick the first set option, preserving
        # the original URI scheme info via a prefix where the field name
        # alone wouldn't be enough to round-trip through ``resolve_skill``.
        for attr, prefix in (
            ("path", ""),
            ("name", "name://"),
            ("git_url", ""),
            ("package", "module://"),
        ):
            value = getattr(skill, attr, None) if skill is not None else None
            if isinstance(skill, dict):
                value = skill.get(attr)
            if isinstance(value, str) and value:
                return f"{prefix}{value}" if prefix else value
        return None

    # =========================================================================
    # Serialization Methods
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the chain to a dictionary.

        Handles both legacy StepDescription and new typed step classes.

        Returns:
            Dictionary representation of the chain
        """
        serialized_steps = []
        for step in self.steps:
            if isinstance(step, StepDescription):
                # Legacy format - use model_dump directly
                serialized_steps.append(step.model_dump(mode="json"))
            elif isinstance(step, StepDescriptionBase):
                # New typed step classes - need to convert to legacy format for JSON
                step_data = {
                    "number": step.number,
                    "title": step.title,
                    "dependencies": step.dependencies,
                    "triggered_by": list(getattr(step, "triggered_by", []) or []),
                    "step_type": step.step_type,
                    "checkpoint": getattr(step, "checkpoint", False),
                    "checkpoint_name": getattr(step, "checkpoint_name", None),
                    "replan_enabled": getattr(step, "replan_enabled", None),
                }
                # Add step_config for non-LLM steps
                if step.step_config is not None:
                    step_data["step_config"] = step.step_config.model_dump(mode="json")
                # Add LLM-specific fields
                if step.is_llm_step():
                    step_data["aim"] = getattr(step, "aim", "")
                    step_data["reasoning_questions"] = getattr(step, "reasoning_questions", "")
                    step_data["step_context_queries"] = getattr(step, "step_context_queries", [])
                    step_data["stage_action"] = getattr(step, "stage_action", "")
                    step_data["example_reasoning"] = getattr(step, "example_reasoning", "")
                    step_data["llm_config"] = getattr(step, "llm_config", None)
                    if step_data["llm_config"] is not None:
                        step_data["llm_config"] = step_data["llm_config"].model_dump(mode="json")
                    step_data["retry_max"] = getattr(step, "retry_max", None)
                    step_data["timeout"] = getattr(step, "timeout", None)
                serialized_steps.append(step_data)
            else:
                # Fallback - try model_dump
                serialized_steps.append(step.model_dump(mode="json"))

        try:
            _carl_version = _pkg_version("mmar-carl")
        except Exception:
            _carl_version = "unknown"

        result = {
            "format_version": self.FORMAT_VERSION,
            "carl_version": _carl_version,
            "max_workers": self.max_workers,
            "enable_progress": self.enable_progress,
            "metadata": self.metadata,
            "timeout": self.timeout,
            "replan_policy": self.replan_policy.model_dump(mode="json") if self.replan_policy else None,
            "search_config": self.prompt_template.search_config.model_dump() if self.prompt_template else None,
            "steps": serialized_steps,
        }
        if self.trace_name:
            result["trace_name"] = self.trace_name
        if self.session_id:
            result["session_id"] = self.session_id
        if self.default_llm_config is not None:
            result["default_llm_config"] = self.default_llm_config.model_dump(mode="json")
        return result

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the chain to a JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def estimate_cost(
        self,
        context: "ReasoningContext",
        *,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        default_output_tokens: int = 512,
        char_per_token: int = 4,
    ) -> "CostEstimate":
        """
        Estimate token usage and cost for this chain without calling any LLM.

        Walks each step in declaration order and produces a structured estimate
        per LLM-calling step (LLM, STRUCTURED_OUTPUT, EVALUATION when
        ``evaluation_method == "llm"``, and PARALLEL_SAMPLING for ``n_samples``
        × cost). Non-LLM steps (Tool, Memory, Transform, Conditional, …) appear
        in the result with ``calls_llm=False`` and zero tokens. Step types
        whose runtime LLM cost isn't fully modelled (``AGENT_SKILL`` LLM_AGENT
        iterations, ``AGENT_HANDOFF`` sub-chains) get a non-zero estimate when
        possible plus an explanatory note — the estimate is a lower bound for
        those.

        Token counts use a flat ``chars / char_per_token`` heuristic
        (default 4 chars/token, a reasonable proxy for English). For precise
        counts, plug in your provider's tokenizer externally.

        Pricing is opt-in. ``pricing`` maps model name to
        ``(input_per_1k_usd, output_per_1k_usd)``. Missing entries are
        surfaced as ``pricing_missing=True`` on the row; total cost in that
        row is reported as ``0.0`` so the chain total stays meaningful.

        Args:
            context: The :class:`ReasoningContext` the chain would execute against
                (used for ``outer_context`` size and the context's default model).
            pricing: Optional ``{model: (input_per_1k, output_per_1k)}`` map.
            default_output_tokens: Used when a step does not pin ``max_tokens``.
            char_per_token: Heuristic for tokens-from-chars (must be positive).

        Returns:
            :class:`CostEstimate` with per-step rows + chain-level totals.
        """
        from .cost import estimate_chain_cost
        return estimate_chain_cost(
            self,
            context,
            pricing=pricing,
            default_output_tokens=default_output_tokens,
            char_per_token=char_per_token,
        )

    def to_mermaid(self) -> str:
        """
        Export the chain as a Mermaid flowchart diagram.

        Each step is rendered as a node colored by step type:
        - LLM / StructuredOutput → blue
        - Tool → green
        - AgentSkill → purple
        - Memory → orange
        - Conditional → yellow
        - Transform → grey
        - MCP → teal

        Dependencies become directed edges.  Steps with no declared
        dependencies have an implicit edge from the previous step's
        number only when the chain is fully linear; otherwise they float
        at the top of the diagram (DAG roots).

        Returns:
            Mermaid flowchart string, suitable for rendering in GitHub
            Markdown or https://mermaid.live.

        Example::

            print(chain.to_mermaid())

            # flowchart TD
            #     S1["1: Analyse data\\n🧠 LLM"]:::llm
            #     S2["2: Search web\\n🔧 Tool"]:::tool
            #     S1 --> S2
            #     classDef llm fill:#3B82F6,color:#fff,...
        """
        _STYLE: dict[str, tuple[str, str]] = {
            "llm":              ("#3B82F6", "🧠 LLM"),
            "structured_output": ("#6366F1", "📋 Structured"),
            "tool":             ("#22C55E", "🔧 Tool"),
            "agent_skill":      ("#A855F7", "🎯 Skill"),
            "memory":           ("#F97316", "💾 Memory"),
            "conditional":      ("#EAB308", "⑂ Conditional"),
            "transform":        ("#6B7280", "⚙ Transform"),
            "mcp":              ("#14B8A6", "🔌 MCP"),
        }

        def _esc(text: str) -> str:
            """Escape double-quotes for Mermaid node labels."""
            return text.replace('"', "'")

        lines: list[str] = ["flowchart TD"]

        # Node definitions
        for step in self.steps:
            step_type = str(getattr(step, "step_type", "llm"))
            _, type_label = _STYLE.get(step_type, ("#6B7280", step_type))
            title = _esc(getattr(step, "title", f"Step {step.number}"))
            node_id = f"S{step.number}"
            label = f'{step.number}: {title}\\n{type_label}'
            css_class = step_type.replace("_", "")  # no hyphens/underscores in classDef names
            lines.append(f'    {node_id}["{label}"]:::{css_class}')

        # Edges from declared dependencies
        has_any_dep = False
        for step in self.steps:
            for dep in (step.dependencies or []):
                lines.append(f"    S{dep} --> S{step.number}")
                has_any_dep = True

        # If no step declared any dependency, draw a simple linear chain
        if not has_any_dep and len(self.steps) > 1:
            for i in range(len(self.steps) - 1):
                lines.append(f"    S{self.steps[i].number} --> S{self.steps[i + 1].number}")

        # classDef declarations (only for types that appear in this chain)
        used_types: set[str] = set()
        for step in self.steps:
            used_types.add(str(getattr(step, "step_type", "llm")))

        for step_type in sorted(used_types):
            color, _ = _STYLE.get(step_type, ("#6B7280", step_type))
            css_class = step_type.replace("_", "")
            lines.append(
                f"    classDef {css_class} fill:{color},color:#fff,"
                f"stroke:{color},stroke-width:2px,rx:6"
            )

        return "\n".join(lines)

    def to_mermaid_critical_path(self, result: "ReasoningResult") -> str:
        """Mermaid flowchart with the critical-path edges highlighted in red.

        The critical path is the *longest cumulative latency chain* through
        the DAG — speeding up any step *not* on this path doesn't reduce
        total wall-clock time. Answers "which step do I optimize first?"

        Uses per-step ``execution_time`` from ``result.step_results``. Steps
        that don't appear in the result (e.g. skipped branches) are
        treated as zero-cost.

        Args:
            result: The :class:`ReasoningResult` from this chain's most
                recent execution. Per-step ``execution_time`` populates
                the path weights.

        Returns:
            Mermaid ``flowchart TD`` string, with the same node-type
            colouring as :meth:`to_mermaid` plus a trailing block of
            ``linkStyle`` directives painting critical edges thick red.
            Includes a small text annotation showing the critical-path
            total time vs the chain's wall-clock total — so the user can
            see how much of the run is on the critical path.
        """
        # Per-step execution time, defaulting to 0 for missing steps.
        timings: dict[int, float] = {
            sr.step_number: float(sr.execution_time or 0.0)
            for sr in result.step_results
        }

        # Adjacency: step_number → list of (dependency_step_numbers).
        # Pre-compute parents so we can do topological / longest-path math.
        step_by_num: dict[int, Any] = {step.number: step for step in self.steps}
        parents: dict[int, list[int]] = {
            step.number: list(step.dependencies or []) for step in self.steps
        }
        # If the chain is purely linear (no deps declared anywhere) the existing
        # to_mermaid synthesises implicit edges — mirror that here.
        any_dep = any(deps for deps in parents.values())
        if not any_dep and len(self.steps) > 1:
            sorted_steps = sorted(self.steps, key=lambda s: s.number)
            for i in range(1, len(sorted_steps)):
                parents[sorted_steps[i].number] = [sorted_steps[i - 1].number]

        # Longest-path DP — for each step, cumulative critical time to reach
        # its END (including the step's own execution_time), plus the parent
        # that gave the longest chain.
        ordered = sorted(step_by_num.keys())  # step numbers are monotonic per CARL convention
        cum_time: dict[int, float] = {}
        best_parent: dict[int, int | None] = {}
        for num in ordered:
            own = timings.get(num, 0.0)
            in_deps = parents.get(num, [])
            if not in_deps:
                cum_time[num] = own
                best_parent[num] = None
            else:
                best_p = max(in_deps, key=lambda p: cum_time.get(p, 0.0))
                cum_time[num] = cum_time.get(best_p, 0.0) + own
                best_parent[num] = best_p

        # The critical path ends at whichever step has the max cum_time.
        end_step = max(cum_time, key=lambda n: cum_time[n]) if cum_time else None
        critical_steps: set[int] = set()
        if end_step is not None:
            cur = end_step
            while cur is not None:
                critical_steps.add(cur)
                cur = best_parent[cur]

        # Critical *edges* are between consecutive critical steps along the
        # chosen parent chain — not every edge between two critical nodes.
        critical_edges: set[tuple[int, int]] = set()
        if end_step is not None:
            cur = end_step
            while best_parent[cur] is not None:
                p = best_parent[cur]
                critical_edges.add((p, cur))  # type: ignore[arg-type]
                cur = p  # type: ignore[assignment]

        # Render — start with the base mermaid block, intercept the edge
        # lines so we can record their indices for linkStyle directives.
        _STYLE: dict[str, tuple[str, str]] = {
            "llm":              ("#3B82F6", "🧠 LLM"),
            "structured_output": ("#6366F1", "📋 Structured"),
            "tool":             ("#22C55E", "🔧 Tool"),
            "agent_skill":      ("#A855F7", "🎯 Skill"),
            "memory":           ("#F97316", "💾 Memory"),
            "conditional":      ("#EAB308", "⑂ Conditional"),
            "transform":        ("#6B7280", "⚙ Transform"),
            "mcp":              ("#14B8A6", "🔌 MCP"),
        }

        def _esc(text: str) -> str:
            return text.replace('"', "'")

        lines: list[str] = ["flowchart TD"]

        # Node definitions — append critical-step timing in the label.
        for step in self.steps:
            step_type = str(getattr(step, "step_type", "llm"))
            _, type_label = _STYLE.get(step_type, ("#6B7280", step_type))
            title = _esc(getattr(step, "title", f"Step {step.number}"))
            t = timings.get(step.number, 0.0)
            critical_marker = " ⭐" if step.number in critical_steps else ""
            label = f"{step.number}: {title}\\n{type_label}\\n{t:.2f}s{critical_marker}"
            node_id = f"S{step.number}"
            css_class = step_type.replace("_", "")
            lines.append(f'    {node_id}["{label}"]:::{css_class}')

        # Emit edges in the same order so linkStyle indexing matches.
        edge_index = 0
        critical_link_indices: list[int] = []
        any_dep_emitted = False
        for step in self.steps:
            for dep in (step.dependencies or []):
                lines.append(f"    S{dep} --> S{step.number}")
                if (dep, step.number) in critical_edges:
                    critical_link_indices.append(edge_index)
                edge_index += 1
                any_dep_emitted = True

        if not any_dep_emitted and len(self.steps) > 1:
            for i in range(len(self.steps) - 1):
                a = self.steps[i].number
                b = self.steps[i + 1].number
                lines.append(f"    S{a} --> S{b}")
                if (a, b) in critical_edges:
                    critical_link_indices.append(edge_index)
                edge_index += 1

        # classDef declarations (only for types that appear in this chain).
        used_types: set[str] = set()
        for step in self.steps:
            used_types.add(str(getattr(step, "step_type", "llm")))
        for step_type in sorted(used_types):
            color, _ = _STYLE.get(step_type, ("#6B7280", step_type))
            css_class = step_type.replace("_", "")
            lines.append(
                f"    classDef {css_class} fill:{color},color:#fff,"
                f"stroke:{color},stroke-width:2px,rx:6"
            )

        # Critical edge styling — red, thick.
        if critical_link_indices:
            indices = ",".join(str(i) for i in critical_link_indices)
            lines.append(f"    linkStyle {indices} stroke:#EF4444,stroke-width:4px")

        # Annotation: critical path vs the "fully serial baseline" (sum of all
        # step times). Parallel savings = how much of the serial total was
        # eliminated by overlapping steps. 0% = chain is fully serial (every
        # step on the critical path); higher % = more was off the critical
        # path and could run concurrently.
        critical_time = cum_time.get(end_step, 0.0) if end_step is not None else 0.0
        serial_baseline = sum(timings.values())
        savings_pct = 0.0
        if serial_baseline > 0:
            savings_pct = max(0.0, (serial_baseline - critical_time) / serial_baseline * 100.0)
        # Mermaid `%%` comments are ignored by renderers but visible when
        # reading the raw source.
        lines.append(
            f"    %% critical path = {critical_time:.2f}s / "
            f"serial baseline = {serial_baseline:.2f}s "
            f"(parallel savings ≈ {savings_pct:.0f}%)"
        )

        return "\n".join(lines)

    def to_mermaid_heatmap(
        self,
        result: "ReasoningResult",
        *,
        metric: str = "latency",
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        default_model: Optional[str] = None,
    ) -> str:
        """Mermaid flowchart with nodes coloured by a chosen metric.

        Surfaces "which step is expensive / slow / token-heavy?" at a
        glance — complements :meth:`to_mermaid_critical_path` (which
        highlights the longest *path*; this method highlights the hottest
        *nodes*).

        Three metrics:

        * ``"latency"`` (default) — per-step wall-clock time from
          ``StepExecutionResult.execution_time``.
        * ``"tokens"`` — per-step total tokens from
          ``result.token_usage_by_step``. Non-LLM steps render in grey
          (no usage recorded).
        * ``"cost"`` — per-step USD via ``pricing`` + ``default_model``
          (same lookup as :meth:`ReasoningResult.format_profiling_table`).
          When ``pricing`` is missing or no step matches, cost defaults
          to 0 (rendered as the coolest colour).

        Colours go from green (0% of max) through yellow (50%) to red
        (100%). Each step's label includes the metric value so the user
        can correlate node colour with absolute magnitude.

        Returns a Mermaid ``flowchart TD`` block. Existing
        :meth:`to_mermaid` is unchanged.
        """
        if metric not in {"latency", "tokens", "cost"}:
            raise ValueError(
                f"Unknown metric {metric!r}. Use 'latency', 'tokens', or 'cost'."
            )

        # Per-step metric values + display strings.
        values: dict[int, float] = {}
        labels: dict[int, str] = {}

        if metric == "latency":
            for sr in result.step_results:
                v = float(sr.execution_time or 0.0)
                values[sr.step_number] = v
                labels[sr.step_number] = f"{v:.2f}s"
        elif metric == "tokens":
            usage = result.token_usage_by_step
            for sr in result.step_results:
                u = usage.get(sr.step_number)
                v = float(u["total"]) if u else 0.0
                values[sr.step_number] = v
                labels[sr.step_number] = (
                    f"{int(v):,} tok" if u else "no llm"
                )
        else:  # cost
            usage = result.token_usage_by_step
            for sr in result.step_results:
                u = usage.get(sr.step_number)
                if u is None or not pricing:
                    values[sr.step_number] = 0.0
                    labels[sr.step_number] = "$0.0000" if pricing else "no $"
                    continue
                # Model resolution mirrors format_profiling_table.
                model = (
                    getattr(getattr(sr, "config", None), "model", None)
                    or getattr(getattr(sr, "llm_config", None), "model", None)
                    or default_model
                )
                if model and model in pricing:
                    in_price, out_price = pricing[model]
                    cost = (u["prompt"] / 1000.0) * in_price + (u["completion"] / 1000.0) * out_price
                    values[sr.step_number] = cost
                    labels[sr.step_number] = f"${cost:.4f}"
                else:
                    values[sr.step_number] = 0.0
                    labels[sr.step_number] = "(no price)"

        # Determine max for normalisation.
        max_value = max(values.values(), default=0.0)

        def _heat_color(value: float) -> str:
            """Green (low) → yellow (mid) → red (high). Returns hex string."""
            if max_value <= 0:
                return "#9CA3AF"  # neutral grey for all-zero
            ratio = max(0.0, min(1.0, value / max_value))
            # 3-stop gradient: green (#22C55E) at 0, yellow (#EAB308) at 0.5,
            # red (#EF4444) at 1.0. Linear interpolation between adjacent stops.
            stops = [
                (0.0, (34, 197, 94)),    # green
                (0.5, (234, 179, 8)),    # yellow
                (1.0, (239, 68, 68)),    # red
            ]
            for (r1, c1), (r2, c2) in zip(stops, stops[1:]):
                if ratio <= r2:
                    t = (ratio - r1) / (r2 - r1) if r2 > r1 else 0.0
                    rgb = tuple(
                        int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)
                    )
                    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            return f"#{stops[-1][1][0]:02X}{stops[-1][1][1]:02X}{stops[-1][1][2]:02X}"

        def _esc(text: str) -> str:
            return text.replace('"', "'")

        lines: list[str] = ["flowchart TD"]
        # Track unique colours per step so we can emit a classDef per step.
        for step in self.steps:
            title = _esc(getattr(step, "title", f"Step {step.number}"))
            v = values.get(step.number, 0.0)
            display = labels.get(step.number, "")
            node_id = f"S{step.number}"
            # Mermaid renders `<br/>` as a line break inside node labels;
            # bare `\n` was being passed through as the literal two-character
            # escape sequence and showed up as `\n` in the rendered diagram.
            label = f"{step.number}: {title}<br/>{display}"
            color = _heat_color(v)
            # Use inline `style` so each node gets its own colour without
            # per-step classDef clutter.
            lines.append(f'    {node_id}["{label}"]')
            lines.append(
                f"    style {node_id} fill:{color},color:#fff,"
                f"stroke:{color},stroke-width:2px,rx:6"
            )

        # Edges — same logic as to_mermaid / to_mermaid_critical_path.
        any_dep_emitted = False
        for step in self.steps:
            for dep in (step.dependencies or []):
                lines.append(f"    S{dep} --> S{step.number}")
                any_dep_emitted = True
        if not any_dep_emitted and len(self.steps) > 1:
            for i in range(len(self.steps) - 1):
                a = self.steps[i].number
                b = self.steps[i + 1].number
                lines.append(f"    S{a} --> S{b}")

        # Summary annotation: max value + heatmap meaning.
        legend_label = {
            "latency": "wall time per step",
            "tokens": "total tokens per step",
            "cost": "USD per step",
        }[metric]
        max_str = labels.get(
            max(values, key=lambda n: values[n]) if values else 0, ""
        )
        lines.append(
            f"    %% heatmap: {legend_label} — green=low, red=high. "
            f"Max: {max_str}"
        )
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the chain to a JSON file.

        Args:
            path: File path to save to
        """
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any], use_typed_steps: bool = False) -> "ReasoningChain":
        """
        Deserialize a chain from a dictionary.

        Args:
            data: Dictionary representation of the chain
            use_typed_steps: If True, create typed step classes (LLMStepDescription, etc.)
                           If False (default), create legacy StepDescription for backward compatibility

        Returns:
            Reconstructed ReasoningChain
        """
        # newer-format detection. Surface a typed
        # :class:`ChainFormatNewerError` so callers (CARE TUI) can prompt
        # the user to upgrade ``mmar-carl`` instead of silently losing
        # fields the older library doesn't understand.
        saved_format = data.get("format_version")
        if saved_format is not None and saved_format > cls.FORMAT_VERSION:
            raise ChainFormatNewerError(
                required_version=saved_format,
                this_version=cls.FORMAT_VERSION,
            )

        steps: list[StepDescription | AnyStepDescription] = []
        for step_data in data.get("steps", []):
            # Handle step_config reconstruction based on step_type
            step_type = step_data.get("step_type", "llm")
            step_config_data = step_data.get("step_config")

            if step_config_data is not None:
                step_config = _reconstruct_step_config(step_type, step_config_data)
                step_data["step_config"] = step_config

            llm_config_data = step_data.get("llm_config")
            if llm_config_data is not None and not isinstance(llm_config_data, LLMStepConfig):
                step_data["llm_config"] = LLMStepConfig.model_validate(llm_config_data)

            if use_typed_steps:
                # Create typed step class
                step = _create_typed_step_from_dict(step_data)
            else:
                # Create legacy StepDescription for backward compatibility
                step = StepDescription.model_validate(step_data)
            steps.append(step)

        # Reconstruct search config
        search_config = None
        if data.get("search_config"):
            search_config = ContextSearchConfig.model_validate(data["search_config"])

        # Reconstruct RE-PLAN policy
        replan_policy = None
        if data.get("replan_policy"):
            replan_policy = ReplanPolicy.model_validate(data["replan_policy"])

        default_llm_config = None
        if data.get("default_llm_config"):
            default_llm_config = LLMStepConfig.model_validate(data["default_llm_config"])

        return cls(
            steps=steps,
            max_workers=data.get("max_workers", 3),
            enable_progress=data.get("enable_progress", False),
            metadata=data.get("metadata", {}),
            search_config=search_config,
            timeout=data.get("timeout"),
            trace_name=data.get("trace_name"),
            session_id=data.get("session_id"),
            replan_policy=replan_policy,
            default_llm_config=default_llm_config,
        )

    @classmethod
    def from_dict_typed(cls, data: dict[str, Any]) -> "ReasoningChain":
        """
        Deserialize a chain from a dictionary using typed step classes.

        This creates LLMStepDescription, ToolStepDescription, etc. instead of
        legacy StepDescription.

        Args:
            data: Dictionary representation of the chain

        Returns:
            Reconstructed ReasoningChain with typed step classes
        """
        return cls.from_dict(data, use_typed_steps=True)

    @classmethod
    def migrate(cls, data: dict[str, Any], to_version: int | None = None) -> dict[str, Any]:
        """
        Migrate a serialized chain dict from an older format to a newer one.

        Call this before passing data to ``from_dict()`` when loading chains
        serialized by an older version of the library.

        Args:
            data: Raw dict from ``chain.to_dict()`` or a JSON file.
            to_version: Target format version.  Defaults to the current
                ``FORMAT_VERSION`` supported by this library.

        Returns:
            A (possibly modified) copy of ``data`` at the requested version.

        Raises:
            ValueError: If the requested ``to_version`` is unknown.

        Example::

            raw = json.loads(path.read_text())
            migrated = ReasoningChain.migrate(raw)
            chain = ReasoningChain.from_dict(migrated)
        """
        target = to_version if to_version is not None else cls.FORMAT_VERSION
        if target > cls.FORMAT_VERSION:
            raise ValueError(
                f"Cannot migrate to format_version={target}: "
                f"this library only knows up to format_version={cls.FORMAT_VERSION}."
            )

        data = dict(data)  # shallow copy — callers keep their original
        current = data.get("format_version", 0)

        # Migration ladder: add an `if current < N:` block for each future bump.
        # Version 0 → 1: the "version" string key was replaced by the integer
        # "format_version" key and the "carl_version" string was added.
        if current < 1:
            data.pop("version", None)  # remove old freeform string
            data["format_version"] = 1

        return data

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningChain":
        """
        Deserialize a chain from a JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            Reconstructed ReasoningChain
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ReasoningChain":
        """
        Load a chain from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Reconstructed ReasoningChain
        """
        path = Path(path)
        json_str = path.read_text(encoding="utf-8")
        return cls.from_json(json_str)


def _reconstruct_step_config(step_type: str, config_data: dict[str, Any]) -> Any:
    """Reconstruct step configuration based on step type."""
    if step_type == StepType.TOOL or step_type == "tool":
        # Reconstruct ToolParameter objects if present
        if "parameters" in config_data:
            config_data["parameters"] = [ToolParameter.model_validate(p) for p in config_data["parameters"]]
        return ToolStepConfig.model_validate(config_data)

    elif step_type == StepType.MCP or step_type == "mcp":
        # Reconstruct MCPServerConfig
        if "server" in config_data:
            config_data["server"] = MCPServerConfig.model_validate(config_data["server"])
        return MCPStepConfig.model_validate(config_data)

    elif step_type == StepType.MEMORY or step_type == "memory":
        return MemoryStepConfig.model_validate(config_data)

    elif step_type == StepType.TRANSFORM or step_type == "transform":
        return TransformStepConfig.model_validate(config_data)

    elif step_type == StepType.CONDITIONAL or step_type == "conditional":
        return ConditionalStepConfig.model_validate(config_data)

    elif step_type == StepType.STRUCTURED_OUTPUT or step_type == "structured_output":
        return StructuredOutputStepConfig.model_validate(config_data)

    elif step_type == StepType.AGENT_SKILL or step_type == "agent_skill":
        # Reconstruct AgentSkillSource if needed
        if "skill" in config_data and isinstance(config_data["skill"], dict):
            config_data["skill"] = AgentSkillSource.model_validate(config_data["skill"])
        return AgentSkillStepConfig.model_validate(config_data)

    elif step_type == StepType.MCP_RESOURCE or step_type == "mcp_resource":
        if "server" in config_data and isinstance(config_data["server"], dict):
            config_data["server"] = MCPServerConfig.model_validate(config_data["server"])
        return MCPResourceStepConfig.model_validate(config_data)

    elif step_type == StepType.EVALUATION or step_type == "evaluation":
        return EvaluationStepConfig.model_validate(config_data)

    elif step_type == StepType.AGENT_HANDOFF or step_type == "agent_handoff":
        return AgentHandoffStepConfig.model_validate(config_data)

    elif step_type == StepType.SUPERVISOR or step_type == "supervisor":
        return SupervisorStepConfig.model_validate(config_data)

    elif step_type == StepType.DEBATE or step_type == "debate":
        return DebateStepConfig.model_validate(config_data)

    elif step_type == StepType.PARALLEL_SAMPLING or step_type == "parallel_sampling":
        return ParallelSamplingStepConfig.model_validate(config_data)

    elif step_type == StepType.TOOL_DISCOVERY or step_type == "tool_discovery":
        return ToolDiscoveryStepConfig.model_validate(config_data)

    elif step_type == StepType.HUMAN_INPUT or step_type == "human_input":
        return HumanInputStepConfig.model_validate(config_data)

    return None


def _create_typed_step_from_dict(step_data: dict[str, Any]) -> AnyStepDescription:
    """
    Create a typed step class from dictionary data.

    Args:
        step_data: Dictionary containing step data

    Returns:
        The appropriate typed step description instance
    """
    step_type = step_data.get("step_type", "llm")

    if step_type == StepType.LLM or step_type == "llm":
        return LLMStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            aim=step_data.get("aim", ""),
            reasoning_questions=step_data.get("reasoning_questions", ""),
            step_context_queries=step_data.get("step_context_queries", []),
            stage_action=step_data.get("stage_action", ""),
            example_reasoning=step_data.get("example_reasoning", ""),
            llm_config=step_data.get("llm_config"),
            retry_max=step_data.get("retry_max"),
            timeout=step_data.get("timeout"),
        )
    elif step_type == StepType.TOOL or step_type == "tool":
        return ToolStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.MCP or step_type == "mcp":
        return MCPStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.MEMORY or step_type == "memory":
        return MemoryStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.TRANSFORM or step_type == "transform":
        return TransformStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.CONDITIONAL or step_type == "conditional":
        return ConditionalStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.STRUCTURED_OUTPUT or step_type == "structured_output":
        return StructuredOutputStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.AGENT_SKILL or step_type == "agent_skill":
        config_data = step_data.get("step_config") or step_data.get("config", {})
        if isinstance(config_data, dict):
            if "skill" in config_data and isinstance(config_data["skill"], dict):
                config_data["skill"] = AgentSkillSource.model_validate(config_data["skill"])
            config = AgentSkillStepConfig.model_validate(config_data)
        else:
            config = config_data
        return AgentSkillStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=config,
        )
    elif step_type == StepType.MCP_RESOURCE or step_type == "mcp_resource":
        return MCPResourceStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.EVALUATION or step_type == "evaluation":
        return EvaluationStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.AGENT_HANDOFF or step_type == "agent_handoff":
        # ``sub_chain`` is exclude=True and not in the dict — caller must
        # rebuild it from an entity_id stored in ``chain.metadata`` (CARE
        # convention). We seed a placeholder so the constructor succeeds.
        from .models.steps import LLMStepDescription as _LLMStep  # local — keep import surface clean
        placeholder_chain = ReasoningChain(
            steps=[_LLMStep(number=1, title="__placeholder__", aim="__placeholder__")]
        )
        return AgentHandoffStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            sub_chain=placeholder_chain,
            config=step_data["step_config"],
        )
    elif step_type == StepType.SUPERVISOR or step_type == "supervisor":
        # ``agents`` is exclude=True (runtime-only) — placeholder empty dict;
        # caller injects the real sub-chains keyed by agent name.
        return SupervisorStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            agents={},
            config=step_data["step_config"],
        )
    elif step_type == StepType.DEBATE or step_type == "debate":
        return DebateStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.PARALLEL_SAMPLING or step_type == "parallel_sampling":
        # ``base_step`` is exclude=True — placeholder LLM step.
        placeholder_base = LLMStepDescription(
            number=step_data["number"],
            title="__placeholder__",
            aim="__placeholder__",
        )
        return ParallelSamplingStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            base_step=placeholder_base,
            config=step_data["step_config"],
        )
    elif step_type == StepType.TOOL_DISCOVERY or step_type == "tool_discovery":
        return ToolDiscoveryStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    elif step_type == StepType.HUMAN_INPUT or step_type == "human_input":
        return HumanInputStepDescription(
            number=step_data["number"],
            title=step_data["title"],
            dependencies=step_data.get("dependencies", []),
            checkpoint=step_data.get("checkpoint", False),
            checkpoint_name=step_data.get("checkpoint_name"),
            replan_enabled=step_data.get("replan_enabled"),
            config=step_data["step_config"],
        )
    else:
        raise ValueError(f"Unknown step type: {step_type}")


class ChainBuilder:
    """
    Builder pattern for constructing reasoning chains.

    Provides a fluent interface for building complex reasoning chains.
    """

    def __init__(self):
        """Initialize the chain builder."""
        self.steps: list[StepDescription | StepDescriptionBase | AnyStepDescription] = []
        self.max_workers: int | str = 3
        self.prompt_template: PromptTemplate | None = None
        self.search_config: ContextSearchConfig | None = None
        self.enable_progress: bool = False
        self.metadata: dict[str, Any] = {}
        self.timeout: float | None = None
        self.trace_name: str | None = None
        self.session_id: str | None = None
        self.replan_policy: ReplanPolicy | None = None
        self.default_llm_config: LLMStepConfig | None = None

    def add_step(
        self,
        number: int,
        title: str,
        aim: str,
        reasoning_questions: str,
        stage_action: str,
        example_reasoning: str,
        dependencies: list[int] | None = None,
        step_context_queries: list[ContextQuery | str] | None = None,
        llm_config: LLMStepConfig | None = None,
        execution_mode: ExecutionMode | str | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add an LLM reasoning step to the chain.

        Args:
            number: Step number
            title: Step title
            aim: Step objective
            reasoning_questions: Key questions to answer
            stage_action: Action to perform
            example_reasoning: Example of expert reasoning
            dependencies: List of step numbers this depends on
            step_context_queries: RAG-like context queries
            llm_config: Optional per-step LLM config
            execution_mode: Optional execution mode shortcut ("fast", "self_critic")

        Returns:
            Self for method chaining
        """
        resolved_llm_config = llm_config
        if execution_mode is not None:
            mode = execution_mode if isinstance(execution_mode, ExecutionMode) else ExecutionMode(str(execution_mode))
            if resolved_llm_config is None:
                resolved_llm_config = LLMStepConfig(execution_mode=mode)
            else:
                resolved_llm_config = resolved_llm_config.model_copy(update={"execution_mode": mode})

        step = LLMStepDescription(
            number=number,
            title=title,
            aim=aim,
            reasoning_questions=reasoning_questions,
            stage_action=stage_action,
            example_reasoning=example_reasoning,
            dependencies=dependencies or [],
            step_context_queries=step_context_queries or [],
            llm_config=resolved_llm_config,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
        )
        self.steps.append(step)
        return self

    def add_tool_step(
        self,
        number: int,
        title: str,
        tool_name: str,
        input_mapping: dict[str, str] | None = None,
        dependencies: list[int] | None = None,
        tool_description: str = "",
        timeout: float = 30.0,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a tool execution step to the chain.

        Args:
            number: Step number
            title: Step title
            tool_name: Name of the registered tool to call
            input_mapping: Maps context keys to tool parameters
            dependencies: List of step numbers this depends on
            tool_description: Description of the tool
            timeout: Execution timeout in seconds

        Returns:
            Self for method chaining
        """
        step = ToolStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=ToolStepConfig(
                tool_name=tool_name,
                tool_description=tool_description,
                input_mapping=input_mapping or {},
                timeout=timeout,
            ),
        )
        self.steps.append(step)
        return self

    def add_mcp_step(
        self,
        number: int,
        title: str,
        server_name: str,
        tool_name: str,
        command: str | None = None,
        args: list[str] | None = None,
        arguments: dict[str, Any] | None = None,
        argument_mapping: dict[str, str] | None = None,
        dependencies: list[int] | None = None,
        timeout: float = 60.0,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add an MCP protocol step to the chain.

        Args:
            number: Step number
            title: Step title
            server_name: MCP server name
            tool_name: MCP tool to call
            command: Command to start stdio server
            args: Arguments for the server command
            arguments: Static arguments for the tool
            argument_mapping: Maps context keys to tool arguments
            dependencies: List of step numbers this depends on
            timeout: Execution timeout in seconds

        Returns:
            Self for method chaining
        """
        step = MCPStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=MCPStepConfig(
                server=MCPServerConfig(
                    server_name=server_name,
                    command=command,
                    args=args or [],
                ),
                tool_name=tool_name,
                arguments=arguments or {},
                argument_mapping=argument_mapping or {},
                timeout=timeout,
            ),
        )
        self.steps.append(step)
        return self

    def add_memory_step(
        self,
        number: int,
        title: str,
        operation: str,
        memory_key: str,
        value_source: str | None = None,
        default_value: Any = None,
        namespace: str = "default",
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a memory operation step to the chain.

        Args:
            number: Step number
            title: Step title
            operation: Memory operation (read, write, append, delete, list)
            memory_key: Key in memory store
            value_source: Source of value for write operations
            default_value: Default value if key not found
            namespace: Memory namespace
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        from .models import MemoryOperation

        step = MemoryStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=MemoryStepConfig(
                operation=MemoryOperation(operation),
                memory_key=memory_key,
                value_source=value_source,
                default_value=default_value,
                namespace=namespace,
            ),
        )
        self.steps.append(step)
        return self

    def add_transform_step(
        self,
        number: int,
        title: str,
        transform_type: str,
        input_key: str = "$history[-1]",
        output_format: str | None = None,
        expression: str | None = None,
        map_template: str | None = None,
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a data transformation step to the chain (no LLM call).

        Args:
            number: Step number
            title: Step title
            transform_type: Type (extract, format, aggregate, filter, map)
            input_key: Source of input data
            output_format: Format template for 'format' type
            expression: Regex for 'extract' or 'filter' types
            map_template: Template for 'map' type
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        step = TransformStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=TransformStepConfig(
                transform_type=transform_type,  # type: ignore
                input_key=input_key,
                output_format=output_format,
                expression=expression,
                map_template=map_template,
            ),
        )
        self.steps.append(step)
        return self

    def add_conditional_step(
        self,
        number: int,
        title: str,
        branches: list[tuple[str, int] | ConditionalBranch],
        default_step: int | None = None,
        condition_context_key: str = "$history[-1]",
        dependencies: list[int] | None = None,
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
        replan_enabled: bool | None = None,
    ) -> "ChainBuilder":
        """
        Add a conditional branching step to the chain.

        Args:
            number: Step number
            title: Step title
            branches: List of (condition, next_step) tuples OR ConditionalBranch objects
            default_step: Default step if no condition matches
            condition_context_key: Context key to evaluate
            dependencies: List of step numbers this depends on

        Returns:
            Self for method chaining
        """
        from .models import ConditionalBranch

        # Normalize branches to ConditionalBranch objects
        normalized_branches = []
        for branch in branches:
            if isinstance(branch, tuple):
                condition, next_step = branch
                normalized_branches.append(
                    ConditionalBranch(condition=condition, next_step=next_step)
                )
            elif isinstance(branch, ConditionalBranch):
                normalized_branches.append(branch)
            else:
                raise ValueError(
                    f"Branch must be tuple (condition, next_step) or ConditionalBranch, "
                    f"got {type(branch)}"
                )

        step = ConditionalStepDescription(
            number=number,
            title=title,
            dependencies=dependencies or [],
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=ConditionalStepConfig(
                branches=normalized_branches,
                default_step=default_step,
                condition_context_key=condition_context_key,
            ),
        )
        self.steps.append(step)
        return self

    def add_if_else(
        self,
        number: int,
        title: str,
        condition: str,
        if_true: int,
        if_false: int,
        condition_step: int | None = None,
        condition_context_key: str = "$history[-1]",
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
    ) -> "ChainBuilder":
        """
        Add a binary If-Else conditional step.

        Convenience wrapper around :meth:`add_conditional_step` for the common case
        of two branches: one taken when ``condition`` is true, another when false.

        Args:
            number: Step number.
            title: Step title.
            condition: Condition expression evaluated by simpleeval against the
                ``condition_context_key`` value.  The value is available as the
                name ``value`` (alias: ``v``).  Example: ``"'urgent' in value"``.
            if_true: Step number to execute when the condition is **true**.
            if_false: Step number to execute when the condition is **false**.
            condition_step: If provided, that step's number is automatically added
                to this step's ``dependencies`` list.
            condition_context_key: Context key whose resolved value is tested.
                Defaults to ``"$history[-1]"`` (most recent history entry).
            checkpoint: Whether to checkpoint this step.
            checkpoint_name: Name for the checkpoint.

        Returns:
            ``self`` for method chaining.

        Example::

            chain = (
                ChainBuilder()
                .add_tool_step(1, "Classify", tool_name="classify", input_mapping={})
                .add_if_else(
                    2, "Route",
                    condition="'urgent' in value",
                    if_true=3,
                    if_false=4,
                    condition_step=1,
                )
                .add_step(3, "Handle urgent", aim="Handle urgent request")
                .add_step(4, "Handle normal", aim="Handle normal request")
                .build()
            )
        """
        dependencies = [condition_step] if condition_step is not None else []
        return self.add_conditional_step(
            number=number,
            title=title,
            branches=[(condition, if_true)],
            default_step=if_false,
            condition_context_key=condition_context_key,
            dependencies=dependencies,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
        )

    def add_switch(
        self,
        number: int,
        title: str,
        branches: dict[str, int],
        condition_step: int | None = None,
        condition_context_key: str = "$history[-1]",
        checkpoint: bool = False,
        checkpoint_name: str | None = None,
    ) -> "ChainBuilder":
        """
        Add a multi-way Switch conditional step.

        Convenience wrapper around :meth:`add_conditional_step` for the common case
        of routing to one of several steps based on a condition value.

        The ``branches`` dict maps condition expressions to step numbers.
        The special key ``"default"`` maps to the ``default_step`` (no condition).

        Supported shorthand condition prefixes (handled by ConditionalStepExecutor):
        - ``"contains:<text>"`` — true if ``<text>`` appears in the value
        - ``"startswith:<text>"`` — true if value starts with ``<text>``
        - ``"endswith:<text>"`` — true if value ends with ``<text>``
        - ``"equals:<text>"`` — exact match
        - ``"empty"`` / ``"nonempty"``
        - Any other expression is evaluated with simpleeval (``value`` variable).

        Args:
            number: Step number.
            title: Step title.
            branches: Mapping of condition → step number.  Use ``"default"`` for
                the fallback branch that runs when no condition matches.
            condition_step: If provided, that step's number is automatically added
                to this step's ``dependencies`` list.
            condition_context_key: Context key whose resolved value is tested.
            checkpoint: Whether to checkpoint this step.
            checkpoint_name: Name for the checkpoint.

        Returns:
            ``self`` for method chaining.

        Example::

            chain = (
                ChainBuilder()
                .add_tool_step(1, "Classify", tool_name="classify", input_mapping={})
                .add_switch(
                    2, "Route",
                    branches={
                        "contains:urgent": 3,
                        "contains:billing": 4,
                        "default": 5,
                    },
                    condition_step=1,
                )
                .add_step(3, "Urgent handler", aim="...")
                .add_step(4, "Billing handler", aim="...")
                .add_step(5, "General handler", aim="...")
                .build()
            )
        """
        default_step = branches.pop("default", None) if "default" in branches else None
        branch_list = [(cond, step) for cond, step in branches.items()]
        dependencies = [condition_step] if condition_step is not None else []
        return self.add_conditional_step(
            number=number,
            title=title,
            branches=branch_list,
            default_step=default_step,
            condition_context_key=condition_context_key,
            dependencies=dependencies,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
        )

    def add_while_loop(
        self,
        body_steps: list[StepDescriptionBase],
        condition_key: str,
        max_iterations: int = 10,
        start_number: int | None = None,
    ) -> "ChainBuilder":
        """
        Add a while-loop: re-execute *body_steps* while *condition_key* resolves to a truthy value.

        The ``body_steps`` are renumbered starting from *start_number* (defaults to the next
        available step number) and inserted into the chain.  The last step in the body
        automatically receives ``loop_back_to`` and a :class:`LoopConfig` that evaluates
        *condition_key* after each iteration.

        ``condition_key`` supports any reference understood by ``resolve_context_reference``:
        - ``$memory.namespace.key`` — read a memory value (falsy → exit loop)
        - ``$history[-1]`` — last history entry (non-empty string → truthy)
        - ``$outer_context`` — the outer context string

        Args:
            body_steps: Ordered list of step descriptions forming the loop body.
            condition_key: Context reference whose value controls iteration.
                Empty string means "always loop" — only the budget guard stops it.
            max_iterations: Maximum re-execution count (budget guard). Default 10.
            start_number: First step number to use. Defaults to max(existing_steps)+1.

        Returns:
            ``self`` for method chaining.

        Example::

            (
                ChainBuilder()
                .add_while_loop(
                    [
                        ToolStepDescription(number=1, title="Fetch", config=ToolStepConfig(tool_name="fetch")),
                        ToolStepDescription(number=2, title="Check", config=ToolStepConfig(tool_name="check")),
                    ],
                    condition_key="$memory.loop.needs_more",
                    max_iterations=5,
                )
                .add_tool_step(3, "Done", tool_name="finalize")
                .build()
            )
        """
        if not body_steps:
            return self

        base = start_number if start_number is not None else (
            max((getattr(s, "number", 0) for s in self.steps), default=0) + 1
        )
        # Renumber body steps and attach loop config to the last step.
        for i, step in enumerate(body_steps):
            step.number = base + i

        last = body_steps[-1]
        last.loop_back_to = base
        last.loop_config = LoopConfig(condition_key=condition_key, max_iterations=max_iterations)

        self.steps.extend(body_steps)
        return self

    def add_until_loop(
        self,
        body_steps: list[StepDescriptionBase],
        condition_key: str,
        max_iterations: int = 10,
        start_number: int | None = None,
    ) -> "ChainBuilder":
        """
        Add an until-loop: re-execute *body_steps* until *condition_key* becomes truthy.

        Semantically the complement of :meth:`add_while_loop`.  The loop continues as long
        as *condition_key* is **falsy**; when it becomes truthy the loop exits.

        Args:
            body_steps: Ordered list of step descriptions forming the loop body.
            condition_key: Context reference whose value is checked each iteration.
                When the resolved value is truthy, the loop exits.
            max_iterations: Maximum re-execution count (budget guard). Default 10.
            start_number: First step number to use. Defaults to max(existing_steps)+1.

        Returns:
            ``self`` for method chaining.

        Example::

            (
                ChainBuilder()
                .add_until_loop(
                    [ToolStepDescription(number=1, title="Try", config=ToolStepConfig(tool_name="try_tool"))],
                    condition_key="$memory.result.ok",
                    max_iterations=3,
                )
                .build()
            )
        """
        if not body_steps:
            return self

        base = start_number if start_number is not None else (
            max((getattr(s, "number", 0) for s in self.steps), default=0) + 1
        )
        for i, step in enumerate(body_steps):
            step.number = base + i

        last = body_steps[-1]
        last.loop_back_to = base
        last.loop_config = LoopConfig(
            condition_key=condition_key,
            max_iterations=max_iterations,
            negate_condition=True,  # loop while NOT condition
        )

        self.steps.extend(body_steps)
        return self

    def with_max_workers(self, max_workers: int | str) -> "ChainBuilder":
        """
        Set maximum number of parallel workers.

        Args:
            max_workers: Maximum workers, or ``"auto"`` to match each batch size.

        Returns:
            Self for method chaining
        """
        self.max_workers = max_workers
        return self

    def with_prompt_template(self, template: PromptTemplate) -> "ChainBuilder":
        """
        Set custom prompt template.

        Args:
            template: Prompt template to use

        Returns:
            Self for method chaining
        """
        self.prompt_template = template
        return self

    def with_search_config(self, config: ContextSearchConfig) -> "ChainBuilder":
        """
        Set search configuration for context extraction.

        Args:
            config: Search configuration to use

        Returns:
            Self for method chaining
        """
        self.search_config = config
        return self

    def with_progress(self, enable: bool = True) -> "ChainBuilder":
        """
        Enable or disable progress tracking.

        Args:
            enable: Whether to enable progress

        Returns:
            Self for method chaining
        """
        self.enable_progress = enable
        return self

    def with_metadata(self, **metadata) -> "ChainBuilder":
        """
        Add metadata to the chain.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for method chaining
        """
        self.metadata.update(metadata)
        return self

    def with_trace_name(self, trace_name: str) -> "ChainBuilder":
        """
        Set the trace name for LangFuse tracing.

        Args:
            trace_name: Name shown in LangFuse UI for this chain's trace

        Returns:
            Self for method chaining
        """
        self.trace_name = trace_name
        return self

    def with_session_id(self, session_id: str) -> "ChainBuilder":
        """
        Set the session ID for LangFuse tracing.

        Multiple chains sharing the same session_id are grouped
        as one session in LangFuse.

        Args:
            session_id: Session identifier for grouping traces

        Returns:
            Self for method chaining
        """
        self.session_id = session_id
        return self

    def with_timeout(self, timeout: float) -> "ChainBuilder":
        """
        Set maximum execution time for the chain.

        Args:
            timeout: Maximum execution time in seconds

        Returns:
            Self for method chaining
        """
        self.timeout = timeout
        return self

    def with_replan_policy(self, policy: ReplanPolicy | None) -> "ChainBuilder":
        """
        Set chain-level RE-PLAN policy.

        Args:
            policy: RE-PLAN policy configuration (None to disable)

        Returns:
            Self for method chaining
        """
        self.replan_policy = policy
        return self

    def with_default_llm_config(self, llm_config: LLMStepConfig) -> "ChainBuilder":
        """
        Set a chain-level default LLM configuration.

        Steps without their own ``llm_config`` inherit this default.
        Per-step ``llm_config`` values merge on top of (override) this default.

        Args:
            llm_config: Default LLM settings (model, temperature, max_tokens, etc.)

        Returns:
            self for chaining
        """
        self.default_llm_config = llm_config
        return self

    def build(self) -> ReasoningChain:
        """
        Build the reasoning chain.

        Returns:
            Constructed reasoning chain

        Raises:
            ValueError: If chain configuration is invalid
        """
        return ReasoningChain(
            steps=self.steps,
            max_workers=self.max_workers,
            prompt_template=self.prompt_template,
            enable_progress=self.enable_progress,
            metadata=self.metadata,
            search_config=self.search_config,
            timeout=self.timeout,
            trace_name=self.trace_name,
            session_id=self.session_id,
            replan_policy=self.replan_policy,
            default_llm_config=self.default_llm_config,
        )

    @classmethod
    async def from_description(
        cls,
        task: str,
        llm_client: Any,
        *,
        available_skills: Optional[list[str]] = None,
        available_tools: Optional[list[str]] = None,
        max_steps: int = 10,
        max_workers: int | str = "auto",
        extra_instructions: str = "",
        max_retries: int = 2,
    ) -> ReasoningChain:
        """
        Generate a :class:`ReasoningChain` from a natural-language task description.

        Asks *llm_client* (anything implementing
        :class:`~mmar_carl.models.llm_client_base.LLMClientBase` or any object
        with an async ``get_response_with_retries(prompt, retries=...)``
        method) to plan a chain of LLM/Tool/Memory/Transform steps for the
        task. The model's JSON spec is parsed via
        :py:meth:`ReasoningChain.from_dict` so all of the existing validation
        (cycles, dependency references, reference syntax warnings) applies.

        The model's reply is expected to be a JSON object of the form::

            {
                "steps": [
                    {
                        "number": 1,
                        "title": "...",
                        "step_type": "llm" | "tool" | "memory" | "transform" | "conditional",
                        "aim": "...",          # for llm steps
                        "dependencies": [...],
                        "step_config": {...},  # for non-llm steps
                    },
                    ...
                ]
            }

        Markdown code-fence wrapping (\\`\\`\\`json … \\`\\`\\`) is tolerated.

        Args:
            task: Natural-language description of what the chain should do.
            llm_client: Async LLM client. Must expose
                ``get_response_with_retries(prompt, retries=int) -> str`` or
                ``get_response(prompt) -> str``.
            available_skills: Optional list of registered skill names the
                planner is allowed to reference (currently surfaced in the
                planning prompt; not enforced).
            available_tools: Optional list of registered tool names the
                planner is allowed to reference (surfaced in the planning
                prompt; ``ToolStep`` entries in the output must use these names).
            max_steps: Upper bound on the number of steps the planner may
                produce. Enforced after parsing — raises ``ValueError`` if
                exceeded.
            max_workers: Parallel-worker setting on the resulting chain.
            extra_instructions: Free-form text appended to the planning
                prompt (project-specific constraints, output format hints, …).

        Returns:
            A fully-constructed :class:`ReasoningChain`.

        Raises:
            ValueError: If the LLM reply can't be parsed as JSON, is missing
                ``"steps"``, or the produced chain fails ``_validate_steps``.
        """
        import json as _json
        import re as _re

        skill_clause = (
            f"\nYou MAY reference these named AgentSkills in `agent_skill` steps: "
            f"{', '.join(available_skills)}."
            if available_skills else ""
        )
        tool_clause = (
            f"\nYou MAY reference these registered tools in `tool` steps "
            f"(use exactly these names): {', '.join(available_tools)}."
            if available_tools else ""
        )

        planning_prompt = (
            "You are a CARL chain planner. Produce a JSON plan for the task "
            "below. Respond with a single JSON object — no prose, no Markdown "
            "fences (or, if you must, only triple-backtick `json` fences which "
            "will be stripped).\n\n"
            "Output schema:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "number": <int, starting at 1, monotonically increasing>,\n'
            '      "title": "<short label>",\n'
            '      "step_type": "llm" | "tool" | "memory" | "transform",\n'
            '      "dependencies": [<numbers of prior steps this one needs>],\n'
            '      "aim": "<one-sentence goal — REQUIRED for step_type=llm>",\n'
            '      "step_config": { ... }   // REQUIRED for non-llm types; '
            "shape matches CARL's typed step configs\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Field types (strict — wrong types get rejected by pydantic):\n"
            "- `number`: int. Sequential starting at 1.\n"
            "- `title`: string. Required, non-empty.\n"
            "- `step_type`: string. One of: \"llm\", \"tool\", \"memory\", \"transform\".\n"
            "- `dependencies`: list of int. Step numbers this one waits on. Use [] for no deps.\n"
            "- `aim`: string (NOT array). Single sentence describing the step's goal.\n"
            "- `reasoning_questions`: string (NOT array). If you have multiple questions, "
            "join them with newlines in one string.\n"
            "- `stage_action`: string (NOT array). Single instruction.\n"
            "- `example_reasoning`: string (NOT array). One example reasoning trace.\n"
            "- `step_config`: object (NOT array). Per-step-type shape — see Rules.\n\n"
            f"Rules:\n"
            f"- Produce at most {max_steps} steps. Fewer is better.\n"
            "- Number steps sequentially starting at 1.\n"
            "- Use `dependencies` to express the DAG; do NOT use `triggered_by`.\n"
            "- For `tool` steps, set `step_config = {\"tool_name\": <name>, "
            "\"input_mapping\": {...}}`.\n"
            "- For `memory` steps, set `step_config = {\"operation\": "
            "\"read\"|\"write\"|\"append\", \"memory_key\": ..., "
            "\"value_source\": ...}`.\n"
            f"{skill_clause}{tool_clause}\n"
            f"{('Additional constraints: ' + extra_instructions) if extra_instructions else ''}\n\n"
            f"Task: {task}"
        )

        # Provenance: capture full prompt + raw reply + every
        # retry attempt so callers can diagnose a failed planning run
        # offline without re-hitting the API. Truncated to keep the
        # chain spec under a sensible serialisation cost; the full text
        # would otherwise survive every ``chain.to_dict()`` round-trip.
        _PROMPT_PREVIEW_LIMIT = 4000
        _REPLY_PREVIEW_LIMIT = 4000

        async def _call_planner(prompt_text: str) -> str:
            """Dispatch through whichever interface the client offers."""
            if hasattr(llm_client, "get_response_with_retries"):
                return await llm_client.get_response_with_retries(prompt_text, retries=2)
            return await llm_client.get_response(prompt_text)

        def _truncate(text: str, limit: int) -> str:
            if len(text) <= limit:
                return text
            return text[:limit] + f"…[truncated, {len(text)} chars total]"

        provenance_attempts: list[dict[str, Any]] = []

        def _parse_and_build(reply: str) -> ReasoningChain:
            """Parse one planner reply into a chain. Raises ``ValueError`` on
            any validation failure — the retry loop catches and re-prompts."""
            text = (reply or "").strip()
            fence = _re.match(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$", text, _re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            try:
                spec = _json.loads(text)
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"LLM reply is not valid JSON: {exc.msg} at line "
                    f"{exc.lineno}, col {exc.colno}. Reply was:\n{reply[:500]!r}"
                ) from exc

            if not isinstance(spec, dict) or "steps" not in spec:
                raise ValueError(
                    "planner output missing top-level 'steps' key. "
                    "Got: " + repr(spec)[:200]
                )
            if not isinstance(spec["steps"], list) or not spec["steps"]:
                raise ValueError(
                    "planner 'steps' must be a non-empty list. "
                    "Got: " + repr(spec["steps"])[:200]
                )
            if len(spec["steps"]) > max_steps:
                raise ValueError(
                    f"planner produced {len(spec['steps'])} steps, "
                    f"exceeding max_steps={max_steps}."
                )

            spec.setdefault("max_workers", max_workers)
            spec.setdefault("metadata", {})
            md = spec["metadata"]
            md.setdefault("generated_from_description", task[:200])
            # provenance: stash the planner trail next to the
            # task tag so it's accessible via the chain itself.
            md.setdefault("planner_prompt", _truncate(planning_prompt, _PROMPT_PREVIEW_LIMIT))
            md.setdefault(
                "planner_reply", _truncate(reply or "", _REPLY_PREVIEW_LIMIT),
            )
            md.setdefault("planner_attempts", list(provenance_attempts))
            return ReasoningChain.from_dict(spec, use_typed_steps=True)

        # Retry loop: ask the planner up to ``max_retries+1`` times, feeding
        # the previous validation error back into the prompt so the LLM can
        # self-correct. ``max_retries=0`` preserves the original single-shot
        # behaviour.
        current_prompt = planning_prompt
        last_error: Optional[ValueError] = None
        attempts = max(1, max_retries + 1)
        for attempt in range(attempts):
            reply = await _call_planner(current_prompt)
            provenance_attempts.append({
                "attempt": attempt + 1,
                "prompt": _truncate(current_prompt, _PROMPT_PREVIEW_LIMIT),
                "reply": _truncate(reply or "", _REPLY_PREVIEW_LIMIT),
                # ``error`` is populated below when validation fails.
                "error": None,
            })
            try:
                return _parse_and_build(reply)
            except ValueError as exc:
                last_error = exc
                provenance_attempts[-1]["error"] = str(exc)[:500]
                if attempt + 1 < attempts:
                    # Append a remediation turn for the next attempt.
                    current_prompt = (
                        planning_prompt
                        + "\n\n---\n"
                        + "Your previous attempt failed validation with this error:\n"
                        + f"  {exc}\n\n"
                        + "Reply with a corrected JSON object that fixes this issue. "
                        + "Do not repeat the previous mistake. Output JSON only — no prose."
                    )
                    continue

        # All retries exhausted — raise the last error with a clear prefix
        # so callers can tell whether the failure came from from_description
        # vs. some other ValueError path.
        assert last_error is not None  # loop ran at least once
        raise ValueError(
            f"ChainBuilder.from_description: {last_error} "
            f"(failed after {attempts} attempt{'s' if attempts != 1 else ''})"
        ) from last_error


def create_chain_from_config(config: dict[str, Any]) -> ReasoningChain:
    """
    Create a reasoning chain from a configuration dictionary.

    This function delegates to ReasoningChain.from_dict() which properly
    handles all step types including tool, MCP, memory, transform, and conditional.

    Args:
        config: Configuration dictionary

    Returns:
        Constructed reasoning chain
    """
    return ReasoningChain.from_dict(config)
