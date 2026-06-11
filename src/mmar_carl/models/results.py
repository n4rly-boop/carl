"""
Result classes for CARL reasoning system.
"""

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .enums import StepType
from .replan import ReplanAction, ReplanAggregationStrategy, ReplanRollbackTarget


class StepExecutionResult(BaseModel):
    """
    Result of executing a single reasoning step.
    """

    step_number: int = Field(..., description="Number of the executed step")
    step_title: str = Field(..., description="Title of the executed step")
    step_type: StepType = Field(default=StepType.LLM, description="Type of step that was executed")
    result: str = Field(..., description="Result content (string representation)")
    result_data: Any = Field(default=None, description="Structured result data (for non-LLM steps)")
    success: bool = Field(..., description="Whether execution succeeded")
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    error_traceback: str | None = Field(default=None, description="Full traceback if execution failed")
    execution_time: float | None = Field(default=None, description="Time taken for execution in seconds")
    updated_history: list[str] = Field(default_factory=list, description="History after this step's execution")
    updated_messages: list[Any] = Field(
        default_factory=list,
        description=(
            "Structured message list after this step's execution. "
            "Populated only when the step uses LLMStepConfig.use_message_history=True. "
            "Contains ChatMessage instances (user+assistant turns appended this step)."
        ),
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage for this step: {'prompt': X, 'completion': Y, 'total': Z}"
    )
    model: str | None = Field(
        default=None,
        description=(
            "The LLM model identifier actually used by this step (e.g. "
            "``'gpt-4o-mini'``, ``'qwen/qwen3-8b'``). Populated by "
            "``LLMStepExecutor`` from the resolved client's ``model_name`` "
            "property when the step makes an LLM call. ``None`` for non-LLM "
            "steps (Tool, Memory, Transform, Conditional) or when the client "
            "doesn't expose its model. Used by "
            ":meth:`ReasoningResult.format_cost_by_model` to group spend by "
            "model when a chain mixes multiple LLMs."
        ),
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Metric scores for this step: {metric_name: score}",
    )
    skipped: bool = Field(default=False, description="Whether this step was skipped due to conditional routing")
    injected_steps: list[Any] = Field(
        default_factory=list,
        description=(
            "Optional list of new ``StepDescription`` / typed step instances "
            "to splice into the DAG after this step completes. The executor "
            "renumbers each injected step to a fresh integer above the "
            "highest existing step number, registers it as ready to run "
            "(its declared ``dependencies`` are remapped through the rename "
            "table), and includes it in subsequent batches. Subject to a "
            "chain-level injection budget guard (``max_injections``, default "
            "50) — exceeding the budget raises ``ValueError`` and stops the "
            "chain. Injected steps cannot reference step numbers that haven't "
            "executed yet (forward references are forbidden)."
        ),
    )
    profiling: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Profiling data captured by DAGExecutor after this step. "
            "Keys: 'history_bytes_added' (bytes of history entry written), "
            "'memory_bytes_after' (estimated total working-memory size after merge), "
            "'history_bytes_after' (total history size in bytes after merge), "
            "'batch_index' (0-based index of the execution batch this step ran in)."
        ),
    )

    def to_dict(self, *, truncate: bool = True) -> dict[str, Any]:
        """
        Convert the step result to a dictionary.

        Args:
            truncate: When True (default), ``result`` is clipped to the
                first 1000 chars — the log-friendly default that keeps
                old log dumps readable. Pass ``truncate=False`` for
                lossless persistence (e.g. CARE's ``RunRecord`` writes
                to gigaevo-memory) — the round-trip via
                :meth:`from_dict` then reconstructs the exact result.

        When ``truncate=False`` the output also includes ``updated_history``,
        ``updated_messages``, ``model``, and ``injected_steps`` so the
        full step record round-trips through :meth:`from_dict`.
        """
        result_text: Any
        if truncate and self.result:
            result_text = self.result[:1000]
        else:
            result_text = self.result if self.result else None

        out: dict[str, Any] = {
            "step_number": self.step_number,
            "step_title": self.step_title,
            "step_type": str(self.step_type),
            "success": self.success,
            "skipped": self.skipped,
            "result": result_text,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "error_traceback": self.error_traceback,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "metrics": self.metrics,
            "profiling": self.profiling,
        }
        if not truncate:
            # Lossless extras for full persistence
            out["updated_history"] = list(self.updated_history)
            # ChatMessage objects → dicts via Pydantic if present
            out["updated_messages"] = [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in self.updated_messages
            ]
            out["model"] = self.model
            # ``injected_steps`` carries StepDescription instances that
            # are not yet wired through pydantic's full dump cycle.
            # Skip on serialise — it's a runtime hook, not state.
            out["injected_steps"] = []
            out["_truncated"] = False
        else:
            out["_truncated"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepExecutionResult":
        """Reconstruct a :class:`StepExecutionResult` from a :meth:`to_dict`
        payload.

        Tolerant of both the truncated (default) and full (``truncate=False``)
        shapes — fields that were dropped by truncation simply default
        to their schema defaults.
        """
        # ``step_type`` round-trips through ``str()``; resolve back to enum.
        raw_type = data.get("step_type")
        if isinstance(raw_type, str) and raw_type.startswith("StepType."):
            raw_type = raw_type[len("StepType."):].lower()
        # Filter to known fields so unexpected keys (e.g. ``_truncated``)
        # don't trip pydantic's strict mode.
        known = {k: v for k, v in data.items() if k in cls.model_fields}
        if raw_type is not None:
            known["step_type"] = raw_type
        # ``result`` may have been serialised as None for empty strings.
        if known.get("result") is None:
            known["result"] = ""
        return cls.model_validate(known)

    # ------------------------------------------------------------------
    # typed views over ``result_data``.
    #
    # Each accessor returns ``None`` when the step's ``step_type``
    # doesn't match the target — so callers can chain via the walrus
    # without an isinstance dance:
    #
    #     if (skill := result.as_skill_output()):
    #         for f in skill.output_files: ...
    #
    # The Pydantic models live in :mod:`mmar_carl.models.result_data`.
    # Validation is permissive (extra="allow"): extras / future fields
    # don't break callers. ``None`` is also returned when the dict
    # fails validation — accessors never raise on a malformed payload.
    # ------------------------------------------------------------------

    def _as_typed_result_data(
        self, step_types: set, model_cls: type,
    ) -> Optional[Any]:
        """Shared dispatch for the four typed accessors below.

        Returns ``None`` when the step type doesn't match OR when the
        underlying ``result_data`` doesn't validate against ``model_cls``.
        """
        # Match on both enum and string-value forms (truncated/full
        # serialised results round-trip via the value form).
        type_values = {
            getattr(t, "value", t) for t in step_types
        } | step_types
        actual = getattr(self.step_type, "value", self.step_type)
        if actual not in type_values:
            return None
        if not isinstance(self.result_data, dict):
            return None
        try:
            return model_cls.model_validate(self.result_data)
        except Exception:
            return None

    def as_skill_output(self) -> Optional[Any]:
        """Return a :class:`SkillOutput` view of an AgentSkill step's
        ``result_data``, or ``None`` when the step type doesn't match."""
        from .result_data import SkillOutput  # noqa: PLC0415
        from .enums import StepType as _StepType  # noqa: PLC0415
        return self._as_typed_result_data(
            {_StepType.AGENT_SKILL}, SkillOutput,
        )

    def as_debate_transcript(self) -> Optional[Any]:
        """Return a :class:`DebateTranscript` view of a Debate step's
        ``result_data``, or ``None`` when the step type doesn't match."""
        from .result_data import DebateTranscript  # noqa: PLC0415
        from .enums import StepType as _StepType  # noqa: PLC0415
        return self._as_typed_result_data(
            {_StepType.DEBATE}, DebateTranscript,
        )

    def as_supervisor_decision(self) -> Optional[Any]:
        """Return a :class:`SupervisorDecision` view of a Supervisor
        step's ``result_data``, or ``None`` when the step type doesn't
        match."""
        from .result_data import SupervisorDecision  # noqa: PLC0415
        from .enums import StepType as _StepType  # noqa: PLC0415
        return self._as_typed_result_data(
            {_StepType.SUPERVISOR}, SupervisorDecision,
        )

    def as_parallel_samples(self) -> Optional[Any]:
        """Return a :class:`ParallelSamples` view of a ParallelSampling
        step's ``result_data``, or ``None`` when the step type doesn't
        match."""
        from .result_data import ParallelSamples  # noqa: PLC0415
        from .enums import StepType as _StepType  # noqa: PLC0415
        return self._as_typed_result_data(
            {_StepType.PARALLEL_SAMPLING}, ParallelSamples,
        )


class ReplanCheckerVote(BaseModel):
    """Single checker verdict recorded for a RE-PLAN evaluation."""

    checker_name: str
    action: ReplanAction
    reason: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    suggested_target: ReplanRollbackTarget | None = None
    regeneration_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplanAggregationOutcome(BaseModel):
    """Aggregated decision outcome across all checker votes."""

    strategy: ReplanAggregationStrategy
    triggered: bool
    trigger_count: int
    total_count: int
    mandatory_satisfied: bool = True
    selected_checker: str | None = None
    selected_action: ReplanAction = ReplanAction.CONTINUE


class ReplanEvent(BaseModel):
    """A recorded RE-PLAN evaluation/action event."""

    sequence: int
    step_number: int
    step_title: str
    checker_votes: list[ReplanCheckerVote] = Field(default_factory=list)
    aggregation: ReplanAggregationOutcome
    final_action: ReplanAction = ReplanAction.CONTINUE
    rollback_target: ReplanRollbackTarget | None = None
    feedback_passed: list[str] = Field(default_factory=list)
    triggering_checkers: list[str] = Field(default_factory=list)
    budget_usage: dict[str, Any] = Field(default_factory=dict)
    budget_exhausted: bool = False
    note: str = ""


class ReasoningResult(BaseModel):
    """
    Final result of executing a complete reasoning chain.
    """

    success: bool = Field(..., description="Whether overall execution succeeded")
    history: list[str] = Field(..., description="Complete reasoning history")
    step_results: list[StepExecutionResult] = Field(..., description="Results from each step")
    total_execution_time: float | None = Field(default=None, description="Total execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    context_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Snapshot of ``ReasoningContext.metadata`` captured at chain "
            "completion. Includes any user keys set before/during execution "
            "(e.g. step outputs accessible via ``$metadata.step_N``, custom "
            "ground-truth keys, etc.). Framework-internal keys are prefixed "
            "with ``__`` (e.g. ``__default_llm_config``, ``__langfuse_trace``) "
            "and can be excluded via :meth:`get_context_metadata`."
        ),
    )
    replan_events: list[ReplanEvent] = Field(default_factory=list, description="Recorded RE-PLAN events")
    token_usage: dict[str, int] = Field(
        default_factory=dict,
        description="Total token usage across all steps: {'prompt': X, 'completion': Y, 'total': Z}"
    )
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Chain-level metric scores: {metric_name: score}",
    )
    trace: Optional[Any] = Field(
        default=None,
        description="Structured execution trace built during chain execution (ExecutionTrace instance).",
    )

    @property
    def error(self) -> str | None:
        """
        Get the error message from the first failed step.

        Returns:
            Error message string if any step failed, None otherwise
        """
        for step in self.step_results:
            if not step.success and step.error_message:
                return step.error_message
        return None

    def _repr_markdown_(self) -> str:
        """Rich-display protocol for Jupyter.

        Returns Markdown that Jupyter / nbviewer / GitHub all render
        natively: a status banner, a per-step profiling table, and a
        Mermaid token-pie diagram (only when at least one step recorded
        usage — keeps the cell tight for pure-tool chains).

        Bare evaluation in a notebook (``result``) triggers this method
        and shows the rich view; no ``print()`` needed.
        """
        status = "✅ success" if self.success else "❌ failed"
        elapsed = (
            f"{self.total_execution_time:.2f}s"
            if self.total_execution_time is not None else "—"
        )
        total_tokens = self.token_usage.get("total", 0)
        n_steps = len(self.step_results)
        n_failed = sum(1 for s in self.step_results if not s.success)

        lines = [
            f"**ReasoningResult** — {status} · {n_steps} step"
            f"{'s' if n_steps != 1 else ''}"
            f"{' (' + str(n_failed) + ' failed)' if n_failed else ''}"
            f" · {elapsed} · {total_tokens:,} tokens",
            "",
        ]
        if self.error:
            lines.append(f"> ⚠ {self.error}")
            lines.append("")

        if self.step_results:
            lines.append("```text")
            lines.append(self.format_profiling_table())
            lines.append("```")
            lines.append("")

        # Mermaid token pie — only when at least one step recorded usage.
        if any(s.token_usage.get("total", 0) for s in self.step_results):
            lines.append("```mermaid")
            lines.append(self.format_token_pie(format="mermaid"))
            lines.append("```")
        return "\n".join(lines)

    def get_full_output(self) -> str:
        """Get the full reasoning output as a single string."""
        return "\n".join(self.history)

    def get_final_output(self) -> str:
        """Get the final reasoning output as a single string without step headers."""
        if not self.history:
            return ""
        last_entry = self.history[-1]
        # Check if it's a step result with header (Russian or English)
        if last_entry.startswith("Шаг ") and "\nРезультат: " in last_entry:
            # Extract content after "Результат: " for Russian steps
            return last_entry.split("\nРезультат: ", 1)[1].strip()
        elif last_entry.startswith("Step ") and "\nResult: " in last_entry:
            # Extract content after "Result: " for English steps
            return last_entry.split("\nResult: ", 1)[1].strip()
        else:
            # Return as-is if it doesn't match expected format
            return last_entry.strip()

    def get_context_metadata(self, *, include_internal: bool = False) -> dict[str, Any]:
        """Return the post-execution context metadata snapshot.

        By default, framework-internal keys (prefixed with ``__``) are
        excluded — these include ``__default_llm_config``,
        ``__langfuse_trace``, ``__expected_answer``, and other CARL plumbing.
        Pass ``include_internal=True`` to receive the unfiltered snapshot
        (useful for debugging).

        Args:
            include_internal: If True, return all keys including ``__``-prefixed
                framework internals.

        Returns:
            dict mapping metadata key → value. Mutating the returned dict
            does not affect the underlying result.
        """
        snapshot = dict(self.context_metadata)
        if include_internal:
            return snapshot
        return {k: v for k, v in snapshot.items() if not k.startswith("__")}

    def get_successful_steps(self) -> list[StepExecutionResult]:
        """Get all successfully executed steps (excludes skipped steps)."""
        return [step for step in self.step_results if step.success and not step.skipped]

    def get_failed_steps(self) -> list[StepExecutionResult]:
        """Get all failed steps."""
        return [step for step in self.step_results if not step.success and not step.skipped]

    def get_skipped_steps(self) -> list[StepExecutionResult]:
        """Get all steps that were skipped due to conditional routing."""
        return [step for step in self.step_results if step.skipped]

    def get_step_result(self, step_number: int) -> StepExecutionResult | None:
        """
        Get the result for a specific step by its number.

        Args:
            step_number: The step number to look up

        Returns:
            The StepExecutionResult for that step, or None if not found
        """
        for result in self.step_results:
            if result.step_number == step_number:
                return result
        return None

    def get_total_tokens(self) -> dict[str, int]:
        """
        Calculate total token usage across all steps.

        Returns:
            Dict with 'prompt', 'completion', and 'total' token counts
        """
        total_prompt = 0
        total_completion = 0

        for step in self.step_results:
            if step.token_usage:
                total_prompt += step.token_usage.get("prompt", 0)
                total_completion += step.token_usage.get("completion", 0)

        return {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_prompt + total_completion,
        }

    @property
    def partial_outputs(self) -> dict[int, str]:
        """Per-step textual outputs from steps that completed successfully.

        Useful after :class:`~mmar_carl.executor.ExecutionCancelledError` to
        recover whatever work was already done — but also valid for normal
        runs (just returns every successful step's text).

        Skipped steps are excluded; failed steps are excluded. Steps with
        empty ``result`` strings are included so callers can still see which
        step completed even if its output was empty.

        Returns:
            ``{step_number: result_text}`` for each successful, non-skipped
            step. Empty when no step completed.

        Example::

            try:
                result = await chain.execute_async(ctx)
            except ExecutionCancelledError as exc:
                partial = exc.result.partial_outputs
                print(f"completed {len(partial)} steps before cancellation")
        """
        return {
            step.step_number: step.result
            for step in self.step_results
            if step.success and not step.skipped
        }

    def get_partial_final_output(self) -> str | None:
        """Return the output of the highest-numbered successful step.

        Designed for the cancellation-recovery flow: after a chain is
        cancelled, you typically want "whatever the chain produced before
        we pulled the plug" — that's the latest-numbered completed step.

        Returns:
            The result string of the step with the highest ``step_number``
            among successful, non-skipped steps. ``None`` if no step
            completed.

        Example::

            try:
                result = await chain.execute_async(ctx)
            except ExecutionCancelledError as exc:
                tail = exc.result.get_partial_final_output()
                if tail:
                    log_recovery("partial work: " + tail[:200])
        """
        partial = self.partial_outputs
        if not partial:
            return None
        latest = max(partial.keys())
        return partial[latest]

    @property
    def token_usage_by_step(self) -> dict[int, dict[str, int]]:
        """Per-step token usage keyed by ``step_number``.

        Convenience accessor that consolidates ``step_results[i].token_usage``
        into a single mapping. Steps without recorded usage (e.g. Tool /
        Memory / Transform / Conditional steps, or LLM steps backed by a
        mock client that doesn't return ``usage``) are omitted.

        Returns:
            ``{step_number: {"prompt": int, "completion": int, "total": int}}``
            with ``total`` filled in defensively even when the underlying
            step dict only carries ``prompt`` and ``completion``.

        Example::

            result = await chain.execute_async(ctx)
            for step_num, usage in result.token_usage_by_step.items():
                print(f"step {step_num}: {usage['total']} tokens")
        """
        per_step: dict[int, dict[str, int]] = {}
        for step in self.step_results:
            if not step.token_usage:
                continue
            prompt = int(step.token_usage.get("prompt", 0))
            completion = int(step.token_usage.get("completion", 0))
            # Honour any explicit "total" the client populated; otherwise compute.
            total = int(step.token_usage.get("total", prompt + completion))
            per_step[step.step_number] = {
                "prompt": prompt,
                "completion": completion,
                "total": total,
            }
        return per_step

    def get_profiling_summary(self) -> dict[str, Any]:
        """
        Return a per-step profiling breakdown and chain-level totals.

        Each entry in ``steps`` contains:
        - ``step_number``, ``step_title``, ``step_type``
        - ``execution_time_s`` — wall-clock time for the step
        - ``history_bytes_added`` — bytes appended to history by this step
        - ``memory_bytes_after`` — estimated total working-memory size after merge
        - ``batch_index`` — which parallel batch this step ran in

        Chain-level totals:
        - ``total_execution_time_s``
        - ``total_history_bytes`` — cumulative history size at chain end
        - ``peak_memory_bytes`` — largest ``memory_bytes_after`` across all steps
        - ``token_usage``
        """
        step_rows = []
        peak_memory = 0
        for sr in self.step_results:
            p = sr.profiling or {}
            mem_after = p.get("memory_bytes_after", 0)
            if mem_after > peak_memory:
                peak_memory = mem_after
            step_rows.append({
                "step_number": sr.step_number,
                "step_title": sr.step_title,
                "step_type": sr.step_type.value if sr.step_type else "unknown",
                "execution_time_s": sr.execution_time,
                "history_bytes_added": p.get("history_bytes_added", 0),
                "memory_bytes_after": mem_after,
                "history_bytes_after": p.get("history_bytes_after", 0),
                "batch_index": p.get("batch_index"),
                "skipped": sr.skipped,
                "success": sr.success,
            })
        # Final history total (last step's history_bytes_after, or sum from history)
        total_history = sum(len(e) for e in self.history)
        return {
            "steps": step_rows,
            "total_execution_time_s": self.total_execution_time,
            "total_history_bytes": total_history,
            "peak_memory_bytes": peak_memory,
            "token_usage": self.token_usage,
        }

    def format_token_pie(
        self,
        *,
        format: str = "text",
        title_width: int = 28,
        bar_width: int = 30,
        png_path: Optional[str] = None,
    ) -> str:
        """Render a per-step token-spend breakdown ("which step burned the tokens?").

        Renders the same underlying data three ways:

        * ``format="text"`` (default) — Unicode horizontal-bar chart ranked by
          total tokens, with absolute counts + percentages. Terminal-only,
          no optional deps.
        * ``format="mermaid"`` — a Mermaid ``pie`` block. Renders natively in
          GitHub README, Markdown notebooks, and most IDEs.
        * ``format="png"`` — matplotlib pie chart written to ``png_path``.
          Requires the ``mmar-carl[viz]`` extra; raises ``ImportError`` with
          a clear install hint if matplotlib is unavailable. Returns the
          absolute path of the written file (also the function's return
          value) so callers can chain ``IPython.display.Image(...)``.

        Non-LLM steps (Tool / Memory / Transform / Conditional) are omitted
        from the breakdown — they don't record token usage and would
        otherwise show as 0%. Empty chains and chains with no LLM-token
        usage return a one-line placeholder rather than an empty chart.

        Args:
            format: ``"text"``, ``"mermaid"``, or ``"png"``.
            title_width: Max characters of each step title (text mode only).
            bar_width: Width of the bar column in chars (text mode only).
            png_path: Required when ``format="png"``; ignored otherwise.

        Returns:
            The rendered string (text + mermaid) or the absolute path of
            the written PNG file (png).
        """
        per_step = self.token_usage_by_step
        if not per_step:
            return "(no token usage recorded — non-LLM-only chain or mock client)"

        # Rank by total descending so the biggest slice comes first.
        rows: list[tuple[int, str, int]] = []
        for sr in self.step_results:
            usage = per_step.get(sr.step_number)
            if usage is None:
                continue
            rows.append((sr.step_number, sr.step_title or f"step {sr.step_number}", usage["total"]))
        rows.sort(key=lambda r: r[2], reverse=True)

        total = sum(r[2] for r in rows)
        if total == 0:
            return "(all LLM steps recorded zero tokens — nothing to chart)"

        if format == "text":
            return self._format_token_pie_text(rows, total, title_width, bar_width)
        if format == "mermaid":
            return self._format_token_pie_mermaid(rows, total)
        if format == "png":
            return self._format_token_pie_png(rows, total, png_path)
        raise ValueError(
            f"Unknown format {format!r}. Use 'text', 'mermaid', or 'png'."
        )

    @staticmethod
    def _format_token_pie_text(
        rows: list[tuple[int, str, int]],
        total: int,
        title_width: int,
        bar_width: int,
    ) -> str:
        # Unicode block characters give us 1/8 resolution per cell.
        blocks = " ▏▎▍▌▋▊▉█"
        lines = [
            f"{'#':>3}  {'step':<{title_width}}  {'tokens':>7}  {'%':>5}  bar"
        ]
        lines.append("-" * (3 + 2 + title_width + 2 + 7 + 2 + 5 + 2 + bar_width))
        for number, title, tokens in rows:
            pct = (tokens / total) * 100.0
            filled = (tokens / total) * bar_width
            full = int(filled)
            remainder = filled - full
            sub = blocks[int(remainder * 8)]
            bar = "█" * full + (sub if full < bar_width else "")
            short_title = (
                title[: title_width - 1] + "…" if len(title) > title_width else title
            )
            lines.append(
                f"{number:>3}  {short_title:<{title_width}}  {tokens:>7,}  {pct:>4.1f}%  {bar}"
            )
        lines.append("-" * (3 + 2 + title_width + 2 + 7 + 2 + 5 + 2 + bar_width))
        lines.append(
            f"{'TOT':>3}  {'':<{title_width}}  {total:>7,}  {'100.0%':>5}"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_token_pie_mermaid(rows: list[tuple[int, str, int]], total: int) -> str:
        lines = ["pie title Token spend by step"]
        for number, title, tokens in rows:
            # Mermaid requires quotes around the slice label and bans
            # double-quotes inside; replace any with single-quotes.
            safe_title = title.replace('"', "'")
            label = f"step {number}: {safe_title}"
            lines.append(f'    "{label}" : {tokens}')
        return "\n".join(lines)

    @staticmethod
    def _format_token_pie_png(
        rows: list[tuple[int, str, int]],
        total: int,
        png_path: Optional[str],
    ) -> str:
        if png_path is None:
            raise ValueError("format='png' requires png_path=<destination>.")
        from .._optional_deps import require_matplotlib  # noqa: PLC0415

        plt = require_matplotlib()

        labels = [f"step {n}: {t}" for n, t, _ in rows]
        sizes = [tokens for _, _, tokens in rows]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        ax.set_title(f"Token spend by step (total: {total:,} tokens)")
        import os  # noqa: PLC0415

        abs_path = os.path.abspath(png_path)
        fig.savefig(abs_path, bbox_inches="tight")
        plt.close(fig)
        return abs_path

    def format_prompt_completion_breakdown(
        self,
        *,
        title_width: int = 28,
        bar_width: int = 40,
    ) -> str:
        """Render a per-step prompt-vs-completion stacked bar chart.

        Answers the routine prompt-engineering question: "is my **prompt**
        bloated, or is the **output** bloated?" Each row shows a step's
        token usage as a two-segment bar — ``▒`` for prompt tokens, ``█``
        for completion tokens — sized proportionally to the step with the
        most total tokens. Steps with no recorded token usage (Tool /
        Memory / Transform / Conditional) are omitted.

        Returns a text-only chart for now; PNG/Mermaid variants would
        require less-readable encodings of the two-color split and are
        deferred (Mermaid pie wouldn't capture per-step grouping;
        matplotlib stacked bar is straightforward to add later under the
        ``mmar-carl[viz]`` extra).

        Args:
            title_width: Max characters of each step title (truncate with
                ``…`` suffix when overflowing).
            bar_width: Width of the bar column in characters. Steps with
                the highest total token count fill the whole bar; others
                scale proportionally.

        Returns:
            Printable string. Empty when no LLM steps recorded usage —
            returns a one-line placeholder.
        """
        per_step = self.token_usage_by_step
        if not per_step:
            return "(no token usage recorded — non-LLM-only chain or mock client)"

        # Pull rows in the chain's natural order (not by-total), since the
        # ratio comparison is more useful when neighboring steps stay near
        # each other on screen.
        rows: list[tuple[int, str, int, int]] = []
        for sr in self.step_results:
            usage = per_step.get(sr.step_number)
            if usage is None:
                continue
            rows.append((sr.step_number, sr.step_title or f"step {sr.step_number}", usage["prompt"], usage["completion"]))

        max_total = max((p + c for _, _, p, c in rows), default=0)
        if max_total == 0:
            return "(all LLM steps recorded zero tokens — nothing to chart)"

        # Sub-character resolution so the bar boundaries are accurate.
        # Use ▒ for prompt, █ for completion. The transition between them
        # marks where prompt ends and completion begins.
        lines = [
            f"{'#':>3}  {'step':<{title_width}}  {'prompt':>7}  {'compl':>7}  bar"
        ]
        lines.append("-" * (3 + 2 + title_width + 2 + 7 + 2 + 7 + 2 + bar_width))
        for number, title, prompt_tok, completion_tok in rows:
            # Bar widths proportional to the chain's heaviest step.
            scale = bar_width / max_total
            prompt_cells = int(round(prompt_tok * scale))
            completion_cells = int(round(completion_tok * scale))
            # Ensure each non-zero segment shows at least one cell.
            if prompt_tok > 0 and prompt_cells == 0:
                prompt_cells = 1
            if completion_tok > 0 and completion_cells == 0:
                completion_cells = 1
            bar = ("▒" * prompt_cells) + ("█" * completion_cells)
            short_title = (
                title[: title_width - 1] + "…" if len(title) > title_width else title
            )
            lines.append(
                f"{number:>3}  {short_title:<{title_width}}  {prompt_tok:>7,}  "
                f"{completion_tok:>7,}  {bar}"
            )
        lines.append("-" * (3 + 2 + title_width + 2 + 7 + 2 + 7 + 2 + bar_width))
        total_prompt = sum(p for _, _, p, _ in rows)
        total_completion = sum(c for _, _, _, c in rows)
        lines.append(
            f"{'TOT':>3}  {'':<{title_width}}  {total_prompt:>7,}  {total_completion:>7,}"
        )
        lines.append("")
        lines.append("legend:  ▒ prompt   █ completion")
        return "\n".join(lines)

    def format_cost_by_model(
        self,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        *,
        default_model: Optional[str] = None,
        format: str = "text",
        bar_width: int = 30,
    ) -> str:
        """Per-model breakdown of tokens + cost when a chain mixes multiple LLMs.

        Groups every LLM step's token usage by ``step.model`` (populated
        by ``LLMStepExecutor`` from the resolved client's ``model_name``).
        Steps without an assigned model fall back to ``default_model``;
        if neither is available, the step is bucketed under ``"(unknown)"``.

        Two output formats:

        * ``"text"`` (default): table with one row per model showing
          ``tokens``, ``cost`` (when pricing supplied), ``% share``, and
          a horizontal bar sized to the share. Followed by a TOTAL row.
          Models without a pricing entry render ``$-`` in the cost column
          and a ``⚠ missing pricing for: <models>`` line below the table.
        * ``"mermaid"``: ``pie title Token spend by model`` block with
          one slice per model labelled by name. Token totals are the
          slice values (Mermaid pies don't natively label by USD).

        Args:
            pricing: Optional ``{model: (input_per_1k_usd,
                output_per_1k_usd)}`` mapping — same shape as
                :py:meth:`format_profiling_table`. Missing entries leave
                the cost column blank.
            default_model: Fallback model name when ``step.model`` is None.
            format: ``"text"`` or ``"mermaid"``.
            bar_width: Width of the bar column in characters (text only).

        Returns:
            Printable string. Empty / non-LLM-only chains return a
            one-line placeholder.
        """
        # Aggregate per-model tokens.
        per_model: dict[str, dict[str, int]] = {}
        for sr in self.step_results:
            if not sr.token_usage:
                continue
            model = sr.model or default_model or "(unknown)"
            bucket = per_model.setdefault(model, {"prompt": 0, "completion": 0, "total": 0})
            bucket["prompt"] += int(sr.token_usage.get("prompt", 0))
            bucket["completion"] += int(sr.token_usage.get("completion", 0))
            bucket["total"] += int(sr.token_usage.get("total", 0)) or (
                bucket["prompt"] + bucket["completion"] - bucket["total"]  # defensive sum fallback
            )

        if not per_model:
            return "(no model-attributed token usage recorded — non-LLM-only chain or mock client)"

        # Recompute totals defensively (the fallback above may double-count;
        # this gives us authoritative numbers).
        for model, b in per_model.items():
            b["total"] = b["prompt"] + b["completion"]

        # Sort rows by total tokens descending.
        rows = sorted(per_model.items(), key=lambda kv: kv[1]["total"], reverse=True)
        grand_total = sum(b["total"] for _, b in rows) or 1  # avoid div-by-zero

        # Cost calculation per model.
        model_costs: dict[str, Optional[float]] = {}
        missing_pricing: list[str] = []
        if pricing:
            for model, b in rows:
                p = pricing.get(model)
                if p is None:
                    model_costs[model] = None
                    missing_pricing.append(model)
                else:
                    in_price, out_price = p
                    model_costs[model] = (
                        (b["prompt"] / 1000.0) * in_price
                        + (b["completion"] / 1000.0) * out_price
                    )

        if format == "text":
            return self._format_cost_by_model_text(
                rows, grand_total, model_costs, missing_pricing,
                bar_width=bar_width, pricing_supplied=pricing is not None,
            )
        if format == "mermaid":
            return self._format_cost_by_model_mermaid(rows)
        raise ValueError(
            f"Unknown format {format!r}. Use 'text' or 'mermaid'."
        )

    @staticmethod
    def _format_cost_by_model_text(
        rows: list[tuple[str, dict[str, int]]],
        grand_total: int,
        model_costs: dict[str, Optional[float]],
        missing_pricing: list[str],
        *,
        bar_width: int,
        pricing_supplied: bool,
    ) -> str:
        model_col = max(len("model"), max(len(m) for m, _ in rows))
        header = (
            f"{'model':<{model_col}}  {'tokens':>9}  "
            f"{'cost':>10}  {'%':>5}  bar"
        )
        sep = "-" * (model_col + 2 + 9 + 2 + 10 + 2 + 5 + 2 + bar_width)
        lines = [header, sep]
        total_cost = 0.0
        for model, b in rows:
            pct = (b["total"] / grand_total) * 100.0
            cells = int(round((b["total"] / grand_total) * bar_width))
            if b["total"] > 0 and cells == 0:
                cells = 1
            bar = "█" * cells
            if pricing_supplied:
                cost = model_costs.get(model)
                if cost is None:
                    cost_str = "$-"
                else:
                    cost_str = f"${cost:.4f}"
                    total_cost += cost
            else:
                cost_str = ""
            lines.append(
                f"{model:<{model_col}}  {b['total']:>9,}  "
                f"{cost_str:>10}  {pct:>4.1f}%  {bar}"
            )
        lines.append(sep)
        total_cost_str = f"${total_cost:.4f}" if pricing_supplied else ""
        lines.append(
            f"{'TOT':<{model_col}}  {grand_total:>9,}  "
            f"{total_cost_str:>10}  {'100.0%':>5}"
        )
        if missing_pricing:
            lines.append("")
            lines.append(
                f"⚠ missing pricing for: {', '.join(sorted(set(missing_pricing)))} "
                f"(supply via `pricing={{model: (in_per_1k, out_per_1k)}}` for accurate cost)"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_cost_by_model_mermaid(
        rows: list[tuple[str, dict[str, int]]],
    ) -> str:
        lines = ["pie title Token spend by model"]
        for model, b in rows:
            # Sanitise quotes in model names so Mermaid's pie syntax doesn't break.
            safe = model.replace('"', "'")
            lines.append(f'    "{safe}" : {b["total"]}')
        return "\n".join(lines)

    def format_profiling_table(
        self,
        *,
        pricing: Optional[dict[str, tuple[float, float]]] = None,
        default_model: Optional[str] = None,
        title_width: int = 28,
    ) -> str:
        """One-line-per-step human-readable "where did the time/spend go?" view.

        Composes columns from ``get_profiling_summary()`` and ``token_usage_by_step``
        so users get a single answer to the common chain-debugging question
        "which step is dominating latency / tokens / cost?" without
        post-processing the dicts themselves.

        Columns:
        - ``#`` — step number
        - ``title`` — truncated step title
        - ``type`` — step type (e.g. ``llm``, ``tool``)
        - ``wall_ms`` — execution time in milliseconds
        - ``tok_in`` / ``tok_out`` — input + output tokens (blank for non-LLM steps)
        - ``cache`` — ``hit`` if the step result was served from cache, else blank
        - ``cost`` — USD cost when ``pricing`` is provided AND the step's
          token usage is recorded (blank otherwise)
        - ``status`` — ``ok`` / ``fail`` / ``skip``

        Args:
            pricing: Optional ``{model: (input_per_1k_usd, output_per_1k_usd)}``
                map — same shape as :py:meth:`ReasoningChain.estimate_cost`.
                When provided, a ``cost`` column is filled in using each step's
                resolved model (looked up via ``step.config.model`` →
                ``step.llm_config.model`` → ``default_model``). Missing
                pricing leaves the cell blank rather than failing.
            default_model: Fallback model name used for pricing lookup when
                a step's own model can't be determined from its result.
                ``StepExecutionResult`` doesn't preserve the resolved model
                identity, so this kwarg is the practical way to get
                non-blank cost cells. Pass the chain's primary model name.
            title_width: Max characters to print for the step title before
                truncation (default 28).

        Returns:
            A printable string with a header row, separator, per-step rows,
            a separator, and a TOTAL row. Empty when the result has no steps.

        Example::

            result = await chain.execute_async(ctx)
            print(result.format_profiling_table(
                pricing={"gpt-4o-mini": (0.00015, 0.0006)},
            ))
        """
        if not self.step_results:
            return "(empty chain)"

        usage_by_step = self.token_usage_by_step

        # Resolve a price per step (best effort).
        def _step_model(sr: StepExecutionResult) -> Optional[str]:
            cfg = getattr(sr, "config", None) or getattr(sr, "llm_config", None)
            if cfg is None:
                return None
            return getattr(cfg, "model", None)

        rows = []
        total_wall_ms = 0.0
        total_in = 0
        total_out = 0
        total_cost = 0.0
        for sr in self.step_results:
            wall_ms = round((sr.execution_time or 0.0) * 1000.0)
            total_wall_ms += wall_ms
            usage = usage_by_step.get(sr.step_number)
            tok_in = usage["prompt"] if usage else 0
            tok_out = usage["completion"] if usage else 0
            total_in += tok_in
            total_out += tok_out

            cache_str = "hit" if (sr.profiling or {}).get("cache_hit") else ""

            if sr.skipped:
                status = "skip"
            elif sr.success:
                status = "ok"
            else:
                status = "fail"

            # Cost (best effort): step's own model → default_model → blank.
            cost_str = ""
            if pricing and usage:
                model = _step_model(sr) or default_model
                if model and model in pricing:
                    in_price, out_price = pricing[model]
                    step_cost = (tok_in / 1000.0) * in_price + (tok_out / 1000.0) * out_price
                    total_cost += step_cost
                    cost_str = f"${step_cost:.4f}"

            rows.append(
                {
                    "n": sr.step_number,
                    "title": sr.step_title or "",
                    "type": sr.step_type.value if sr.step_type else "?",
                    "wall_ms": wall_ms,
                    "tok_in": tok_in if usage else None,
                    "tok_out": tok_out if usage else None,
                    "cache": cache_str,
                    "cost": cost_str,
                    "status": status,
                }
            )

        # Render
        header = (
            f"{'#':>3}  {'title':<{title_width}}  {'type':<10}  {'wall_ms':>8}  "
            f"{'tok_in':>7}  {'tok_out':>7}  {'cache':>5}  {'cost':>10}  {'status':<6}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            title = r["title"][: title_width - 1] + "…" if len(r["title"]) > title_width else r["title"]
            tin = "" if r["tok_in"] is None else f"{r['tok_in']:,}"
            tout = "" if r["tok_out"] is None else f"{r['tok_out']:,}"
            lines.append(
                f"{r['n']:>3}  {title:<{title_width}}  {r['type']:<10}  {r['wall_ms']:>8,}  "
                f"{tin:>7}  {tout:>7}  {r['cache']:>5}  {r['cost']:>10}  {r['status']:<6}"
            )
        lines.append(sep)
        cost_total_str = f"${total_cost:.4f}" if pricing else ""
        lines.append(
            f"{'TOT':>3}  {'':<{title_width}}  {'':<10}  {int(total_wall_ms):>8,}  "
            f"{total_in:>7,}  {total_out:>7,}  {'':>5}  {cost_total_str:>10}  "
        )
        return "\n".join(lines)

    def to_dict(self, *, full: bool = False) -> dict[str, Any]:
        """
        Convert the reasoning result to a dictionary.

        Args:
            full: When ``False`` (default) emits the existing log-friendly
                summary shape — preserves backward compatibility for any
                caller already consuming `to_dict()`. When ``True`` emits a
                **lossless** payload suitable for persistence (CARE writes
                this to gigaevo-memory as a ``RunRecord``):

                * Untruncated per-step results (each step uses
                  ``StepExecutionResult.to_dict(truncate=False)``).
                * Full ``history`` list.
                * ``context_metadata`` snapshot.
                * ``replan_events`` already round-tripped via
                  ``model_dump(mode="json")``.

                ``trace`` (an ``Optional[Any]`` `ExecutionTrace` instance)
                is included only when it exposes ``to_dict()`` — keeps
                serialisation safe when callers attach exotic trace
                objects.

        :meth:`from_dict` accepts either shape.
        """
        if not full:
            return {
                "success": self.success,
                "total_execution_time": self.total_execution_time,
                "total_steps": len(self.step_results),
                "successful_steps": len(self.get_successful_steps()),
                "failed_steps": len(self.get_failed_steps()),
                "skipped_steps": len(self.get_skipped_steps()),
                "token_usage": self.token_usage or self.get_total_tokens(),
                "metrics": self.metrics,
                "step_results": [r.to_dict() for r in self.step_results],
                "replan_events": [event.model_dump(mode="json") for event in self.replan_events],
                "metadata": self.metadata,
                "_full": False,
            }
        trace_payload: Any = None
        if self.trace is not None and hasattr(self.trace, "to_dict"):
            try:
                trace_payload = self.trace.to_dict()
            except Exception:
                trace_payload = None
        return {
            "_full": True,
            "success": self.success,
            "history": list(self.history),
            "step_results": [r.to_dict(truncate=False) for r in self.step_results],
            "total_execution_time": self.total_execution_time,
            "metadata": self.metadata,
            "context_metadata": self.context_metadata,
            "replan_events": [event.model_dump(mode="json") for event in self.replan_events],
            "token_usage": self.token_usage,
            "metrics": self.metrics,
            "trace": trace_payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningResult":
        """Rebuild a :class:`ReasoningResult` from a :meth:`to_dict` payload.

        Supports both the summary (default) and full (``full=True``)
        shapes. The full shape round-trips losslessly; the summary
        shape loses ``history`` / ``context_metadata`` / per-step
        ``updated_history`` (those fields default to empty).
        """
        is_full = bool(data.get("_full"))
        step_results = [
            StepExecutionResult.from_dict(d) for d in data.get("step_results", [])
        ]
        replan_events = [
            ReplanEvent.model_validate(e) for e in data.get("replan_events", [])
        ]
        history = data.get("history") if is_full else None
        if history is None:
            # Fallback for the summary shape — reconstruct from per-step
            # ``updated_history`` if present (the executor's contract is
            # that each step's ``updated_history`` is the cumulative
            # history *after* that step). The last successful step's
            # ``updated_history`` is therefore the final history.
            for sr in reversed(step_results):
                if sr.updated_history:
                    history = list(sr.updated_history)
                    break
            if history is None:
                history = []
        ctor: dict[str, Any] = {
            "success": data.get("success", False),
            "history": history,
            "step_results": step_results,
            "total_execution_time": data.get("total_execution_time"),
            "metadata": data.get("metadata", {}),
            "context_metadata": data.get("context_metadata", {}),
            "replan_events": replan_events,
            "token_usage": data.get("token_usage", {}),
            "metrics": data.get("metrics", {}),
        }
        # ``trace`` is intentionally left as None on rebuild — callers
        # that want a typed ExecutionTrace can call
        # ``ExecutionTrace.from_json`` on ``data["trace"]`` themselves.
        return cls.model_validate(ctor)

    def to_json(self, *, indent: int | None = None, full: bool = True) -> str:
        """Serialise to a JSON string. Defaults to the lossless ``full``
        shape — direct counterpart for :meth:`from_json`."""
        import json as _json  # noqa: PLC0415
        return _json.dumps(self.to_dict(full=full), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ReasoningResult":
        """Inverse of :meth:`to_json`. Accepts either shape."""
        import json as _json  # noqa: PLC0415
        return cls.from_dict(_json.loads(json_str))

    def save(self, path: "str | Path", *, indent: int | None = 2) -> Path:
        """Persist the result to a JSON file (lossless shape).

        Returns the absolute path written. Mirrors the ergonomics of
        :meth:`ReasoningChain.save` so CARE can pair a chain spec with
        its execution result on disk under matching filenames.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(indent=indent, full=True), encoding="utf-8")
        return p.resolve()

    @classmethod
    def load(cls, path: "str | Path") -> "ReasoningResult":
        """Inverse of :meth:`save` — reads a JSON file and reconstructs."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
