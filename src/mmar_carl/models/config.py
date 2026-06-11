"""
Step configuration models for CARL reasoning system.
"""

from enum import Enum
from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field

from .enums import MemoryOperation


class ExecutionMode(str, Enum):
    """Execution strategy for LLM step."""

    FAST = "fast"
    SELF_CRITIC = "self_critic"


class ToolParameter(BaseModel):
    """Parameter definition for tool calls."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type (string, int, float, bool, list, dict)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether the parameter is required")
    default: Any = Field(default=None, description="Default value if not required")


class ToolErrorRecovery(BaseModel):
    """
    Error recovery configuration for tool execution.

    Provides retry logic and fallback strategies when a tool fails or times out.

    Example::

        ToolStepConfig(
            tool_name="web_search",
            error_recovery=ToolErrorRecovery(
                retry_max=2,
                retry_delay=1.0,
                on_timeout="cached_search",       # fallback tool name
                on_exception="cached_search",     # fallback tool name
            )
        )
    """

    retry_max: int = Field(
        default=0,
        ge=0,
        description="Number of extra attempts after the first failure (0 = no retries).",
    )
    retry_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds to wait between retry attempts.",
    )
    on_timeout: Optional[str] = Field(
        default=None,
        description=(
            "Name of a registered fallback tool to call when the primary tool times out. "
            "The fallback receives the same kwargs as the primary tool."
        ),
    )
    on_exception: Optional[str] = Field(
        default=None,
        description=(
            "Name of a registered fallback tool to call when the primary tool raises an exception "
            "(after all retries are exhausted). The fallback receives the same kwargs."
        ),
    )


class ToolStepConfig(BaseModel):
    """Configuration for tool/function call steps."""

    tool_name: str = Field(..., description="Name of the tool to call")
    tool_description: str = Field(default="", description="Description of what the tool does")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool input parameters")
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps step context keys to tool parameter names. Use '$history' for previous results.",
    )
    output_key: str = Field(default="result", description="Key to store tool output in step result")
    timeout: float = Field(default=30.0, description="Timeout in seconds for tool execution")
    retry_on_error: bool = Field(default=True, description="Whether to retry on error")
    error_recovery: Optional[ToolErrorRecovery] = Field(
        default=None,
        description="Error recovery strategy: retries, fallback tool on timeout or exception.",
    )
    allowed_tool_tags: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of tags. When set, the executor refuses to call the "
            "configured tool unless its registered tag set intersects this list. "
            "Use to keep destructive tools out of steps that only need informational access."
        ),
    )

    # The actual callable is set at runtime, not serialized
    _tool_callable: Optional[Callable] = None

    model_config = {"arbitrary_types_allowed": True}


class MCPServerConfig(BaseModel):
    """Configuration for MCP server connection."""

    server_name: str = Field(..., description="Name of the MCP server")
    transport: Literal["stdio", "http", "sse"] = Field(
        default="stdio",
        description=(
            "Transport type: 'stdio' (subprocess pipe), 'http' (streamable HTTP — MCP 2025-03-26 spec), "
            "'sse' (HTTP + Server-Sent Events — legacy MCP transport)."
        ),
    )
    command: Optional[str] = Field(default=None, description="Command to start stdio server")
    args: list[str] = Field(default_factory=list, description="Arguments for stdio server")
    url: Optional[str] = Field(default=None, description="Base URL for HTTP/SSE server")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers sent with every request (e.g. Authorization: Bearer …). "
            "Applied to both 'http' and 'sse' transports."
        ),
    )


class MCPStepConfig(BaseModel):
    """Configuration for MCP protocol steps."""

    server: MCPServerConfig = Field(..., description="MCP server configuration")
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Static arguments for the tool")
    argument_mapping: dict[str, str] = Field(
        default_factory=dict, description="Maps step context keys to MCP tool arguments"
    )
    timeout: float = Field(default=60.0, description="Timeout in seconds")


class MCPResourceStepConfig(BaseModel):
    """
    Configuration for MCP resource-reading steps.

    Resources are read-only data exposed by an MCP server (files,
    documentation, schemas, etc.) — distinct from MCP tools, which take
    arguments and perform actions. This step type fetches a named resource
    and writes it to memory (and history) without making any tool call.

    Example::

        MCPResourceStepDescription(
            number=2,
            title="Load API docs",
            config=MCPResourceStepConfig(
                server=MCPServerConfig(server_name="docs", transport="sse", url="..."),
                resource_uri="docs://api/reference.md",
                output_memory_key="api_docs",
            ),
        )
    """

    server: MCPServerConfig = Field(..., description="MCP server configuration")
    resource_uri: str = Field(
        ...,
        min_length=1,
        description="URI of the resource to read (e.g. 'docs://api/reference.md').",
    )
    output_memory_key: str = Field(
        default="",
        description=(
            "Memory key to store the resource content. When non-empty, the "
            "content is written to ``memory[output_namespace][output_memory_key]``. "
            "When empty, the content is only available via ``$history[-1]`` / "
            "``$metadata.step_N``."
        ),
    )
    output_namespace: str = Field(
        default="mcp_resource",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Timeout in seconds for the MCP read_resource call.",
    )


class MemoryStepConfig(BaseModel):
    """Configuration for memory read/write steps."""

    operation: MemoryOperation = Field(..., description="Memory operation type")
    memory_key: str = Field(..., description="Key in memory store")
    value_source: Optional[str] = Field(
        default=None, description="Source of value for write operations (context key or '$history[-1]')"
    )
    default_value: Any = Field(default=None, description="Default value if key not found on read")
    namespace: str = Field(default="default", description="Memory namespace for isolation")


class TransformStepConfig(BaseModel):
    """Configuration for data transformation steps (no LLM call)."""

    transform_type: Literal["extract", "format", "aggregate", "filter", "map"] = Field(
        ..., description="Type of transformation"
    )
    input_key: str = Field(default="$history[-1]", description="Key to get input from")
    output_format: Optional[str] = Field(default=None, description="Output format template")
    expression: Optional[str] = Field(default=None, description="Transformation expression (for extract/filter)")
    # For 'map' operations - applies a simple template to each item
    map_template: Optional[str] = Field(default=None, description="Template for map operations")


class ConditionalBranch(BaseModel):
    """A single conditional branch."""

    condition: str = Field(..., description="Condition expression (evaluated against context)")
    next_step: int = Field(..., description="Step number to execute if condition is true")


class ConditionalStepConfig(BaseModel):
    """Configuration for conditional branching steps."""

    branches: list[ConditionalBranch] = Field(..., description="List of conditional branches")
    default_step: Optional[int] = Field(default=None, description="Default step if no condition matches")
    condition_context_key: str = Field(
        default="$history[-1]", description="Context key to evaluate conditions against"
    )


class StructuredOutputStepConfig(BaseModel):
    """Configuration for schema-constrained structured output steps."""

    input_source: str = Field(
        default="$history[-1]",
        description="Input source used to build the structured output prompt",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema that model output must match",
    )
    schema_name: str = Field(default="StructuredOutput", description="Human-readable schema name")
    instruction: str = Field(
        default="",
        description="Additional instruction for the model before schema conversion",
    )
    strict_json: bool = Field(
        default=True,
        description="If true, executor asks model to return only raw JSON",
    )

    @classmethod
    def from_pydantic_model(
        cls,
        model_cls: type[BaseModel],
        input_source: str = "$history[-1]",
        instruction: str = "",
        strict_json: bool = True,
    ) -> "StructuredOutputStepConfig":
        """Create structured config from a Pydantic model class."""
        return cls(
            input_source=input_source,
            output_schema=model_cls.model_json_schema(),
            schema_name=model_cls.__name__,
            instruction=instruction,
            strict_json=strict_json,
        )


class AgentHandoffStepConfig(BaseModel):
    """
    Configuration for agent handoff steps.

    An agent handoff step runs a complete sub-chain inside the parent chain's
    execution, with isolated context derived from the parent. The result is
    merged back into the parent via ``output_memory_key``.

    Example::

        AgentHandoffStepDescription(
            number=3,
            title="Delegate to research agent",
            sub_chain=research_chain,
            config=AgentHandoffStepConfig(
                input_mapping={
                    "input.topic": "$memory.input.topic",
                    "input.lang": "'english'",
                },
                output_memory_key="research_result",
            ),
        )

    Input mapping
    -------------
    Keys are dotted ``namespace.key`` paths written into the sub-chain's memory.
    Values are parent context references resolved by ``resolve_context_reference``:
    ``$memory.ns.key``, ``$history[-1]``, ``$outer_context``, or quoted literals.

    If a key has no dot the value is written to the ``"input"`` namespace.

    Output
    ------
    On success, the sub-chain's final result (last history entry) is written to
    ``context.memory_write(output_memory_key, result, namespace=output_namespace)``.
    The full ``ReasoningResult`` is always available in ``result_data["sub_result"]``.
    """

    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps resolved parent values to sub-chain memory entries. "
            "Keys are 'namespace.key' in sub-chain memory; values are parent context references."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context to store the sub-chain's result. "
                    "Empty string disables automatic result storage.",
    )
    output_namespace: str = Field(
        default="handoff",
        description="Namespace to use when writing the output to parent memory.",
    )
    propagate_failure: bool = Field(
        default=True,
        description="If True, a failed sub-chain marks this step as failed. "
                    "If False, sub-chain failures are recorded in result_data but the parent step succeeds.",
    )
    inherit_tools: bool = Field(
        default=True,
        description="If True, the sub-chain's context receives a copy of the parent tool registry.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum seconds to allow the sub-chain to run. None = no timeout.",
    )


class LoopConfig(BaseModel):
    """
    Configuration for loop-back execution in a reasoning chain.

    Attach to a step via ``loop_back_to`` / ``loop_config`` on
    ``StepDescriptionBase``.  After the step completes successfully the
    executor checks ``condition_key``; when the resolved value is truthy
    (and ``max_iterations`` is not exhausted) it resets the loop body steps
    and re-runs them.

    Example::

        # Steps 1-2 form the loop body; step 2 drives iteration.
        ToolStepDescription(
            number=2,
            title="Refine answer",
            config=ToolStepConfig(tool_name="refiner", input_mapping={}),
            loop_back_to=1,
            loop_config=LoopConfig(condition_key="$memory.loop.needs_retry", max_iterations=5),
        )
    """

    condition_key: str = Field(
        default="",
        description=(
            "Reference to the value that controls loop continuation. "
            "Accepts any context reference supported by resolve_context_reference: "
            "``$memory.namespace.key``, ``$history[-1]``, or ``$outer_context``. "
            "The resolved value is cast to bool; truthy → loop continues. "
            "Empty string (default) means 'always loop' up to max_iterations."
        ),
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum number of times the loop body may be re-executed (budget guard).",
    )
    negate_condition: bool = Field(
        default=False,
        description=(
            "If True, the continuation decision is negated: loop continues when the resolved "
            "value is *falsy* (until-loop semantics). Default False = while-loop semantics."
        ),
    )


class EvalFailAction(str, Enum):
    """Action to take when an EvaluationStep's criteria are not met."""

    CONTINUE = "continue"
    """Log the failure and let the chain continue normally."""

    ABORT = "abort"
    """Fail the chain immediately (step result marked as failure)."""

    RETRY_WITH_FEEDBACK = "retry_with_feedback"
    """Re-run an improved LLM response using the critique as additional context,
    then re-evaluate up to *max_retries* times before falling through to CONTINUE."""


class EvaluationStepConfig(BaseModel):
    """
    Configuration for inline quality-gate (evaluation) steps.

    An evaluation step assesses the output of a previously executed step and
    reacts according to ``on_fail`` when one or more criteria are not satisfied.

    Example::

        EvaluationStepDescription(
            number=5,
            title="Check output quality",
            dependencies=[4],
            config=EvaluationStepConfig(
                evaluates_step=4,
                criteria=[
                    "The response is at least 200 words",
                    "All numerical claims have sources",
                ],
                on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
                max_retries=2,
            ),
        )

    Criteria evaluation methods
    ---------------------------
    ``"rule"`` (default when all criteria are simpleeval expressions):
        Each criterion is a boolean expression evaluated by simpleeval over
        a ``value`` variable holding the target step's output string.
        Example: ``"len(value) >= 200"`` or ``"'source' in value"``.

        Shorthand patterns (no eval required):
        - ``"min_words:<N>"``  — at least N whitespace-separated words
        - ``"contains:<text>"`` — case-insensitive substring check
        - ``"startswith:<prefix>"`` / ``"endswith:<suffix>"``

    ``"llm"``:
        Passes all criteria to an LLM judge in a structured prompt.
        The LLM returns ``PASS`` or ``FAIL`` with a critique string.
        Criteria can be plain English sentences.
    """

    evaluates_step: int = Field(
        ...,
        description=(
            "Step number whose output is evaluated. The evaluation step must declare "
            "this step in its dependencies so the output is available in history."
        ),
    )
    criteria: list[str] = Field(
        ...,
        min_length=1,
        description="List of evaluation criteria. Interpretation depends on evaluation_method.",
    )
    on_fail: EvalFailAction = Field(
        default=EvalFailAction.CONTINUE,
        description="Action to take when one or more criteria are not satisfied.",
    )
    evaluation_method: Literal["rule", "llm"] = Field(
        default="rule",
        description="How to evaluate criteria: 'rule' uses simpleeval expressions, 'llm' uses an LLM judge.",
    )
    input_source: str = Field(
        default="$history[-1]",
        description=(
            "Context reference pointing to the evaluated step's output "
            "(e.g. '$history[-1]', '$history[-2]'). Defaults to the most recent history entry."
        ),
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        description=(
            "Maximum number of LLM retry calls when on_fail=RETRY_WITH_FEEDBACK. "
            "After exhausting retries the chain continues normally."
        ),
    )


class ParallelSamplingAggregation(str, Enum):
    """Aggregation strategy for ParallelSamplingStep."""

    MAJORITY_VOTE = "majority_vote"  # Most common response wins (exact/normalized match)
    BEST_OF_N = "best_of_n"  # LLM judge picks the best candidate
    LLM_JUDGE = "llm_judge"  # Alias for best_of_n with explicit judge prompt


class ParallelSamplingStepConfig(BaseModel):
    """Configuration for ParallelSamplingStep."""

    n_samples: int = Field(
        default=5,
        ge=2,
        description="Number of independent LLM samples to draw in parallel.",
    )
    aggregation: ParallelSamplingAggregation = Field(
        default=ParallelSamplingAggregation.MAJORITY_VOTE,
        description="Strategy used to select the winning response from all samples.",
    )
    judge_prompt: str = Field(
        default="",
        description=(
            "Optional custom judge prompt for BEST_OF_N / LLM_JUDGE aggregation. "
            "Receives the candidates as a numbered list. Leave empty to use the default."
        ),
    )
    normalize_for_vote: bool = Field(
        default=True,
        description=(
            "When True, responses are lowercased and stripped before majority-vote comparison, "
            "reducing spurious mismatches from whitespace/case differences."
        ),
    )


class ModuleToolSource(BaseModel):
    """
    Discover tools by scanning a Python module for public callables.

    Example::

        ModuleToolSource(module="mypackage.tools")
        ModuleToolSource(module="mypackage.tools", name_prefix="search_")
        ModuleToolSource(module="mypackage.tools", tag="information")
    """

    module: str = Field(..., description="Fully-qualified Python module path to import and scan.")
    name_prefix: str = Field(
        default="",
        description=(
            "Only register callables whose names start with this prefix. "
            "Empty string (default) matches all public callables."
        ),
    )
    tag: str = Field(
        default="",
        description=(
            "Only register callables that have the given tag in their "
            "``__carl_tool_tags__`` attribute (set by the @carl_tool decorator). "
            "Empty string (default) disables tag filtering."
        ),
    )
    strip_prefix: bool = Field(
        default=False,
        description=(
            "When True, ``name_prefix`` is stripped from registered tool names. "
            "E.g. prefix='tool_', strip_prefix=True → 'tool_search' registered as 'search'."
        ),
    )


class CallableToolSource(BaseModel):
    """
    Discover tools by calling a factory function.

    The factory must return a ``dict[str, Callable]`` mapping tool names to callables.

    Example::

        def my_factory() -> dict:
            return {"search": search_fn, "summarize": summarize_fn}

        CallableToolSource(factory=my_factory)
    """

    model_config = {"arbitrary_types_allowed": True}

    factory: Callable[[], Dict[str, Callable]] = Field(
        ..., description="Zero-argument callable that returns a dict of {tool_name: callable}."
    )


class DictToolSource(BaseModel):
    """
    Register a static dict of tools.

    Example::

        DictToolSource(tools={"search": search_fn, "calc": calc_fn})
    """

    model_config = {"arbitrary_types_allowed": True}

    tools: Dict[str, Callable] = Field(
        ..., description="Mapping of tool names to callables to register."
    )


ToolSource = Union[ModuleToolSource, CallableToolSource, DictToolSource]


class ToolDiscoveryStepConfig(BaseModel):
    """
    Configuration for a ToolDiscoveryStep.

    Discovers tools from an external source and registers them in the context
    so that subsequent ToolSteps can use them.

    Example::

        ToolDiscoveryStepDescription(
            number=1,
            title="Load search tools",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="myapp.search_tools", name_prefix="tool_"),
                output_memory_key="discovered_tools",
            ),
        )
    """

    source: ToolSource = Field(
        ...,
        description="Source of tools to discover and register.",
    )
    output_memory_key: str = Field(
        default="",
        description=(
            "If set, writes the list of discovered tool names to "
            "``context.memory['tools'][output_memory_key]`` for downstream inspection."
        ),
    )
    tool_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Optional per-call timeout applied to all discovered sync tools "
            "(wraps them with AsyncToolWrapper). Has no effect on async tools."
        ),
    )


class HumanInputStepConfig(BaseModel):
    """
    Configuration for ``HumanInputStepDescription``.

    Pauses chain execution and waits for a human to provide a string value.
    The value is written to chain history and optionally to memory.

    Workflow:

    1. Executor calls ``context.on_human_input_requested(prompt, future)`` (if set).
    2. Executor awaits the asyncio ``Future`` — this suspends only the current coroutine;
       the event loop remains live so the caller can schedule
       ``context.provide_human_input(value)``.
    3. When ``provide_human_input`` is called, the future resolves and execution resumes.
    4. If ``timeout`` seconds elapse before a value is provided, ``fallback_value`` is used.

    Example::

        HumanInputStepDescription(
            number=4,
            title="User review required",
            config=HumanInputStepConfig(
                prompt="Please review the draft and provide feedback:",
                timeout=300,
                fallback_value="[no feedback provided]",
                output_memory_key="user_feedback",
            ),
        )
    """

    prompt: str = Field(
        default="Please provide input:",
        description="Human-readable prompt displayed when asking for input.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum seconds to wait for human input. "
            "If elapsed, ``fallback_value`` is used and execution continues. "
            "None means wait indefinitely."
        ),
    )
    fallback_value: str = Field(
        default="[no input provided]",
        description="Value used when timeout elapses before input is provided.",
    )
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "If set, write the collected input to this key in the 'human_input' "
            "memory namespace. Useful for downstream steps to read via "
            "``$memory.human_input.<key>``."
        ),
    )


# Union type for all step configurations
StepConfig = Union[
    ToolStepConfig,
    MCPStepConfig,
    MemoryStepConfig,
    TransformStepConfig,
    ConditionalStepConfig,
    StructuredOutputStepConfig,
    EvaluationStepConfig,
    AgentHandoffStepConfig,
    ParallelSamplingStepConfig,
    ToolDiscoveryStepConfig,
    HumanInputStepConfig,
    LoopConfig,
    None,
]


class StepCache(BaseModel):
    """
    Step result memoization configuration.

    When attached to a step via the ``cache`` field, the DAG executor checks a
    per-executor in-memory cache **before** running the step.  On a cache hit
    the stored result is returned immediately — no LLM or tool call is made.
    On a miss the step runs normally and the result is stored for future hits.

    The cache lives on the ``DAGExecutor`` instance, so it is shared across
    all batches within a single chain execution (useful for loops that revisit
    the same inputs) but reset on each new ``chain.execute()`` call.

    Example::

        LLMStepDescription(
            number=2,
            title="Classify intent",
            aim="Classify the user intent",
            cache=StepCache(
                ttl=300,
                key_fn=lambda ctx: ctx.outer_context[:512],
            ),
        )

    A step is considered a cache hit when:
    1. A cache entry exists for the computed key, AND
    2. The entry has not expired (``ttl=None`` → never expires).
    """

    model_config = {"arbitrary_types_allowed": True}

    ttl: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cache time-to-live in seconds. After this many seconds the cached "
            "result expires and the step will be re-executed on the next call. "
            "None (default) means the cache never expires within the executor's lifetime."
        ),
    )
    key_fn: Optional[Callable[..., str]] = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable that receives the current ``ReasoningContext`` and returns "
            "a string cache key.  When None, the default key combines the step number "
            "and the first 256 characters of ``context.outer_context``."
        ),
    )


class ContextQuery(BaseModel):
    """
    Individual context query with optional search configuration override.

    Allows fine-grained control over search strategy for specific queries.
    """

    query: str = Field(..., description="The query text for context extraction")
    search_strategy: Optional[Literal["substring", "vector"]] = Field(
        default=None, description="Override search strategy for this query"
    )
    search_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional search configuration for this query"
    )

    def __str__(self) -> str:
        return self.query


class LLMStepConfig(BaseModel):
    """
    Configuration for per-step LLM overrides.

    Allows specifying a different model for specific LLM steps,
    overriding the default set in ReasoningContext.

    Example usage:
        ```python
        # Override model for a specific step (OpenAI-compatible APIs)
        LLMStepDescription(
            number=1,
            title="Complex Analysis",
            aim="Perform complex analysis",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```
    """

    # Model override (for OpenAI-compatible APIs)
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use for this step (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')",
    )

    # Temperature override
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for this step (overrides default)",
    )

    # Max tokens override
    max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens for this step (overrides default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    # Token budget warning threshold
    token_budget_warning: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Emit a warning when total tokens consumed by this step exceed this threshold. "
            "None = no warning. Only works with clients that return usage info (e.g. OpenAICompatibleClient)."
        ),
    )

    # Execution mode override
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.FAST,
        description="Execution strategy for this step",
    )

    # SELF_CRITIC configuration
    self_critic_evaluators: list[str] = Field(
        default_factory=lambda: ["llm"],
        min_length=1,
        description="Ordered evaluator names for SELF_CRITIC mode; all must approve",
    )
    self_critic_max_revisions: int = Field(
        default=1,
        ge=0,
        description="Maximum candidate regeneration rounds when evaluator chain disapproves",
    )
    self_critic_instruction: str = Field(
        default="",
        description="Optional extra instruction for built-in 'llm' self-critic evaluator.",
    )
    self_critic_disapprove_feedback: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Static feedback appended to regeneration notes when evaluator disapproves. "
            "Keys are evaluator names; optional '*' applies to any evaluator."
        ),
    )

    # Multi-turn message history
    use_message_history: bool = Field(
        default=False,
        description=(
            "When True, the LLM step sends a structured message list instead of a flat prompt. "
            "System prompt and outer context become the first 'system' message; "
            "previous turns from ``context.messages`` are included as-is; "
            "the current step prompt is appended as a 'user' message; "
            "the LLM response is stored back into ``context.messages`` as an 'assistant' message. "
            "Requires a client that supports :meth:`get_response_with_messages` "
            "(e.g. OpenAICompatibleClient)."
        ),
    )


class SupervisorStepConfig(BaseModel):
    """
    Configuration for supervisor / hierarchical routing steps.

    A supervisor step asks an LLM to pick *one* of N named sub-chains
    (registered as ``agents`` on the :class:`SupervisorStepDescription`) and
    runs the chosen sub-chain. The result is folded back into the parent's
    history / memory just like an :class:`AgentHandoffStepConfig`.

    Example::

        SupervisorStepDescription(
            number=1,
            title="Route to specialist",
            agents={"pdf": pdf_chain, "search": search_chain, "code": code_chain},
            config=SupervisorStepConfig(
                routing_prompt=(
                    "Choose ONE specialist for the task below. "
                    "Reply with just the specialist name.\\n\\nTask: {task}"
                ),
                task_source="$outer_context",
                fallback_agent="search",
                output_memory_key="specialist_result",
            ),
        )

    Routing prompt
    --------------
    ``{task}`` is replaced with the resolved task string before the LLM call;
    ``{agents}`` is replaced with a comma-joined list of available agent names.

    The LLM's reply is matched (case-insensitive, whitespace-trimmed, prefix-
    tolerant) against the registered agent names. If no match is found and
    ``fallback_agent`` is set, that agent is used; otherwise the step fails
    with a descriptive error.
    """

    routing_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Prompt template the supervisor LLM sees. Supports ``{task}`` (the "
            "resolved task string) and ``{agents}`` (comma-joined agent names)."
        ),
    )
    task_source: str = Field(
        default="$outer_context",
        description=(
            "Where to resolve the task string from. Same syntax as input_mapping "
            "values: ``$outer_context``, ``$memory.ns.key``, ``$history[-1]``, "
            "or a quoted literal."
        ),
    )
    fallback_agent: Optional[str] = Field(
        default=None,
        description=(
            "Agent name to use when the LLM's reply doesn't match any registered "
            "agent. When ``None`` (default), an unparseable routing reply fails the step."
        ),
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Forwarded to the chosen sub-chain — same semantics as "
            ":class:`AgentHandoffStepConfig.input_mapping`."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context to store the sub-chain's result.",
    )
    output_namespace: str = Field(
        default="supervisor",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    propagate_failure: bool = Field(
        default=True,
        description=(
            "If True, a failed sub-chain marks this step as failed. "
            "If False, sub-chain failures are recorded in result_data but the parent step succeeds."
        ),
    )
    inherit_tools: bool = Field(
        default=True,
        description="If True, the sub-chain inherits the parent context's tool registry.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional total timeout (seconds) for the sub-chain.",
    )
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for the *routing* call itself.",
    )


class DebateStepConfig(BaseModel):
    """
    Configuration for round-robin multi-agent debate steps.

    A debate step runs N round-robin turns over a list of named roles
    (e.g. ``["proponent", "critic"]``). Each role's LLM call sees the topic
    plus the running transcript and produces its next argument. After all
    rounds complete, a single judge LLM call synthesises the transcript into
    a final verdict.

    Example::

        DebateStepDescription(
            number=3,
            title="Debate the approach",
            config=DebateStepConfig(
                roles=["proponent", "critic"],
                role_prompts={
                    "proponent": "Argue IN FAVOUR of the proposal. Be concrete.",
                    "critic":    "Argue AGAINST the proposal. Flag risks.",
                },
                rounds=2,
                judge_prompt="Given the debate above, what is the best answer? Topic: {task}",
                output_memory_key="verdict",
            ),
        )

    Prompt templating
    -----------------
    Per-role prompt sees ``{task}``, ``{role}``, ``{round}``, ``{transcript}``.
    Judge prompt sees ``{task}`` and ``{transcript}``.

    LLM-call count
    --------------
    ``len(roles) * rounds + 1`` (the final ``+1`` is the judge).
    """

    roles: list[str] = Field(
        ...,
        min_length=2,
        description="Names of the debating roles (≥ 2). Used as transcript speaker labels.",
    )
    role_prompts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-role instruction template. Missing roles fall back to a "
            "neutral 'Argue your position' instruction. Supports ``{task}``, "
            "``{role}``, ``{round}``, ``{transcript}`` placeholders."
        ),
    )
    rounds: int = Field(
        default=2,
        ge=1,
        description="Number of round-robin rounds.",
    )
    task_source: str = Field(
        default="$outer_context",
        description=(
            "Where to resolve the debate topic from — same reference syntax as "
            "``input_mapping`` values."
        ),
    )
    judge_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Prompt for the final synthesis call. Supports ``{task}`` and "
            "``{transcript}`` placeholders. Required."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context for the judge's verdict.",
    )
    output_namespace: str = Field(
        default="debate",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="LLM config applied to every role call AND the judge call (default).",
    )
    role_llm_configs: dict[str, LLMStepConfig] = Field(
        default_factory=dict,
        description=(
            "Optional per-role LLM config overrides. Used in place of ``llm_config`` "
            "for the specified role's turns. The judge always uses ``llm_config``."
        ),
    )


class StepGroup(BaseModel):
    """
    Apply ``llm_config`` overrides to a named subset of steps in a chain.

    Useful for separating phases — e.g. give all "analysis" steps
    ``temperature=0.0`` while leaving "creative" steps at the chain default::

        ReasoningChain(
            default_llm_config=LLMStepConfig(model="gpt-4o", temperature=0.7),
            step_groups=[
                StepGroup(
                    name="analysis",
                    steps=[2, 3, 4],
                    llm_config=LLMStepConfig(temperature=0.0),
                ),
            ],
            steps=[...],
        )

    Precedence at runtime (highest → lowest):

    1. Per-step ``llm_config`` (explicitly set fields)
    2. ``StepGroup.llm_config`` (explicitly set fields on the group)
    3. ``ReasoningChain.default_llm_config``
    4. Context default client

    Group config is "baked into" each member step's ``llm_config`` at chain
    construction time using pydantic's ``model_fields_set`` to detect which
    fields the caller actually set — defaults are not propagated, so the
    chain default still wins for unset fields.

    Validation rules (enforced by :class:`ReasoningChain`):

    - Every step number listed must exist in the chain.
    - A step may appear in at most one group.
    """

    name: str = Field(..., min_length=1, description="Human-readable group label (used in errors).")
    steps: list[int] = Field(
        ...,
        min_length=1,
        description="Step numbers that receive this group's llm_config overrides.",
    )
    llm_config: LLMStepConfig = Field(
        ...,
        description="LLM config whose explicitly-set fields are merged into each member step.",
    )
