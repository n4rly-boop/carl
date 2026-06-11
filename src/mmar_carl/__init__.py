"""
MMAR CARL (Collaborative Agent Reasoning Library)

A library for building chain-of-thought reasoning systems with DAG-based parallel execution.
Supports multiple step types: LLM, Tool, MCP, Memory, Transform, and Conditional.

Module Structure:
- mmar_carl.models: Package containing all data models (re-exports all for convenience)
  - mmar_carl.models.enums: StepType, MemoryOperation, Language
  - mmar_carl.models.base: SearchStrategy, SelfCriticDecision, SelfCriticEvaluatorBase
  - mmar_carl.models.llm_client_base: LLMClientBase
  - mmar_carl.models.search: SubstringSearchStrategy, VectorSearchStrategy, ContextSearchConfig
  - mmar_carl.models.config: ToolParameter, ToolStepConfig, MCPStepConfig, MemoryStepConfig, etc.
  - mmar_carl.models.steps: StepDescriptionBase, LLMStepDescription, ToolStepDescription, etc.
  - mmar_carl.models.context: ReasoningContext
  - mmar_carl.models.results: StepExecutionResult, ReasoningResult
  - mmar_carl.models.prompts: PromptTemplate

Step Description Classes (New API):
- StepDescriptionBase: Abstract base class for all steps
- LLMStepDescription: LLM reasoning steps
- ToolStepDescription: External tool/function execution
- MCPStepDescription: MCP protocol calls
- MemoryStepDescription: Memory operations
- TransformStepDescription: Data transformations
- ConditionalStepDescription: Conditional branching
- AnyStepDescription: Union type of all step types
- create_step(): Factory function for creating typed steps

Legacy API (Backward Compatible):
- StepDescription: Unified class supporting all step types
"""

from .chain import (
    ChainBuilder,
    ChainFormatNewerError,
    ReasoningChain,
    ReflectionOptions,
    create_chain_from_config,
)
from .dataset_evaluator import DatasetEvaluator
from .metrics import MetricBase, MetricOutput, call_metric_async
from .executor import DAGExecutor, ExecutionCancelledError
from .tracing import flush as langfuse_flush, is_langfuse_enabled
from .replan import (
    LLMReplanChecker,
    RuleBasedReplanChecker,
)
from .llm import (
    OpenAIClientConfig,
    OpenAICompatibleClient,
    RetryPolicy,
    create_openai_client,
)
from .anthropic_client import (
    AnthropicClient,
    AnthropicClientConfig,
    AnthropicClientError,
    openai_tools_to_anthropic,
)
from .record_replay import (
    CassetteMissError,
    PlayingLLMClient,
    RecordingLLMClient,
)
from .skill_loader import SkillLoader, SkillNotFoundError, SkillParseError
from .skill_runtime import (
    LocalSkillRuntime,
    NetworkPolicy,
    RuntimeRunResult,
    SkillRuntime,
    SkillRuntimeError,
    SkillRuntimeHandle,
    get_skill_runtime,
    list_skill_runtimes,
    parse_network_allowlist_from_allowed_tools,
    register_skill_runtime,
    resolve_network_policy,
)
# Import for side effect: self-registers DockerSkillRuntime in the
# runtime registry so AgentSkillStepConfig(runtime="docker") works.
from .docker_skill_runtime import DockerSkillRuntime  # noqa: F401
# Same pattern for the Firejail sandbox (Linux-only at runtime, but
# the module itself imports clean everywhere — the firejail CLI check
# happens at `prepare()` time).
from .firejail_skill_runtime import FirejailSkillRuntime  # noqa: F401
# E2B remote sandbox — self-registers; lazy-imports the `e2b` SDK
# only inside `prepare()` so users without the extra don't pay it.
from .e2b_skill_runtime import E2BSkillRuntime  # noqa: F401
from .mcp_pool import MCPSessionPool
from .skill_resolver import (
    GithubResolver,
    HttpsTarballResolver,
    LocalResolver,
    ModuleResolver,
    ResolvedSkill,
    SkillFrontmatterData,
    SkillIntegrityError,
    SkillResolveError,
    SkillResolverRegistry,
    get_default_registry,
    list_cached_skills,
    resolve_skill,
)
from .chain_evolution import (
    ChainEvolver,
    ChainMutator,
    EvolutionCostEstimate,
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
    MutationKind,
    format_runs_pareto,
)
from .execution_trace import ExecutionTrace, TraceAggregator, TraceEvent
from .visualizer import ChainVisualizer
from .lazy_memory import LazyMemoryValue, unwrap_lazy
from .ltm import InMemoryLTM, JsonFileLTM, LTMBase
from .models import (
    # AgentSkill models
    AgentSkillExecutionMode,
    AgentSkillSource,
    AgentSkillStepConfig,
    SkillManifest,
    # Dataset abstractions
    AbstractDataset,
    CARE_METADATA_NAMESPACE,
    CareChainMetadata,
    CareContextFile,
    CaseEvaluationResult,
    PreflightReport,
    RunRecord,
    DataCase,
    DataFrameDataset,
    DatasetEvaluationReport,
    SelectionStrategy,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
    # Enums
    Language,
    MemoryOperation,
    StepType,
    # Abstract base classes
    ChatMessage,
    LLMClientBase,
    SearchStrategy,
    SelfCriticDecision,
    SelfCriticEvaluatorBase,
    # Search & Context
    ContextQuery,
    ContextSearchConfig,
    SubstringSearchStrategy,
    VectorSearchStrategy,
    # Step Configurations
    AgentHandoffStepConfig,
    ConditionalBranch,
    ConditionalStepConfig,
    LoopConfig,
    EvalFailAction,
    EvaluationStepConfig,
    ExecutionMode,
    LLMStepConfig,
    ParallelSamplingAggregation,
    ParallelSamplingStepConfig,
    LLMReplanCheckerConfig,
    MCPServerConfig,
    MCPStepConfig,
    MCPResourceStepConfig,
    MemoryStepConfig,
    StepCache,
    StepGroup,
    StepConfig,
    ToolErrorRecovery,
    ToolParameter,
    ToolStepConfig,
    TransformStepConfig,
    StructuredOutputStepConfig,
    SupervisorStepConfig,
    DebateStepConfig,
    CallableToolSource,
    DictToolSource,
    ModuleToolSource,
    ToolDiscoveryStepConfig,
    HumanInputStepConfig,
    ToolSource,
    # Step Description Classes (New API)
    AgentHandoffStepDescription,
    AgentSkillStepDescription,
    HumanInputStepDescription,
    ToolDiscoveryStepDescription,
    AnyStepDescription,
    ConditionalStepDescription,
    EvaluationStepDescription,
    LLMStepDescription,
    MCPStepDescription,
    MCPResourceStepDescription,
    MemoryStepDescription,
    ParallelSamplingStepDescription,
    StepDescriptionBase,
    ToolStepDescription,
    TransformStepDescription,
    StructuredOutputStepDescription,
    SupervisorStepDescription,
    DebateStepDescription,
    create_step,
    # Legacy Step Description (Backward Compatible)
    StepDescription,
    # Execution Context & Results
    PromptTemplate,
    RegisteredReplanCheckerConfig,
    ContextSnapshot,
    ReasoningContext,
    ReasoningResult,
    ReplanAggregationOutcome,
    ReplanCheckerVote,
    ReplanEvent,
    ReplanAction,
    ReplanAggregationConfig,
    ReplanAggregationStrategy,
    ReplanBudgetConfig,
    ReplanCheckerBase,
    ReplanCheckerInput,
    ReplanCheckerSpec,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanTriggerConfig,
    ReplanVerdict,
    RuleBasedReplanCheckerConfig,
    StepExecutionResult,
)
# typed views over StepExecutionResult.result_data.
from .models.result_data import (
    DebateTranscript,
    DebateTurn,
    ParallelSamples,
    SkillOutput,
    SupervisorDecision,
)
from .step_executors import (
    AgentHandoffStepExecutor,
    AgentSkillStepExecutor,
    AsyncToolWrapper,
    ConditionalStepExecutor,
    EvaluationStepExecutor,
    LLMStepExecutor,
    MCPStepExecutor,
    MemoryStepExecutor,
    ParallelSamplingStepExecutor,
    StepExecutorBase,
    ToolStepExecutor,
    TransformStepExecutor,
    StructuredOutputStepExecutor,
    ToolDiscoveryStepExecutor,
    HumanInputStepExecutor,
    carl_tool,
    get_executor,
    register_executor,
)
from .logging_utils import (
    get_logger,
    set_log_level,
    log_chain_start,
    log_chain_complete,
    log_step_start,
    log_step_complete,
    log_batch_start,
    log_error,
    log_warning,
    log_debug,
    log_info,
)
from .cost import CostEstimate, StepCostEstimate
from .eval_suite import (
    CaseInsensitiveMatchMetric,
    ContainsMetric,
    EvalCaseResult,
    EvalSuite,
    EvalSuiteDiff,
    EvalSuiteReport,
    ExactMatchMetric,
    MetricSummary,
    RegexMatchMetric,
)
from .memory_schema import MemorySchemaError
from .skill_output_schema import SkillOutputSchemaError, parse_and_validate_skill_output
from .tool_definition import ToolDefinition
from .streaming import StreamingBuffer
from .testing import ChainTestHarness

__version__ = "0.3.0"
__all__ = [
    # Core Chain Classes
    "ChainFormatNewerError",
    "ReasoningChain",
    "ChainBuilder",
    "DAGExecutor",
    "create_chain_from_config",
    # Metrics
    "MetricBase",
    "call_metric_async",
    "MetricOutput",
    # Reflection
    "ReflectionOptions",
    # Dataset abstractions
    "DataCase",
    "AbstractDataset",
    "SimpleDataset",
    "DataFrameDataset",
    "ThresholdStrategy",
    "TopKWorstStrategy",
    "SelectionStrategy",
    "CARE_METADATA_NAMESPACE",
    "CareChainMetadata",
    "CareContextFile",
    "CaseEvaluationResult",
    "PreflightReport",
    "RunRecord",
    "DatasetEvaluationReport",
    "DatasetEvaluator",
    # Exceptions
    "ExecutionCancelledError",
    # Enums
    "Language",
    "MemoryOperation",
    "StepType",
    # Abstract Base Classes
    "ChatMessage",
    "LLMClientBase",
    "SearchStrategy",
    "SelfCriticDecision",
    "SelfCriticEvaluatorBase",
    # Search & Context
    "ContextQuery",
    "ContextSearchConfig",
    "SubstringSearchStrategy",
    "VectorSearchStrategy",
    # Step Configurations
    "AgentHandoffStepConfig",
    "ConditionalBranch",
    "ConditionalStepConfig",
    "LoopConfig",
    "EvalFailAction",
    "EvaluationStepConfig",
    "ExecutionMode",
    "LLMStepConfig",
    "ParallelSamplingAggregation",
    "ParallelSamplingStepConfig",
    "ReplanAction",
    "ReplanTargetType",
    "ReplanRollbackTarget",
    "ReplanVerdict",
    "ReplanAggregationStrategy",
    "ReplanAggregationConfig",
    "ReplanTriggerConfig",
    "ReplanBudgetConfig",
    "RuleBasedReplanCheckerConfig",
    "LLMReplanCheckerConfig",
    "RegisteredReplanCheckerConfig",
    "ReplanCheckerSpec",
    "ReplanPolicy",
    "ReplanCheckerInput",
    "ReplanCheckerBase",
    "MCPServerConfig",
    "MCPStepConfig",
    "MemoryStepConfig",
    "StepConfig",
    "ToolErrorRecovery",
    "ToolParameter",
    "ToolStepConfig",
    "TransformStepConfig",
    "StructuredOutputStepConfig",
    "SupervisorStepConfig",
    "DebateStepConfig",
    # Tool discovery
    "ModuleToolSource",
    "CallableToolSource",
    "DictToolSource",
    "ToolSource",
    "HumanInputStepConfig",
    "HumanInputStepDescription",
    "HumanInputStepExecutor",
    "StepCache",
    "StepGroup",
    "ToolDiscoveryStepConfig",
    "ToolDiscoveryStepDescription",
    "ToolDiscoveryStepExecutor",
    "carl_tool",
    # AgentSkill
    "AgentSkillExecutionMode",
    "AgentSkillSource",
    "AgentSkillStepConfig",
    "AgentSkillStepDescription",
    "AgentSkillStepExecutor",
    "SkillLoader",
    "SkillManifest",
    "SkillNotFoundError",
    "SkillParseError",
    # SkillRuntime — pluggable sandbox backends
    "DockerSkillRuntime",
    "E2BSkillRuntime",
    "FirejailSkillRuntime",
    "LocalSkillRuntime",
    "MCPSessionPool",
    "RuntimeRunResult",
    "SkillRuntime",
    "SkillRuntimeError",
    "SkillRuntimeHandle",
    "NetworkPolicy",
    "get_skill_runtime",
    "list_skill_runtimes",
    "parse_network_allowlist_from_allowed_tools",
    "register_skill_runtime",
    "resolve_network_policy",
    # SkillResolver
    "GithubResolver",
    "HttpsTarballResolver",
    "LocalResolver",
    "ModuleResolver",
    "ResolvedSkill",
    "SkillFrontmatterData",
    "SkillIntegrityError",
    "SkillResolveError",
    "SkillResolverRegistry",
    "get_default_registry",
    "list_cached_skills",
    "resolve_skill",
    # Step Description Classes (New API)
    "AgentHandoffStepDescription",
    "AnyStepDescription",
    "ConditionalStepDescription",
    "EvaluationStepDescription",
    "LLMStepDescription",
    "MCPStepDescription",
    "MCPResourceStepDescription",
    "MCPResourceStepConfig",
    "MemoryStepDescription",
    "ParallelSamplingStepDescription",
    "StepDescriptionBase",
    "ToolStepDescription",
    "TransformStepDescription",
    "StructuredOutputStepDescription",
    "SupervisorStepDescription",
    "DebateStepDescription",
    "create_step",
    # Legacy Step Description (Backward Compatible)
    "StepDescription",
    # Execution Context & Results
    "ContextSnapshot",
    "PromptTemplate",
    "ReasoningContext",
    "ReasoningResult",
    "ReplanCheckerVote",
    "ReplanAggregationOutcome",
    "ReplanEvent",
    "DebateTranscript",
    "DebateTurn",
    "ParallelSamples",
    "SkillOutput",
    "StepExecutionResult",
    "SupervisorDecision",
    # Execution trace
    "ChainVisualizer",
    "ExecutionTrace",
    "TraceAggregator",
    "TraceEvent",
    # Lazy memory wrapper
    "LazyMemoryValue",
    "unwrap_lazy",
    # Chain evolution
    "ChainEvolver",
    "ChainMutator",
    "EvolutionCostEstimate",
    "EvolutionResult",
    "GenerationStats",
    "IndividualMetrics",
    "MutationKind",
    "format_runs_pareto",
    # Long-term memory
    "LTMBase",
    "InMemoryLTM",
    "JsonFileLTM",
    # Tracing
    "langfuse_flush",
    "is_langfuse_enabled",
    # LLM
    "create_openai_client",
    "OpenAIClientConfig",
    "OpenAICompatibleClient",
    "RetryPolicy",
    "AnthropicClient",
    "AnthropicClientConfig",
    "AnthropicClientError",
    "openai_tools_to_anthropic",
    "CassetteMissError",
    "PlayingLLMClient",
    "RecordingLLMClient",
    # RE-PLAN runtime checkers
    "RuleBasedReplanChecker",
    "LLMReplanChecker",
    # Step Executors
    "AgentHandoffStepExecutor",
    "AgentSkillStepExecutor",
    "AsyncToolWrapper",
    "ConditionalStepExecutor",
    "EvaluationStepExecutor",
    "LLMStepExecutor",
    "MCPStepExecutor",
    "MemoryStepExecutor",
    "ParallelSamplingStepExecutor",
    "StepExecutorBase",
    "ToolStepExecutor",
    "TransformStepExecutor",
    "StructuredOutputStepExecutor",
    "get_executor",
    "register_executor",
    # Logging
    "get_logger",
    "set_log_level",
    "log_chain_start",
    "log_chain_complete",
    "log_step_start",
    "log_step_complete",
    "log_batch_start",
    "log_error",
    "log_warning",
    "log_debug",
    "log_info",
    # Testing
    "ChainTestHarness",
    "StreamingBuffer",
    "CostEstimate",
    "StepCostEstimate",
    "MemorySchemaError",
    "EvalSuite",
    "EvalSuiteReport",
    "EvalSuiteDiff",
    "EvalCaseResult",
    "MetricSummary",
    "ExactMatchMetric",
    "CaseInsensitiveMatchMetric",
    "ContainsMetric",
    "RegexMatchMetric",
    "SkillOutputSchemaError",
    "parse_and_validate_skill_output",
    "ToolDefinition",
]
