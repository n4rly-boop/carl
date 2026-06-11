# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MMAR CARL (Collaborative Agent Reasoning Library)** is a Python library for building sophisticated chain-of-thought reasoning systems with DAG-based parallel execution. It enables developers to create expert reasoning chains that can execute reasoning steps in parallel where dependencies allow, with automatic RAG-like context extraction for each step.

### Core Purpose

CARL provides a structured framework for:
- Defining multi-step reasoning chains with dependencies
- Automatically parallelizing steps based on a DAG (Directed Acyclic Graph)
- Extracting relevant context from input data for each reasoning step
- Supporting multiple step types: LLM reasoning, Tool calls, MCP protocol, Memory operations, Data transformations, and Conditional branching
- Running chains asynchronously with configurable LLM clients (OpenRouter, Azure OpenAI, local LLMs, or mmar-llm)

## Tech Stack

- **Language**: Python 3.12+
- **Core Dependencies**:
  - `pydantic>=2.11.7` - Data validation and serialization
  - `mmar-utils~=1.1.18` - Utility functions including async gathering
  - `openai>=1.0.0` - OpenAI-compatible client support
  - `simpleeval>=1.0.0` - Expression evaluation for conditional steps
- **Optional Features**:
  - `faiss-cpu`, `fastembed`, `numpy` - Vector-based semantic search for context extraction
  - `mcp>=1.0.0` - Model Context Protocol server integration
  - `langfuse>=3.0.0` - Tracing and observability
  - `pandas` - Dataset support
  - `matplotlib` (`viz`) - PNG chart output
  - `agent-skills` (`agent-skills`) - AgentSkills catalog integration (`SkillLoader.catalog_from_agent_skills`)
  - `pypdf`, `pdfplumber` (`pdf`) / `python-pptx` (`pptx`) - bundled PDF→PPTX AgentSkill workflow; the `skills` extra pulls all three. Not imported eagerly — skill scripts also self-install via `extra_pip`
- **Build**: `uv_build>=0.8.14` - Fast Python package builder
- **Testing**: `pytest>=8.2`, `pytest-asyncio>=1.0.0`
- **Linting**: `ruff>=0.4`

## Architecture Overview

### Core Components

1. **ReasoningChain** (`chain.py`)
   - Main public API for defining and executing reasoning chains
   - Supports JSON serialization/deserialization for persistence
   - Manages reflection capabilities for analyzing chain execution
   - Integrates with DAGExecutor for parallel execution
   - Supports both synchronous and asynchronous execution

2. **DAGExecutor** (`executor.py`)
   - Core execution engine that builds and executes a directed acyclic graph
   - Analyzes step dependencies and creates execution batches for maximum parallelization
   - Manages step execution through pluggable executors
   - Handles memory isolation between parallel steps (deep copy)
   - Supports RE-PLAN (replanning) with checkpoint rollback capabilities
   - Traces execution with structured logging

3. **Step Execution System** (`step_executors.py`)
   - **StepExecutorBase**: Abstract base class for all executors
   - **LLMStepExecutor**: LLM reasoning steps with context extraction, prompt generation, FAST / SELF_CRITIC execution modes
   - **ToolStepExecutor**: Registered Python functions with dynamic input mapping, retry, `ToolErrorRecovery` policies (RAISE / SKIP / RETRY / FALLBACK)
   - **MemoryStepExecutor**: Memory read/write/append/delete/list with namespace isolation
   - **TransformStepExecutor**: Data transformations without LLM calls
   - **ConditionalStepExecutor**: Conditional branching (true routing — winner step executes, others skip)
   - **MCPStepExecutor / MCPResourceStepExecutor**: Model Context Protocol tool calls + resource fetches (stdio / SSE / streamable_http transports)
   - **StructuredOutputStepExecutor**: LLM output matching JSON schemas
   - **AgentSkillStepExecutor**: Executes [AgentSkills](https://agentskills.io) — portable skill folders with `SKILL.md` instructions (LLM / SCRIPT / HYBRID / SUBAGENT / LLM_AGENT modes)
   - **AgentHandoffStepExecutor**: Explicit task delegation to a sub-chain with input/output mapping
   - **SupervisorStepExecutor**: Hierarchical routing to specialist sub-chains by LLM-chosen route
   - **DebateStepExecutor**: Round-robin role-based debate + judge synthesis
   - **ParallelSamplingStepExecutor**: N-sample voting / LLM-judge aggregation
   - **HumanInputStepExecutor**: Pauses for human input (in-process callable or webhook)
   - **ToolDiscoveryStepExecutor**: Dynamic tool discovery from `ModuleToolSource` / `CallableToolSource` / `DictToolSource`
   - **EvaluationStepExecutor**: Inline metric evaluation as a chain step (fail/pass gates downstream branches)
   - **Loop support**: Steps can declare `loop_config` (while/until pattern) — the executor re-runs the step until the condition is met or `max_iterations` exhausts
   - Executor registry pattern: `get_executor()` and `register_executor()` for extensibility

4. **ReasoningContext** (`models/context.py`)
   - Manages execution state: outer_context (input data), history, memory, metadata
   - Holds references to LLM client, language preference, system prompts
   - Provides tool registry for function calls
   - Supports callbacks for monitoring (on_step_start, on_step_complete, on_progress, on_llm_chunk)
   - Memory storage organized by namespace for state management
   - History management with configurable entry limits

5. **Step Definitions** (`models/steps.py`)
   - **Typed Step Classes (New API)**:
     - `LLMStepDescription`: Chain-of-thought reasoning steps
     - `ToolStepDescription`: External function/tool execution
     - `MemoryStepDescription`: Memory operations (read/write/append/delete/list)
     - `TransformStepDescription`: Data transformations
     - `ConditionalStepDescription`: Conditional branching
     - `MCPStepDescription` / `MCPResourceStepDescription`: MCP protocol calls + resource fetches
     - `StructuredOutputStepDescription`: Schema-constrained JSON output
     - `AgentSkillStepDescription`: Executes [AgentSkills](https://agentskills.io) from local path, name lookup, git URL, or Python package
     - `AgentHandoffStepDescription`: Sub-chain delegation with input/output mapping
     - `SupervisorStepDescription`: Hierarchical routing to specialist sub-chains
     - `DebateStepDescription`: Multi-role round-robin debate + judge synthesis
     - `ParallelSamplingStepDescription`: N-sample voting / LLM-judge aggregation
     - `HumanInputStepDescription`: Human-in-the-loop pause point
     - `ToolDiscoveryStepDescription`: Discover and register tools dynamically
     - `EvaluationStepDescription`: Inline metric evaluation as a chain step
   - **Legacy API**: `StepDescription` (backward compatible, unified class)
   - All steps share: `number`, `title`, `dependencies`, checkpoint support, `metrics`, per-step `llm_config` override, `retry_max`, `timeout`, `cache` (per-step caching), `loop_config` (while/until), `replan_enabled`

6. **Context Extraction (Search)** (`models/search.py`)
   - **SubstringSearchStrategy**: Fast text-based keyword search (default, no dependencies)
   - **VectorSearchStrategy**: Semantic similarity search using FAISS embeddings (requires optional deps)
   - **ContextSearchConfig**: Chain-level search configuration
   - **ContextQuery**: Per-query override with custom search strategy and parameters
   - Automatic extraction of relevant context for each step's `step_context_queries`

7. **Evaluation & Metrics** (`metrics.py`, `dataset_evaluator.py`, `eval_suite.py`)
   - **MetricBase**: Abstract class for step-level and chain-level metrics; supports case-aware dispatch (metrics whose `compute_async(output, *, case=None)` opts in get per-case ground truth via `call_metric_async`)
   - Attach metrics to steps or chains: `metrics=[WordCountMetric(), LLMJudgeMetric()]`
   - Built-in match metrics: `ExactMatchMetric`, `CaseInsensitiveMatchMetric`, `ContainsMetric`, `RegexMatchMetric`
   - **DatasetEvaluator**: Batch evaluation against a `DataCase`-based dataset; produces `DatasetEvaluationReport` with per-case scores, step outcomes, step latencies, step-level metric scores
   - **EvalSuite** (`eval_suite.py`): Lightweight golden-output regression harness for examples & docs
   - **SelectionStrategy**: `ThresholdStrategy` / `TopKWorstStrategy` for picking problem cases for reflection
   - **`DatasetEvaluationReport` formatters** (all return printable strings — no `[viz]` required):
     - `format_failure_heatmap()` — cases × steps grid of ✓ / ✗ / · / `-` outcomes with always-fail warning
     - `format_step_metric_heatmap(metric_name)` — cases × steps shaded Unicode grid for any step-level metric
     - `format_score_distribution()` — one-line Unicode box plot (min / Q1 / median / Q3 / max)
     - `format_latency_histogram()` — per-step inline sparkline histogram with p50/p95/max
     - `format_cost_trend()` — per-run sparkline with regression detector (`run > factor × median ⇒ ⚠`)

8. **RE-PLAN System** (`replan.py`, `models/replan.py`)
   - Runtime replanning with checkpoint rollback support
   - **RuleBasedReplanChecker**: Deterministic rules (error substrings, result patterns)
   - **LLMReplanChecker**: Uses LLM to decide when/how to replan
   - **ReplanPolicy**: Chain-level policy with per-step override
   - **Aggregation Strategies**: Unanimous, majority voting on checker decisions
   - **Budget Guards**: Prevent infinite replanning loops with cost tracking

9. **LLM Client Integration** (`llm.py`, `models/llm_client_base.py`, `record_replay.py`)
   - **LLMClientBase**: Abstract base for LLM implementations; covers `get_response`, `get_response_with_retries`, `get_response_with_system`, `get_response_with_usage`, `get_response_with_messages`, `get_response_with_tools`, `stream_response`. Typed introspection: `model_name`, `temperature`, `max_tokens`, `supports_streaming`
   - **ChatMessage**: Pydantic model for multi-turn message history (`role: system|user|assistant`)
   - **OpenAICompatibleClient**: Works with OpenRouter, Azure OpenAI, local LLMs (Ollama, vLLM, LM Studio); supports tool calls, streaming, message history
   - **RetryPolicy** (`llm.py`): Transient-only retries with exponential backoff + jitter; configurable `retry_on_status`, `max_attempts`, `initial_delay_s`, `max_delay_s`. Wires through `OpenAIClientConfig.retry_policy` — 401/403/404/422 are NOT retried by default
   - **RecordingLLMClient / PlayingLLMClient** (`record_replay.py`): pytest-vcr-style cassette wrappers. Record once against a real API, replay deterministically from a JSONL cassette (zero API calls). Cassette key = `sha256(method + prompt/messages + model + temperature)`. Missing key → `CassetteMissError` with prompt preview
   - **Automatic Client Detection**: Creates appropriate client from API object type (LLMHub, LLMHubAPI, OpenAICompatibleClient, mock objects)
   - Per-step LLM config override: `llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")`
   - Execution modes: FAST (single pass) and SELF_CRITIC (reasoning + evaluation)

10. **Tracing & Observability** (`tracing.py`, `execution_trace.py`)
    - **ExecutionTrace** (`execution_trace.py`): full structured trace built automatically during every chain run; attached to `ReasoningResult.trace`. Serialisable (`to_json` / `from_json`), diffable (`trace.diff(other_trace)`), replayable
    - **TraceAggregator**: aggregates per-step `{p50, p95, p99, mean, max}_ms` latency and `{p50, p95}` token usage across N traces — useful for capacity planning and regression-catching after batch runs
    - **`ExecutionTrace.format_gantt(format="text"|"mermaid")`**: parallel-batch-aware Gantt chart
    - **`ExecutionTrace.to_html(path=None)`**: standalone animated HTML/JS playback (zero external deps, inline CSS+JS) — drop into a PR description
    - Optional Langfuse integration for tracing dashboards
    - Helps debug execution flow and identify bottlenecks

11. **Chain Generation & Evolution** (`chain.py`, `chain_evolution.py`)
    - **`ChainBuilder.from_description(task, llm_client, ...)`**: meta-agent that asks an LLM to plan a chain from a natural-language description; with `max_retries` for self-correction loops. Captures full planner provenance (`planner_prompt`, `planner_reply`, per-attempt log with errors) in `chain.metadata`
    - **`ChainEvolver`** (`chain_evolution.py`): evolutionary search over chain variants. Population, elitism, mutator, generations. Smoke check, atomic checkpoint/resume (`checkpoint_path=...`), concurrent individual evaluation
      - **Multi-objective**: accepts `metric: MetricBase | list[MetricBase]` + optional `fitness_fn(scores: dict) -> float` (defaults to mean across metrics)
      - **`IndividualMetrics`**: per-individual `score`, `wall_time_s`, `total_tokens`, `llm_calls`, `mutation_kind`, `parent_score`, `scores_by_metric` (multi-objective)
      - **`EvolutionResult`** formatters: `format_score_evolution` (text + PNG), `format_pareto`, `format_spend_vs_quality`, `format_mutation_effectiveness`, `to_lineage_mermaid` (parent edges + gold border on best chain)
      - **`format_runs_pareto(results, ...)`** (top-level): cross-run Pareto chart for multiple evolution runs
      - **`EvolutionCostEstimate`**: pre-flight cost projection (smoke + pop × gens × cases)
    - **`ChainMutator`**: model swap / temperature swap / prompt rewrite / max-workers / **structural mutations** (`DELETE_STEP` removes a leaf step; `INSERT_STEP` splices a verification step from a template pool). All mutations are wrapped in `from_dict` validation; failed mutations roll back transparently

12. **Visualization Framework** (`models/results.py`, `chain.py`, `visualizer.py`)
    - **`ReasoningResult` formatters** (text-only by default; opt into PNG via `mmar-carl[viz]`):
      - `format_token_pie(format="text"|"mermaid"|"png")`
      - `format_prompt_completion_breakdown()`
      - `format_profiling_table()` — per-step cost / latency / cache table
      - `format_cost_by_model(pricing=..., format="text"|"mermaid")` — per-model cost breakdown
      - `token_usage_by_step` (property), `partial_outputs`, `get_partial_final_output`, `context_metadata`
    - **`ReasoningChain` formatters**: `to_mermaid()`, `to_mermaid_critical_path(result)`, `to_mermaid_heatmap(result, metric="tokens"|"latency"|"cost")` (uses `<br/>` for in-node line breaks)
    - **`ChainVisualizer(result, *, chain=None, evolution_result=None)`**: fluent facade that composes many viz methods into a single buffer
      ```python
      ChainVisualizer(result, chain=chain).token_pie().gantt().heatmap(metric="tokens").print()
      ```
    - **Jupyter rich display**: `_repr_markdown_` on `ReasoningResult`, `EvolutionResult`, `DatasetEvaluationReport`, `CostEstimate`, `ChainVisualizer` — typing `result` bare in a notebook cell renders a status banner + tables + Mermaid diagrams
    - **`mmar-carl[viz]` extra**: pulls in `matplotlib>=3.8.0` for PNG output paths

13. **Memory Systems** (`models/context.py`, `cow_memory.py`, `ltm.py`, `memory_schema.py`, `lazy_memory.py`)
    - **Three-layer memory**: namespaced short-term (`context.memory[namespace][key]`), session-level metadata, and optional long-term (LTM) backends
    - **COW (`cow_memory.py`)**: copy-on-write memory store replaces deep-copy isolation for parallel steps; bytes-saved profiling under `result.metadata`
    - **LTM (`ltm.py`)**: `LTMBase` abstract class + `InMemoryLTM` + `JsonFileLTM` backends for cross-session persistence
    - **`memory_schema`** (`ReasoningContext(memory_schema={...})`): validates `memory[namespace][key]` types at write-time via `validate_memory_write`; raises `MemorySchemaError`
    - **`LazyMemoryValue`**: lazy reference into memory; `unwrap_lazy(value)` materialises
    - **Pre-execution `$memory.*` reference validation**: warns when a step reads a key no prior step writes

14. **Cost Estimation** (`cost.py`)
    - **`CostEstimate`** / **`StepCostEstimate`**: dry-run projection of token / USD spend before executing a chain
    - Per-step `format_table()` text output + `_repr_markdown_` for notebook display
    - Pricing map: `{model: (input_per_1k_usd, output_per_1k_usd)}`

15. **Event Bus** (`event_bus.py`)
    - Intra-chain step communication: a step can `emit_event(name, data)`; another step can react via `event_dependencies=[...]` or `on_event` handlers
    - Enables fan-out patterns where many downstream steps watch the same event

### Data Flow

1. User creates steps (LLMStepDescription, ToolStepDescription, etc.)
2. User builds ReasoningChain with steps, search config, metrics, replan policy
3. User creates ReasoningContext with input data and LLM client
4. `chain.execute(context)` or `chain.execute_async(context)` invokes DAGExecutor
5. DAGExecutor:
   - Builds execution DAG from step dependencies
   - Creates execution batches (topological order)
   - For each batch, executes steps in parallel using asyncio
   - Each step uses appropriate executor (get_executor by step type)
6. Executor (e.g., LLMStepExecutor):
   - Resolves dynamic references ($history, $memory) from context
   - Extracts relevant context using search strategy
   - Generates prompt from template
   - Calls LLM client
   - Stores result in history
   - Applies metrics if attached
7. DAGExecutor collects results and returns ReasoningResult

### Memory & State Management

- **History**: Flat list of step outputs (strings), indexed by execution order
- **Memory**: Namespaced dictionary storage (write/read/append/delete/list operations)
- **Metadata**: Arbitrary key-value state for custom extensions
- **Isolation**: During parallel execution, each step gets deep copy of memory; writes only visible to subsequent batches

### Multi-language Support

- `Language.RUSSIAN` and `Language.ENGLISH` enums
- Prompt templates automatically adjusted per language
- System prompts prefixed appropriately ("System Instructions:" in English, "Системные инструкции:" in Russian)

## Development Commands

### Build & Install
```bash
# Build the package
make build

# Build and install development wheel
make install
```

### Testing
Tests are organised into **topic submodules** under `tests/`: `agents`, `chain_lifecycle`, `evaluation`, `llm_inference`, `mcp`, `memory`, `orchestration`, `replan`, `tool_calling`. Each has a generated `README.md` (regenerated via `make docs-topic-index`).

```bash
# Run all hermetic tests (default — live tests are deselected via pytest `-m 'not live'`)
make test

# Run a single test file
uv run pytest tests/orchestration/test_loop_execution.py -v

# Run a specific test function
uv run pytest tests/evaluation/test_format_cost_by_model.py::TestPricing -v

# Run tests matching a pattern
uv run pytest tests/ -k "llm_step" -v

# Run live tests against a real LLM (requires OPENAI_API_KEY)
make test-live   # equivalent to `pytest -m live`
```

**Topic index regeneration**: `make docs-topic-index` rewrites every `tests/<topic>/README.md` and `examples/<topic>/README.md` from each module's docstring.

### Examples
```bash
# Run all examples (set OPENAI_API_KEY for LLM examples)
make examples

# Run specific example
PYTHONPATH=$(pwd) uv run python examples/basic_chain_example.py
PYTHONPATH=$(pwd) uv run python examples/tool_steps_example.py
PYTHONPATH=$(pwd) uv run python examples/replan_deterministic_example.py

# Run examples requiring API key (set OPENAI_API_KEY first)
make example-openrouter
make example-structured
make example-council
```

### Code Quality
```bash
# Lint with ruff (configuration in pyproject.toml, line-length=120)
uv run ruff check src/

# Format with ruff
uv run ruff format src/

# Type checking (mypy configured in pyproject.toml)
uv run mypy src/
```

### Version Management
```bash
# Show current version
make version

# Increment patch version in pyproject.toml
make v++

# Check published version on PyPI
make pypi
```

### Notebook & Docs Tooling
```bash
# Rebuild notebooks/02_visualizations_demo.ipynb from its Python source
make notebook-build

# Record cassettes against OpenRouter so the notebook can run offline (requires API key)
make notebook-record

# Execute the demo notebook in cassette mode (no API calls)
make notebook-smoke

# Re-execute and save outputs (committed rendered Mermaid blocks)
make notebook-smoke-save

# Regenerate notebooks/README.md index from each notebook's metadata.demo_info
make notebooks-readme

# Regenerate per-topic README.md under tests/ and examples/
make docs-topic-index
```

## File Structure

```
src/mmar_carl/
├── __init__.py                 # Main API exports
├── chain.py                    # ReasoningChain + ChainBuilder + reflection + stream_async
├── chain_evolution.py          # ChainEvolver, ChainMutator, EvolutionResult, formatters
├── executor.py                 # DAGExecutor (core execution engine, loops, events)
├── step_executors.py           # All step executor implementations
├── llm.py                      # OpenAICompatibleClient, RetryPolicy, factory
├── record_replay.py            # RecordingLLMClient / PlayingLLMClient cassette wrappers
├── metrics.py                  # MetricBase, call_metric_async, MetricOutput
├── dataset_evaluator.py        # DatasetEvaluator for batch evaluation
├── eval_suite.py               # EvalSuite golden-output regression harness
├── replan.py                   # RE-PLAN checker implementations
├── cost.py                     # CostEstimate / StepCostEstimate (dry-run projections)
├── tracing.py                  # Langfuse integration
├── execution_trace.py          # ExecutionTrace, TraceEvent, TraceAggregator, to_html
├── visualizer.py               # ChainVisualizer fluent facade
├── streaming.py                # StreamingBuffer
├── cow_memory.py               # Copy-on-write memory store for parallel steps
├── lazy_memory.py              # LazyMemoryValue / unwrap_lazy
├── ltm.py                      # Long-term memory backends (InMemoryLTM, JsonFileLTM)
├── memory_schema.py            # memory write-time schema validation (MemorySchemaError)
├── event_bus.py                # Intra-chain event-driven step triggers
├── testing.py                  # ChainTestHarness
├── skill_loader.py             # SkillLoader + manifest parsing
├── skill_resolver.py           # URI-based skill resolver (GitHub, HTTPS, local, module)
├── skill_output_schema.py      # AgentSkill output schema validation
├── tool_definition.py          # ToolDefinition for tool composition
├── logging_utils.py            # Structured logging utilities
├── _optional_deps.py           # Optional dependency checking (`viz`, `vector_search`)
└── models/
    ├── __init__.py             # Model exports
    ├── base.py                 # SearchStrategy, SelfCritic classes
    ├── enums.py                # StepType, MemoryOperation, Language
    ├── steps.py                # Step description classes (typed + legacy)
    ├── config.py               # Step configuration classes + LoopConfig, ToolErrorRecovery
    ├── context.py              # ReasoningContext
    ├── results.py              # StepExecutionResult, ReasoningResult (+ formatters)
    ├── search.py               # Search strategies
    ├── prompts.py              # PromptTemplate
    ├── llm_client_base.py      # LLMClientBase, ChatMessage
    ├── agent_skill.py          # AgentSkill step config + manifest
    ├── replan.py               # RE-PLAN configuration models
    └── dataset.py              # Dataset abstractions + DatasetEvaluationReport (+ formatters)

tests/                          # Organised by topic — see make docs-topic-index
├── agents/                     # Supervisor, debate, handoff, skill, human input, parallel sampling
├── chain_lifecycle/            # Builder, serialization, reflection, cancellation, evolver, harness
├── evaluation/                 # Metrics, dataset evaluator, formatters, eval suite, evaluation step
├── llm_inference/              # Retry, streaming, structured output, council, record/replay, trace aggregator, live OpenRouter
├── mcp/                        # MCP tool / resource / transport tests
├── memory/                     # COW, LTM, schema, history truncation, parallel isolation
├── orchestration/              # DAG executor, conditional, loops, tool steps, transforms, caching, stream_async
├── replan/                     # Rule-based + LLM-based replan
└── tool_calling/               # Tool registration, error recovery, discovery, advanced cases

examples/                       # Same topic structure as tests
├── agents/                     # Supervisor, debate, council, human-in-the-loop, agent skills
├── evaluation/                 # Reflection, dataset eval, custom metrics, structured output
├── llm_inference/              # OpenRouter, execution modes, streaming, council voting
├── orchestration/              # Basic, parallel branches, conditional, loops, transforms
├── replan/                     # Deterministic / LLM / voting / checkpoint / budget
├── skills/                     # AgentSkill loading variants
└── tool_calling/               # Tool registration, argument mapping, recovery, discovery

scripts/                        # Validation helpers — exercise specific features against live OpenRouter
└── validate_*.py               # Each script names its target feature

notebooks/
├── 01_test.ipynb               # Original sandbox
├── 02_visualizations_demo.ipynb  # Rebuilt by scripts/build_visualizations_notebook.py
├── cassettes/                  # Committed JSONL replays so the notebook runs offline
└── README.md                   # Auto-generated index (make notebooks-readme)
```

## Key Concepts & Patterns

### Step Dependencies & Parallelization
- Steps can declare `dependencies=[step_num1, step_num2]`
- DAGExecutor builds batches: steps with no dependencies run first, then steps whose dependencies are met
- Parallel steps share shallow-copied tools but get deep-copied memory for isolation
- Memory writes from parallel steps only visible to next batch

### Context Extraction (RAG Pattern)
```python
LLMStepDescription(
    number=1,
    title="Analysis",
    step_context_queries=["revenue growth", "profit margins"],  # Searches outer_context
    # Each query extracts matching text, included in prompt
)
```

### AgentSkills Integration

**Skill identity — URI string support (v0.3+):**
```python
# GitHub tarball (recommended — fetches Anthropic's official skills)
AgentSkillStepConfig(skill="github://anthropics/skills/skills/pdf@main", ...)

# Local path
AgentSkillStepConfig(skill="/path/to/skill", ...)

# Python package
AgentSkillStepConfig(skill="module://my_pkg.skills.pdf", ...)

# Skill name (searched in ~/.agents/skills/, ./.claude/skills/, etc.)
AgentSkillStepConfig(skill="pdf", ...)
```

**Execution modes:**
| Mode | Behaviour |
|------|-----------|
| `LLM` | SKILL.md as system prompt; single LLM call (default) |
| `SCRIPT` | Bundled script directly; no LLM call |
| `HYBRID` | Script first; LLM fallback |
| `SUBAGENT` | Script collects data; LLM synthesises |
| `LLM_AGENT` | **Iterative tool-calling loop**: LLM calls `run_script/read_file/write_file/list_resources` until final answer; workspace-isolated |

**LLM_AGENT mode** (matches the AgentSkills spec's progressive-disclosure model):
```python
AgentSkillStepConfig(
    skill="github://anthropics/skills/skills/pdf@main",
    task="Extract text from {pdf_path} and summarise.",
    execution_mode=AgentSkillExecutionMode.LLM_AGENT,
    input_mapping={"pdf_path": "$memory.input.pdf_path"},
    llm_max_iterations=8,         # max tool-call rounds
    output_capture="both",        # "stdout" | "files" | "both"
    output_files_glob=["*.json"], # filter workspace/out files
    trust_policy="sha_pinned",    # "any" | "sha_pinned"
    skill_sha256="<hex>",         # SKILL.md digest for pinning
    extra_pip=["pdfplumber"],     # install before running scripts
)
```
Input files from `input_mapping` are staged in `/workspace/in/`.
Output files written by the LLM to `/workspace/out/` are collected in `result_data["output_files"]`.

**SkillResolver (v0.3+) — URI-based resolution:**
```python
from mmar_carl import resolve_skill, GithubResolver, SkillIntegrityError

# Simple top-level function
skill = resolve_skill("github://anthropics/skills/skills/pdf@main")
print(skill.local_root)   # ~/.cache/mmar_carl/skills/github/<key>/skills/pdf

# With SHA pinning
skill = resolve_skill(
    "github://anthropics/skills/skills/pdf@main",
    sha256="<hex-of-SKILL.md>",
    trust_policy="sha_pinned",
)

# Force re-download
resolver = GithubResolver()
skill = resolver.resolve("github://anthropics/skills/skills/pdf", force_refresh=True)
```
Supported schemes: `github://`, `local://`, `https://`, `module://`, plain paths.
Cache lives in `~/.cache/mmar_carl/skills/github/<key16>/`.

**Key facts:**
- `SkillResolverRegistry` dispatches by URI scheme; extend with custom resolvers
- `GithubResolver` downloads tarballs from `codeload.github.com` (no `git` required)
- `trust_policy="sha_pinned"` verifies SHA256 of the local SKILL.md after extraction
- `filter_security_terms=True` (default) strips password/encrypt/decrypt sections from LLM prompts
- `SkillManifest.get_allowed_tools()` / `get_allowed_tool_names()` parse the `allowed-tools` frontmatter
- `SkillLoader.catalog_all()` returns installed skills from SKILL.md dirs + `agent-skills` library
- `make skills` prints all installed skills
- Anthropic's `pdf` and `pptx` skills are **source-available, not open-source** — fetch at runtime, don't redistribute
- See `examples/agent_skill_example.py` for the full PDF→analysis→PPTX chain (`make example-agent-skill`)

**File structure additions (v0.3):**
```
src/mmar_carl/
├── skill_resolver.py   # URI-based resolver (GitHub, HTTPS, local, module)
```

### Dynamic Value Resolution
- `$history[-1]` or `$history[0]` — Previous step outputs
- `$memory.namespace.key` — Read from memory
- `$metadata.step_N` / `$metadata.<key>` — Read structured metadata
- `$outer_context` — Original input data
- `$ltm.key` — Long-term memory (when an `LTMBase` backend is wired into the context)
- `$event.<name>` — Most-recent payload from `event_bus` for that event name
- String literals: `'"literal_value"'` or `"'literal_value'"`
- Used in input_mapping, value_source, condition expressions

### Chain Generation From Natural Language
```python
from mmar_carl import ChainBuilder

chain = await ChainBuilder.from_description(
    task="Build a 2-step pipeline that outlines key arguments then condenses them.",
    llm_client=client,
    max_steps=4,
    max_retries=2,            # self-correction on validation errors
    available_tools=["fetch", "summarise"],
)
# Full provenance in chain.metadata: planner_prompt, planner_reply, planner_attempts
```

### Evolutionary Chain Search
```python
from mmar_carl import ChainEvolver
from mmar_carl.chain_evolution import ChainMutator, MutationKind

evolver = ChainEvolver(
    base_chain=chain,
    dataset=dataset,
    metric=[AccuracyMetric(), BrevityMetric()],          # multi-objective
    fitness_fn=lambda s: 0.7 * s["accuracy"] + 0.3 * s["brevity"],
    mutator=ChainMutator(
        temperature_pool=[0.1, 0.5],
        aim_suffix_pool=[" Be brief."],
        step_template_pool=[{"step_type": "llm", "title": "Verify", "aim": "..."}],
        allow_step_deletion=True,
        enabled_kinds=[MutationKind.PROMPT_REWRITE, MutationKind.INSERT_STEP, MutationKind.DELETE_STEP],
    ),
    population_size=4, generations=3, elitism=1,
    checkpoint_path="evolution.json",                     # atomic resume on crash
)
result = await evolver.evolve(context_factory=factory)
```

### Streaming Execution
```python
from mmar_carl.models.results import StepExecutionResult, ReasoningResult

async for item in chain.stream_async(ctx):
    if isinstance(item, StepExecutionResult):
        print(f"step {item.step_number} done")
    else:                                                 # terminal ReasoningResult
        print(f"chain success={item.success}")
```

### Record / Replay LLM Cassettes
```python
from mmar_carl import RecordingLLMClient, PlayingLLMClient

# Record once with real API
real = OpenAICompatibleClient(...)
rec = RecordingLLMClient(real, "cassette.jsonl", overwrite=True)
ctx = ReasoningContext(outer_context="...", api=rec)
await chain.execute_async(ctx)

# Replay deterministically with zero API calls
play = PlayingLLMClient("cassette.jsonl")
ctx2 = ReasoningContext(outer_context="...", api=play)
await chain.execute_async(ctx2)                            # ~22,000× faster
```

### Cost Estimation (Dry Run)
```python
estimate = chain.estimate_cost(
    pricing={"qwen/qwen3-8b": (0.00002, 0.00006)},
    default_output_tokens=512,
)
print(estimate.format_table())
# In Jupyter: just type `estimate` — _repr_markdown_ renders banner + table
```

### Loops, Events, Conditional Routing
```python
# Loop until condition met
LLMStepDescription(
    number=1, title="Refine",
    aim="Refine the answer.",
    loop_config=LoopConfig(condition="$history[-1].endswith('OK')", max_iterations=5),
)

# Event-driven trigger
LLMStepDescription(
    number=3, title="Notify",
    event_dependencies=["error_detected"],  # waits for emit_event("error_detected", ...)
)

# Conditional routing (true branch / else branch — non-winning step's downstream is skipped)
ConditionalStepDescription(
    number=2,
    config=ConditionalStepConfig(branches=[
        ConditionalBranch(condition="$history[-1] == 'yes'", next_step=3),
    ], default_step=4),
)
```

### LLM Client Detection
Pass any LLM API object to ReasoningContext; CARL auto-detects type and creates appropriate client:
- LLMHub (from mmar-llm)
- LLMHubAPI (from mmar-mapi)
- OpenAICompatibleClient (OpenRouter, Azure, local)
- Mock objects with `__getitem__` or `get_response` methods

### Adding Custom Steps
1. Create step class extending `StepDescriptionBase`
2. Implement `step_type` property
3. Create executor extending `StepExecutorBase`
4. Register with `register_executor(StepType.CUSTOM, CustomExecutor)`

### Configuration Overrides
- Per-step LLM model: `llm_config=LLMStepConfig(model="...")`
- Per-step retry: `retry_max=5`
- Per-step timeout: `timeout=30.0`
- Per-step metrics: `metrics=[...] `
- Per-step RE-PLAN: `replan_enabled=True/False`

## Common Development Tasks

### Adding a New Reasoning Step Type
1. Define config in `models/config.py` (e.g., `CustomStepConfig`)
2. Create step class in `models/steps.py` extending `StepDescriptionBase`
3. Implement executor in `step_executors.py` extending `StepExecutorBase`
4. Register in executor registry within executor module
5. Add tests in `tests/test_*.py`
6. Export from `__init__.py`

### Implementing a Custom Metric
1. Extend `MetricBase` in `metrics.py` or custom module
2. Implement `name` property and `compute_async()` method
3. Attach to step/chain: `LLMStepDescription(..., metrics=[MyMetric()])`
4. Access output via `isinstance(output, StepExecutionResult)` or `ReasoningResult`

### Adding LLM Provider Support
1. Extend `LLMClientBase` in `llm.py` or custom module
2. Implement async methods: `get_response()`, `get_response_stream()`
3. Register in auto-detection logic in `llm.py` (type checking)
4. Test with existing examples using custom API object

### Debugging Execution
- Enable logging: `set_log_level(logging.DEBUG)` before execution
- Check `ReasoningResult.get_failed_steps()` for errors with tracebacks
- Use callbacks: `on_step_start`, `on_step_complete` for monitoring (preserved across `stream_async`)
- Inspect `result.trace` — full `ExecutionTrace`; persist via `trace.to_json()`, diff via `trace.diff(other)`, animate via `trace.to_html("playback.html")`
- Use `TraceAggregator(traces)` to spot tail-latency outliers across N runs
- Enable Langfuse: Set LANGFUSE_PUBLIC_KEY for tracing dashboard
- Print history: `context.get_current_history()` or indexing `context.history[i]`

### Visualisation & Reporting Quick Reference
- `result.format_token_pie()`, `format_prompt_completion_breakdown()`, `format_profiling_table()`, `format_cost_by_model(pricing={...})`
- `chain.to_mermaid()`, `to_mermaid_critical_path(result)`, `to_mermaid_heatmap(result, metric=...)`
- `trace.format_gantt()`, `trace.to_html(path)`
- `report.format_failure_heatmap()`, `format_step_metric_heatmap(name)`, `format_score_distribution()`, `format_latency_histogram()`, `format_cost_trend(pricing=..., default_model=...)`
- `evolution_result.format_score_evolution()`, `format_pareto()`, `format_spend_vs_quality()`, `format_mutation_effectiveness()`, `to_lineage_mermaid()`
- `ChainVisualizer(result, chain=chain).token_pie().gantt().heatmap().print()` — fluent composition
- All result types have `_repr_markdown_` so they render as banners + tables + Mermaid in Jupyter without a `print()`

## Testing Best Practices

- Use mocked LLM clients (MagicMock with AsyncMock) in unit tests
- Tests automatically marked async if they use `await` (pytest-asyncio auto mode)
- Optional dependency tests: Use `check_vector_search_available()` to skip when deps missing
- See `tests/test_mmar_carl.py` for patterns with mocked APIs

## Release & Publishing

- Version in `pyproject.toml` (format: MAJOR.MINOR.PATCH)
- Bump with `make v++`
- Build with `make build`
- PyPI check: `make pypi`
- Automated PyPI publishing configured elsewhere

## Important Notes

- **Backward Compatibility**: Legacy `StepDescription` class still supported; prefer typed classes for new code
- **Memory Semantics**: Parallel-step isolation now uses copy-on-write (`cow_memory.py`); writes only visible to subsequent batches. Bytes-saved profiling under `result.metadata`
- **Experimental Features**: MCP steps are experimental; prefer registered Python tools for production
- **Async Safety**: All tools in parallel steps must be stateless; thread safety not guaranteed
- **Vector Search**: Optional FAISS-based semantic search; substring search always available
- **History Overflow**: Use `max_history_entries` and `history_truncation_strategy` (recency / token-budget) in long chains to prevent context bloat
- **Retries**: Don't blindly retry — `OpenAICompatibleClient` honours `RetryPolicy.retry_on_status`; 401/403/404/422 fail fast by default
- **Live Tests**: `@pytest.mark.live` tests are deselected by default. Use `make test-live` (or `pytest -m live`) when an `OPENAI_API_KEY` is set
- **Mermaid Labels**: Mermaid renders `<br/>` (not `\n`) as a line break inside node labels — every CARL `to_mermaid_*` method uses `<br/>`
- **Provenance**: `ChainBuilder.from_description` writes `planner_prompt` / `planner_reply` / `planner_attempts` into `chain.metadata` for offline failure diagnosis
- **Cassettes**: `RecordingLLMClient` / `PlayingLLMClient` enable deterministic notebook & test replay with zero API calls
