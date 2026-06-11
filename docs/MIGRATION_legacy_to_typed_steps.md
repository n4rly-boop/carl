# Migration guide: legacy `StepDescription` → typed step classes

CARL's original API used a single `StepDescription` class for every step
type, switching on a `step_type` enum. That class is now **deprecated** (it
emits `DeprecationWarning` on every instantiation) but still works — your
existing code keeps running. New code should use the typed step classes,
which are validated at construction time and give better IDE / type-checker
support.

This guide shows the equivalent typed-class form for every legacy pattern.
The chain definition, executor, and serialization layers all accept the
typed classes natively; no migration is needed beyond updating the call
sites.

---

## Quick reference

| Legacy `StepDescription` shape | Typed class |
|--------------------------------|-------------|
| `step_type=StepType.LLM` (default) | `LLMStepDescription` |
| `step_type=StepType.TOOL, step_config=ToolStepConfig(...)` | `ToolStepDescription(config=ToolStepConfig(...))` |
| `step_type=StepType.MCP, step_config=MCPStepConfig(...)` | `MCPStepDescription(config=MCPStepConfig(...))` |
| `step_type=StepType.MEMORY, step_config=MemoryStepConfig(...)` | `MemoryStepDescription(config=MemoryStepConfig(...))` |
| `step_type=StepType.TRANSFORM, step_config=TransformStepConfig(...)` | `TransformStepDescription(config=TransformStepConfig(...))` |
| `step_type=StepType.CONDITIONAL, step_config=ConditionalStepConfig(...)` | `ConditionalStepDescription(config=ConditionalStepConfig(...))` |
| `step_type=StepType.STRUCTURED_OUTPUT, step_config=StructuredOutputStepConfig(...)` | `StructuredOutputStepDescription(config=StructuredOutputStepConfig(...))` |

**Field-name change**: the legacy `step_config` field is named `config` on
every typed class. Configuration objects themselves (`ToolStepConfig`,
`MemoryStepConfig`, …) are unchanged.

Step types added after the typed-class refactor — `AGENT_SKILL`,
`EVALUATION`, `AGENT_HANDOFF`, `PARALLEL_SAMPLING`, `TOOL_DISCOVERY`,
`HUMAN_INPUT`, `SUPERVISOR`, `DEBATE` — are typed-class only; the legacy
`StepDescription` cannot express them.

---

## Before / after by step type

### LLM step

```python
# Legacy (deprecated)
StepDescription(
    number=1,
    title="Analyse claims",
    step_type=StepType.LLM,        # default — could be omitted
    aim="Extract every factual claim from the input.",
    reasoning_questions="What are the verifiable facts?",
    stage_action="enumerate",
    example_reasoning="...",
    llm_config=LLMStepConfig(model="gpt-4o", temperature=0.3),
)

# Typed (recommended)
LLMStepDescription(
    number=1,
    title="Analyse claims",
    aim="Extract every factual claim from the input.",
    reasoning_questions="What are the verifiable facts?",
    stage_action="enumerate",
    example_reasoning="...",
    llm_config=LLMStepConfig(model="gpt-4o", temperature=0.3),
)
```

Just drop `step_type=StepType.LLM` (it was the default) and rename the
class.

### Tool step

```python
# Legacy
StepDescription(
    number=2,
    title="Fetch data",
    step_type=StepType.TOOL,
    step_config=ToolStepConfig(
        tool_name="web_search",
        input_mapping={"query": "$memory.input.q"},
    ),
    dependencies=[1],
)

# Typed
ToolStepDescription(
    number=2,
    title="Fetch data",
    config=ToolStepConfig(                 # rename: step_config → config
        tool_name="web_search",
        input_mapping={"query": "$memory.input.q"},
    ),
    dependencies=[1],
)
```

### Memory step

```python
# Legacy
StepDescription(
    number=3,
    title="Store result",
    step_type=StepType.MEMORY,
    step_config=MemoryStepConfig(
        operation=MemoryOperation.WRITE,
        memory_key="summary",
        value_source="$history[-1]",
        namespace="output",
    ),
)

# Typed
MemoryStepDescription(
    number=3,
    title="Store result",
    config=MemoryStepConfig(
        operation=MemoryOperation.WRITE,
        memory_key="summary",
        value_source="$history[-1]",
        namespace="output",
    ),
)
```

### Transform step

```python
# Legacy
StepDescription(
    number=4,
    title="Extract emails",
    step_type=StepType.TRANSFORM,
    step_config=TransformStepConfig(
        transform_type="extract",
        expression=r"[\w.+-]+@[\w-]+\.[\w.-]+",
        input_key="$history[-1]",
    ),
)

# Typed
TransformStepDescription(
    number=4,
    title="Extract emails",
    config=TransformStepConfig(
        transform_type="extract",
        expression=r"[\w.+-]+@[\w-]+\.[\w.-]+",
        input_key="$history[-1]",
    ),
)
```

### Conditional step

```python
# Legacy
StepDescription(
    number=5,
    title="Route by sentiment",
    step_type=StepType.CONDITIONAL,
    step_config=ConditionalStepConfig(
        branches=[
            ConditionalBranch(condition="contains:positive", next_step=6),
            ConditionalBranch(condition="contains:negative", next_step=7),
        ],
        default_step=8,
        condition_context_key="$history[-1]",
    ),
)

# Typed
ConditionalStepDescription(
    number=5,
    title="Route by sentiment",
    config=ConditionalStepConfig(
        branches=[
            ConditionalBranch(condition="contains:positive", next_step=6),
            ConditionalBranch(condition="contains:negative", next_step=7),
        ],
        default_step=8,
        condition_context_key="$history[-1]",
    ),
)
```

A faster builder-level helper is also available — see `ChainBuilder.add_if_else`
and `ChainBuilder.add_switch` in `chain.py`.

### MCP step

```python
# Legacy
StepDescription(
    number=6,
    title="Call MCP tool",
    step_type=StepType.MCP,
    step_config=MCPStepConfig(
        server=MCPServerConfig(server_name="local", transport="stdio", command="..."),
        tool_name="search",
        argument_mapping={"query": "$outer_context"},
    ),
)

# Typed
MCPStepDescription(
    number=6,
    title="Call MCP tool",
    config=MCPStepConfig(
        server=MCPServerConfig(server_name="local", transport="stdio", command="..."),
        tool_name="search",
        argument_mapping={"query": "$outer_context"},
    ),
)
```

### Structured output step

```python
# Legacy
StepDescription(
    number=7,
    title="Extract structured data",
    step_type=StepType.STRUCTURED_OUTPUT,
    step_config=StructuredOutputStepConfig(
        output_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        input_source="$history[-1]",
    ),
)

# Typed
StructuredOutputStepDescription(
    number=7,
    title="Extract structured data",
    config=StructuredOutputStepConfig(
        output_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        input_source="$history[-1]",
    ),
)
```

---

## Mechanical refactor (sed-style)

For most codebases the migration is mechanical:

```bash
# Quick search to find every legacy site
grep -rn 'StepDescription(' --include='*.py' your_project/
```

For each match:

1. Identify `step_type=StepType.XXX` — drop it and switch the class name to `XXXStepDescription`.
2. Rename `step_config=` → `config=` (when the config field is present).
3. For LLM steps with no `step_type` argument, just rename `StepDescription` → `LLMStepDescription`.

If a step's config is a literal `None` (i.e. no `step_config=…`), it is an
LLM step — rename to `LLMStepDescription` and keep going.

---

## Cross-cutting differences

### Validation timing

Both forms validate at construction time, but the typed classes' errors are
narrower and clearer. For example, building a `MemoryStepDescription` with
a `ToolStepConfig` fails immediately with a pydantic discriminator error,
while the legacy form fails with a generic
`"MEMORY steps require MemoryStepConfig"` message after the model validator
fires.

### Per-step LLM config

The `llm_config: Optional[LLMStepConfig]` field is **only** on
`LLMStepDescription` and `StructuredOutputStepDescription` (and a few other
LLM-calling step types like `EvaluationStepDescription`,
`SupervisorStepDescription`, etc.). The legacy class accepts `llm_config`
on every step regardless of `step_type` but silently ignores it for non-LLM
steps. The typed classes make this an attribute error, which is the more
honest behaviour.

### Retry / timeout

`retry_max` and `timeout` are present on `StepDescriptionBase` (i.e. on
every typed class). The migration is field-for-field; no change.

### Metrics

`metrics: list[MetricBase] = []` is on `StepDescriptionBase` and excluded
from JSON serialization (same as legacy). No change.

### `dependencies` / `checkpoint` / `checkpoint_name` / `replan_enabled`

All on `StepDescriptionBase`. No change.

---

## Chain construction is unchanged

`ReasoningChain` accepts both legacy and typed steps in the same list, so
you can migrate incrementally:

```python
chain = ReasoningChain(
    steps=[
        LLMStepDescription(number=1, title="...", aim="..."),  # typed (new)
        StepDescription(                                       # legacy (old, still works)
            number=2, title="...", step_type=StepType.TOOL,
            step_config=ToolStepConfig(...),
        ),
    ],
)
```

The `DeprecationWarning` from the legacy entry fires once per construction;
filter it via standard `warnings.filterwarnings("ignore", DeprecationWarning, …)`
if it gets noisy mid-migration.

---

## Serialization

`chain.to_dict()` round-trips both forms. The serialized output always
includes a `step_type` discriminator, so a legacy-serialised chain loaded
with `ReasoningChain.from_dict()` is automatically reconstructed using
typed classes when `step_type` matches a typed-class type and the
discriminated-union machinery picks the right class. No data loss.

---

## Migration checklist

- [ ] Replace every `StepDescription(step_type=StepType.X, …)` with `XStepDescription(…)`.
- [ ] Rename `step_config=` → `config=` on typed classes.
- [ ] Drop `step_type=` since it's encoded in the class.
- [ ] Move `llm_config=` off non-LLM step calls (it was silently dropped before).
- [ ] Run your test suite. The default `DeprecationWarning` filter (`pytest`'s
      `-W` flag, your `pytest.ini`, etc.) should now be silent.
- [ ] Optional: remove the `# type: ignore` pragmas you might have on
      `step_config` reads — typed classes expose `step_config` as a
      property that already returns the correctly-typed config.
