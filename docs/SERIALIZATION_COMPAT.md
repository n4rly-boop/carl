# Chain Serialization Compatibility Policy

CARE saves `ReasoningChain` instances to gigaevo-memory and may load
them years later — often against a newer release of CARL than the one
that wrote them. This document codifies the contract CARL keeps with
serialised chains and the rules every CARL change has to follow when
touching the chain dict shape.

If you're a CARL contributor changing the chain dict layout, **read section 3
first**.

## 1. Format version

Every chain dict carries:

- `format_version: int` — incremented when the wire shape changes in a
  way that older readers can't ignore. Current value:
  `ReasoningChain.FORMAT_VERSION = 1` (defined in `src/mmar_carl/chain.py`).
- `carl_version: str` — informational, the `mmar-carl` version that
  wrote the dict. Used for diagnostics only; readers never branch on it.

The version is stamped by `chain.to_dict()` and inspected by
`ReasoningChain.from_dict(data)` and `ReasoningChain.migrate(data)`.

## 2. Compatibility contract

CARL guarantees the following across minor releases:

| Direction      | Guarantee                                                                                                                                     |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Read older** | A newer CARL **always** reads chain dicts saved by an older CARL — the migration ladder fills in any missing fields with sensible defaults.   |
| **Read newer** | A newer wire format raises `ChainFormatNewerError(required_version, this_version)`. Callers (CARE) catch this and prompt the user to upgrade. |
| **Read same**  | Round-trip is lossless for every field documented in section 4. Runtime-only fields are excluded by design.                                     |

A chain saved at `format_version = N` MUST stay loadable on every
released CARL version that ships `FORMAT_VERSION ≥ N`.

## 3. Rules for CARL contributors

When you change the serialised shape, follow these rules:

### 3.1 Adding a field

Most field additions are **backward-compatible at the read level** and
do not require a `format_version` bump. The new field must:

- Have a Pydantic `default` (a literal or `default_factory`) so older
  dicts that lack the field load cleanly.
- Survive `to_dict()` → `from_dict()` round-trip — exercise via the
  parametrised `test_chain_roundtrip_all_step_types.py` (auto-runs
  against every `StepType`).

### 3.2 Removing or renaming a field

This is a **breaking** change at the read level. You MUST:

1. Bump `ReasoningChain.FORMAT_VERSION` by 1.
2. Add a new migration rung to `ReasoningChain.migrate(data)`:
   ```python
   if current < N:
       data["new_name"] = data.pop("old_name", default)
       data["format_version"] = N
   ```
   The rung's job is to rewrite an older dict into the current shape.
   Existing rungs MUST stay in place forever — newer CARL versions still
   read older chains by walking every rung in order.
3. Add a legacy fixture to `tests/fixtures/legacy_chains/` that exercises
   the migration. The parametrised
   `tests/chain_lifecycle/test_legacy_chain_fixtures.py` picks it up
   automatically and asserts a clean round-trip after migration.

### 3.3 Adding a new step type

A new step type is **backward-compatible** as long as older readers
that don't know the type can either skip it or fail loudly. Today
`_reconstruct_step_config` raises `ValueError("Unknown step type:
<x>")` for unknown types — that's the "fail loudly" path. If a chain
serialised with a brand-new step type lands on an older reader, the
error is local to that step's reconstruction.

The parametrised round-trip test in
`tests/chain_lifecycle/test_chain_roundtrip_all_step_types.py` includes
an exhaustiveness guard that fails if a new `StepType` value is added
without a fixture, so the round-trip path is enforced by CI.

### 3.4 Changing field semantics

Same as section 3.2 — bump `FORMAT_VERSION` and add a migration rung. A field
that keeps its name but changes its meaning (e.g. units, encoding,
defaults) is more dangerous than a rename because the reader can't
tell from the dict shape alone. Migrate the _value_ explicitly.

## 4. Round-trip surface

The following fields are documented and guaranteed to round-trip via
`ReasoningChain.to_dict()` / `from_dict()`:

### Chain-level

- `format_version` (int)
- `carl_version` (str, informational)
- `max_workers` (int / `"auto"`)
- `enable_progress` (bool)
- `metadata` (free-form dict — CARE namespaces its keys under
  `metadata["care"]`)
- `timeout` (Optional[float])
- `replan_policy` (full `ReplanPolicy.model_dump`)
- `search_config` (full `ContextSearchConfig.model_dump`)
- `default_llm_config` (full `LLMStepConfig.model_dump`)
- `trace_name`, `session_id` (optional strs)
- `steps` (list of step dicts — shape per step type)

### Step-level (every type)

- `number` (int)
- `title` (str)
- `dependencies` (list[int])
- `triggered_by` (list[str])
- `step_type` (str enum value: `"llm"`, `"tool"`, …)
- `checkpoint`, `checkpoint_name`, `replan_enabled`

### LLM step extras

- `aim`, `reasoning_questions`, `step_context_queries`,
  `stage_action`, `example_reasoning`
- `llm_config` (Optional[LLMStepConfig])
- `retry_max`, `timeout`

### Non-LLM step extras

- `step_config` (full Pydantic dump of the appropriate `*StepConfig`
  class — `ToolStepConfig`, `MCPStepConfig`, `MemoryStepConfig`, etc.)

## 5. Runtime-only fields (NOT round-tripped)

These fields are marked `exclude=True` on the Pydantic model and are
**not** serialised. Callers must rebuild them after `from_dict()`:

- `AgentHandoffStepDescription.sub_chain` — placeholder is a
  `ReasoningChain` with a single `__placeholder__` LLM step. CARE
  rebuilds the real sub-chain from a gigaevo-memory entity_id stored
  in chain metadata.
- `SupervisorStepDescription.agents` — placeholder is `{}`. CARE
  rebuilds the `{name: sub_chain}` mapping by resolving entity_id refs.
- `ParallelSamplingStepDescription.base_step` — placeholder is a
  minimal `LLMStepDescription` titled `__placeholder__`. CARE rebuilds
  the real base step from chain metadata.
- `metrics` — list of `MetricBase` instances. Callers re-attach.
- `cache` — `StepCache` instances. Callers re-attach.

This contract is asserted by the
`TestRuntimeOnlyPlaceholders` class in
`tests/chain_lifecycle/test_chain_roundtrip_all_step_types.py`.

## 6. Newer-format detection

When `from_dict()` encounters a serialised chain with
`format_version > FORMAT_VERSION`, it raises:

```python
class ChainFormatNewerError(Exception):
    required_version: int    # what the dict says
    this_version: int        # what this CARL supports
```

CARE's TUI catches this and surfaces a "upgrade mmar-carl from X to
≥Y" prompt. The exception is exported from the top-level
`mmar_carl.ChainFormatNewerError` so callers can `except` it without
a deep import.

Older formats are handled silently by the migration ladder in
`ReasoningChain.migrate()`; you only get the error when the writer is
newer than the reader.

## 7. CI guards

The following tests run on every PR and catch regressions in this
policy:

- `tests/chain_lifecycle/test_chain_roundtrip_all_step_types.py` —
  parametrised over every `StepType` (16 step types × 2 paths = 32
  cases plus 4 documentation tests = 36 tests). Each step type
  round-trips via `to_dict()` → `from_dict(..., use_typed_steps=True)`
  and asserts `model_dump(mode="json")` is identical.
- `tests/chain_lifecycle/test_legacy_chain_fixtures.py` —
  parametrised over every JSON file under
  `tests/fixtures/legacy_chains/`. Each fixture survives
  `migrate()` → `from_dict()` losslessly.
- Newer-format detection — `TestChainFormatNewerError` (4 cases) in
  the same test module.

## 8. Glossary

| Term               | Meaning                                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------- |
| **Wire shape**     | The exact dict produced by `chain.to_dict()`. The contract is on this — not on internal Python types.       |
| **Migration rung** | One `if current < N: …` block in `ReasoningChain.migrate(data)` that rewrites an older shape into the next. |
| **Round-trip**     | `from_dict(to_dict(chain))` ≡ `chain` for every documented field.                                           |
| **Runtime-only**   | A Pydantic field marked `exclude=True` — deliberately not serialised because it can't be JSON-encoded.      |
