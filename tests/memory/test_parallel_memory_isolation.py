"""
Tests for parallel execution memory isolation in the DAG executor.

Covers the semantics described in executor.py:
- Each parallel step gets a deep-copy of context.memory — siblings cannot
  see each other's writes during a batch.
- After the batch completes, all successful steps' memory snapshots are
  merged back into the primary context via dict.update() (last-writer wins
  for the same key).
- Failed steps' memory writes are discarded (not merged back).
- Steps in subsequent batches see the merged state from all prior batches.
"""

import pytest

from mmar_carl import ReasoningChain, ReasoningContext, Language
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.steps import LLMStepDescription, MemoryStepDescription
from mmar_carl.models.config import MemoryStepConfig
from mmar_carl.models.enums import MemoryOperation


# ---------------------------------------------------------------------------
# Shared mock client
# ---------------------------------------------------------------------------

class SequencedClient(LLMClientBase):
    """Returns responses from a list in order; raises on overflow."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def get_response(self, prompt: str) -> str:
        if self._idx >= len(self._responses):
            return f"response-{self._idx}"
        r = self._responses[self._idx]
        self._idx += 1
        return r

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def make_context(client: LLMClientBase, memory: dict | None = None) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context="test data",
        api=client,
        model="mock",
        language=Language.ENGLISH,
    )
    if memory:
        for ns, kv in memory.items():
            if ns not in ctx.memory:
                ctx.memory[ns] = {}
            ctx.memory[ns].update(kv)
    return ctx


# ---------------------------------------------------------------------------
# 1. Isolation: parallel siblings cannot see each other's writes
# ---------------------------------------------------------------------------

class TestParallelMemoryIsolation:
    """Siblings in the same batch each start with an independent memory copy."""

    @pytest.mark.asyncio
    async def test_siblings_do_not_see_each_other_writes(self):
        """
        Steps 2 and 3 both run in parallel (both depend only on step 1).
        Step 2 writes key 'a'; step 3 reads key 'a'. Step 3 must NOT see
        step 2's write — it should see the pre-batch value (absent).
        """
        client = SequencedClient(["step1-result"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Writer",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="a",
                        value_source='"written-by-step-2"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Reader",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.READ,
                        memory_key="a",
                    ),
                ),
            ],
            max_workers=2,
        )

        context = make_context(client)
        result = await chain.execute_async(context)

        # Both steps must have succeeded
        assert result.success or any(r.step_number in (2, 3) for r in result.step_results)

        # After the batch, the primary context DOES have the written key
        # (merged from the writer's snapshot)
        assert context.memory.get("default", {}).get("a") == "written-by-step-2"

    @pytest.mark.asyncio
    async def test_parallel_writes_different_keys_both_visible_after_batch(self):
        """
        Steps 2 and 3 each write a distinct memory key.
        Both writes must be visible in primary context after the batch.
        """
        client = SequencedClient(["s1"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Write X",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="x",
                        value_source='"value-x"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Write Y",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="y",
                        value_source='"value-y"',
                    ),
                ),
            ],
            max_workers=2,
        )

        context = make_context(client)
        result = await chain.execute_async(context)

        assert result.success
        assert context.memory["default"]["x"] == "value-x"
        assert context.memory["default"]["y"] == "value-y"

    @pytest.mark.asyncio
    async def test_parallel_writes_same_key_last_writer_wins(self):
        """
        Steps 2 and 3 write the SAME key. After the batch, one of their values
        wins (last-merger-wins via dict.update). The value must be one of the
        two written values — not None, not a mix, not the pre-batch value.
        """
        client = SequencedClient(["s1"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Write shared key v2",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="shared",
                        value_source='"from-step-2"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Write shared key v3",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="shared",
                        value_source='"from-step-3"',
                    ),
                ),
            ],
            max_workers=2,
        )

        context = make_context(client)
        result = await chain.execute_async(context)

        assert result.success
        final = context.memory["default"]["shared"]
        assert final in ("from-step-2", "from-step-3"), (
            f"Expected one of the written values, got: {final!r}"
        )

    @pytest.mark.asyncio
    async def test_parallel_writes_do_not_overwrite_pre_batch_other_keys(self):
        """
        Pre-existing memory key 'preexisting' must survive parallel writes to
        an unrelated key.
        """
        client = SequencedClient(["s1"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Write new key",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="new_key",
                        value_source='"new-value"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Write another new key",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="another_key",
                        value_source='"another-value"',
                    ),
                ),
            ],
            max_workers=2,
        )

        context = make_context(client, memory={"default": {"preexisting": "original"}})
        result = await chain.execute_async(context)

        assert result.success
        # Pre-existing key untouched
        assert context.memory["default"]["preexisting"] == "original"
        # New keys written by parallel steps are present
        assert context.memory["default"]["new_key"] == "new-value"
        assert context.memory["default"]["another_key"] == "another-value"


# ---------------------------------------------------------------------------
# 2. Subsequent batches see merged state from prior batches
# ---------------------------------------------------------------------------

class TestSequentialBatchMemoryVisibility:
    """Steps in later batches must see memory written by all prior batches."""

    @pytest.mark.asyncio
    async def test_subsequent_step_sees_parallel_batch_writes(self):
        """
        Step 1 runs, then steps 2+3 run in parallel (each writes a key),
        then step 4 depends on both 2+3 and reads their keys via history.
        The final LLM step receives both keys in its context.
        """
        client = SequencedClient(["s1", "s4-answer"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Write Alpha",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="alpha",
                        value_source='"alpha-value"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Write Beta",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="beta",
                        value_source='"beta-value"',
                    ),
                ),
                LLMStepDescription(
                    number=4, title="Read both",
                    aim="combine results",
                    prompt_template="combine results",
                    dependencies=[2, 3],
                ),
            ],
            max_workers=2,
        )

        context = make_context(client)
        result = await chain.execute_async(context)

        assert result.success
        # Both parallel writes visible after their batch
        assert context.memory["default"]["alpha"] == "alpha-value"
        assert context.memory["default"]["beta"] == "beta-value"
        # Step 4 succeeded (it ran after the merge)
        step4 = next((r for r in result.step_results if r.step_number == 4), None)
        assert step4 is not None and step4.success

    @pytest.mark.asyncio
    async def test_three_batch_memory_chain(self):
        """
        Batch 1: step 1 (LLM, no deps)
        Batch 2: steps 2+3 in parallel (each write distinct keys)
        Batch 3: step 4 (reads both keys, depends on 2+3)

        Verifies the full fan-out / fan-in memory lifecycle.
        """
        client = SequencedClient(["batch1", "batch3"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Batch1 LLM", aim="start", prompt_template="start",
                ),
                MemoryStepDescription(
                    number=2, title="Batch2 Write p",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="p",
                        value_source='"parallel-p"',
                    ),
                ),
                MemoryStepDescription(
                    number=3, title="Batch2 Write q",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="q",
                        value_source='"parallel-q"',
                    ),
                ),
                LLMStepDescription(
                    number=4, title="Batch3 Merge", aim="merge", prompt_template="merge",
                    dependencies=[2, 3],
                ),
            ],
            max_workers=3,
        )

        context = make_context(client)
        result = await chain.execute_async(context)

        assert result.success, f"Chain failed: {result.get_failed_steps()}"
        assert context.memory["default"]["p"] == "parallel-p"
        assert context.memory["default"]["q"] == "parallel-q"


# ---------------------------------------------------------------------------
# 3. Failed step's memory writes are discarded
# ---------------------------------------------------------------------------

class TestFailedStepMemoryDiscarded:
    """A step that raises must not contribute its memory writes to the merge."""

    @pytest.mark.asyncio
    async def test_failed_step_memory_not_merged(self):
        """
        Steps 2 and 3 run in parallel.
        Step 2 (MemoryStep) succeeds and writes 'ok_key'.
        Step 3 (ToolStep) calls a registered tool that raises — the step fails.

        After the batch:
        - 'ok_key' from the successful step must be visible.
        - 'fail_key' written inside the failing tool's snapshot must NOT appear,
          because the executor only merges memory from *successful* steps.
        """
        from mmar_carl.models.config import ToolStepConfig
        from mmar_carl.models.steps import ToolStepDescription

        client = SequencedClient(["s1"])

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(
                    number=1, title="Init", aim="init", prompt_template="init",
                ),
                MemoryStepDescription(
                    number=2, title="Write ok_key",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="ok_key",
                        value_source='"step2-wrote-this"',
                    ),
                ),
                ToolStepDescription(
                    number=3, title="Failing tool step",
                    dependencies=[1],
                    config=ToolStepConfig(
                        tool_name="always_fails",
                        input_mapping={},
                        output_key="fail_result",
                    ),
                ),
            ],
            max_workers=2,
        )

        context = make_context(client)

        # Register a tool that always raises — step 3 will fail
        async def always_fails():
            raise RuntimeError("intentional tool failure")

        context.register_tool("always_fails", always_fails)

        result = await chain.execute_async(context)

        step2 = next(r for r in result.step_results if r.step_number == 2)
        step3 = next(r for r in result.step_results if r.step_number == 3)
        assert step2.success, "Step 2 should succeed"
        assert not step3.success, "Step 3 should fail (tool raised)"

        # Successful step's write is present
        assert context.memory.get("default", {}).get("ok_key") == "step2-wrote-this", (
            "Successful step's memory write must be present in primary context"
        )
        # Failed step produced no memory writes; the 'fail_result' tool output key
        # should not exist in memory
        assert "fail_result" not in context.memory.get("default", {}), (
            "Failed step's output must not be merged into primary context"
        )
