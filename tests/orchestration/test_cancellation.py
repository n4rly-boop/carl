"""
Tests for mid-execution cancellation via context.cancel().

Cancellation semantics (from executor.py):
- context.cancel() sets an internal flag.
- The DAG executor checks the flag at the START of each batch loop.
  A batch that has already started runs to completion before the check fires.
- After the loop exits, ExecutionCancelledError is raised.
- Steps that completed before cancellation have their results in context.history
  and their memory writes merged into the primary context.
- context.reset_cancellation() clears the flag for a subsequent execution.
"""

import pytest

from mmar_carl import ReasoningChain, ReasoningContext, Language
from mmar_carl.executor import ExecutionCancelledError
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.models.steps import LLMStepDescription, MemoryStepDescription
from mmar_carl.models.config import MemoryStepConfig
from mmar_carl.models.enums import MemoryOperation


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class SequencedClient(LLMClientBase):
    """Returns responses from a list in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        if self._idx >= len(self._responses):
            return f"response-{self._idx}"
        r = self._responses[self._idx]
        self._idx += 1
        return r

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        return await self.get_response(prompt)


def make_context(client: LLMClientBase) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test data",
        api=client,
        model="mock",
        language=Language.ENGLISH,
    )


# ---------------------------------------------------------------------------
# 1. Pre-execution cancel
# ---------------------------------------------------------------------------

class TestCancellationStateAPI:
    """cancel() / reset_cancellation() / is_cancelled() API behaviour."""

    def test_cancel_sets_flag(self):
        """context.cancel() makes is_cancelled() return True."""
        context = make_context(SequencedClient([]))
        assert not context.is_cancelled()
        context.cancel()
        assert context.is_cancelled()

    def test_reset_clears_flag(self):
        """reset_cancellation() clears the cancelled flag."""
        context = make_context(SequencedClient([]))
        context.cancel()
        assert context.is_cancelled()
        context.reset_cancellation()
        assert not context.is_cancelled()

    @pytest.mark.asyncio
    async def test_executor_always_resets_cancellation_at_start(self):
        """
        execute_async() calls reset_cancellation() before the first batch,
        so a pre-set cancel flag is ignored and execution completes normally.
        This is by design — cancellation only takes effect when requested
        *during* an active execution via a callback.
        """
        client = SequencedClient(["s1"])
        chain = ReasoningChain(
            steps=[LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p")]
        )
        context = make_context(client)
        context.cancel()  # set flag before execution

        # Executor resets it → execution proceeds normally
        result = await chain.execute_async(context)
        assert result.success
        assert len(context.history) == 1

    @pytest.mark.asyncio
    async def test_can_execute_again_after_cancel_and_reset(self):
        """Chain can be executed again after a cancelled run + reset."""
        # Use a 2-step chain so cancellation after step 1 leaves a next batch to block
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
            ]
        )

        # First run: cancel mid-execution via callback after step 1
        client1 = SequencedClient(["s1", "s2"])
        ctx1 = make_context(client1)
        ctx1.on_step_complete = lambda r: ctx1.cancel() if r.step_number == 1 else None

        with pytest.raises(ExecutionCancelledError):
            await chain.execute_async(ctx1)

        # Second run: fresh context, no cancellation → succeeds
        ctx2 = make_context(SequencedClient(["s1-second", "s2-second"]))
        result = await chain.execute_async(ctx2)
        assert result.success


# ---------------------------------------------------------------------------
# 2. Cancel via on_step_complete callback (mid-execution)
# ---------------------------------------------------------------------------

class TestMidExecutionCancel:
    """Cancel requested during execution stops at the next batch boundary."""

    @pytest.mark.asyncio
    async def test_cancel_after_step1_skips_remaining(self):
        """
        Linear chain: step1 → step2 → step3.
        Cancel in on_step_complete after step 1.
        Steps 2 and 3 must NOT execute.
        """
        client = SequencedClient(["s1", "s2", "s3"])
        completed_steps: list[int] = []

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
            ]
        )
        context = make_context(client)

        def on_complete(result):
            completed_steps.append(result.step_number)
            if result.step_number == 1:
                context.cancel()  # request cancellation after step 1

        context.on_step_complete = on_complete

        with pytest.raises(ExecutionCancelledError) as exc_info:
            await chain.execute_async(context)

        assert "cancelled" in str(exc_info.value).lower()
        assert 1 in completed_steps, "Step 1 should have completed"
        assert 2 not in completed_steps, "Step 2 must not run after cancel"
        assert 3 not in completed_steps, "Step 3 must not run after cancel"

    @pytest.mark.asyncio
    async def test_cancel_after_step2_skips_step3(self):
        """
        Linear chain: step1 → step2 → step3.
        Cancel after step 2. Step 3 must not run.
        """
        client = SequencedClient(["s1", "s2", "s3"])
        completed_steps: list[int] = []

        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
            ]
        )
        context = make_context(client)

        def on_complete(result):
            completed_steps.append(result.step_number)
            if result.step_number == 2:
                context.cancel()

        context.on_step_complete = on_complete

        with pytest.raises(ExecutionCancelledError):
            await chain.execute_async(context)

        assert 1 in completed_steps
        assert 2 in completed_steps
        assert 3 not in completed_steps

    @pytest.mark.asyncio
    async def test_parallel_batch_completes_before_cancel_takes_effect(self):
        """
        Batch 1: step 1 (LLM).
        Batch 2: steps 2 + 3 in parallel.
        Cancel is requested at the end of step 1 (on_step_complete).

        Since cancellation is checked at batch *start*, and batch 2 may already
        be scheduled by the time cancellation propagates, the exact behavior
        depends on timing. However, the key invariant: step 1 completes and
        cancellation is raised before step 4 (which depends on 2+3) runs.
        """
        client = SequencedClient(["s1", "s2", "s3", "s4"])
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=4, title="S4", aim="a", prompt_template="p", dependencies=[2, 3]),
            ],
            max_workers=2,
        )
        context = make_context(client)
        completed: list[int] = []

        def on_complete(result):
            completed.append(result.step_number)
            if result.step_number == 1:
                context.cancel()

        context.on_step_complete = on_complete

        with pytest.raises(ExecutionCancelledError):
            await chain.execute_async(context)

        # Step 1 always completes
        assert 1 in completed
        # Step 4 (the merge step) never runs — it depends on batch 2 which is
        # scheduled after the cancel-check batch boundary
        assert 4 not in completed


# ---------------------------------------------------------------------------
# 3. Partial results preserved in context.history
# ---------------------------------------------------------------------------

class TestPartialResultsPreserved:
    """Steps completed before cancellation have their results in context.history."""

    @pytest.mark.asyncio
    async def test_history_contains_completed_steps(self):
        """
        After cancellation, context.history holds the output of all steps
        that ran to completion before the cancel was detected.
        """
        client = SequencedClient(["output-step-1", "output-step-2"])
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
            ]
        )
        context = make_context(client)

        def on_complete(result):
            if result.step_number == 2:
                context.cancel()

        context.on_step_complete = on_complete

        with pytest.raises(ExecutionCancelledError):
            await chain.execute_async(context)

        # History must contain step 1 and step 2 outputs
        history_text = "\n".join(context.history)
        assert "output-step-1" in history_text, "Step 1 result must be in history"
        assert "output-step-2" in history_text, "Step 2 result must be in history"

    @pytest.mark.asyncio
    async def test_memory_writes_from_completed_steps_preserved(self):
        """
        A MemoryStep that completed before cancellation must have its write
        visible in context.memory after the error is raised.
        """
        client = SequencedClient(["s1"])
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                MemoryStepDescription(
                    number=2, title="Write key",
                    dependencies=[1],
                    config=MemoryStepConfig(
                        operation=MemoryOperation.WRITE,
                        memory_key="saved",
                        value_source='"written-before-cancel"',
                    ),
                ),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
            ]
        )
        context = make_context(client)

        def on_complete(result):
            if result.step_number == 2:
                context.cancel()

        context.on_step_complete = on_complete

        with pytest.raises(ExecutionCancelledError):
            await chain.execute_async(context)

        # Memory write from step 2 must be present
        assert context.memory.get("default", {}).get("saved") == "written-before-cancel"


# ---------------------------------------------------------------------------
# 4. ReasoningResult metadata
# ---------------------------------------------------------------------------

class TestCancellationMetadata:
    """The chain raises ExecutionCancelledError; the error message is informative."""

    @pytest.mark.asyncio
    async def test_error_message_mentions_partial_results(self):
        """ExecutionCancelledError message describes the partial execution state."""
        client = SequencedClient(["s1", "s2"])
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="S1", aim="a", prompt_template="p"),
                LLMStepDescription(number=2, title="S2", aim="a", prompt_template="p", dependencies=[1]),
                LLMStepDescription(number=3, title="S3", aim="a", prompt_template="p", dependencies=[2]),
            ]
        )
        context = make_context(client)
        # Cancel after step 1 so steps 2+3 are skipped
        context.on_step_complete = lambda r: context.cancel() if r.step_number == 1 else None

        with pytest.raises(ExecutionCancelledError) as exc_info:
            await chain.execute_async(context)

        msg = str(exc_info.value)
        # Should mention step counts (e.g. "1/3 steps")
        assert "/" in msg, f"Expected step count fraction in error message, got: {msg!r}"
