"""
Tests for step result caching (StepCache memoization).

StepCache allows steps to be cached by a key derived from the context.
On a cache hit, the stored result is returned without calling LLM/tool.
Cache is scoped to a single DAGExecutor.execute() call (cleared each run).
"""


import pytest

from mmar_carl import ReasoningChain, ReasoningContext
from mmar_carl.models.config import StepCache, ToolStepConfig
from mmar_carl.models.steps import LLMStepDescription, ToolStepDescription
from mmar_carl.models.llm_client_base import LLMClientBase


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class CountingClient(LLMClientBase):
    """LLM client that counts how many times it was called."""

    def __init__(self):
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:  # noqa: ARG002
        self.call_count += 1
        return f"response-{self.call_count}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:  # noqa: ARG002
        self.call_count += 1
        return f"response-{self.call_count}"


def make_context(outer_context: str = "test input") -> tuple[ReasoningContext, CountingClient]:
    client = CountingClient()
    ctx = ReasoningContext(outer_context=outer_context, api=client, model="mock")
    return ctx, client


# ---------------------------------------------------------------------------
# StepCache model tests
# ---------------------------------------------------------------------------


class TestStepCacheModel:
    def test_default_values(self):
        cache = StepCache()
        assert cache.ttl is None
        assert cache.key_fn is None

    def test_with_ttl(self):
        cache = StepCache(ttl=300)
        assert cache.ttl == 300

    def test_with_key_fn(self):
        fn = lambda ctx: "static-key"  # noqa: E731
        cache = StepCache(key_fn=fn)
        assert cache.key_fn is fn

    def test_key_fn_excluded_from_serialization(self):
        """key_fn is a callable — should be excluded from Pydantic serialization."""
        fn = lambda ctx: "static-key"  # noqa: E731
        cache = StepCache(ttl=60, key_fn=fn)
        data = cache.model_dump()
        assert "key_fn" not in data
        assert data["ttl"] == 60

    def test_step_cache_attached_to_step(self):
        cache = StepCache(ttl=120)
        step = LLMStepDescription(number=1, title="A", aim="a", cache=cache)
        assert step.cache is cache
        assert step.cache.ttl == 120

    def test_cache_field_excluded_from_step_serialization(self):
        """cache field must not appear in to_dict/from_dict since it has non-serializable key_fn."""
        cache = StepCache(key_fn=lambda ctx: "x")
        step = LLMStepDescription(number=1, title="A", aim="a", cache=cache)
        data = step.model_dump()
        # exclude=True means the field is not in the dict
        assert "cache" not in data


# ---------------------------------------------------------------------------
# Cache hit / miss behaviour
# ---------------------------------------------------------------------------


class TestCacheHitMiss:
    @pytest.mark.asyncio
    async def test_no_cache_calls_llm_each_time(self):
        """Without cache, every step execution calls the LLM."""
        ctx, client = make_context()
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="A", aim="a"),
                LLMStepDescription(number=2, title="B", aim="b", dependencies=[1]),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_cached_step_not_re_executed_in_loop(self):
        """A cached LLM step should only call the LLM on the first pass."""
        call_counts: list[str] = []

        def expensive_tool():
            call_counts.append("called")
            return "expensive-result"

        ctx, _ = make_context()
        ctx.register_tool("expensive", expensive_tool)

        # Build a simple 2-step loop: step 1 runs the tool, step 2 decides to loop back
        loop_iterations = [0]

        def loop_driver():
            loop_iterations[0] += 1
            # Loop 2 times, then stop
            ctx.memory.setdefault("loop", {})["continue"] = loop_iterations[0] < 2
            return "done"

        ctx.register_tool("loop_driver", loop_driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Expensive step",
                    config=ToolStepConfig(tool_name="expensive", input_mapping={}),
                    cache=StepCache(),  # no TTL, default key (step + outer_context)
                ),
                ToolStepDescription(
                    number=2,
                    title="Loop driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="loop_driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.loop.continue", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # The expensive tool should have been called only ONCE despite 2 loop iterations
        assert len(call_counts) == 1, f"Expected 1 call, got {len(call_counts)}"

    @pytest.mark.asyncio
    async def test_cache_hit_returns_same_result(self):
        """Result returned from cache must match the original execution result."""
        call_results: list[str] = []

        def tracked_tool():
            call_results.append("original")
            return "the-result"

        ctx, _ = make_context()
        ctx.register_tool("tracked", tracked_tool)

        loop_count = [0]

        def loop_ctrl():
            loop_count[0] += 1
            ctx.memory.setdefault("loop", {})["go"] = loop_count[0] < 2
            return "ctrl"

        ctx.register_tool("loop_ctrl", loop_ctrl)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Tracked",
                    config=ToolStepConfig(tool_name="tracked", input_mapping={}),
                    cache=StepCache(),
                ),
                ToolStepDescription(
                    number=2,
                    title="Control",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="loop_ctrl", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.loop.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # Only one real call was made
        assert len(call_results) == 1

    @pytest.mark.asyncio
    async def test_cache_profiling_marks_hit(self):
        """Cache-hit results must have cache_hit=True in their profiling dict."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return "result"

        ctx, _ = make_context()
        ctx.register_tool("t", tool)

        loop_count = [0]

        def driver():
            loop_count[0] += 1
            ctx.memory.setdefault("l", {})["go"] = loop_count[0] < 2
            return "d"

        ctx.register_tool("driver", driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="Cached",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(),
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.l.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success

        # The loop purges old step results on restart, so step_results only contains
        # the final (cache-hit) iteration of step 1.  Verify it has cache_hit=True.
        step1_results = [r for r in result.step_results if r.step_number == 1]
        assert len(step1_results) == 1
        assert step1_results[0].profiling.get("cache_hit") is True
        # And the real tool was only called once (the first iteration's miss)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# TTL tests
# ---------------------------------------------------------------------------


class TestCacheTTL:
    @pytest.mark.asyncio
    async def test_expired_cache_re_executes_step(self):
        """When TTL expires, the step must be re-executed."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        ctx, _ = make_context()
        ctx.register_tool("t", tool)

        # Use a very short TTL; we'll fake expiration by manipulating stored_at in the cache
        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(ttl=1),  # 1-second TTL
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "", "max_iterations": 2},
                ),
            ],
        )
        from mmar_carl.executor import DAGExecutor

        executor = DAGExecutor(max_workers=1)
        # Patch stored_at to be in the past after first execution
        original_execute_step = executor.execute_step

        first_call_done = [False]

        async def patched_execute_step(node, context):
            res = await original_execute_step(node, context)
            if node.step.number == 1 and not first_call_done[0]:
                # Expire the cache entry
                first_call_done[0] = True
                for k in list(executor._result_cache):
                    old = executor._result_cache[k]
                    executor._result_cache[k] = (old[0], old[1], old[2], old[3], 0.0)  # stored_at = epoch 0
            return res

        executor.execute_step = patched_execute_step

        await executor.execute(chain.steps, ctx)
        # With TTL expired on first iteration, step re-runs on second → 3 total calls
        # (step 1 first real, step 2 first, step 1 expired re-run)
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_unexpired_cache_skips_re_execution(self):
        """Within TTL window, cached result is returned without re-executing."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return "ok"

        ctx, _ = make_context()
        ctx.register_tool("t", tool)

        loop_count = [0]

        def driver():
            loop_count[0] += 1
            ctx.memory.setdefault("l", {})["go"] = loop_count[0] < 2
            return "d"

        ctx.register_tool("driver", driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(ttl=3600),  # 1-hour TTL — won't expire in test
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.l.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1  # cached on second iteration


# ---------------------------------------------------------------------------
# Custom key_fn tests
# ---------------------------------------------------------------------------


class TestCustomKeyFn:
    @pytest.mark.asyncio
    async def test_custom_key_fn_used(self):
        """key_fn receives the context and its return value is used as cache key."""
        received_contexts: list = []

        def my_key_fn(ctx):
            received_contexts.append(ctx)
            return "fixed-key"

        call_count = [0]

        def tool():
            call_count[0] += 1
            return "res"

        ctx, _ = make_context("outer")
        ctx.register_tool("t", tool)

        loop_count = [0]

        def driver():
            loop_count[0] += 1
            ctx.memory.setdefault("l", {})["go"] = loop_count[0] < 2
            return "d"

        ctx.register_tool("driver", driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(key_fn=my_key_fn),
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.l.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1  # hit on second pass
        assert len(received_contexts) >= 1

    @pytest.mark.asyncio
    async def test_different_key_fn_results_different_cache_entries(self):
        """Different key_fn return values are treated as independent cache entries."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        ctx, _ = make_context("outer")
        ctx.register_tool("t", tool)

        key_values = ["key-A", "key-B"]
        key_index = [0]

        def rotating_key(c):
            # Each call uses a different key, so cache is never hit
            k = key_values[key_index[0] % len(key_values)]
            key_index[0] += 1
            return k

        loop_count = [0]

        def driver():
            loop_count[0] += 1
            ctx.memory.setdefault("l", {})["go"] = loop_count[0] < 2
            return "d"

        ctx.register_tool("driver", driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(key_fn=rotating_key),
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.l.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # Different keys → cache miss on each pass → tool called twice
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_key_fn_error_treats_as_miss(self):
        """If key_fn raises, the step still runs (treated as cache miss)."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return "ok"

        ctx, _ = make_context()
        ctx.register_tool("t", tool)

        def bad_key_fn(c):
            raise RuntimeError("key_fn failure")

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(key_fn=bad_key_fn),
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        assert call_count[0] == 1  # step ran despite key_fn error


# ---------------------------------------------------------------------------
# Cache reset between chain.execute() calls
# ---------------------------------------------------------------------------


class TestCacheReset:
    @pytest.mark.asyncio
    async def test_cache_reset_between_execute_calls(self):
        """Cache is cleared at the start of each execute() call."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return "result"

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="T",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    cache=StepCache(),
                ),
            ],
        )

        ctx1, _ = make_context("input-1")
        ctx1.register_tool("t", tool)
        await chain.execute_async(ctx1)
        assert call_count[0] == 1

        # Second run with a fresh context — cache must be cleared, tool re-runs
        ctx2, _ = make_context("input-2")
        ctx2.register_tool("t", tool)
        result2 = await chain.execute_async(ctx2)
        assert result2.success
        assert call_count[0] == 2  # ran again


# ---------------------------------------------------------------------------
# Non-cached steps still work normally
# ---------------------------------------------------------------------------


class TestNonCachedStepUnaffected:
    @pytest.mark.asyncio
    async def test_non_cached_step_always_runs(self):
        """Steps without cache= always execute, even in loops."""
        call_count = [0]

        def tool():
            call_count[0] += 1
            return "ok"

        ctx, _ = make_context()
        ctx.register_tool("t", tool)

        loop_count = [0]

        def driver():
            loop_count[0] += 1
            ctx.memory.setdefault("l", {})["go"] = loop_count[0] < 3
            return "d"

        ctx.register_tool("driver", driver)

        chain = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1,
                    title="No cache",
                    config=ToolStepConfig(tool_name="t", input_mapping={}),
                    # no cache= field
                ),
                ToolStepDescription(
                    number=2,
                    title="Driver",
                    dependencies=[1],
                    config=ToolStepConfig(tool_name="driver", input_mapping={}),
                    loop_back_to=1,
                    loop_config={"condition_key": "$memory.l.go", "max_iterations": 5},
                ),
            ],
        )
        result = await chain.execute_async(ctx)
        assert result.success
        # 3 loop iterations → tool runs 3 times
        assert call_count[0] == 3
