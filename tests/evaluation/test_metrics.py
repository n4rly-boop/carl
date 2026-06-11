"""
Tests for the MetricBase evaluation system.

Covers:
- MetricBase abstract contract
- Step-level metrics: computed after successful execution, stored in StepExecutionResult
- Chain-level metrics: computed from final output, stored in ReasoningResult
- Multiple metrics on the same step / chain
- Metrics are skipped for failed steps
- Metric exceptions are swallowed and do not abort execution
- to_dict() serialization includes metric scores
"""

import pytest

from mmar_carl import (
    Language,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    ReasoningResult,
    StepDescription,
    StepExecutionResult,
)
from mmar_carl.metrics import MetricOutput


# ---------------------------------------------------------------------------
# Shared mock LLM client
# ---------------------------------------------------------------------------


class SimpleMockClient(LLMClientBase):
    """Returns a fixed, deterministic string for every request."""

    def __init__(self, response: str = "hello world mock response with some words"):
        self.response = response
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.call_count += 1
        return self.response


class FailingMockClient(LLMClientBase):
    """Always raises an exception to simulate step failure."""

    async def get_response(self, prompt: str) -> str:
        raise RuntimeError("LLM call failed")

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        raise RuntimeError("LLM call failed")


# ---------------------------------------------------------------------------
# Concrete metric implementations for testing
# ---------------------------------------------------------------------------


def _text_from(output: MetricOutput) -> str:
    """Extract plain text from a StepExecutionResult or ReasoningResult."""
    if isinstance(output, ReasoningResult):
        return output.get_final_output()
    return output.result


class WordCountMetric(MetricBase):
    """Returns the number of words in the output text."""

    @property
    def name(self) -> str:
        return "word_count"

    async def compute_async(self, output: MetricOutput) -> float:
        return float(len(_text_from(output).split()))


class CharCountMetric(MetricBase):
    """Returns the number of characters in the output text."""

    @property
    def name(self) -> str:
        return "char_count"

    async def compute_async(self, output: MetricOutput) -> float:
        return float(len(_text_from(output)))


class KeywordPresentMetric(MetricBase):
    """Returns 1.0 if a keyword is present, 0.0 otherwise."""

    def __init__(self, keyword: str):
        self.keyword = keyword.lower()

    @property
    def name(self) -> str:
        return f"has_{self.keyword}"

    async def compute_async(self, output: MetricOutput) -> float:
        return 1.0 if self.keyword in _text_from(output).lower() else 0.0


class AlwaysErrorMetric(MetricBase):
    """Always raises an exception — used to verify graceful error handling."""

    @property
    def name(self) -> str:
        return "always_error"

    async def compute_async(self, output: MetricOutput) -> float:
        raise ValueError("intentional metric error")


class ConstantMetric(MetricBase):
    """Returns a fixed value — useful for quick assertions."""

    def __init__(self, value: float = 42.0, metric_name: str = "constant"):
        self._value = value
        self._name = metric_name

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output: MetricOutput) -> float:
        return self._value


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _step_result(text: str) -> StepExecutionResult:
    """Minimal StepExecutionResult for contract-level tests."""
    return StepExecutionResult(step_number=1, step_title="t", result=text, success=True)


def make_context(client: LLMClientBase) -> ReasoningContext:
    return ReasoningContext(
        outer_context="test input data",
        api=client,
        model="test",
        retry_max=1,
        language=Language.ENGLISH,
    )


def make_step(**kwargs) -> LLMStepDescription:
    defaults = dict(number=1, title="Test Step", aim="Do something")
    defaults.update(kwargs)
    return LLMStepDescription(**defaults)


# ---------------------------------------------------------------------------
# MetricBase contract tests
# ---------------------------------------------------------------------------


class TestMetricBaseContract:
    """Verify the abstract base class contract."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            MetricBase()  # type: ignore[abstract]

    def test_name_property_required(self):
        class NoName(MetricBase):
            async def compute_async(self, output: MetricOutput) -> float:
                return 0.0

        with pytest.raises(TypeError):
            NoName()  # type: ignore[abstract]

    def test_compute_async_required(self):
        class NoCompute(MetricBase):
            @property
            def name(self) -> str:
                return "no_compute"

        with pytest.raises(TypeError):
            NoCompute()  # type: ignore[abstract]

    def test_concrete_metric_instantiates(self):
        m = WordCountMetric()
        assert m.name == "word_count"

    @pytest.mark.asyncio
    async def test_compute_async_returns_float(self):
        m = WordCountMetric()
        result = await m.compute_async(_step_result("hello world"))
        assert isinstance(result, float)
        assert result == 2.0

    def test_sync_compute_wrapper(self):
        m = ConstantMetric(7.0, "c")
        assert m.compute(_step_result("any text")) == 7.0


# ---------------------------------------------------------------------------
# Step-level metrics
# ---------------------------------------------------------------------------


class TestStepMetrics:
    """Metrics attached to individual steps."""

    @pytest.mark.asyncio
    async def test_single_metric_on_step(self):
        client = SimpleMockClient("hello world mock")
        step = make_step(metrics=[WordCountMetric()])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert result.success
        sr = result.step_results[0]
        assert "word_count" in sr.metrics
        assert sr.metrics["word_count"] == 3.0

    @pytest.mark.asyncio
    async def test_multiple_metrics_on_step(self):
        response = "alpha beta gamma"
        client = SimpleMockClient(response)
        step = make_step(metrics=[WordCountMetric(), CharCountMetric()])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        sr = result.step_results[0]
        assert sr.metrics["word_count"] == 3.0
        assert sr.metrics["char_count"] == float(len(response))

    @pytest.mark.asyncio
    async def test_keyword_metric_found(self):
        client = SimpleMockClient("the quick brown fox jumps over the lazy dog")
        step = make_step(metrics=[KeywordPresentMetric("fox")])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert result.step_results[0].metrics["has_fox"] == 1.0

    @pytest.mark.asyncio
    async def test_keyword_metric_not_found(self):
        client = SimpleMockClient("hello world")
        step = make_step(metrics=[KeywordPresentMetric("fox")])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert result.step_results[0].metrics["has_fox"] == 0.0

    @pytest.mark.asyncio
    async def test_no_metrics_gives_empty_dict(self):
        client = SimpleMockClient("hello")
        step = make_step()
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert result.step_results[0].metrics == {}

    @pytest.mark.asyncio
    async def test_metric_error_is_swallowed(self):
        """A crashing metric must not abort execution."""
        client = SimpleMockClient("hello")
        step = make_step(metrics=[AlwaysErrorMetric(), ConstantMetric(5.0)])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert result.success
        sr = result.step_results[0]
        # The broken metric should be absent
        assert "always_error" not in sr.metrics
        # The good metric should still be present
        assert sr.metrics["constant"] == 5.0

    @pytest.mark.asyncio
    async def test_metrics_not_run_on_failed_step(self):
        """Metrics must not be run when the step itself failed."""
        client = FailingMockClient()
        step = make_step(metrics=[ConstantMetric(99.0)])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        assert not result.success
        sr = result.step_results[0]
        assert sr.metrics == {}

    @pytest.mark.asyncio
    async def test_metrics_on_multiple_steps_independently(self):
        client = SimpleMockClient("one two three")
        steps = [
            make_step(number=1, metrics=[WordCountMetric()]),
            LLMStepDescription(
                number=2,
                title="Step 2",
                aim="second step",
                dependencies=[1],
                metrics=[CharCountMetric()],
            ),
        ]
        chain = ReasoningChain(steps=steps)
        result = await chain.execute_async(make_context(client))

        assert result.success
        assert "word_count" in result.step_results[0].metrics
        assert "char_count" in result.step_results[1].metrics
        # The first step should not have char_count
        assert "char_count" not in result.step_results[0].metrics

    def test_metrics_field_on_step_description(self):
        step = make_step(metrics=[WordCountMetric(), CharCountMetric()])
        assert len(step.metrics) == 2

    def test_metrics_excluded_from_serialization(self):
        """Metrics (code objects) must not appear in the JSON-serialized step."""
        step = make_step(metrics=[WordCountMetric()])
        data = step.model_dump()
        assert "metrics" not in data

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_step_description_legacy_accepts_metrics(self):
        """The deprecated StepDescription class should also accept metrics."""
        step = StepDescription(
            number=1,
            title="Legacy",
            aim="test",
            metrics=[ConstantMetric(3.0)],
        )
        assert len(step.metrics) == 1


# ---------------------------------------------------------------------------
# Chain-level metrics
# ---------------------------------------------------------------------------


class TestChainMetrics:
    """Metrics attached to the chain (evaluated on final output)."""

    @pytest.mark.asyncio
    async def test_chain_metric_is_computed(self):
        client = SimpleMockClient("one two three four five")
        step = make_step()
        chain = ReasoningChain(steps=[step], metrics=[WordCountMetric()])
        result = await chain.execute_async(make_context(client))

        assert "word_count" in result.metrics
        assert result.metrics["word_count"] > 0

    @pytest.mark.asyncio
    async def test_multiple_chain_metrics(self):
        client = SimpleMockClient("alpha beta gamma delta")
        step = make_step()
        chain = ReasoningChain(
            steps=[step],
            metrics=[WordCountMetric(), CharCountMetric()],
        )
        result = await chain.execute_async(make_context(client))

        assert "word_count" in result.metrics
        assert "char_count" in result.metrics

    @pytest.mark.asyncio
    async def test_no_chain_metrics_gives_empty_dict(self):
        client = SimpleMockClient("hello")
        chain = ReasoningChain(steps=[make_step()])
        result = await chain.execute_async(make_context(client))

        assert result.metrics == {}

    @pytest.mark.asyncio
    async def test_chain_metric_error_is_swallowed(self):
        client = SimpleMockClient("hello")
        chain = ReasoningChain(
            steps=[make_step()],
            metrics=[AlwaysErrorMetric(), ConstantMetric(7.0)],
        )
        result = await chain.execute_async(make_context(client))

        assert result.success
        assert "always_error" not in result.metrics
        assert result.metrics["constant"] == 7.0

    @pytest.mark.asyncio
    async def test_step_and_chain_metrics_coexist(self):
        client = SimpleMockClient("step output text")
        step = make_step(metrics=[ConstantMetric(1.0, "step_metric")])
        chain = ReasoningChain(
            steps=[step],
            metrics=[ConstantMetric(2.0, "chain_metric")],
        )
        result = await chain.execute_async(make_context(client))

        assert result.step_results[0].metrics["step_metric"] == 1.0
        assert result.metrics["chain_metric"] == 2.0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestMetricsSerialization:
    """Metric scores must survive to_dict() and be visible to callers."""

    @pytest.mark.asyncio
    async def test_step_result_to_dict_includes_metrics(self):
        client = SimpleMockClient("hello world")
        step = make_step(metrics=[WordCountMetric()])
        chain = ReasoningChain(steps=[step])
        result = await chain.execute_async(make_context(client))

        d = result.step_results[0].to_dict()
        assert "metrics" in d
        assert d["metrics"]["word_count"] == 2.0

    @pytest.mark.asyncio
    async def test_reasoning_result_to_dict_includes_chain_metrics(self):
        client = SimpleMockClient("a b c d")
        chain = ReasoningChain(steps=[make_step()], metrics=[WordCountMetric()])
        result = await chain.execute_async(make_context(client))

        d = result.to_dict()
        assert "metrics" in d
        assert d["metrics"]["word_count"] > 0
