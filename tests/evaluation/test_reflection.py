"""Tests for reflection feature."""


import pytest

from mmar_carl import (
    Language,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    ReflectionOptions,
)


class MockLLMHub:
    """Mock LLMHub for testing."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    def __getitem__(self, key: str):
        return self

    def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.calls.append(prompt)
        return self.response


class MockLLMClient(LLMClientBase):
    """Mock LLM client for testing."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[str] = []

    async def get_response(self, prompt: str, request: str | None = None, **kwargs) -> str:
        # Handle both positional and keyword argument styles
        actual_prompt = request if request is not None else prompt
        self.calls.append(actual_prompt)
        return self.response

    async def get_response_with_retries(self, prompt: str, retries: int = 3, request: str | None = None, **kwargs) -> str:
        # Handle both positional and keyword argument styles
        actual_prompt = request if request is not None else prompt
        self.calls.append(actual_prompt)
        return self.response


class TestReflection:
    """Test reflection feature."""

    def test_reflect_without_execution_raises(self):
        """Test that reflect raises error when no execution has been performed."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        with pytest.raises(RuntimeError, match="No execution result available"):
            chain.reflect(task_description="Test task")

    @pytest.mark.asyncio
    async def test_reflect_async_without_execution_raises(self):
        """Test that reflect_async raises error when no execution has been performed."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        with pytest.raises(RuntimeError, match="No execution result available"):
            await chain.reflect_async(task_description="Test task")

    @pytest.mark.asyncio
    async def test_reflect_async_success(self):
        """Test successful reflection after execution."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("Step result")
        context = ReasoningContext(
            outer_context="test data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        # Execute first
        result = await chain.execute_async(context)
        assert result.success

        # Mock reflection response
        mock_client.response = "Reflection: The chain successfully completed the task."

        # Reflect
        reflection = await chain.reflect_async(task_description="Analyze the test data")

        assert reflection == "Reflection: The chain successfully completed the task."
        # Should have called LLM at least once for execution + once for reflection
        assert len(mock_client.calls) >= 2

    @pytest.mark.asyncio
    async def test_reflect_includes_task_description(self):
        """Test that reflection prompt includes task description."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("Step result")
        context = ReasoningContext(
            outer_context="test data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        # Execute
        await chain.execute_async(context)

        # Reflect
        mock_client.response = "Reflection response"
        await chain.reflect_async(task_description="My specific task description")

        # Check that the reflection prompt includes the task description
        reflection_prompt = mock_client.calls[-1]
        assert "My specific task description" in reflection_prompt

    @pytest.mark.asyncio
    async def test_reflect_russian_language(self):
        """Test reflection in Russian language."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Тестовый шаг",
                aim="Тестовая цель",
                reasoning_questions="Тестовые вопросы",
                stage_action="Тестовое действие",
                example_reasoning="Тестовый пример",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("Результат шага")
        context = ReasoningContext(
            outer_context="тестовые данные",
            api=mock_client,
            model="test",
            language=Language.RUSSIAN,
        )

        # Execute
        await chain.execute_async(context)

        # Reflect
        mock_client.response = "Рефлексия: Цепочка успешно выполнила задачу."
        reflection = await chain.reflect_async(task_description="Проанализировать данные")  # noqa: F841

        # Check that the reflection prompt contains Russian text
        reflection_prompt = mock_client.calls[-1]
        assert "Исходная задача" in reflection_prompt or "рефлексии" in reflection_prompt.lower()

    @pytest.mark.asyncio
    async def test_reflect_with_failed_step(self):
        """Test reflection includes information about failed steps."""
        steps = [
            LLMStepDescription(
                number=1,
                title="First Step",
                aim="First aim",
                reasoning_questions="First questions",
                stage_action="First action",
                example_reasoning="First example",
            ),
            LLMStepDescription(
                number=2,
                title="Second Step",
                aim="Second aim",
                reasoning_questions="Second questions",
                stage_action="Second action",
                example_reasoning="Second example",
                dependencies=[1],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("Result")
        context = ReasoningContext(
            outer_context="test data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        # Execute
        result = await chain.execute_async(context)  # noqa: F841

        # Reflect
        mock_client.response = "Reflection on execution"
        await chain.reflect_async(task_description="Test task")

        # Verify reflection prompt mentions execution stats
        reflection_prompt = mock_client.calls[-1]
        assert "Total steps" in reflection_prompt or "Successful" in reflection_prompt

    def test_get_last_result(self):
        """Test getting last execution result."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        # Initially None
        assert chain.get_last_result() is None

        # After execution
        mock_client = MockLLMClient("Result")
        context = ReasoningContext(
            outer_context="test data",
            api=mock_client,
            model="test",
        )

        chain.execute(context)
        result = chain.get_last_result()

        assert result is not None
        assert result.success

    def test_reflect_sync(self):
        """Test synchronous reflection."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Test Step",
                aim="Test aim",
                reasoning_questions="Test questions",
                stage_action="Test action",
                example_reasoning="Test example",
            )
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("Step result")
        context = ReasoningContext(
            outer_context="test data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        # Execute
        chain.execute(context)

        # Reflect
        mock_client.response = "Reflection response"
        reflection = chain.reflect(task_description="Test task")

        assert reflection == "Reflection response"


# ---------------------------------------------------------------------------
# Metric-enhanced reflection tests
# ---------------------------------------------------------------------------

class ConstantMetric(MetricBase):
    """Returns a fixed score for deterministic testing."""

    def __init__(self, name_: str, value: float):
        self._name = name_
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output) -> float:
        return self._value


def _make_step(**kwargs):
    defaults = dict(
        number=1,
        title="Test Step",
        aim="Test aim",
        reasoning_questions="Test questions",
        stage_action="Test action",
        example_reasoning="Test example",
    )
    defaults.update(kwargs)
    return LLMStepDescription(**defaults)


class TestReflectionMetricScores:
    """Metric scores appear in / are excluded from the reflection prompt."""

    @pytest.mark.asyncio
    async def test_step_metric_scores_in_prompt(self):
        """Step metric scores are included in the reflection prompt by default."""
        step = _make_step(
            metrics=[ConstantMetric("coverage", 0.42)],
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("step output")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(task_description="task")

        prompt = mock_client.calls[-1]
        assert "coverage" in prompt
        assert "0.42" in prompt

    @pytest.mark.asyncio
    async def test_chain_metric_scores_in_prompt(self):
        """Chain-level metric scores are included in the reflection prompt by default."""
        step = _make_step()
        chain = ReasoningChain(
            steps=[step],
            metrics=[ConstantMetric("judge_score", 7.5)],
        )

        mock_client = MockLLMClient("chain output")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(task_description="task")

        prompt = mock_client.calls[-1]
        assert "judge_score" in prompt
        assert "7.5" in prompt

    @pytest.mark.asyncio
    async def test_metric_scores_excluded_when_disabled(self):
        """Metric scores are absent from the prompt when include_metric_scores=False."""
        step = _make_step(
            metrics=[ConstantMetric("my_metric", 0.99)],
        )
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("step output")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(
            task_description="task",
            options=ReflectionOptions(include_metric_scores=False),
        )

        prompt = mock_client.calls[-1]
        assert "my_metric" not in prompt

    @pytest.mark.asyncio
    async def test_no_metrics_section_when_no_metrics_attached(self):
        """No metric section added to the prompt when no metrics are defined."""
        step = _make_step()
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("result")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(task_description="task")

        prompt = mock_client.calls[-1]
        assert "Evaluation Metric Scores" not in prompt

    @pytest.mark.asyncio
    async def test_multiple_step_metrics_all_appear(self):
        """All metric scores from multiple steps appear in the prompt."""
        steps = [
            _make_step(number=1, title="Step A", metrics=[ConstantMetric("score_a", 1.0)]),
            _make_step(number=2, title="Step B", metrics=[ConstantMetric("score_b", 2.0)], dependencies=[1]),
        ]
        chain = ReasoningChain(steps=steps, max_workers=1)

        mock_client = MockLLMClient("output")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(task_description="task")

        prompt = mock_client.calls[-1]
        assert "score_a" in prompt
        assert "score_b" in prompt


class TestReflectionExtraFeedback:
    """User-provided extra_feedback appears in the reflection prompt."""

    @pytest.mark.asyncio
    async def test_extra_feedback_dict_in_prompt(self):
        """Dict extra_feedback is included with labelled entries."""
        step = _make_step()
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("result")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(
            task_description="task",
            options=ReflectionOptions(
                extra_feedback={"domain": "medical", "note": "focus on precision"},
            ),
        )

        prompt = mock_client.calls[-1]
        assert "domain" in prompt
        assert "medical" in prompt
        assert "note" in prompt
        assert "focus on precision" in prompt

    @pytest.mark.asyncio
    async def test_extra_feedback_string_in_prompt(self):
        """String extra_feedback is included verbatim."""
        step = _make_step()
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("result")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(
            task_description="task",
            options=ReflectionOptions(
                extra_feedback="Please focus on improving step 2 specifically.",
            ),
        )

        prompt = mock_client.calls[-1]
        assert "Please focus on improving step 2 specifically." in prompt

    @pytest.mark.asyncio
    async def test_extra_feedback_none_no_section(self):
        """No extra feedback section when extra_feedback=None."""
        step = _make_step()
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("result")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(task_description="task")

        prompt = mock_client.calls[-1]
        assert "Additional Feedback" not in prompt

    @pytest.mark.asyncio
    async def test_extra_feedback_russian_header(self):
        """Russian language uses Russian section header for extra feedback."""
        step = _make_step(title="Шаг теста", aim="Тест")
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("результат")
        context = ReasoningContext(
            outer_context="данные",
            api=mock_client,
            model="test",
            language=Language.RUSSIAN,
        )

        await chain.execute_async(context)
        mock_client.response = "рефлексия"
        await chain.reflect_async(
            task_description="задача",
            options=ReflectionOptions(extra_feedback="важный контекст"),
        )

        prompt = mock_client.calls[-1]
        assert "Дополнительный контекст" in prompt
        assert "важный контекст" in prompt

    @pytest.mark.asyncio
    async def test_metrics_and_extra_feedback_combined(self):
        """Both metric scores and extra_feedback appear together in the prompt."""
        step = _make_step(metrics=[ConstantMetric("precision", 0.88)])
        chain = ReasoningChain(steps=[step], max_workers=1)

        mock_client = MockLLMClient("result")
        context = ReasoningContext(
            outer_context="data",
            api=mock_client,
            model="test",
            language=Language.ENGLISH,
        )

        await chain.execute_async(context)
        mock_client.response = "reflection"
        await chain.reflect_async(
            task_description="task",
            options=ReflectionOptions(
                include_metric_scores=True,
                extra_feedback={"priority": "high", "reviewer": "alice"},
            ),
        )

        prompt = mock_client.calls[-1]
        assert "precision" in prompt
        assert "0.88" in prompt
        assert "priority" in prompt
        assert "reviewer" in prompt
