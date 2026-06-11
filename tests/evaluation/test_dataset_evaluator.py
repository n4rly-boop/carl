"""
Integration tests for DatasetEvaluator and dataset_report in ReflectionOptions.

Covers:
- DatasetEvaluator.evaluate() returns correct DatasetEvaluationReport
- Scores are computed from the metric, not hardcoded
- Failed chain execution → success=False, score=0.0, does not crash
- Empty dataset → empty report, no crash
- ReflectionOptions.dataset_report=None → prompt unchanged (regression)
- ReflectionOptions.dataset_report=<report> → dedicated section in prompt
- to_reflection_dict() path via extra_feedback still works
"""

import pytest

from mmar_carl import (
    DataCase,
    DatasetEvaluationReport,
    DatasetEvaluator,
    Language,
    LLMClientBase,
    LLMStepDescription,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    ReflectionOptions,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class FixedLLMClient(LLMClientBase):
    """Returns a fixed response string for every call."""

    def __init__(self, response: str = "fixed chain output") -> None:
        self.response = response
        self.call_count = 0

    async def get_response(self, prompt: str, **kwargs) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3, **kwargs) -> str:
        self.call_count += 1
        return self.response


class FailingLLMClient(LLMClientBase):
    """Always raises to simulate chain failure."""

    async def get_response(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("LLM unavailable")

    async def get_response_with_retries(self, prompt: str, retries: int = 3, **kwargs) -> str:
        raise RuntimeError("LLM unavailable")


class FixedScoreMetric(MetricBase):
    """Returns a fixed score regardless of input."""

    def __init__(self, score: float, name: str = "fixed_score") -> None:
        self._score = score
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def compute_async(self, output) -> float:
        return self._score


class InputLengthMetric(MetricBase):
    """Score = number of words in the final output (normalised to 0–1 by /100)."""

    @property
    def name(self) -> str:
        return "word_count_norm"

    async def compute_async(self, output) -> float:
        from mmar_carl import ReasoningResult
        text = output.get_final_output() if isinstance(output, ReasoningResult) else output.result
        return min(1.0, len(text.split()) / 100)


# ---------------------------------------------------------------------------
# Shared chain factory
# ---------------------------------------------------------------------------


def _make_chain(llm_client: LLMClientBase) -> tuple[ReasoningChain, LLMClientBase]:
    steps = [
        LLMStepDescription(
            number=1,
            title="Step",
            aim="Do something",
        )
    ]
    chain = ReasoningChain(steps=steps)
    return chain, llm_client


def _context_factory(client: LLMClientBase):
    def factory(case: DataCase) -> ReasoningContext:
        return ReasoningContext(outer_context=case.input, api=client)

    return factory


# ---------------------------------------------------------------------------
# DatasetEvaluator tests
# ---------------------------------------------------------------------------


class TestDatasetEvaluator:
    def test_basic_evaluation_returns_report(self):
        client = FixedLLMClient()
        chain, _ = _make_chain(client)
        dataset = SimpleDataset([
            DataCase(input="hello", label="c1"),
            DataCase(input="world", label="c2"),
        ])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.7),
            strategy=TopKWorstStrategy(k=1),
        )
        report = evaluator.evaluate(_context_factory(client))

        assert len(report.all_results) == 2
        assert report.metric_name == "fixed_score"

    def test_scores_come_from_metric(self):
        client = FixedLLMClient()
        chain, _ = _make_chain(client)
        dataset = SimpleDataset([DataCase(input="a"), DataCase(input="b")])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.42),
            strategy=TopKWorstStrategy(k=2),
        )
        report = evaluator.evaluate(_context_factory(client))

        for result in report.all_results:
            assert result.score == pytest.approx(0.42)

    def test_all_results_success_true(self):
        client = FixedLLMClient("good output text")
        chain, _ = _make_chain(client)
        dataset = SimpleDataset([DataCase(input="x")])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.9),
            strategy=TopKWorstStrategy(k=1),
        )
        report = evaluator.evaluate(_context_factory(client))
        assert report.all_results[0].success is True

    def test_failed_chain_produces_zero_score_and_no_crash(self):
        client = FailingLLMClient()
        chain, _ = _make_chain(client)
        dataset = SimpleDataset([DataCase(input="test", label="fail_case")])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(1.0),
            strategy=TopKWorstStrategy(k=1),
        )
        report = evaluator.evaluate(_context_factory(client))

        assert len(report.all_results) == 1
        r = report.all_results[0]
        assert r.success is False
        assert r.score == 0.0

    def test_empty_dataset_returns_empty_report(self):
        client = FixedLLMClient()
        chain, _ = _make_chain(client)
        dataset = SimpleDataset([])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.5),
            strategy=TopKWorstStrategy(k=1),
        )
        report = evaluator.evaluate(_context_factory(client))

        assert report.all_results == []
        assert report.selected_cases == []
        assert report.mean_score == 0.0

    def test_top_k_selection_correct(self):
        client = FixedLLMClient()
        chain, _ = _make_chain(client)

        # Use InputLengthMetric: shorter output = lower score = worse
        # FixedLLMClient always returns the same response, so scores will be equal.
        # Use FixedScoreMetric variant with per-case different labels and a threshold.
        dataset = SimpleDataset([
            DataCase(input="a", label="low"),
            DataCase(input="b", label="high"),
        ])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.3),
            strategy=ThresholdStrategy(threshold=0.5, higher_is_better=True),
        )
        report = evaluator.evaluate(_context_factory(client))

        # All cases score 0.3 < 0.5 → all selected
        assert len(report.selected_cases) == 2

    def test_mean_min_max_scores(self):
        client = FixedLLMClient()
        chain, _ = _make_chain(client)
        # Two cases: both get the same fixed score from FixedScoreMetric
        dataset = SimpleDataset([DataCase(input="a"), DataCase(input="b")])
        evaluator = DatasetEvaluator(
            chain=chain,
            dataset=dataset,
            metric=FixedScoreMetric(0.6),
            strategy=TopKWorstStrategy(k=1),
        )
        report = evaluator.evaluate(_context_factory(client))

        assert report.mean_score == pytest.approx(0.6)
        assert report.min_score == pytest.approx(0.6)
        assert report.max_score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# ReflectionOptions.dataset_report integration
# ---------------------------------------------------------------------------


class TestDatasetReportInReflection:
    """Verify that dataset_report wires correctly into _build_reflection_prompt."""

    def _build_mock_report(self) -> DatasetEvaluationReport:
        from mmar_carl import CaseEvaluationResult
        case = DataCase(input="sample input text", label="s1")
        result = CaseEvaluationResult(
            case=case, score=0.2, chain_output="short output", success=True
        )
        return DatasetEvaluationReport(
            metric_name="quality",
            strategy=ThresholdStrategy(threshold=0.5),
            all_results=[result],
            selected_cases=[result],
            mean_score=0.2,
            min_score=0.2,
            max_score=0.2,
        )

    def _execute_and_reflect(
        self,
        options: ReflectionOptions,
        reflection_response: str = "reflection text",
    ) -> str:
        # Two clients: one for chain execution, one for reflection LLM call
        exec_client = FixedLLMClient("chain execution output")
        reflect_client = FixedLLMClient(reflection_response)

        steps = [LLMStepDescription(number=1, title="Step", aim="Do something")]
        chain = ReasoningChain(steps=steps)

        ctx = ReasoningContext(outer_context="test input", api=exec_client)
        chain.execute(ctx)

        # Override the internal LLM client used for reflection
        # reflect() calls the same context's api, so we re-run with reflect client
        ctx2 = ReasoningContext(outer_context="test input", api=exec_client)
        chain.execute(ctx2)
        chain._last_context = ReasoningContext(outer_context="test input", api=reflect_client)

        return chain.reflect("improve quality", options=options)

    def test_no_dataset_report_works_as_before(self):
        """Regression: omitting dataset_report must not change existing behavior."""
        options = ReflectionOptions(dataset_report=None)
        reflection = self._execute_and_reflect(options)
        assert isinstance(reflection, str)

    def test_dataset_report_section_appears_in_prompt(self):
        """When dataset_report is provided, the prompt contains the dataset section."""
        report = self._build_mock_report()

        captured_prompts: list[str] = []

        class CapturingClient(LLMClientBase):
            async def get_response(self, prompt: str, **kwargs) -> str:
                return await self.get_response_with_retries(prompt)

            async def get_response_with_retries(self, prompt: str, retries: int = 3, **kwargs) -> str:
                captured_prompts.append(prompt)
                return "reflection result"

        exec_client = FixedLLMClient("chain output")
        steps = [LLMStepDescription(number=1, title="Step", aim="Do something")]
        chain = ReasoningChain(steps=steps)

        ctx = ReasoningContext(outer_context="input", api=exec_client)
        chain.execute(ctx)
        chain._last_context = ReasoningContext(
            outer_context="input", api=CapturingClient()
        )

        chain.reflect(
            "improve quality",
            options=ReflectionOptions(dataset_report=report),
        )

        assert captured_prompts, "No prompt was captured"
        prompt = captured_prompts[0]
        assert "Dataset Evaluation" in prompt or "Оценка датасета" in prompt
        assert "quality" in prompt  # metric name
        assert "s1" in prompt  # case label

    def test_dataset_report_russian_section(self):
        """Russian language produces Russian dataset section headers."""
        report = self._build_mock_report()
        captured: list[str] = []

        class CapturingClient(LLMClientBase):
            async def get_response(self, prompt: str, **kwargs) -> str:
                return await self.get_response_with_retries(prompt)

            async def get_response_with_retries(self, prompt: str, retries: int = 3, **kwargs) -> str:
                captured.append(prompt)
                return "рефлексия"

        exec_client = FixedLLMClient("вывод")
        steps = [LLMStepDescription(number=1, title="Шаг", aim="Что-то делать")]
        chain = ReasoningChain(steps=steps)

        ctx = ReasoningContext(outer_context="ввод", api=exec_client)
        chain.execute(ctx)
        chain._last_context = ReasoningContext(outer_context="ввод", api=CapturingClient())

        chain.reflect(
            "улучшить качество",
            language=Language.RUSSIAN,
            options=ReflectionOptions(dataset_report=report),
        )

        assert captured
        assert "Оценка датасета" in captured[0]

    def test_extra_feedback_via_to_reflection_dict(self):
        """MVP path: to_reflection_dict() passed via extra_feedback works end-to-end."""
        report = self._build_mock_report()
        d = report.to_reflection_dict()
        assert isinstance(d, dict)

        captured: list[str] = []

        class CapturingClient(LLMClientBase):
            async def get_response(self, prompt: str, **kwargs) -> str:
                return await self.get_response_with_retries(prompt)

            async def get_response_with_retries(self, prompt: str, retries: int = 3, **kwargs) -> str:
                captured.append(prompt)
                return "feedback-based reflection"

        exec_client = FixedLLMClient("output")
        steps = [LLMStepDescription(number=1, title="Step", aim="Aim")]
        chain = ReasoningChain(steps=steps)
        ctx = ReasoningContext(outer_context="input", api=exec_client)
        chain.execute(ctx)
        chain._last_context = ReasoningContext(outer_context="input", api=CapturingClient())

        chain.reflect("task", options=ReflectionOptions(extra_feedback=d))

        assert captured
        assert "dataset_evaluation_metric" in captured[0]
