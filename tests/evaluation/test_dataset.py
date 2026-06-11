"""
Unit tests for dataset abstractions (models/dataset.py).

Covers:
- DataCase Pydantic model construction and defaults
- SimpleDataset iteration and __len__
- ThresholdStrategy.select — higher_is_better=True and False, empty input,
  no match, all match
- TopKWorstStrategy.select — k < N, k > N, k == 0 (validator), ties at boundary
- DatasetEvaluationReport.to_reflection_dict() — structure and content
"""

import pytest
from pydantic import ValidationError

from mmar_carl import (
    CaseEvaluationResult,
    DataCase,
    DatasetEvaluationReport,
    SimpleDataset,
    ThresholdStrategy,
    TopKWorstStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(score: float, label: str = "case") -> CaseEvaluationResult:
    return CaseEvaluationResult(
        case=DataCase(input="some input", label=label),
        score=score,
        chain_output="some output",
        success=True,
    )


# ---------------------------------------------------------------------------
# DataCase
# ---------------------------------------------------------------------------


class TestDataCase:
    def test_required_input(self):
        case = DataCase(input="hello")
        assert case.input == "hello"
        assert case.label is None
        assert case.expected is None
        assert case.metadata == {}

    def test_all_fields(self):
        case = DataCase(
            input="data",
            label="c1",
            expected="expected output",
            metadata={"source": "test"},
        )
        assert case.label == "c1"
        assert case.expected == "expected output"
        assert case.metadata == {"source": "test"}

    def test_missing_input_raises(self):
        with pytest.raises(ValidationError):
            DataCase()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# SimpleDataset
# ---------------------------------------------------------------------------


class TestSimpleDataset:
    def test_iteration(self):
        cases = [DataCase(input=f"input {i}") for i in range(3)]
        ds = SimpleDataset(cases)
        assert list(ds) == cases

    def test_len(self):
        ds = SimpleDataset([DataCase(input="a"), DataCase(input="b")])
        assert len(ds) == 2

    def test_empty_dataset(self):
        ds = SimpleDataset([])
        assert list(ds) == []
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# ThresholdStrategy
# ---------------------------------------------------------------------------


class TestThresholdStrategy:
    def test_higher_is_better_selects_below_threshold(self):
        results = [_make_result(0.9), _make_result(0.5), _make_result(0.3)]
        strategy = ThresholdStrategy(threshold=0.6, higher_is_better=True)
        selected = strategy.select(results)
        scores = {r.score for r in selected}
        assert scores == {0.5, 0.3}

    def test_lower_is_better_selects_above_threshold(self):
        # e.g. "error rate" — lower is better → high scores are bad
        results = [_make_result(0.1), _make_result(0.7), _make_result(0.9)]
        strategy = ThresholdStrategy(threshold=0.6, higher_is_better=False)
        selected = strategy.select(results)
        scores = {r.score for r in selected}
        assert scores == {0.7, 0.9}

    def test_empty_input(self):
        strategy = ThresholdStrategy(threshold=0.5)
        assert strategy.select([]) == []

    def test_no_cases_match(self):
        # All scores above threshold → nothing selected
        results = [_make_result(0.8), _make_result(0.9)]
        strategy = ThresholdStrategy(threshold=0.5, higher_is_better=True)
        assert strategy.select(results) == []

    def test_all_cases_match(self):
        results = [_make_result(0.1), _make_result(0.2)]
        strategy = ThresholdStrategy(threshold=0.5, higher_is_better=True)
        selected = strategy.select(results)
        assert len(selected) == 2

    def test_exact_threshold_boundary_excluded(self):
        # score == threshold is NOT selected (strict less-than)
        results = [_make_result(0.5), _make_result(0.4)]
        strategy = ThresholdStrategy(threshold=0.5, higher_is_better=True)
        selected = strategy.select(results)
        assert len(selected) == 1
        assert selected[0].score == 0.4


# ---------------------------------------------------------------------------
# TopKWorstStrategy
# ---------------------------------------------------------------------------


class TestTopKWorstStrategy:
    def test_k_less_than_n(self):
        results = [_make_result(s) for s in [0.9, 0.1, 0.5, 0.3, 0.7]]
        strategy = TopKWorstStrategy(k=2, higher_is_better=True)
        selected = strategy.select(results)
        scores = sorted(r.score for r in selected)
        assert scores == [0.1, 0.3]

    def test_k_greater_than_n_returns_all(self):
        results = [_make_result(s) for s in [0.9, 0.4]]
        strategy = TopKWorstStrategy(k=10, higher_is_better=True)
        selected = strategy.select(results)
        assert len(selected) == 2

    def test_k_equal_to_n(self):
        results = [_make_result(s) for s in [0.8, 0.2, 0.5]]
        strategy = TopKWorstStrategy(k=3, higher_is_better=True)
        assert len(strategy.select(results)) == 3

    def test_k_zero_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TopKWorstStrategy(k=0)

    def test_k_negative_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TopKWorstStrategy(k=-1)

    def test_ties_included_by_default(self):
        # k=2 but three cases share the worst score
        results = [
            _make_result(0.1, "a"),
            _make_result(0.1, "b"),
            _make_result(0.1, "c"),
            _make_result(0.9, "d"),
        ]
        strategy = TopKWorstStrategy(k=2, higher_is_better=True, include_ties=True)
        selected = strategy.select(results)
        # All three tied-worst cases should be included
        assert len(selected) == 3

    def test_ties_excluded_when_disabled(self):
        results = [
            _make_result(0.1, "a"),
            _make_result(0.1, "b"),
            _make_result(0.1, "c"),
            _make_result(0.9, "d"),
        ]
        strategy = TopKWorstStrategy(k=2, higher_is_better=True, include_ties=False)
        selected = strategy.select(results)
        assert len(selected) == 2

    def test_lower_is_better_direction(self):
        # lower_is_better=False → higher scores are worse
        results = [_make_result(s) for s in [0.1, 0.9, 0.7, 0.3]]
        strategy = TopKWorstStrategy(k=2, higher_is_better=False)
        selected = strategy.select(results)
        scores = sorted((r.score for r in selected), reverse=True)
        assert scores == [0.9, 0.7]

    def test_empty_input(self):
        strategy = TopKWorstStrategy(k=3)
        assert strategy.select([]) == []


# ---------------------------------------------------------------------------
# DatasetEvaluationReport
# ---------------------------------------------------------------------------


class TestDatasetEvaluationReport:
    def _make_report(
        self,
        all_scores: list[float],
        selected_scores: list[float],
    ) -> DatasetEvaluationReport:
        all_results = [_make_result(s, f"c{i}") for i, s in enumerate(all_scores)]
        selected = [r for r in all_results if r.score in selected_scores]
        return DatasetEvaluationReport(
            metric_name="test_metric",
            strategy=ThresholdStrategy(threshold=0.5),
            all_results=all_results,
            selected_cases=selected,
            mean_score=sum(all_scores) / len(all_scores),
            min_score=min(all_scores),
            max_score=max(all_scores),
        )

    def test_to_reflection_dict_structure(self):
        report = self._make_report([0.8, 0.3, 0.4], [0.3, 0.4])
        d = report.to_reflection_dict()
        assert "dataset_evaluation_metric" in d
        assert "dataset_evaluation_stats" in d
        assert "dataset_problem_cases" in d
        assert "dataset_overfitting_warning" in d

    def test_to_reflection_dict_metric_name(self):
        report = self._make_report([0.7, 0.2], [0.2])
        d = report.to_reflection_dict()
        assert d["dataset_evaluation_metric"] == "test_metric"

    def test_to_reflection_dict_no_problem_cases(self):
        report = self._make_report([0.8, 0.9], [])
        d = report.to_reflection_dict()
        assert d["dataset_problem_cases"] == "none"

    def test_to_reflection_dict_stats_format(self):
        report = self._make_report([0.6, 0.4], [0.4])
        stats = report.to_reflection_dict()["dataset_evaluation_stats"]
        assert "total=2" in stats
        assert "mean=" in stats
        assert "min=" in stats
        assert "max=" in stats
