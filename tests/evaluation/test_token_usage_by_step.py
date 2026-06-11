"""Tests for ``ReasoningResult.token_usage_by_step`` convenience accessor."""

from __future__ import annotations

import pytest

from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _step(
    number: int,
    *,
    title: str = "s",
    step_type: StepType = StepType.LLM,
    usage: dict[str, int] | None = None,
    success: bool = True,
) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=number,
        step_title=title,
        step_type=step_type,
        result="",
        success=success,
        token_usage=usage or {},
    )


def _result(*steps: StepExecutionResult) -> ReasoningResult:
    return ReasoningResult(success=True, history=[], step_results=list(steps))


class TestTokenUsageByStep:
    def test_empty_chain_returns_empty_dict(self) -> None:
        assert _result().token_usage_by_step == {}

    def test_step_with_no_usage_omitted(self) -> None:
        """Tool / Memory / Transform steps don't record token usage —
        they should NOT appear in the by-step mapping."""
        result = _result(
            _step(1, step_type=StepType.TOOL),  # no usage
            _step(2, step_type=StepType.LLM, usage={"prompt": 50, "completion": 25, "total": 75}),
        )
        per_step = result.token_usage_by_step
        assert 1 not in per_step
        assert 2 in per_step

    def test_complete_usage_preserved(self) -> None:
        result = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150}),
        )
        assert result.token_usage_by_step[1] == {
            "prompt": 100,
            "completion": 50,
            "total": 150,
        }

    def test_missing_total_computed(self) -> None:
        """Some clients populate only prompt + completion. Fill in total
        defensively as prompt+completion."""
        result = _result(_step(1, usage={"prompt": 200, "completion": 100}))
        usage = result.token_usage_by_step[1]
        assert usage["prompt"] == 200
        assert usage["completion"] == 100
        assert usage["total"] == 300

    def test_explicit_total_overrides_sum(self) -> None:
        """If the client returns a `total` that doesn't equal prompt+completion
        (e.g. provider counts cached tokens differently), preserve it."""
        result = _result(
            _step(1, usage={"prompt": 100, "completion": 50, "total": 142})
        )
        # Explicit total wins over the prompt+completion sum
        assert result.token_usage_by_step[1]["total"] == 142

    def test_missing_prompt_or_completion_defaults_to_zero(self) -> None:
        """Partial usage dicts should not raise."""
        result = _result(_step(1, usage={"prompt": 50}))
        usage = result.token_usage_by_step[1]
        assert usage["prompt"] == 50
        assert usage["completion"] == 0
        assert usage["total"] == 50

    def test_all_zero_usage_still_included(self) -> None:
        """A step that ran the LLM and reports {0, 0, 0} should appear in
        the breakdown (it ran), unlike a step that didn't call LLM at all
        (empty dict, omitted)."""
        result = _result(
            _step(1, usage={"prompt": 0, "completion": 0, "total": 0}),
        )
        assert result.token_usage_by_step == {1: {"prompt": 0, "completion": 0, "total": 0}}

    def test_per_step_keyed_by_step_number_not_position(self) -> None:
        """If a chain has steps numbered 5, 10, 15 (non-contiguous), the
        mapping uses those step numbers as keys."""
        result = _result(
            _step(5, usage={"prompt": 10, "completion": 5, "total": 15}),
            _step(10, usage={"prompt": 20, "completion": 10, "total": 30}),
            _step(15, usage={"prompt": 30, "completion": 15, "total": 45}),
        )
        per_step = result.token_usage_by_step
        assert set(per_step.keys()) == {5, 10, 15}
        assert per_step[10]["total"] == 30

    def test_failed_step_with_usage_still_recorded(self) -> None:
        """A step that failed mid-LLM-call may still have a partial usage
        record (e.g. the API returned usage before the post-processing
        error). Include it — useful for cost analysis on failed runs."""
        result = _result(
            _step(
                1,
                success=False,
                usage={"prompt": 200, "completion": 0, "total": 200},
            )
        )
        assert result.token_usage_by_step == {1: {"prompt": 200, "completion": 0, "total": 200}}

    def test_returns_int_values(self) -> None:
        """Coerce stringy or float inputs to int defensively (some mocks
        return floats)."""
        result = _result(_step(1, usage={"prompt": 100.0, "completion": 50.0, "total": 150.0}))  # type: ignore[arg-type]
        per_step = result.token_usage_by_step
        assert per_step[1]["prompt"] == 100
        assert isinstance(per_step[1]["prompt"], int)

    def test_sum_across_breakdown_matches_chain_total(self) -> None:
        """Sanity: summing per-step totals should equal the chain-level
        ``token_usage`` populated by DAGExecutor (when both are populated
        from the same source)."""
        steps = [
            _step(1, usage={"prompt": 100, "completion": 50, "total": 150}),
            _step(2, usage={"prompt": 60, "completion": 30, "total": 90}),
            _step(3, step_type=StepType.TOOL),  # no usage, ignored
            _step(4, usage={"prompt": 40, "completion": 20, "total": 60}),
        ]
        result = ReasoningResult(
            success=True,
            history=[],
            step_results=steps,
            token_usage={"prompt": 200, "completion": 100, "total": 300},
        )
        per_step = result.token_usage_by_step
        # Chain-level totals
        assert result.token_usage["total"] == 300
        # Per-step sum
        assert sum(u["total"] for u in per_step.values()) == 300
        assert sum(u["prompt"] for u in per_step.values()) == 200
        assert sum(u["completion"] for u in per_step.values()) == 100


# ---------------------------------------------------------------------------
# Documentation: the public API surface
# ---------------------------------------------------------------------------


def test_is_a_property_not_a_method() -> None:
    """``token_usage_by_step`` is a property — callers should NOT need
    ``.token_usage_by_step()``."""
    result = _result(_step(1, usage={"prompt": 1, "completion": 1, "total": 2}))
    # No parens
    usage = result.token_usage_by_step
    assert isinstance(usage, dict)
    # Calling without parens AND then with parens would fail with TypeError
    with pytest.raises(TypeError):
        result.token_usage_by_step()  # type: ignore[operator]
