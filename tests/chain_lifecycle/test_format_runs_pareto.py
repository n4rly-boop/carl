"""Tests for ``format_runs_pareto`` — cross-run Pareto front across a
budget sweep of evolutions.

Each ``EvolutionResult`` in the input list
contributes one point on ``(total_tokens, best_score)``; Pareto-dominant
runs marked distinctly so users see which budget is the best ROI.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from mmar_carl import (
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
    format_runs_pareto,
)


def _run(score: float, tokens: int, *, n: int = 1) -> EvolutionResult:
    """Build an EvolutionResult with a single generation of `n` identical individuals."""
    per_ind = tokens // max(n, 1)
    return EvolutionResult(
        best_chain_spec={},
        best_score=score,
        best_generation=0,
        history=[
            GenerationStats(
                generation=0,
                best_score=score,
                mean_score=score,
                population_scores=[score] * n,
                population_metrics=[
                    IndividualMetrics(score=score, total_tokens=per_ind)
                    for _ in range(n)
                ],
            )
        ],
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_list_returns_placeholder(self) -> None:
        out = format_runs_pareto([])
        assert "no runs supplied" in out

    def test_all_inf_score_runs_return_placeholder(self) -> None:
        out = format_runs_pareto([
            _run(float("-inf"), 100),
            _run(float("-inf"), 200),
        ])
        assert "no runs had a finite best_score" in out

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown format"):
            format_runs_pareto([_run(0.5, 100)], format="svg")

    def test_labels_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="labels length"):
            format_runs_pareto(
                [_run(0.5, 100), _run(0.7, 200)],
                labels=["only_one_label"],
            )

    def test_single_run_renders(self) -> None:
        out = format_runs_pareto([_run(0.5, 100)])
        assert "★" in out
        assert "(1 / 1 on front" in out


# ---------------------------------------------------------------------------
# Pareto-dominance across runs
# ---------------------------------------------------------------------------


class TestParetoMath:
    def test_cheaper_higher_score_dominates(self) -> None:
        """Run A: 100 tokens, score 0.9. Run B: 200 tokens, score 0.5.
        B is dominated."""
        out = format_runs_pareto([_run(0.9, 100), _run(0.5, 200)])
        assert "(1 / 2 on front" in out

    def test_staircase_all_on_front(self) -> None:
        """Each run trades cost for score — no one dominates anyone."""
        out = format_runs_pareto([
            _run(0.3, 100),
            _run(0.5, 200),
            _run(0.7, 300),
            _run(0.9, 400),
        ])
        assert "(4 / 4 on front" in out

    def test_costly_flop_dominated_by_cheaper_runs(self) -> None:
        """A run that's both expensive AND scores poorly is dominated by
        any cheaper run with a better score."""
        out = format_runs_pareto([
            _run(0.6, 100),  # cheap, decent → front
            _run(0.8, 200),  # mid, great → front
            _run(0.4, 500),  # expensive, mediocre → dominated by both above
        ])
        assert "(2 / 3 on front" in out


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


class TestLabels:
    def test_default_labels_are_r0_r1_etc(self) -> None:
        """No labels supplied → use rN indexing."""
        out = format_runs_pareto([
            _run(0.5, 100),
            _run(0.7, 200),
        ])
        # Both runs on the front (staircase), labels in legend
        assert "r0" in out or "r1" in out

    def test_custom_labels_appear_in_legend(self) -> None:
        out = format_runs_pareto(
            [_run(0.5, 100), _run(0.7, 200)],
            labels=["tiny", "huge"],
        )
        assert "tiny" in out
        assert "huge" in out


# ---------------------------------------------------------------------------
# Axis modes
# ---------------------------------------------------------------------------


class TestAxisModes:
    def test_default_axis_is_tokens(self) -> None:
        out = format_runs_pareto([_run(0.5, 100), _run(0.7, 200)])
        assert "total tokens" in out

    def test_cost_per_1k_switches_to_usd(self) -> None:
        out = format_runs_pareto(
            [_run(0.5, 1000), _run(0.7, 2000)],
            cost_per_1k=0.05,
        )
        assert "total cost (USD)" in out
        assert "$0.0500" in out  # 1000 * 0.05/1000 = $0.05

    def test_thousands_separator_on_token_axis(self) -> None:
        out = format_runs_pareto([_run(0.5, 1000), _run(0.7, 25000)])
        # Larger label uses thousands separator
        assert "25,000" in out


# ---------------------------------------------------------------------------
# PNG format
# ---------------------------------------------------------------------------


class TestPngFormat:
    def test_png_requires_path(self) -> None:
        with pytest.raises(ValueError, match="png_path"):
            format_runs_pareto([_run(0.5, 100)], format="png")

    def test_png_raises_install_hint_when_matplotlib_missing(self) -> None:
        try:
            import matplotlib  # noqa: F401
            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        except ImportError:
            pass
        with pytest.raises(ImportError, match="mmar-carl\\[viz\\]"):
            format_runs_pareto(
                [_run(0.5, 100), _run(0.7, 200)],
                format="png", png_path="/tmp/x.png",
            )

    def test_png_writes_when_matplotlib_available(self) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            pytest.skip("matplotlib not installed")
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "runs_pareto.png")
            written = format_runs_pareto(
                [_run(0.3, 100), _run(0.5, 200), _run(0.8, 500)],
                format="png", png_path=path,
            )
            assert os.path.exists(written)
            assert os.path.getsize(written) > 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_inf_runs_filtered_keep_others() -> None:
    """Mix of finite and -inf runs: -inf runs skipped, others kept."""
    out = format_runs_pareto([
        _run(float("-inf"), 100),
        _run(0.5, 200),
        _run(0.8, 300),
    ])
    # Only 2 of original 3 retained; both on the front (staircase)
    assert "(2 / 2 on front" in out


def test_total_tokens_summed_across_generations() -> None:
    """An evolution with multiple generations should sum all individuals'
    tokens to compute the run's x-axis position."""
    multi_gen = EvolutionResult(
        best_chain_spec={},
        best_score=0.8,
        best_generation=1,
        history=[
            GenerationStats(
                generation=0, best_score=0.5, mean_score=0.5,
                population_scores=[0.5],
                population_metrics=[IndividualMetrics(score=0.5, total_tokens=500)],
            ),
            GenerationStats(
                generation=1, best_score=0.8, mean_score=0.8,
                population_scores=[0.8],
                population_metrics=[IndividualMetrics(score=0.8, total_tokens=700)],
            ),
        ],
    )
    cheap = _run(0.3, 200)
    out = format_runs_pareto([multi_gen, cheap], labels=["multi", "cheap"])
    # multi_gen's total = 500 + 700 = 1200, displayed somewhere
    assert "1,200" in out
