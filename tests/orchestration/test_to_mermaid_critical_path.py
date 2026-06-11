"""Tests for ``ReasoningChain.to_mermaid_critical_path(result)``.

Highlights the longest cumulative-latency chain
through the DAG so users see "which step do I optimize first?" — speeding
up any step *off* this path doesn't reduce total wall-clock time.
"""

from __future__ import annotations

from mmar_carl import LLMStepDescription, ReasoningChain
from mmar_carl.models.enums import StepType
from mmar_carl.models.results import ReasoningResult, StepExecutionResult


def _chain(*steps: LLMStepDescription) -> ReasoningChain:
    return ReasoningChain(steps=list(steps))


def _make_step(number: int, title: str, deps: list[int] | None = None) -> LLMStepDescription:
    return LLMStepDescription(
        number=number, title=title, aim="x", dependencies=deps or []
    )


def _result(timings: dict[int, float], wall: float | None = None) -> ReasoningResult:
    """Build a ReasoningResult with synthetic per-step execution_time."""
    return ReasoningResult(
        success=True,
        history=[],
        total_execution_time=wall if wall is not None else sum(timings.values()),
        step_results=[
            StepExecutionResult(
                step_number=n,
                step_title=f"step {n}",
                step_type=StepType.LLM,
                result="",
                success=True,
                execution_time=t,
            )
            for n, t in timings.items()
        ],
    )


# ---------------------------------------------------------------------------
# Simple linear chain — every edge is critical
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_linear_chain_all_edges_critical(self) -> None:
        chain = _chain(
            _make_step(1, "A"),
            _make_step(2, "B", deps=[1]),
            _make_step(3, "C", deps=[2]),
        )
        result = _result({1: 1.0, 2: 1.0, 3: 1.0})
        out = chain.to_mermaid_critical_path(result)
        # All 3 nodes carry a ⭐ since every step is on the critical path
        assert out.count("⭐") == 3
        # Both edges are critical → linkStyle 0,1
        assert "linkStyle 0,1" in out

    def test_linear_chain_zero_parallel_savings(self) -> None:
        chain = _chain(
            _make_step(1, "A"),
            _make_step(2, "B", deps=[1]),
        )
        result = _result({1: 2.0, 2: 3.0})
        out = chain.to_mermaid_critical_path(result)
        # serial baseline = critical path = 5.0s → 0% savings
        assert "parallel savings ≈ 0%" in out
        assert "5.00s" in out


# ---------------------------------------------------------------------------
# Diamond DAG with one fast / one slow branch
# ---------------------------------------------------------------------------


class TestDiamondCriticalPath:
    """1 → [2 (fast), 3 (slow)] → 4. Critical path is 1 → 3 → 4."""

    def _chain(self) -> ReasoningChain:
        return _chain(
            _make_step(1, "Plan"),
            _make_step(2, "Fast", deps=[1]),
            _make_step(3, "Slow", deps=[1]),
            _make_step(4, "Synth", deps=[2, 3]),
        )

    def test_slow_branch_marked_critical_fast_branch_not(self) -> None:
        result = _result({1: 2.0, 2: 0.5, 3: 3.0, 4: 1.0})
        chain_out = self._chain().to_mermaid_critical_path(result)
        # Step 3 (slow) has ⭐; step 2 (fast) does not
        # Find the node-definition line for each
        s2_line = next(line for line in chain_out.splitlines() if line.lstrip().startswith("S2["))
        s3_line = next(line for line in chain_out.splitlines() if line.lstrip().startswith("S3["))
        assert "⭐" not in s2_line
        assert "⭐" in s3_line

    def test_critical_edges_correctly_identified(self) -> None:
        """Edge index ordering follows declaration order:
        0: S1→S2, 1: S1→S3, 2: S2→S4, 3: S3→S4
        Critical edges are 1 (S1→S3) and 3 (S3→S4)."""
        result = _result({1: 2.0, 2: 0.5, 3: 3.0, 4: 1.0})
        out = self._chain().to_mermaid_critical_path(result)
        assert "linkStyle 1,3" in out

    def test_parallel_savings_calculated_against_serial_baseline(self) -> None:
        """Critical = 6.0s, serial sum = 6.5s, savings = 7.7% ≈ 8%."""
        result = _result({1: 2.0, 2: 0.5, 3: 3.0, 4: 1.0})
        out = self._chain().to_mermaid_critical_path(result)
        assert "critical path = 6.00s" in out
        assert "serial baseline = 6.50s" in out
        assert "parallel savings ≈ 8%" in out

    def test_red_edge_styling_present(self) -> None:
        result = _result({1: 2.0, 2: 0.5, 3: 3.0, 4: 1.0})
        out = self._chain().to_mermaid_critical_path(result)
        # Critical edges use red (#EF4444) thick (4px)
        assert "#EF4444" in out
        assert "stroke-width:4px" in out


# ---------------------------------------------------------------------------
# Reverse-symmetric DAG — verify the algorithm picks the correct branch
# ---------------------------------------------------------------------------


def test_critical_path_switches_when_branch_speed_inverts() -> None:
    """Same DAG, but swap timings: now branch 2 is slow, 3 is fast.
    Critical path should switch to 1 → 2 → 4."""
    chain = _chain(
        _make_step(1, "P"),
        _make_step(2, "Slow", deps=[1]),
        _make_step(3, "Fast", deps=[1]),
        _make_step(4, "S", deps=[2, 3]),
    )
    result = _result({1: 1.0, 2: 5.0, 3: 0.1, 4: 1.0})
    out = chain.to_mermaid_critical_path(result)
    # Now S2 has the star, S3 doesn't
    s2_line = next(line for line in out.splitlines() if line.lstrip().startswith("S2["))
    s3_line = next(line for line in out.splitlines() if line.lstrip().startswith("S3["))
    assert "⭐" in s2_line
    assert "⭐" not in s3_line
    # And critical edges flip: 0 (S1→S2), 2 (S2→S4)
    assert "linkStyle 0,2" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_step_chain_no_critical_edges(self) -> None:
        """One step → no edges → no linkStyle directive."""
        chain = _chain(_make_step(1, "Only"))
        result = _result({1: 2.0})
        out = chain.to_mermaid_critical_path(result)
        # The only step is critical
        assert "⭐" in out
        # No linkStyle line (no edges to colour)
        assert "linkStyle" not in out

    def test_missing_execution_time_treated_as_zero(self) -> None:
        chain = _chain(
            _make_step(1, "A"),
            _make_step(2, "B", deps=[1]),
            _make_step(3, "C", deps=[2]),
        )
        # Result only has step 1's timing; 2 and 3 default to 0.
        partial = ReasoningResult(
            success=True,
            history=[],
            total_execution_time=1.0,
            step_results=[
                StepExecutionResult(
                    step_number=1,
                    step_title="A",
                    step_type=StepType.LLM,
                    result="",
                    success=True,
                    execution_time=1.0,
                )
            ],
        )
        out = chain.to_mermaid_critical_path(partial)
        # Still produces a valid mermaid block; step 1 dominates the path
        assert "flowchart TD" in out
        assert "0.00s" in out  # missing timings render as 0.00s

    def test_implicit_linear_chain_when_no_deps_declared(self) -> None:
        """Mirrors to_mermaid's behaviour: no declared deps → synthetic
        linear edges. Critical path lights them all up."""
        chain = _chain(
            _make_step(1, "A"),
            _make_step(2, "B"),  # no deps
            _make_step(3, "C"),  # no deps
        )
        result = _result({1: 1.0, 2: 1.0, 3: 1.0})
        out = chain.to_mermaid_critical_path(result)
        # Implicit edges S1→S2 and S2→S3
        assert "S1 --> S2" in out
        assert "S2 --> S3" in out
        # Both critical → linkStyle 0,1
        assert "linkStyle 0,1" in out

    def test_all_zero_times_produces_safe_output(self) -> None:
        """Defensive: every step has 0 execution_time."""
        chain = _chain(
            _make_step(1, "A"),
            _make_step(2, "B", deps=[1]),
        )
        result = _result({1: 0.0, 2: 0.0})
        out = chain.to_mermaid_critical_path(result)
        # No crash, valid mermaid, 0% savings (no time to save)
        assert "flowchart TD" in out
        assert "critical path = 0.00s" in out


# ---------------------------------------------------------------------------
# Mermaid structural sanity
# ---------------------------------------------------------------------------


class TestMermaidStructure:
    def test_output_starts_with_flowchart_directive(self) -> None:
        chain = _chain(_make_step(1, "A"))
        out = chain.to_mermaid_critical_path(_result({1: 1.0}))
        assert out.startswith("flowchart TD")

    def test_node_labels_include_time(self) -> None:
        chain = _chain(_make_step(1, "MyStep"))
        out = chain.to_mermaid_critical_path(_result({1: 2.5}))
        assert "2.50s" in out

    def test_classDef_block_still_present(self) -> None:
        """Existing colouring is preserved alongside the new linkStyle."""
        chain = _chain(_make_step(1, "A"))
        out = chain.to_mermaid_critical_path(_result({1: 1.0}))
        assert "classDef llm fill:" in out

    def test_annotation_comment_uses_mermaid_comment_syntax(self) -> None:
        chain = _chain(_make_step(1, "A"))
        out = chain.to_mermaid_critical_path(_result({1: 1.0}))
        # `%%` is the mermaid comment marker — renderers ignore it,
        # raw readers see the stats.
        assert "    %% critical path" in out
