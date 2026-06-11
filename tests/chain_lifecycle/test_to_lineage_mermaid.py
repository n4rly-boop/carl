"""Tests for ``EvolutionResult.to_lineage_mermaid``.

Mermaid tree of every generation's individuals + parent edges. The
parent relationship is recovered heuristically from
``IndividualMetrics.parent_score`` matched against the prior
generation's scores — no explicit parent-index pointer is stored on
the schema, which keeps the evolver's data model simple.
"""

from __future__ import annotations

from mmar_carl.chain_evolution import (
    EvolutionResult,
    GenerationStats,
    IndividualMetrics,
)


def _result(
    *generations: tuple[list[float], list[IndividualMetrics] | None],
    best_score: float = 0.0,
    best_generation: int = 0,
) -> EvolutionResult:
    history = []
    for i, (scores, metrics) in enumerate(generations):
        history.append(GenerationStats(
            generation=i,
            best_score=max(scores, default=0.0),
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            population_scores=list(scores),
            population_metrics=metrics or [],
        ))
    return EvolutionResult(
        best_chain_spec={},
        best_score=best_score,
        best_generation=best_generation,
        history=history,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_history_returns_placeholder(self) -> None:
        ev = EvolutionResult(
            best_chain_spec={}, best_score=0.0, best_generation=0, history=[],
        )
        out = ev.to_lineage_mermaid()
        assert "no history" in out

    def test_no_population_metrics_still_renders(self) -> None:
        """Falls back to ``population_scores`` for the labels and
        renders every individual as a seed (since no mutation metadata
        is available)."""
        ev = _result(
            ([0.5, 0.4], None),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        assert "flowchart TD" in out
        assert "seed" in out
        # Best-chain styling applied
        assert "stroke:#B45309" in out


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_one_subgraph_per_generation(self) -> None:
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            ([0.6], [IndividualMetrics(score=0.6, mutation_kind="prompt_rewrite", parent_score=0.5)]),
            ([0.7], [IndividualMetrics(score=0.7, mutation_kind="prompt_rewrite", parent_score=0.6)]),
            best_score=0.7, best_generation=2,
        )
        out = ev.to_lineage_mermaid()
        assert 'subgraph gen0["Generation 0"]' in out
        assert 'subgraph gen1["Generation 1"]' in out
        assert 'subgraph gen2["Generation 2"]' in out

    def test_starts_with_flowchart_directive(self) -> None:
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        assert out.splitlines()[0] == "flowchart TD"

    def test_node_label_uses_br_tag_not_literal_backslash_n(self) -> None:
        """Same Mermaid gotcha as in ``to_mermaid_heatmap``: labels need
        ``<br/>`` rather than ``\\n`` to actually wrap."""
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        label_lines = [
            line for line in out.splitlines()
            if "[" in line and '"' in line and "G0I" in line
        ]
        for line in label_lines:
            assert "<br/>" in line
            assert "\\n" not in line


class TestNodeLabels:
    def test_seed_label_shown_for_generation_0(self) -> None:
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        assert "seed" in out

    def test_mutation_kind_shown_in_label(self) -> None:
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            ([0.7], [IndividualMetrics(score=0.7, mutation_kind="temperature_swap", parent_score=0.5)]),
            best_score=0.7, best_generation=1,
        )
        out = ev.to_lineage_mermaid()
        # Mutation appears inside the node label
        assert "temperature_swap" in out

    def test_score_rendered_with_default_two_decimals(self) -> None:
        ev = _result(
            ([0.123456], [IndividualMetrics(score=0.123456)]),
            best_score=0.123456, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        assert "0.12" in out
        # Not the full precision
        assert "0.123456" not in out

    def test_custom_decimals_respected(self) -> None:
        ev = _result(
            ([0.123456], [IndividualMetrics(score=0.123456)]),
            best_score=0.123456, best_generation=0,
        )
        out = ev.to_lineage_mermaid(score_decimals=4)
        assert "0.1235" in out  # rounded to 4 dp

    def test_double_quotes_in_mutation_name_replaced(self) -> None:
        """Defensive: a mutation name containing a double quote would
        break the Mermaid label string. Should be sanitised."""
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            ([0.6], [IndividualMetrics(
                score=0.6, mutation_kind='has "quoted" word', parent_score=0.5,
            )]),
            best_score=0.6, best_generation=1,
        )
        out = ev.to_lineage_mermaid()
        # No raw double quote inside any label
        assert 'has "quoted"' not in out
        assert "has 'quoted'" in out


# ---------------------------------------------------------------------------
# Parent-edge recovery
# ---------------------------------------------------------------------------


class TestParentEdges:
    def test_edge_drawn_when_parent_score_matches(self) -> None:
        ev = _result(
            ([0.5, 0.4], [
                IndividualMetrics(score=0.5),
                IndividualMetrics(score=0.4),
            ]),
            ([0.7], [
                IndividualMetrics(score=0.7, mutation_kind="prompt_rewrite",
                                   parent_score=0.5),
            ]),
            best_score=0.7, best_generation=1,
        )
        out = ev.to_lineage_mermaid()
        assert "G0I0 -->|prompt_rewrite| G1I0" in out

    def test_no_edge_for_seed(self) -> None:
        ev = _result(
            ([0.5, 0.4], [
                IndividualMetrics(score=0.5),
                IndividualMetrics(score=0.4),
            ]),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        assert "-->" not in out

    def test_no_edge_for_elitism_clone(self) -> None:
        """Elitism clones carry no ``parent_score`` → no edge."""
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            ([0.5, 0.6], [
                IndividualMetrics(score=0.5),  # elitism clone — no parent_score
                IndividualMetrics(score=0.6, mutation_kind="prompt_rewrite",
                                   parent_score=0.5),
            ]),
            best_score=0.6, best_generation=1,
        )
        out = ev.to_lineage_mermaid()
        # Only one edge (to the mutated child), not to the elitism clone
        edges = [line for line in out.splitlines() if "-->" in line]
        assert len(edges) == 1
        assert "G1I1" in edges[0]

    def test_unmatched_parent_score_dropped_safely(self) -> None:
        """If the parent_score doesn't match anyone in the prior
        generation, no edge is drawn — better than guessing wrong."""
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            ([0.9], [
                IndividualMetrics(score=0.9, mutation_kind="prompt_rewrite",
                                   parent_score=0.99),  # no match
            ]),
            best_score=0.9, best_generation=1,
        )
        out = ev.to_lineage_mermaid(score_tolerance=1e-6)
        # No parent edges
        edges = [line for line in out.splitlines() if "-->" in line]
        assert not edges


# ---------------------------------------------------------------------------
# Best-chain highlight
# ---------------------------------------------------------------------------


class TestBestChainHighlight:
    def test_best_node_gets_gold_border(self) -> None:
        ev = _result(
            ([0.5, 0.4], [
                IndividualMetrics(score=0.5),
                IndividualMetrics(score=0.4),
            ]),
            ([0.9], [
                IndividualMetrics(score=0.9, mutation_kind="prompt_rewrite",
                                   parent_score=0.5),
            ]),
            best_score=0.9, best_generation=1,
        )
        out = ev.to_lineage_mermaid()
        # Style line targets G1I0 specifically
        style_line = next(
            line for line in out.splitlines() if line.lstrip().startswith("style")
        )
        assert "G1I0" in style_line
        # Gold/amber fill + dark amber stroke
        assert "FCD34D" in style_line
        assert "B45309" in style_line

    def test_caption_lists_best_node(self) -> None:
        ev = _result(
            ([0.5], [IndividualMetrics(score=0.5)]),
            best_score=0.5, best_generation=0,
        )
        out = ev.to_lineage_mermaid()
        # The trailing %% caption mentions the best node id + score
        caption_line = next(
            line for line in out.splitlines()
            if line.lstrip().startswith("%% lineage tree")
        )
        assert "G0I0" in caption_line
        assert "0.50" in caption_line


# ---------------------------------------------------------------------------
# Method exposure
# ---------------------------------------------------------------------------


class TestExposure:
    def test_method_exists_on_class(self) -> None:
        assert callable(EvolutionResult.to_lineage_mermaid)
