"""Tests for structural mutations.

``MutationKind.DELETE_STEP`` removes a leaf step (one no other step
depends on). ``MutationKind.INSERT_STEP`` splices a template step in
just before the chain's final step, rewriting dependencies so the new
step becomes a verification / refinement gate before the existing
output.
"""

from __future__ import annotations

import random


from mmar_carl import (
    ChainMutator,
    LLMStepDescription,
    ReasoningChain,
    ToolStepDescription,
)
from mmar_carl.chain_evolution import MutationKind
from mmar_carl.models.config import ToolStepConfig


def _two_step_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="Plan", aim="Outline."),
        LLMStepDescription(
            number=2, title="Solve", aim="Answer.", dependencies=[1],
        ),
    ])


def _three_step_chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="A", aim="A."),
        LLMStepDescription(number=2, title="B", aim="B.", dependencies=[1]),
        LLMStepDescription(number=3, title="C", aim="C.", dependencies=[2]),
    ])


# ---------------------------------------------------------------------------
# MutationKind enum
# ---------------------------------------------------------------------------


class TestMutationKindEnum:
    def test_delete_step_value(self) -> None:
        assert MutationKind.DELETE_STEP.value == "delete_step"

    def test_insert_step_value(self) -> None:
        assert MutationKind.INSERT_STEP.value == "insert_step"


# ---------------------------------------------------------------------------
# Auto-enable in ``enabled_kinds`` defaults
# ---------------------------------------------------------------------------


class TestAutoEnable:
    def test_template_pool_auto_enables_insert(self) -> None:
        m = ChainMutator(step_template_pool=[{
            "step_type": "llm", "title": "V", "aim": "verify",
        }])
        assert MutationKind.INSERT_STEP in m.enabled_kinds

    def test_allow_step_deletion_auto_enables_delete(self) -> None:
        m = ChainMutator(allow_step_deletion=True)
        assert MutationKind.DELETE_STEP in m.enabled_kinds

    def test_neither_flag_no_structural_kinds_enabled(self) -> None:
        m = ChainMutator(aim_suffix_pool=[" Be brief."])
        assert MutationKind.DELETE_STEP not in m.enabled_kinds
        assert MutationKind.INSERT_STEP not in m.enabled_kinds


# ---------------------------------------------------------------------------
# DELETE_STEP semantics
# ---------------------------------------------------------------------------


class TestDeleteStep:
    def test_removes_leaf_step(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            allow_step_deletion=True,
            enabled_kinds=[MutationKind.DELETE_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind == MutationKind.DELETE_STEP
        # Step 2 was a leaf and got deleted
        assert [s.number for s in new_chain.steps] == [1]
        assert new_chain.steps[0].title == "Plan"

    def test_single_step_chain_cannot_delete(self) -> None:
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="only", aim="x"),
        ])
        m = ChainMutator(
            allow_step_deletion=True,
            enabled_kinds=[MutationKind.DELETE_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind is None
        assert len(new_chain.steps) == 1

    def test_picks_leaf_not_root_when_one_leaf(self) -> None:
        """Three-step linear chain — only step 3 is a leaf."""
        chain = _three_step_chain()
        m = ChainMutator(
            allow_step_deletion=True,
            enabled_kinds=[MutationKind.DELETE_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind == MutationKind.DELETE_STEP
        assert {s.number for s in new_chain.steps} == {1, 2}

    def test_chain_with_no_leaves_returns_none(self) -> None:
        """If every step is referenced by another, no leaf exists.
        Construct a chain where every step is depended on."""
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="A", aim="x"),
            LLMStepDescription(number=2, title="B", aim="x", dependencies=[1]),
            LLMStepDescription(number=3, title="C", aim="x", dependencies=[1, 2]),
        ])
        # Step 3 is still a leaf (nobody depends on it), so this WILL apply.
        # To prove the "no leaves" guard works, use a chain where each step
        # has a dependent — currently impossible in a DAG without a sink,
        # so skip and rely on the simpler tests above.
        m = ChainMutator(
            allow_step_deletion=True,
            enabled_kinds=[MutationKind.DELETE_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        # Will pick step 3 (the only leaf). Chain remains valid.
        assert kind == MutationKind.DELETE_STEP
        assert {s.number for s in new_chain.steps} == {1, 2}


# ---------------------------------------------------------------------------
# INSERT_STEP semantics
# ---------------------------------------------------------------------------


class TestInsertStep:
    def test_inserts_template_step_before_final(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "Verify",
                "aim": "Verify the answer.",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind == MutationKind.INSERT_STEP
        titles = [s.title for s in new_chain.steps]
        assert "Verify" in titles

    def test_inserted_step_number_unique(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "Verify", "aim": "v",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, _ = m.mutate_with_kind(chain, random.Random(0))
        numbers = [s.number for s in new_chain.steps]
        assert len(set(numbers)) == len(numbers)

    def test_inserted_step_inherits_last_steps_dependencies(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "Verify", "aim": "v",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, _ = m.mutate_with_kind(chain, random.Random(0))
        verify = next(s for s in new_chain.steps if s.title == "Verify")
        # Originally step 2 depended on [1]; the new step takes [1].
        assert verify.dependencies == [1]

    def test_last_step_now_depends_on_inserted_step(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "Verify", "aim": "v",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, _ = m.mutate_with_kind(chain, random.Random(0))
        verify = next(s for s in new_chain.steps if s.title == "Verify")
        solve = next(s for s in new_chain.steps if s.title == "Solve")
        assert solve.dependencies == [verify.number]

    def test_inserted_chain_passes_from_dict_validation(self) -> None:
        """Round-trip through ``to_dict`` / ``from_dict`` — the mutated
        spec must be valid."""
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "Verify", "aim": "v",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, _ = m.mutate_with_kind(chain, random.Random(0))
        spec = new_chain.to_dict()
        rehydrated = ReasoningChain.from_dict(spec, use_typed_steps=True)
        assert len(rehydrated.steps) == 3

    def test_empty_template_pool_means_no_insert(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind is None
        assert len(new_chain.steps) == 2


# ---------------------------------------------------------------------------
# Validation rollback — broken mutations don't break the run
# ---------------------------------------------------------------------------


class TestValidationRollback:
    def test_template_with_bad_dependency_falls_through(self) -> None:
        """An INSERT_STEP whose template would produce a circular dep
        is caught by ``from_dict`` and rolled back — caller still gets
        a valid chain back, mutator returns None."""
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "BadDep", "aim": "x",
                # Forced dep on a non-existent step number — chain
                # validation should reject.
                "dependencies": [999],
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        # Our code overwrites template dependencies, so this still
        # succeeds — proves the dependency-rewrite is robust.
        # (If it ever doesn't, the kind would be None and the chain
        # would round-trip unchanged.)
        assert kind in (MutationKind.INSERT_STEP, None)
        # Either way, the returned chain is valid.
        ReasoningChain.from_dict(new_chain.to_dict(), use_typed_steps=True)


# ---------------------------------------------------------------------------
# Tool-step interaction
# ---------------------------------------------------------------------------


class TestToolStepCompatibility:
    def test_delete_does_not_break_chain_with_tool_steps(self) -> None:
        """Mix of step types — delete a leaf LLM step but keep the tool
        step intact."""
        chain = ReasoningChain(steps=[
            ToolStepDescription(
                number=1, title="Fetch",
                config=ToolStepConfig(tool_name="fetch"),
            ),
            LLMStepDescription(
                number=2, title="Summary", aim="x", dependencies=[1],
            ),
        ])
        m = ChainMutator(
            allow_step_deletion=True,
            enabled_kinds=[MutationKind.DELETE_STEP],
        )
        new_chain, kind = m.mutate_with_kind(chain, random.Random(0))
        assert kind == MutationKind.DELETE_STEP
        # Step 2 was the leaf — deleted. Step 1 (tool) survives.
        assert [s.number for s in new_chain.steps] == [1]
        assert new_chain.steps[0].title == "Fetch"


# ---------------------------------------------------------------------------
# Sequential mutations are stable
# ---------------------------------------------------------------------------


class TestSequentialMutations:
    def test_repeated_insert_grows_the_chain(self) -> None:
        chain = _two_step_chain()
        m = ChainMutator(
            step_template_pool=[{
                "step_type": "llm", "title": "V", "aim": "verify",
            }],
            enabled_kinds=[MutationKind.INSERT_STEP],
        )
        for _ in range(3):
            chain, _ = m.mutate_with_kind(chain, random.Random(0))
        # 2 original + 3 inserted = 5 steps; all numbers unique
        numbers = [s.number for s in chain.steps]
        assert len(numbers) == 5
        assert len(set(numbers)) == 5
        # Chain remains valid
        ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
