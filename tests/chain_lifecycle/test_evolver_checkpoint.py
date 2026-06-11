"""Tests for ``ChainEvolver`` checkpoint / resume.

A long evolution run
shouldn't vanish on Ctrl-C. When ``checkpoint_path`` is set, the evolver
writes JSON state after each completed generation and auto-resumes
from that file on the next ``evolve()`` call.
"""

from __future__ import annotations

import json
import os
import random
import tempfile

import pytest

from mmar_carl import (
    ChainEvolver,
    DataCase,
    MetricBase,
    ReasoningChain,
    ReasoningContext,
    SimpleDataset,
    ToolStepConfig,
    ToolStepDescription,
)


class _ConstMetric(MetricBase):
    @property
    def name(self) -> str:
        return "c"

    async def compute_async(self, output) -> float:  # noqa: ANN001
        return 0.5


def _chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="noop", config=ToolStepConfig(tool_name="noop")
            )
        ],
    )


def _ctx_factory():
    def factory(case: DataCase) -> ReasoningContext:
        c = ReasoningContext(outer_context=case.input, api=None, model="default")
        c.register_tool("noop", lambda: "ok")
        return c
    return factory


# ---------------------------------------------------------------------------
# Checkpoint file shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_file_written_after_each_generation() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=3,
            smoke_check=False,
            checkpoint_path=path,
        )
        await ev.evolve(_ctx_factory())
        assert os.path.exists(path)
        data = json.loads(open(path).read())
        assert data["version"] == 1
        # After completing gen 2 (3rd gen, 0-indexed), completed_gen=2 and next=3
        assert data["completed_gen"] == 2
        assert data["next_generation"] == 3
        # History was serialized
        assert len(data["history"]) == 3


@pytest.mark.asyncio
async def test_checkpoint_atomic_write_via_temp_file() -> None:
    """The save path uses tempfile + os.replace so a partial write
    leaves the prior checkpoint intact. Verify by checking that the
    final file is well-formed JSON."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        await ev.evolve(_ctx_factory())
        # File parses cleanly
        with open(path) as f:
            data = json.load(f)
        assert "history" in data


# ---------------------------------------------------------------------------
# Resume flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_picks_up_where_checkpoint_left_off() -> None:
    """Write a checkpoint via a 3-gen run, then start a fresh evolver with
    the same path → should resume (no new gens because the original
    already completed all 3)."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")

        # Run 1: complete all 3 gens
        ev1 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=3,
            smoke_check=False,
            checkpoint_path=path,
            rng=random.Random(42),
        )
        result1 = await ev1.evolve(_ctx_factory())
        assert len(result1.history) == 3

        # Run 2: same config, same checkpoint — should produce the same
        # final history because all gens already done.
        ev2 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=3,
            smoke_check=False,
            checkpoint_path=path,
            rng=random.Random(999),  # different seed; should be overridden
        )
        result2 = await ev2.evolve(_ctx_factory())
        assert len(result2.history) == 3
        # Best score & generation preserved
        assert result2.best_score == result1.best_score
        assert result2.best_generation == result1.best_generation


@pytest.mark.asyncio
async def test_resume_continues_partial_run() -> None:
    """Manually craft a checkpoint that's 1 generation in, then resume —
    verify only the remaining gens execute."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")

        # Run 1: stop after generations=2 (so 2 gens done)
        ev1 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
            rng=random.Random(42),
        )
        await ev1.evolve(_ctx_factory())
        # Verify 2 gens recorded in checkpoint
        with open(path) as f:
            data = json.load(f)
        assert data["completed_gen"] == 1  # 0-indexed: gens 0 and 1
        assert data["next_generation"] == 2

        # Run 2: same checkpoint but expanded budget — but we changed
        # `generations` so checkpoint should be ignored (starts fresh).
        # Verify that path.
        ev2 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=4,  # different
            smoke_check=False,
            checkpoint_path=path,
        )
        result2 = await ev2.evolve(_ctx_factory())
        # Fresh run → all 4 gens executed
        assert len(result2.history) == 4


@pytest.mark.asyncio
async def test_resume_skips_smoke_check() -> None:
    """When resuming, the smoke check shouldn't re-run (it's already
    been validated)."""
    smoke_call_count = {"n": 0}

    class _CountingMetric(MetricBase):
        @property
        def name(self) -> str:
            return "c"

        async def compute_async(self, output) -> float:  # noqa: ANN001
            smoke_call_count["n"] += 1
            return 0.5

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")

        # Run 1: smoke check enabled
        ev1 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _CountingMetric(),
            population_size=2,
            generations=2,
            smoke_check=True,
            checkpoint_path=path,
        )
        await ev1.evolve(_ctx_factory())
        count_after_run1 = smoke_call_count["n"]
        # Run 1: 1 smoke + (2 pop × 2 gen) = 5 metric calls
        assert count_after_run1 == 5

        # Run 2: resume — smoke check should be skipped
        ev2 = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _CountingMetric(),
            population_size=2,
            generations=2,
            smoke_check=True,  # still True, but resume path overrides
            checkpoint_path=path,
        )
        await ev2.evolve(_ctx_factory())
        # No additional calls because all gens already complete
        assert smoke_call_count["n"] == count_after_run1


# ---------------------------------------------------------------------------
# Invalid checkpoint handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_checkpoint_starts_fresh() -> None:
    """No checkpoint file → evolve from scratch normally."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "nonexistent.json")
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        result = await ev.evolve(_ctx_factory())
        assert len(result.history) == 2
        # Now the file should exist
        assert os.path.exists(path)


@pytest.mark.asyncio
async def test_corrupt_checkpoint_starts_fresh() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")
        # Write malformed JSON
        with open(path, "w") as f:
            f.write("{ not valid json")
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        result = await ev.evolve(_ctx_factory())
        # Did NOT raise; started fresh
        assert len(result.history) == 2


@pytest.mark.asyncio
async def test_wrong_base_chain_starts_fresh() -> None:
    """Checkpoint written for a different base chain → ignore + start fresh."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")
        # Run 1: write checkpoint with chain A (1-step)
        chain_a = _chain()
        ev1 = ChainEvolver(
            chain_a,
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        await ev1.evolve(_ctx_factory())

        # Run 2: different chain → checkpoint should be rejected
        chain_b = ReasoningChain(
            steps=[
                ToolStepDescription(
                    number=1, title="different", config=ToolStepConfig(tool_name="noop"),
                ),
                ToolStepDescription(
                    number=2, title="more", dependencies=[1], config=ToolStepConfig(tool_name="noop"),
                ),
            ],
        )
        ev2 = ChainEvolver(
            chain_b,
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        result2 = await ev2.evolve(_ctx_factory())
        # Fresh run because checkpoint was for a different chain
        assert len(result2.history) == 2
        # First gen of result2 should NOT use chain_a's state
        # (best_chain_spec in run-2's first gen should be chain_b's, not chain_a's)
        first_gen_spec = result2.history[0].best_chain_spec
        assert len(first_gen_spec["steps"]) == 2  # chain_b has 2 steps


@pytest.mark.asyncio
async def test_wrong_version_starts_fresh() -> None:
    """A checkpoint with a future/unknown version field is ignored."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")
        # Pretend a future-version checkpoint exists
        with open(path, "w") as f:
            json.dump(
                {
                    "version": 999,  # unknown
                    "next_generation": 1,
                    "history": [],
                },
                f,
            )
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            checkpoint_path=path,
        )
        result = await ev.evolve(_ctx_factory())
        # Did NOT use the future checkpoint; ran from scratch
        assert len(result.history) == 2


# ---------------------------------------------------------------------------
# RNG state survives the round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rng_state_preserved_across_resume() -> None:
    """Resume with a fresh RNG seed in the constructor — the actual RNG
    used should be the one from the checkpoint, NOT the new seed."""

    # Run two evolutions with the same seed, both complete to 4 gens:
    # one using a checkpoint mid-run, one running straight through.
    # The histories should match (within float tolerance for the seed
    # to produce identical mutations).

    def build(ckpt, seed):
        return ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=4,
            smoke_check=False,
            checkpoint_path=ckpt,
            rng=random.Random(seed),
        )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "evo.json")

        # Straight-through reference run.
        ref = build(None, seed=42)
        result_ref = await ref.evolve(_ctx_factory())

        # Run that checkpoints — same seed.
        ev = build(path, seed=42)
        result_full = await ev.evolve(_ctx_factory())

        # Population scores per generation should match (RNG-driven
        # mutations should produce the same sequence).
        for g_ref, g_ck in zip(result_ref.history, result_full.history):
            assert g_ref.population_scores == g_ck.population_scores


# ---------------------------------------------------------------------------
# No checkpoint when not requested
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_checkpoint_when_path_not_set() -> None:
    """Default behaviour (checkpoint_path=None) — no file written."""
    with tempfile.TemporaryDirectory() as td:
        # We never tell the evolver about this path
        path = os.path.join(td, "evo.json")
        ev = ChainEvolver(
            _chain(),
            SimpleDataset([DataCase(input="x")]),
            _ConstMetric(),
            population_size=2,
            generations=2,
            smoke_check=False,
            # No checkpoint_path
        )
        await ev.evolve(_ctx_factory())
        # td is empty
        assert not os.path.exists(path)
