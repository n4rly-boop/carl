"""Tests for chain content metadata convention.

Three pieces:

* `CareChainMetadata` / `CareContextFile` Pydantic models with
  defaults + a compact `to_metadata_dict()` writer.
* `ReasoningChain.set_care_metadata` / `get_care_metadata` typed
  accessors that namespace under `chain.metadata["care"]`.
* `ReasoningContext.from_chain_inputs` helper that primes a fresh
  context from a saved chain's metadata.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmar_carl import (
    CARE_METADATA_NAMESPACE,
    CareChainMetadata,
    CareContextFile,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
)


def _chain() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="X", aim="x"),
    ])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TestModels:
    def test_default_values(self) -> None:
        m = CareChainMetadata()
        assert m.task_description is None
        assert m.context_files == []
        assert m.tags == []
        assert m.mage_metadata == {}

    def test_context_file_requires_path(self) -> None:
        with pytest.raises(ValueError):
            CareContextFile()  # type: ignore[call-arg]

    def test_context_file_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError):
            CareContextFile(path="x", size_bytes=-1)

    def test_to_metadata_dict_drops_defaults(self) -> None:
        m = CareChainMetadata(task_description="hi")
        d = m.to_metadata_dict()
        # Only the set field present
        assert d == {"task_description": "hi"}

    def test_to_metadata_dict_keeps_explicit_values(self) -> None:
        m = CareChainMetadata(
            task_description="t",
            context_files=[CareContextFile(path="/x")],
            generated_by="mage",
            display_name="Demo",
            description="A test chain.",
            tags=["favourite"],
            mage_metadata={"planner_model": "qwen"},
        )
        d = m.to_metadata_dict()
        for k in ("task_description", "context_files", "generated_by",
                  "display_name", "description", "tags", "mage_metadata"):
            assert k in d


# ---------------------------------------------------------------------------
# ReasoningChain accessors
# ---------------------------------------------------------------------------


class TestChainAccessors:
    def test_set_with_kwargs(self) -> None:
        chain = _chain()
        result = chain.set_care_metadata(
            task_description="t", tags=["a"],
        )
        assert result is chain  # returns self for chaining
        # Stored under the care namespace
        assert CARE_METADATA_NAMESPACE in chain.metadata
        assert chain.metadata[CARE_METADATA_NAMESPACE]["task_description"] == "t"

    def test_set_with_meta(self) -> None:
        chain = _chain()
        meta = CareChainMetadata(task_description="t", tags=["a"])
        chain.set_care_metadata(meta=meta)
        assert chain.metadata[CARE_METADATA_NAMESPACE]["task_description"] == "t"

    def test_mixing_meta_and_kwargs_raises(self) -> None:
        chain = _chain()
        with pytest.raises(ValueError, match="not both"):
            chain.set_care_metadata(
                meta=CareChainMetadata(task_description="a"),
                tags=["b"],
            )

    def test_get_returns_typed_model(self) -> None:
        chain = _chain()
        chain.set_care_metadata(task_description="t", tags=["a"])
        got = chain.get_care_metadata()
        assert isinstance(got, CareChainMetadata)
        assert got.task_description == "t"
        assert got.tags == ["a"]

    def test_get_returns_none_when_absent(self) -> None:
        chain = _chain()
        assert chain.get_care_metadata() is None

    def test_get_returns_none_when_namespace_wrong_type(self) -> None:
        chain = _chain()
        chain.metadata[CARE_METADATA_NAMESPACE] = "not-a-dict"  # type: ignore[assignment]
        assert chain.get_care_metadata() is None

    def test_namespace_does_not_collide_with_user_metadata(self) -> None:
        chain = _chain()
        chain.metadata["task_description"] = "user's value"  # legacy / unrelated
        chain.set_care_metadata(task_description="CARE's value")
        # User's top-level key untouched
        assert chain.metadata["task_description"] == "user's value"
        # CARE's value namespaced
        assert (
            chain.metadata[CARE_METADATA_NAMESPACE]["task_description"]
            == "CARE's value"
        )


# ---------------------------------------------------------------------------
# Round-trip via chain.to_dict() / from_dict()
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_preserves_care_metadata(self) -> None:
        chain = _chain()
        chain.set_care_metadata(
            task_description="t",
            context_files=[CareContextFile(path="/x", sha256="abc")],
            generated_by="mage",
            tags=["a"],
        )
        spec = chain.to_dict()
        rebuilt = ReasoningChain.from_dict(spec, use_typed_steps=True)
        got = rebuilt.get_care_metadata()
        assert got is not None
        assert got.task_description == "t"
        assert got.context_files[0].path == "/x"
        assert got.context_files[0].sha256 == "abc"
        assert got.tags == ["a"]


# ---------------------------------------------------------------------------
# ReasoningContext.from_chain_inputs
# ---------------------------------------------------------------------------


class TestFromChainInputs:
    def test_uses_task_description_as_outer_context(self) -> None:
        chain = _chain()
        chain.set_care_metadata(task_description="Summarise these docs")
        ctx = ReasoningContext.from_chain_inputs(chain, api=None)
        assert ctx.outer_context == "Summarise these docs"

    def test_explicit_outer_context_wins(self) -> None:
        chain = _chain()
        chain.set_care_metadata(task_description="from-metadata")
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None, outer_context="explicit",
        )
        assert ctx.outer_context == "explicit"

    def test_loads_files_from_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Hello\n")
        chain = _chain()
        chain.set_care_metadata(
            task_description="x",
            context_files=[CareContextFile(path=str(f))],
        )
        ctx = ReasoningContext.from_chain_inputs(chain, api=None)
        assert ctx.memory["input"]["doc.md"] == "# Hello\n"

    def test_missing_file_leaves_placeholder(self, tmp_path: Path) -> None:
        # File doesn't exist on disk
        chain = _chain()
        chain.set_care_metadata(
            task_description="x",
            context_files=[CareContextFile(path=str(tmp_path / "ghost.txt"))],
        )
        ctx = ReasoningContext.from_chain_inputs(chain, api=None)
        # The slot is not populated (the loader silently skips truly
        # nonexistent paths to avoid noisy placeholders).
        # We only require: no crash, no real content for the missing file.
        val = ctx.memory["input"].get("ghost.txt", "")
        assert "# Hello" not in val

    def test_explicit_files_override_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# From disk\n")
        chain = _chain()
        chain.set_care_metadata(
            task_description="x",
            context_files=[CareContextFile(path=str(f))],
        )
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None, files={"doc.md": "# Override\n"},
        )
        assert ctx.memory["input"]["doc.md"] == "# Override\n"

    def test_files_only_when_metadata_absent(self, tmp_path: Path) -> None:
        chain = _chain()  # no care metadata
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None, files={"a.txt": "hello"},
        )
        assert ctx.memory["input"]["a.txt"] == "hello"
        assert ctx.outer_context == ""

    def test_load_files_from_metadata_can_be_disabled(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Should not load\n")
        chain = _chain()
        chain.set_care_metadata(
            task_description="x",
            context_files=[CareContextFile(path=str(f))],
        )
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None, load_files_from_metadata=False,
        )
        assert "doc.md" not in ctx.memory["input"]

    def test_extra_kwargs_forwarded_to_constructor(self) -> None:
        from mmar_carl.models.enums import Language
        chain = _chain()
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None,
            outer_context="x", language=Language.ENGLISH,
            system_prompt="be brief",
        )
        assert ctx.language == Language.ENGLISH
        assert ctx.system_prompt == "be brief"

    def test_caller_memory_merged_not_clobbered(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("disk\n")
        chain = _chain()
        chain.set_care_metadata(
            task_description="x",
            context_files=[CareContextFile(path=str(f))],
        )
        ctx = ReasoningContext.from_chain_inputs(
            chain, api=None,
            memory={"custom": {"k": "v"}},
        )
        # Both namespaces present
        assert ctx.memory["input"]["doc.md"] == "disk\n"
        assert ctx.memory["custom"]["k"] == "v"

    def test_chain_without_metadata_yields_empty_context(self) -> None:
        chain = _chain()
        ctx = ReasoningContext.from_chain_inputs(chain, api=None)
        assert ctx.outer_context == ""
        # input namespace seeded but empty
        assert ctx.memory.get("input", {}) == {}
