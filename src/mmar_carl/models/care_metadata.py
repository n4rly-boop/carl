"""Typed accessor for CARE-specific keys stored on ``ReasoningChain.metadata``.

CARE (`Collaborative Agent Reasoning Ecosystem`) saves chains to
gigaevo-memory with a small standard payload of provenance keys so the
"Re-run from library" workflow can deterministically replay a chain
with the same inputs.

This module ships the **typed schema** for those keys plus the
namespace under which they live on the raw dict
(``chain.metadata["care"]``). The dict-on-the-wire shape stays
backward-compatible — older versions of CARE / CARL that read
``chain.metadata`` directly continue to work.

"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# Single namespace key on ``chain.metadata`` so CARE's keys don't
# collide with chain-author-supplied metadata. CARE writes everything
# under this key; older readers that accessed the same key names at
# the top level can be migrated by reading
# ``chain.metadata.get("care", {}).get(key)``.
CARE_METADATA_NAMESPACE: str = "care"


class CareContextFile(BaseModel):
    """One file the user attached when generating / executing the chain.

    Path is captured verbatim (CARE stores absolute paths from the user's
    machine plus a SHA so a re-run from a different machine can detect
    when the file has moved or changed).
    """

    path: str = Field(..., description="Path the user attached, verbatim.")
    sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 of the file contents at capture time.",
    )
    size_bytes: Optional[int] = Field(
        default=None, ge=0,
        description="File size at capture time (bytes).",
    )


class CareChainMetadata(BaseModel):
    """Typed view of the CARE-namespace block on ``chain.metadata``.

    Stored at ``chain.metadata["care"]`` as a plain dict; round-trips
    through ``model_dump()`` / ``model_validate()``. None / empty fields
    are omitted from the dict when written via
    :meth:`ReasoningChain.set_care_metadata`.
    """

    task_description: Optional[str] = Field(
        default=None,
        description="Original user query / natural-language task statement.",
    )
    context_files: list[CareContextFile] = Field(
        default_factory=list,
        description="Files the user attached when generating the chain.",
    )
    generated_by: Optional[str] = Field(
        default=None,
        description=(
            "Origin tag — e.g. ``'mage'`` for chains produced by CARE's "
            "MAGE planner, ``'user'`` for hand-authored chains, or any "
            "other string the caller wants to track."
        ),
    )
    mage_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Full ``MAGEMetadata.model_dump()`` payload from CARE's "
            "planning step (planner model, planner prompt hash, plan "
            "alternatives considered, etc.). Empty for non-MAGE chains."
        ),
    )
    display_name: Optional[str] = Field(
        default=None,
        description="Human-friendly title shown in CARE's library view.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-friendly description shown in CARE's library view.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="User tags (e.g. ``'favourite'``, ``'experimental'``).",
    )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Compact dict for writing back into ``chain.metadata['care']``.

        Drops fields at their default to keep the dict small and to
        avoid noisy diffs when CARE re-saves a chain. ``context_files``,
        ``tags``, and ``mage_metadata`` are kept when non-empty even
        though Pydantic's defaults are empty containers — readers may
        want to distinguish "no files attached" from "this chain was
        authored before the field existed".
        """
        raw = self.model_dump(mode="json")
        return {
            k: v for k, v in raw.items()
            if not (v is None or v == [] or v == {})
        }


__all__ = [
    "CARE_METADATA_NAMESPACE",
    "CareChainMetadata",
    "CareContextFile",
]
