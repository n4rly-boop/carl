"""Tests for ``notebooks/02_visualizations_demo.ipynb``.

The notebook itself is built by ``scripts/build_visualizations_notebook.py``
and consumes cassettes recorded by ``scripts/record_notebook_cassettes.py``.
These tests confirm:

1. The committed notebook matches what the builder emits (catches
   hand-edits that diverge from the source script).
2. The notebook executes end-to-end in cassette mode (RUN_LIVE=False)
   with no errors and produces sensible outputs in every code cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "02_visualizations_demo.ipynb"
CASSETTES = REPO_ROOT / "notebooks" / "cassettes"

pytestmark = pytest.mark.skipif(
    not NOTEBOOK.exists(),
    reason="notebooks/ is gitignored; demo notebook only present in local dev setups",
)


def test_notebook_exists() -> None:
    assert NOTEBOOK.exists(), (
        f"{NOTEBOOK} not found — run `make notebook-build` to regenerate."
    )


def test_notebook_has_run_live_false_default() -> None:
    """The committed notebook must default to RUN_LIVE = False so the
    smoke test (and curious readers) don't burn tokens on import."""
    body = NOTEBOOK.read_text(encoding="utf-8")
    assert "RUN_LIVE = False" in body, (
        "Default RUN_LIVE flag must be False in the committed notebook."
    )


def test_notebook_is_valid_nbformat() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert "cells" in nb
    assert len(nb["cells"]) > 0
    # Every cell has a unique id (nbformat 5.1+ requires this)
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids), "every cell must have an id"
    assert len(set(ids)) == len(ids), "cell ids must be unique"


def test_committed_notebook_has_rendered_mermaid_outputs() -> None:
    """The committed notebook must contain rendered Mermaid display_data
    so that GitHub / JupyterLab / nbviewer show the diagrams without a
    fresh execution. Catches a regression where someone runs the
    notebook without ``--save`` and commits a stripped version."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    mermaid_blocks: list[str] = []
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") != "display_data":
                continue
            md_data = out.get("data", {}).get("text/markdown")
            if md_data is None:
                continue
            md_str = "".join(md_data) if isinstance(md_data, list) else md_data
            if "```mermaid" in md_str:
                mermaid_blocks.append(md_str)
    # At minimum: token-pie, gantt, two heatmaps = 4 rendered blocks
    assert len(mermaid_blocks) >= 4, (
        f"Expected at least 4 rendered Mermaid outputs in the committed "
        f"notebook, found {len(mermaid_blocks)}. Run "
        f"`make notebook-smoke-save` to refresh outputs."
    )


def test_cassette_committed() -> None:
    """At least one cassette must be present so cassette-mode runs."""
    assert (CASSETTES / "cost_demo.jsonl").exists(), (
        "cost_demo.jsonl cassette missing — run `make notebook-record` "
        "(requires OPENAI_API_KEY) to regenerate."
    )


def _cell_sources(nb: dict) -> list[tuple[str, str, str]]:
    """Extract (id, cell_type, source) tuples for sync comparison.

    Ignores ``outputs`` and ``execution_count`` so that re-executing
    the notebook (which populates outputs) doesn't trip the sync test
    — only structural drift between the builder and the committed
    file is flagged.
    """
    cells = []
    for c in nb["cells"]:
        src = c["source"]
        src_str = "".join(src) if isinstance(src, list) else src
        cells.append((c.get("id", ""), c["cell_type"], src_str))
    return cells


def test_builder_output_matches_committed_notebook() -> None:
    """Catch hand-edits that diverge from the builder source.

    Compares cell ids + types + sources only. Outputs and
    ``execution_count`` are intentionally ignored so a fresh execution
    pass doesn't show up as drift — those are captured by
    ``test_notebook_executes_in_cassette_mode``.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_visualizations_notebook",
        REPO_ROOT / "scripts" / "build_visualizations_notebook.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    expected = _cell_sources(mod.NOTEBOOK)
    actual = _cell_sources(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    assert actual == expected, (
        "Committed notebook is out of sync with its builder. Run "
        "`make notebook-build` to regenerate."
    )


def test_notebook_executes_in_cassette_mode() -> None:
    """End-to-end smoke test: run every cell and assert no failures.

    Skips when ``nbclient``/``nbformat`` aren't installed (they ship
    with the dev extras but aren't strictly required at runtime).
    """
    pytest.importorskip("nbclient")
    pytest.importorskip("nbformat")

    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    client = NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    client.execute()

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells, "notebook has no code cells"

    # Every code cell must have produced some output (stream or display_data)
    empty = []
    for i, cell in enumerate(code_cells):
        outputs = cell.get("outputs", [])
        has_output = any(
            o.get("output_type") in ("stream", "execute_result", "display_data")
            for o in outputs
        )
        if not has_output:
            empty.append(i + 1)
    assert not empty, f"code cells {empty} produced no output"
