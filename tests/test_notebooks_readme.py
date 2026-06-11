"""Tests for the notebooks/ README index generator.

``scripts/generate_notebooks_readme.py`` writes ``notebooks/README.md``,
pulling title / summary / runtime / cost / prerequisites from each
notebook's ``metadata.demo_info`` block (with sensible fallbacks for
notebooks that don't yet declare one).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_notebooks_readme.py"
README_PATH = REPO_ROOT / "notebooks" / "README.md"


@pytest.fixture(scope="module")
def gen_mod():
    spec = importlib.util.spec_from_file_location(
        "generate_notebooks_readme", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_nb(path: Path, *, demo_info: dict | None = None,
              first_md: str = "# Stub\n\nA stub notebook.") -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "id": "c1", "metadata": {}, "source": first_md},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                            "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if demo_info is not None:
        nb["metadata"]["demo_info"] = demo_info
    path.write_text(json.dumps(nb))


# ---------------------------------------------------------------------------
# extract_demo_info — metadata-first, with markdown fallbacks
# ---------------------------------------------------------------------------


class TestExtractDemoInfo:
    def test_explicit_demo_info_used(self, gen_mod, tmp_path: Path) -> None:
        nb = tmp_path / "a.ipynb"
        _write_nb(nb, demo_info={
            "title": "Custom Title",
            "summary": "Custom summary.",
            "runtime_offline": "1 s",
            "runtime_live": "2 s",
            "estimated_cost": "$0.001",
            "prerequisites": ["foo", "bar"],
        })
        info = gen_mod.extract_demo_info(nb)
        assert info["title"] == "Custom Title"
        assert info["summary"] == "Custom summary."
        assert info["prerequisites"] == ["foo", "bar"]

    def test_title_falls_back_to_first_h1(self, gen_mod, tmp_path: Path) -> None:
        nb = tmp_path / "b.ipynb"
        _write_nb(nb, demo_info=None,
                  first_md="# Fallback Title\n\nSummary text.")
        info = gen_mod.extract_demo_info(nb)
        assert info["title"] == "Fallback Title"

    def test_title_falls_back_to_stem_when_no_h1(self, gen_mod, tmp_path: Path) -> None:
        nb = tmp_path / "my_demo.ipynb"
        _write_nb(nb, demo_info=None, first_md="No header here.")
        info = gen_mod.extract_demo_info(nb)
        assert info["title"] == "My Demo"

    def test_summary_falls_back_to_first_paragraph(self, gen_mod, tmp_path: Path) -> None:
        nb = tmp_path / "c.ipynb"
        _write_nb(nb, demo_info=None,
                  first_md="# T\n\nFirst para.\nSame para.\n\nSecond para.")
        info = gen_mod.extract_demo_info(nb)
        # Collapsed first paragraph, second paragraph dropped
        assert info["summary"] == "First para. Same para."

    def test_missing_fields_get_dash_default(self, gen_mod, tmp_path: Path) -> None:
        nb = tmp_path / "d.ipynb"
        _write_nb(nb, demo_info=None, first_md="# T\n\nS.")
        info = gen_mod.extract_demo_info(nb)
        assert info["runtime_offline"] == "—"
        assert info["estimated_cost"] == "—"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_includes_title_summary_and_table(
        self, gen_mod, tmp_path: Path
    ) -> None:
        # Build entries manually to bypass file discovery
        info = {
            "title": "My Demo",
            "summary": "Demo summary.",
            "runtime_offline": "1 s",
            "runtime_live": "2 s",
            "estimated_cost": "$0.001",
            "prerequisites": ["dep1", "dep2"],
        }
        # Path used only for the link target — doesn't need to exist
        out = gen_mod.render_readme([(gen_mod.NOTEBOOKS_DIR / "foo.ipynb", info)])
        assert "# Notebooks" in out
        assert "## [My Demo](foo.ipynb)" in out
        assert "Demo summary." in out
        assert "| Runtime (offline) | 1 s |" in out
        assert "| Estimated cost | $0.001 |" in out
        # Prerequisites comma-joined
        assert "dep1, dep2" in out

    def test_empty_list_yields_placeholder(self, gen_mod) -> None:
        out = gen_mod.render_readme([])
        assert "(no notebooks yet)" in out

    def test_prerequisites_string_passes_through(self, gen_mod) -> None:
        info = {
            "title": "T", "summary": "S",
            "runtime_offline": "—", "runtime_live": "—", "estimated_cost": "—",
            "prerequisites": "just one string",
        }
        out = gen_mod.render_readme(
            [(gen_mod.NOTEBOOKS_DIR / "a.ipynb", info)]
        )
        assert "just one string" in out

    def test_empty_prerequisites_renders_dash(self, gen_mod) -> None:
        info = {
            "title": "T", "summary": "S",
            "runtime_offline": "—", "runtime_live": "—", "estimated_cost": "—",
            "prerequisites": [],
        }
        out = gen_mod.render_readme(
            [(gen_mod.NOTEBOOKS_DIR / "a.ipynb", info)]
        )
        assert "| Prerequisites | — |" in out


# ---------------------------------------------------------------------------
# Repo-state sync
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not README_PATH.exists(),
    reason="notebooks/ is gitignored; README only present in local dev setups",
)
class TestRepoSync:
    def test_readme_exists(self) -> None:
        assert README_PATH.exists(), (
            "notebooks/README.md missing. Run `make notebooks-readme`."
        )

    def test_readme_lists_every_committed_notebook(self, gen_mod) -> None:
        names = [p.name for p in sorted(gen_mod.NOTEBOOKS_DIR.glob("*.ipynb"))]
        body = README_PATH.read_text(encoding="utf-8")
        missing = [n for n in names if n not in body]
        assert not missing, (
            f"README.md is missing entries for: {missing}. "
            "Run `make notebooks-readme`."
        )

    def test_readme_matches_generator_output(self, gen_mod) -> None:
        """Catch contributors who add a notebook without regenerating."""
        nb_files = sorted(gen_mod.NOTEBOOKS_DIR.glob("*.ipynb"))
        entries = [(p, gen_mod.extract_demo_info(p)) for p in nb_files]
        expected = gen_mod.render_readme(entries)
        actual = README_PATH.read_text(encoding="utf-8")
        assert actual == expected, (
            "notebooks/README.md is out of sync with its generator. "
            "Run `make notebooks-readme` to regenerate."
        )
