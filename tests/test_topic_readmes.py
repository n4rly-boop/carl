"""Tests for the per-topic README generator.

The generator at ``scripts/generate_topic_readmes.py`` writes one
``README.md`` per topic submodule under ``tests/`` and ``examples/``,
auto-extracting each file's docstring summary. These tests verify the
extractor, the rendering, and that the on-disk READMEs are in sync
(catches contributors who add files without re-running the generator).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_topic_readmes.py"


@pytest.fixture(scope="module")
def gen_mod():
    """Load the generator script as a module for direct unit testing."""
    spec = importlib.util.spec_from_file_location(
        "generate_topic_readmes", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# extract_first_paragraph
# ---------------------------------------------------------------------------


class TestExtractFirstParagraph:
    def test_single_line_docstring(self, gen_mod, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text('"""Brief summary."""\n')
        assert gen_mod.extract_first_paragraph(f) == "Brief summary."

    def test_multi_line_first_paragraph_collapses(
        self, gen_mod, tmp_path: Path
    ) -> None:
        f = tmp_path / "b.py"
        f.write_text('"""Line one.\nStill the first paragraph."""\n')
        assert gen_mod.extract_first_paragraph(f) == (
            "Line one. Still the first paragraph."
        )

    def test_second_paragraph_dropped(self, gen_mod, tmp_path: Path) -> None:
        f = tmp_path / "c.py"
        f.write_text(
            '"""First paragraph.\n\nSecond paragraph, ignored."""\n'
        )
        assert gen_mod.extract_first_paragraph(f) == "First paragraph."

    def test_no_docstring_returns_empty(self, gen_mod, tmp_path: Path) -> None:
        f = tmp_path / "d.py"
        f.write_text("x = 1\n")
        assert gen_mod.extract_first_paragraph(f) == ""

    def test_syntax_error_returns_empty(self, gen_mod, tmp_path: Path) -> None:
        f = tmp_path / "e.py"
        f.write_text("def broken(:\n")
        assert gen_mod.extract_first_paragraph(f) == ""

    def test_nonexistent_file_returns_empty(
        self, gen_mod, tmp_path: Path
    ) -> None:
        assert gen_mod.extract_first_paragraph(tmp_path / "nope.py") == ""


# ---------------------------------------------------------------------------
# render_readme
# ---------------------------------------------------------------------------


class TestRenderReadme:
    def test_renders_title_blurb_and_file_list(
        self, gen_mod, tmp_path: Path
    ) -> None:
        d = tmp_path / "mytopic"
        d.mkdir()
        (d / "test_one.py").write_text('"""Test one summary."""\n')
        (d / "test_two.py").write_text('"""Test two summary."""\n')
        out = gen_mod.render_readme(d, "Topic blurb.", "Tests")
        assert "# Mytopic Tests" in out
        assert "Topic blurb." in out
        assert "test_one.py" in out
        assert "Test one summary." in out
        assert "test_two.py" in out
        # Generation footer present
        assert "make docs-topic-index" in out

    def test_skips_init_and_dunder_files(self, gen_mod, tmp_path: Path) -> None:
        d = tmp_path / "mytopic"
        d.mkdir()
        (d / "__init__.py").write_text('"""init"""\n')
        (d / "_internal.py").write_text('"""internal"""\n')
        (d / "test_visible.py").write_text('"""visible"""\n')
        out = gen_mod.render_readme(d, "blurb", "Tests")
        assert "test_visible.py" in out
        assert "__init__.py" not in out
        assert "_internal.py" not in out

    def test_acronym_title_override(self, gen_mod, tmp_path: Path) -> None:
        d = tmp_path / "mcp"
        d.mkdir()
        (d / "test_x.py").write_text('"""x"""\n')
        out = gen_mod.render_readme(d, "blurb", "Tests")
        assert "# MCP Tests" in out
        # Not the title-cased "Mcp"
        assert "# Mcp Tests" not in out

    def test_empty_dir_shows_placeholder(self, gen_mod, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        out = gen_mod.render_readme(d, "blurb", "Tests")
        assert "(no files yet)" in out

    def test_missing_docstring_shows_no_description(
        self, gen_mod, tmp_path: Path
    ) -> None:
        d = tmp_path / "x"
        d.mkdir()
        (d / "test_undocumented.py").write_text("x = 1\n")
        out = gen_mod.render_readme(d, "blurb", "Tests")
        assert "(no description)" in out

    def test_files_listed_in_sorted_order(self, gen_mod, tmp_path: Path) -> None:
        d = tmp_path / "x"
        d.mkdir()
        (d / "test_z.py").write_text('"""z"""\n')
        (d / "test_a.py").write_text('"""a"""\n')
        (d / "test_m.py").write_text('"""m"""\n')
        out = gen_mod.render_readme(d, "blurb", "Tests")
        a_idx = out.index("test_a.py")
        m_idx = out.index("test_m.py")
        z_idx = out.index("test_z.py")
        assert a_idx < m_idx < z_idx


# ---------------------------------------------------------------------------
# Repo-state sync: every topic dir should have a current README
# ---------------------------------------------------------------------------


class TestReposReadmesAreInSync:
    @pytest.fixture
    def regenerate_in_tmp(self, gen_mod, tmp_path: Path):
        """Run the generator into a tmp tree mirroring the repo layout."""
        # We don't run against the real repo (that would mutate it); we
        # build a minimal mirror of every topic dir with one stub file
        # each, then run the generator inside.
        tests_mirror = tmp_path / "tests"
        examples_mirror = tmp_path / "examples"
        for topic in gen_mod.TEST_TOPICS:
            d = tests_mirror / topic
            d.mkdir(parents=True)
            (d / "test_stub.py").write_text('"""stub"""\n')
        for topic in gen_mod.EXAMPLE_TOPICS:
            d = examples_mirror / topic
            d.mkdir(parents=True)
            (d / "stub_example.py").write_text('"""stub"""\n')
        gen_mod.regenerate(tests_mirror, gen_mod.TEST_TOPICS, kind="Tests")
        gen_mod.regenerate(
            examples_mirror, gen_mod.EXAMPLE_TOPICS, kind="Examples"
        )
        return tests_mirror, examples_mirror

    def test_every_test_topic_writes_a_readme(
        self, regenerate_in_tmp, gen_mod
    ) -> None:
        tests_mirror, _ = regenerate_in_tmp
        for topic in gen_mod.TEST_TOPICS:
            assert (tests_mirror / topic / "README.md").exists()

    def test_every_example_topic_writes_a_readme(
        self, regenerate_in_tmp, gen_mod
    ) -> None:
        _, examples_mirror = regenerate_in_tmp
        for topic in gen_mod.EXAMPLE_TOPICS:
            assert (examples_mirror / topic / "README.md").exists()

    def test_repo_readmes_match_generator_output(self, gen_mod) -> None:
        """Catch contributors who add files without regenerating.

        We re-render each topic's README into memory and compare against
        the on-disk file. Failure means someone added/removed a file or
        changed a docstring without running ``make docs-topic-index``.
        """
        roots = [
            (REPO_ROOT / "tests", gen_mod.TEST_TOPICS, "Tests"),
            (REPO_ROOT / "examples", gen_mod.EXAMPLE_TOPICS, "Examples"),
        ]
        stale: list[str] = []
        for root, topics, kind in roots:
            for topic_name, blurb in topics.items():
                topic_dir = root / topic_name
                if not topic_dir.is_dir():
                    continue
                expected = gen_mod.render_readme(topic_dir, blurb, kind)
                actual_path = topic_dir / "README.md"
                if not actual_path.exists():
                    stale.append(f"missing: {actual_path.relative_to(REPO_ROOT)}")
                    continue
                actual = actual_path.read_text(encoding="utf-8")
                if actual != expected:
                    stale.append(
                        f"out-of-sync: {actual_path.relative_to(REPO_ROOT)}"
                    )
        assert not stale, (
            "Per-topic README(s) are stale. Run `make docs-topic-index` to "
            "regenerate.\n  " + "\n  ".join(stale)
        )
