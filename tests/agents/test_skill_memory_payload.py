"""Tests for Skill catalog → gigaevo-memory payload.

`SkillManifest.to_memory_payload()` and `ResolvedSkill.to_memory_payload()`
each produce a dict matching the gigaevo-memory `agent_skill` card
shape. `SkillLoader.catalog_all(*, payloads=True)` returns the same
shape for every locally-installed skill in one call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmar_carl import SkillLoader
from mmar_carl.models.agent_skill import (
    SkillManifest,
    _sanitize_description,
)
from mmar_carl.skill_resolver import ResolvedSkill, SkillFrontmatterData


# ---------------------------------------------------------------------------
# _sanitize_description helper
# ---------------------------------------------------------------------------


class TestSanitizeDescription:
    def test_none_yields_empty(self) -> None:
        assert _sanitize_description(None) == ""

    def test_empty_string_yields_empty(self) -> None:
        assert _sanitize_description("") == ""

    def test_strips_outer_whitespace(self) -> None:
        assert _sanitize_description("  hello  ") == "hello"

    def test_keeps_newlines_and_tabs(self) -> None:
        s = _sanitize_description("line one\nline two\tindented")
        assert "\n" in s
        assert "\t" in s

    def test_strips_other_control_chars(self) -> None:
        # 0x07 = bell, 0x1b = escape
        s = _sanitize_description("hello\x07world\x1b!")
        assert "\x07" not in s
        assert "\x1b" not in s
        # Visible chars preserved
        assert "hello" in s and "world" in s

    def test_caps_at_500_chars(self) -> None:
        s = _sanitize_description("x" * 600)
        assert len(s) <= 500
        assert s.endswith("…")


# ---------------------------------------------------------------------------
# SkillManifest.to_memory_payload
# ---------------------------------------------------------------------------


def _manifest(**overrides) -> SkillManifest:
    defaults = dict(
        name="pdf",
        description="Extract text from PDFs",
        license="MIT",
        compatibility=">=0.1",
        allowed_tools="Bash(git:*) Read Write",
        metadata={"version": "1.0"},
        instructions="# How to use\n\nRead the PDF.",
        skill_dir="/tmp/pdf",
        skill_md_path="/tmp/pdf/SKILL.md",
    )
    defaults.update(overrides)
    return SkillManifest(**defaults)


class TestManifestPayload:
    def test_payload_keys(self) -> None:
        m = _manifest()
        p = m.to_memory_payload(uri="github://x/y@main", sha256="abc")
        assert set(p.keys()) == {
            "name", "description", "uri", "sha256", "manifest",
            "instructions", "allowed_tools", "tags", "compatibility",
            "tarball_url", "tarball_sha256",
        }

    def test_simple_fields_round_trip(self) -> None:
        m = _manifest()
        p = m.to_memory_payload(uri="github://x/y@main", sha256="abc")
        assert p["name"] == "pdf"
        assert p["description"] == "Extract text from PDFs"
        assert p["uri"] == "github://x/y@main"
        assert p["sha256"] == "abc"
        assert p["compatibility"] == ">=0.1"
        assert p["manifest"] == {"version": "1.0"}
        assert p["instructions"].startswith("# How to use")

    def test_allowed_tools_parsed_as_list(self) -> None:
        m = _manifest(allowed_tools="Bash(git:*) Read Write")
        p = m.to_memory_payload()
        assert p["allowed_tools"] == ["Bash(git:*)", "Read", "Write"]

    def test_allowed_tools_empty_when_unset(self) -> None:
        m = _manifest(allowed_tools=None)
        p = m.to_memory_payload()
        assert p["allowed_tools"] == []

    def test_tags_from_comma_separated_string(self) -> None:
        # ``SkillManifest.metadata`` is typed as ``dict[str, str]`` so
        # tags arrive as a comma-separated string from YAML frontmatter.
        m = _manifest(metadata={"tags": "docs, pdf, extraction"})
        p = m.to_memory_payload()
        assert p["tags"] == ["docs", "pdf", "extraction"]

    def test_tags_from_metadata_list_via_resolved_skill(self) -> None:
        """The list-form parsing path is exercised through
        ``ResolvedSkill.to_memory_payload`` where the underlying
        ``SkillFrontmatterData`` dataclass doesn't enforce
        ``dict[str, str]``."""
        # Covered fully in TestResolvedSkillPayload; this assertion
        # documents that the manifest-side parser tolerates a list
        # too when an upstream loader hands one in via a permissive
        # dict (defensive code path).
        from mmar_carl.models.agent_skill import _sanitize_description  # noqa: F401
        # Directly invoke the parsing branch via the same helper.
        m = _manifest()
        m.metadata = {}  # reset
        # Bypass pydantic by patching attribute directly (defensive
        # code path that must not crash if a list sneaks through).
        object.__setattr__(m, "metadata", {"tags": ["x", "y"]})
        p = m.to_memory_payload()
        assert p["tags"] == ["x", "y"]

    def test_tags_empty_when_missing(self) -> None:
        m = _manifest(metadata={})
        p = m.to_memory_payload()
        assert p["tags"] == []

    def test_optional_args_default_to_none(self) -> None:
        m = _manifest()
        p = m.to_memory_payload()
        assert p["uri"] is None
        assert p["sha256"] is None
        assert p["tarball_url"] is None
        assert p["tarball_sha256"] is None

    def test_tarball_fields_passed_through(self) -> None:
        m = _manifest()
        p = m.to_memory_payload(
            tarball_url="https://codeload.github.com/x/tar.gz",
            tarball_sha256="def456",
        )
        assert p["tarball_url"] == "https://codeload.github.com/x/tar.gz"
        assert p["tarball_sha256"] == "def456"

    def test_long_description_truncated(self) -> None:
        m = _manifest(description="x" * 700)
        p = m.to_memory_payload()
        assert len(p["description"]) <= 500
        assert p["description"].endswith("…")


# ---------------------------------------------------------------------------
# ResolvedSkill.to_memory_payload
# ---------------------------------------------------------------------------


def _resolved(**overrides) -> ResolvedSkill:
    fm = SkillFrontmatterData(
        name="pdf",
        description="Extract text",
        license="MIT",
        compatibility=">=0.1",
        allowed_tools="Bash Read",
        metadata={"tags": ["documents"]},
    )
    defaults = dict(
        name="pdf",
        local_root=Path("/tmp/pdf"),
        skill_md_path=Path("/tmp/pdf/SKILL.md"),
        sha256="abc123",
        resolved_version="anthropics/skills@main",
        frontmatter=fm,
        instructions="# Body",
        source_uri="github://anthropics/skills/skills/pdf@main",
        tarball_url="https://codeload.github.com/anthropics/skills/tar.gz/main",
        tarball_sha256="def456",
    )
    defaults.update(overrides)
    return ResolvedSkill(**defaults)


class TestResolvedSkillPayload:
    def test_payload_keys_match_manifest_shape(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert set(p.keys()) == {
            "name", "description", "uri", "sha256", "manifest",
            "instructions", "allowed_tools", "tags", "compatibility",
            "tarball_url", "tarball_sha256",
        }

    def test_uri_from_source_uri_field(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert p["uri"] == "github://anthropics/skills/skills/pdf@main"

    def test_sha256_from_resolved_skill(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert p["sha256"] == "abc123"

    def test_tarball_fields_carried_through(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert p["tarball_url"].endswith("/main")
        assert p["tarball_sha256"] == "def456"

    def test_allowed_tools_parsed_from_frontmatter(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert p["allowed_tools"] == ["Bash", "Read"]

    def test_tags_from_frontmatter_metadata(self) -> None:
        rs = _resolved()
        p = rs.to_memory_payload()
        assert p["tags"] == ["documents"]

    def test_optional_uri_and_tarball_default_none(self) -> None:
        """A local skill resolved without a URI should produce a
        payload with ``uri=None`` etc."""
        rs = _resolved(source_uri=None, tarball_url=None, tarball_sha256=None)
        p = rs.to_memory_payload()
        assert p["uri"] is None
        assert p["tarball_url"] is None
        assert p["tarball_sha256"] is None


# ---------------------------------------------------------------------------
# SkillLoader.catalog_all(payloads=True)
# ---------------------------------------------------------------------------


class TestCatalogAllPayloads:
    def test_returns_list_of_dicts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Patch defaults so we only see our test skill
        monkeypatch.setattr(SkillLoader, "DEFAULT_SEARCH_PATHS", [])

        skill_dir = tmp_path / "mytool"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: mytool\ndescription: A test skill\ntags: dev, test\n---\n"
            "# Body\nUse this tool to do stuff.\n"
        )

        loader = SkillLoader()
        payloads = loader.catalog_all(
            search_paths=[str(tmp_path)], payloads=True,
        )
        assert isinstance(payloads, list)
        assert len(payloads) == 1
        p = payloads[0]
        assert p["name"] == "mytool"
        assert p["description"] == "A test skill"
        assert p["tags"] == ["dev", "test"]
        assert p["uri"] is not None  # local:// URI synthesised by catalog

    def test_legacy_default_shape_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SkillLoader, "DEFAULT_SEARCH_PATHS", [])

        skill_dir = tmp_path / "mytool"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: mytool\ndescription: A test skill\n---\n# Body"
        )
        loader = SkillLoader()
        tuples = loader.catalog_all(search_paths=[str(tmp_path)])
        assert isinstance(tuples, list)
        assert all(isinstance(t, tuple) for t in tuples)
        assert tuples == [("mytool", "A test skill")]

    def test_sorts_payloads_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SkillLoader, "DEFAULT_SEARCH_PATHS", [])
        for name in ("zebra", "alpha", "mango"):
            d = tmp_path / name
            d.mkdir()
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: A skill\n---\n# Body"
            )
        loader = SkillLoader()
        payloads = loader.catalog_all(
            search_paths=[str(tmp_path)], payloads=True,
        )
        names = [p["name"] for p in payloads]
        assert names == sorted(names)

    def test_skill_with_broken_manifest_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SkillLoader, "DEFAULT_SEARCH_PATHS", [])

        ok = tmp_path / "ok"
        ok.mkdir()
        (ok / "SKILL.md").write_text(
            "---\nname: ok\ndescription: works\n---\n# Body"
        )
        broken = tmp_path / "broken"
        broken.mkdir()
        # Malformed YAML frontmatter
        (broken / "SKILL.md").write_text(
            "---\nname:\n  - invalid\n: yaml::\n---\n"
        )

        loader = SkillLoader()
        payloads = loader.catalog_all(
            search_paths=[str(tmp_path)], payloads=True,
        )
        # The good skill survives; the broken one is silently dropped
        names = [p["name"] for p in payloads]
        assert "ok" in names

    def test_empty_search_path_yields_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(SkillLoader, "DEFAULT_SEARCH_PATHS", [])
        loader = SkillLoader()
        payloads = loader.catalog_all(
            search_paths=[str(tmp_path)], payloads=True,
        )
        assert payloads == []
