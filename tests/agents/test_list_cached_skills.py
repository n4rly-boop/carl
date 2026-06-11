"""``SkillResolverRegistry.list_cached()`` enumerates
every skill currently in the resolver cache.

CARE's catalog screen reads this list to render the local inventory
(sizes, last-used times, source labels) without re-downloading anything.
The test stubs out the cache directory with ``tmp_path`` so we don't
touch the user's real ``~/.cache/mmar_carl/skills`` and stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmar_carl import (
    ResolvedSkill,
    SkillResolverRegistry,
    list_cached_skills,
)


# ---------------------------------------------------------------------------
# Cache fixtures
# ---------------------------------------------------------------------------


def _write_skill(parent: Path, *, name: str, description: str = "test skill") -> Path:
    """Materialise a minimal valid skill directory under ``parent``."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n"
        f"# {name}\n\nDo the thing.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Empty cache root the registry will use."""
    root = tmp_path / "cache"
    root.mkdir()
    return root


@pytest.fixture
def registry(cache_dir: Path) -> SkillResolverRegistry:
    return SkillResolverRegistry(cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestListCached:
    def test_empty_cache_returns_empty_list(
        self, registry: SkillResolverRegistry,
    ) -> None:
        assert registry.list_cached() == []

    def test_nonexistent_cache_dir_returns_empty_list(
        self, tmp_path: Path,
    ) -> None:
        # Point at a path that doesn't exist yet.
        reg = SkillResolverRegistry(cache_dir=tmp_path / "does_not_exist")
        assert reg.list_cached() == []

    def test_lists_single_github_cached_skill(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        _write_skill(
            cache_dir / "github" / "abcdef1234567890",
            name="pdf", description="extract text",
        )
        cached = registry.list_cached()
        assert len(cached) == 1
        skill = cached[0]
        assert isinstance(skill, ResolvedSkill)
        assert skill.name == "pdf"
        assert skill.frontmatter.description == "extract text"
        # The resolved_version is scheme-tagged for cache entries.
        assert skill.resolved_version == "cache:github"

    def test_lists_https_and_github_combined(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        _write_skill(cache_dir / "github" / "key1", name="alpha")
        _write_skill(cache_dir / "https" / "key2", name="beta")
        cached = registry.list_cached()
        names = [s.name for s in cached]
        assert sorted(names) == ["alpha", "beta"]
        # Scheme tag survives.
        schemes = {s.name: s.resolved_version for s in cached}
        assert schemes["alpha"] == "cache:github"
        assert schemes["beta"] == "cache:https"

    def test_results_sorted_by_name_case_insensitively(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        _write_skill(cache_dir / "github" / "k1", name="Zeta")
        _write_skill(cache_dir / "github" / "k2", name="alpha")
        _write_skill(cache_dir / "github" / "k3", name="Mu")
        cached = registry.list_cached()
        assert [s.name for s in cached] == ["alpha", "Mu", "Zeta"]

    def test_broken_skill_md_is_skipped_not_fatal(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        # One valid skill + one with a malformed frontmatter.
        _write_skill(cache_dir / "github" / "ok", name="good")
        bad_dir = cache_dir / "github" / "bad"
        bad_dir.mkdir(parents=True)
        # Header without closing `---` line + missing required fields:
        # parser should reject this.
        (bad_dir / "SKILL.md").write_text(
            "---\nname:\n",  # broken — yaml-ish but unclosed
            encoding="utf-8",
        )
        cached = registry.list_cached()
        names = [s.name for s in cached]
        # Listing didn't crash; "good" survived.
        assert "good" in names

    def test_skill_md_in_subpath_is_found(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        """The github cache often stores skills under
        ``<key>/<subpath>/SKILL.md`` when the URI has a subpath."""
        _write_skill(
            cache_dir / "github" / "key1" / "skills" / "pdf",
            name="pdf",
        )
        cached = registry.list_cached()
        names = [s.name for s in cached]
        assert names == ["pdf"]
        # Local root points at the skill dir, not the cache root.
        assert cached[0].local_root.name == "pdf"

    def test_dedup_when_same_skill_dir_resolved_twice(
        self, registry: SkillResolverRegistry, cache_dir: Path,
    ) -> None:
        """A duplicate ``SKILL.md`` at the same path shouldn't surface
        twice (defensive against symlinks / re-extraction races)."""
        skill_dir = _write_skill(cache_dir / "github" / "k1", name="solo")
        # Touch a second time — same path, same content.
        (skill_dir / "SKILL.md").write_text(
            (skill_dir / "SKILL.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        cached = registry.list_cached()
        assert [s.name for s in cached] == ["solo"]


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


class TestTopLevelHelper:
    def test_list_cached_skills_uses_supplied_cache_dir(
        self, tmp_path: Path,
    ) -> None:
        cache = tmp_path / "user_cache"
        cache.mkdir()
        _write_skill(cache / "github" / "key1", name="convenient")
        # Pass cache_dir explicitly so the helper builds a fresh
        # registry against our tmp path instead of touching the
        # process-wide default.
        cached = list_cached_skills(cache_dir=cache)
        assert [s.name for s in cached] == ["convenient"]

    def test_list_cached_skills_empty_when_cache_dir_missing(
        self, tmp_path: Path,
    ) -> None:
        cached = list_cached_skills(cache_dir=tmp_path / "no_such_dir")
        assert cached == []


# ---------------------------------------------------------------------------
# cache_dir property
# ---------------------------------------------------------------------------


def test_registry_cache_dir_property(tmp_path: Path) -> None:
    """The ``cache_dir`` property surfaces the resolver's root so CARE
    can compute disk usage / last-used-times alongside the listing."""
    cache = tmp_path / "user_cache"
    reg = SkillResolverRegistry(cache_dir=cache)
    assert reg.cache_dir == cache.expanduser()
