"""
SkillLoader — discovers and loads AgentSkills from various sources.

Supports loading from:
  - Local filesystem path
  - Skill name (searched in standard directories)
  - Git repository URL (clone on demand)
  - Installed Python package
"""

import asyncio
import hashlib
import importlib.util
import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .models.agent_skill import AgentSkillSource, SkillManifest


class SkillNotFoundError(Exception):
    """Raised when a skill cannot be located."""


class SkillParseError(Exception):
    """Raised when SKILL.md cannot be parsed."""


def _parse_skill_md(skill_md_path: str) -> dict:
    """
    Parse a SKILL.md file, returning frontmatter fields + instructions body.

    SKILL.md format:
        ---
        name: pdf
        description: Use this skill for PDF files.
        license: Apache-2.0
        ---
        # PDF Processing Guide
        ...instructions...

    Uses stdlib only (no pyyaml dependency).
    """
    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        raise SkillParseError(f"Cannot read {skill_md_path}: {e}") from e

    lines = content.splitlines()

    # Find opening '---'
    if not lines or lines[0].strip() != "---":
        raise SkillParseError(f"SKILL.md at {skill_md_path} does not start with '---' frontmatter delimiter")

    # Find closing '---'
    closing_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_idx = i
            break

    if closing_idx is None:
        raise SkillParseError(f"SKILL.md at {skill_md_path} is missing closing '---' frontmatter delimiter")

    frontmatter_lines = lines[1:closing_idx]
    body_lines = lines[closing_idx + 1:]
    instructions = "\n".join(body_lines).strip()

    # Parse frontmatter leniently (no pyyaml — simple key: value lines)
    # Handles multi-word values, unquoted colons in values, etc.
    frontmatter: dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: list[str] = []

    def _flush_current():
        nonlocal current_key, current_value_lines
        if current_key is not None:
            frontmatter[current_key] = "\n".join(current_value_lines).strip()
        current_key = None
        current_value_lines = []

    for line in frontmatter_lines:
        # Check for key: value pattern (key must start at column 0, no leading space)
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)', line)
        if m:
            _flush_current()
            current_key = m.group(1).lower().replace("-", "_")
            current_value_lines = [m.group(2)]
        elif line.startswith("  ") or line.startswith("\t"):
            # Continuation line (multi-line value)
            if current_key is not None:
                current_value_lines.append(line.strip())
        else:
            # Blank line or comment
            if line.strip().startswith("#"):
                pass  # ignore comment
            elif not line.strip():
                pass  # ignore blank

    _flush_current()

    return {
        "frontmatter": frontmatter,
        "instructions": instructions,
    }


def _discover_resources(skill_dir: str) -> dict[str, list[str]]:
    """
    Walk scripts/, references/, assets/ subdirs and collect relative paths.
    """
    result: dict[str, list[str]] = {"scripts": [], "references": [], "assets": []}
    for subdir in ("scripts", "references", "assets"):
        full_subdir = os.path.join(skill_dir, subdir)
        if os.path.isdir(full_subdir):
            key = subdir
            for root, _, files in os.walk(full_subdir):
                for fname in sorted(files):
                    abs_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(abs_path, skill_dir)
                    result[key].append(rel_path)
    return result


def _build_manifest(skill_dir: str, skill_md_path: str) -> "SkillManifest":
    """Build a SkillManifest from a parsed SKILL.md file."""
    from .models.agent_skill import SkillManifest

    parsed = _parse_skill_md(skill_md_path)
    fm = parsed["frontmatter"]
    instructions = parsed["instructions"]

    name = fm.get("name", "")
    description = fm.get("description", "")

    if not name:
        raise SkillParseError(f"SKILL.md at {skill_md_path} is missing required 'name' field")
    if not description:
        raise SkillParseError(f"SKILL.md at {skill_md_path} is missing required 'description' field")

    # Extract known fields; rest goes to metadata
    known_fields = {"name", "description", "license", "compatibility", "allowed_tools"}
    metadata = {k: v for k, v in fm.items() if k not in known_fields}

    resources = _discover_resources(skill_dir)

    return SkillManifest(
        name=name,
        description=description,
        license=fm.get("license"),
        compatibility=fm.get("compatibility"),
        allowed_tools=fm.get("allowed_tools"),
        metadata=metadata,
        instructions=instructions,
        skill_dir=str(Path(skill_dir).resolve()),
        skill_md_path=str(Path(skill_md_path).resolve()),
        scripts=resources["scripts"],
        references=resources["references"],
        assets=resources["assets"],
    )


class SkillLoader:
    """
    Discovers and loads AgentSkills from various sources.

    Usage:
        loader = SkillLoader()
        manifest = await loader.load(AgentSkillSource(name="pdf"))
        print(manifest.instructions)
    """

    DEFAULT_SEARCH_PATHS = [
        "./.agents/skills",
        "./.claude/skills",
        "~/.agents/skills",
        "~/.claude/skills",
    ]

    def __init__(
        self,
        extra_search_paths: Optional[list[str]] = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self._extra_search_paths: list[str] = extra_search_paths or []
        self._cache_dir = cache_dir or os.path.expanduser("~/.cache/mmar_carl/skills")
        self._enable_cache = enable_cache
        self._memory_cache: dict[str, "SkillManifest"] = {}

    def _get_search_paths(self, extra: Optional[list[str]] = None) -> list[str]:
        """Return ordered search paths (extra first, then defaults)."""
        paths = list(extra or []) + list(self._extra_search_paths) + list(self.DEFAULT_SEARCH_PATHS)
        return paths

    def _resolve_path(self, source: "AgentSkillSource") -> Optional[str]:
        """Resolve a skill from an explicit path."""
        raw = source.path
        if not raw:
            return None
        expanded = os.path.expanduser(raw)
        if not os.path.isabs(expanded):
            expanded = os.path.abspath(expanded)
        skill_md = os.path.join(expanded, "SKILL.md")
        if os.path.isfile(skill_md):
            return expanded
        raise SkillNotFoundError(f"No SKILL.md found at path: {expanded}")

    def _resolve_name(self, source: "AgentSkillSource") -> Optional[str]:
        """
        Search for a skill by name in SKILL.md-based search paths.
        Also tries agent-skills SkillDirectory if the SKILL.md lookup fails.
        """
        name = source.name
        if not name:
            return None

        search_paths = self._get_search_paths(source.search_paths)

        for raw_search_path in search_paths:
            expanded = os.path.expanduser(raw_search_path)
            if not os.path.isabs(expanded):
                expanded = os.path.abspath(expanded)
            candidate = os.path.join(expanded, name)
            skill_md = os.path.join(candidate, "SKILL.md")
            if os.path.isfile(skill_md):
                return candidate

        return None

    @staticmethod
    def _agent_skills_search_paths() -> list[str]:
        """Return directories known to agent-skills SkillDirectory."""
        try:
            dirs: list[str] = []
            for search_path in SkillLoader.DEFAULT_SEARCH_PATHS:
                expanded = os.path.expanduser(search_path)
                if os.path.isdir(expanded):
                    dirs.append(expanded)
            return dirs
        except Exception:
            return []

    def _resolve_git(self, source: "AgentSkillSource") -> Optional[str]:
        """
        Clone a skill from a git repository URL into the cache dir.

        If the URL points to GitHub (``https://github.com/...``), the tarball-based
        :class:`~mmar_carl.skill_resolver.GithubResolver` is used instead of
        ``git clone`` so that no ``git`` binary is required.
        """
        if not source.git_url:
            return None

        # Route GitHub URLs through GithubResolver (tarball, no git required)
        if source.git_url.startswith("https://github.com/"):
            return self._resolve_git_via_github_resolver(source)

        cache_key_str = f"{source.git_url}#{source.git_ref}#{source.git_subdirectory or ''}"
        cache_key = hashlib.sha256(cache_key_str.encode()).hexdigest()[:16]
        clone_dir = os.path.join(self._cache_dir, "git", cache_key)

        if not os.path.isdir(clone_dir):
            os.makedirs(clone_dir, exist_ok=True)
            ref = source.git_ref if source.git_ref and source.git_ref != "HEAD" else None
            cmd = ["git", "clone", "--depth=1"]
            if ref:
                cmd += ["--branch", ref]
            cmd += [source.git_url, clone_dir]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            except subprocess.CalledProcessError as e:
                raise SkillNotFoundError(
                    f"Failed to clone skill from {source.git_url}: {e.stderr.decode()}"
                ) from e
            except FileNotFoundError:
                raise SkillNotFoundError("git is not available on this system") from None

        skill_dir = clone_dir
        if source.git_subdirectory:
            skill_dir = os.path.join(clone_dir, source.git_subdirectory)

        if not os.path.isfile(os.path.join(skill_dir, "SKILL.md")):
            raise SkillNotFoundError(
                f"No SKILL.md found in git clone at {skill_dir} "
                f"(url={source.git_url}, subdirectory={source.git_subdirectory!r})"
            )
        return skill_dir

    def _resolve_git_via_github_resolver(self, source: "AgentSkillSource") -> str:
        """
        Resolve a GitHub ``git_url`` via :class:`~mmar_carl.skill_resolver.GithubResolver`.

        Reconstructs a ``github://owner/repo[/subpath][@ref]`` URI from the
        ``AgentSkillSource`` fields and delegates to ``GithubResolver``, which
        downloads a tarball and caches the extracted tree — no ``git`` binary needed.
        """
        from mmar_carl.skill_resolver import GithubResolver, SkillResolveError

        # Strip "https://github.com/" prefix → "owner/repo"
        git_url: str = source.git_url  # type: ignore[assignment]  # guarded by caller
        path = git_url.removeprefix("https://github.com/").rstrip("/")
        ref = source.git_ref if source.git_ref else "HEAD"
        if source.git_subdirectory:
            uri = f"github://{path}/{source.git_subdirectory}@{ref}"
        else:
            uri = f"github://{path}@{ref}"

        try:
            resolved = GithubResolver().resolve(uri)
        except SkillResolveError as e:
            raise SkillNotFoundError(
                f"Failed to resolve GitHub skill '{source.git_url}' via GithubResolver: {e}"
            ) from e

        return str(resolved.local_root)

    def _resolve_package(self, source: "AgentSkillSource") -> Optional[str]:
        """
        Locate a skill from an installed Python package.
        """
        if not source.package:
            return None

        spec = importlib.util.find_spec(source.package)
        if spec is None or spec.origin is None:
            raise SkillNotFoundError(f"Python package '{source.package}' is not installed")

        pkg_dir = os.path.dirname(spec.origin)

        if source.package_subpath:
            candidate = os.path.join(pkg_dir, source.package_subpath)
        else:
            # Convention: <package>/skills/<name>/SKILL.md
            # or just the package root itself
            candidate = pkg_dir

        if os.path.isfile(os.path.join(candidate, "SKILL.md")):
            return candidate

        # Try <pkg_dir>/skills/<package_name>/
        candidate2 = os.path.join(pkg_dir, "skills", source.package)
        if os.path.isfile(os.path.join(candidate2, "SKILL.md")):
            return candidate2

        raise SkillNotFoundError(
            f"No SKILL.md found in package '{source.package}' at {pkg_dir}"
        )

    async def load(self, source: "AgentSkillSource") -> "SkillManifest":
        """
        Resolve source and return parsed SkillManifest.

        Raises SkillNotFoundError or SkillParseError.
        """
        skill_dir = await asyncio.get_event_loop().run_in_executor(None, self._resolve_sync, source)
        cache_key = str(Path(skill_dir).resolve())

        if self._enable_cache and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        manifest = _build_manifest(skill_dir, skill_md_path)

        if self._enable_cache:
            self._memory_cache[cache_key] = manifest

        return manifest

    def _resolve_sync(self, source: "AgentSkillSource") -> str:
        """Synchronously resolve the skill directory path from source."""
        if source.path is not None:
            result = self._resolve_path(source)
            if result:
                return result
            raise SkillNotFoundError(f"Skill path not found: {source.path}")

        if source.name is not None:
            result = self._resolve_name(source)
            if result:
                return result
            raise SkillNotFoundError(
                f"Skill '{source.name}' not found in any search path. "
                f"Search paths: {self._get_search_paths(source.search_paths)}"
            )

        if source.git_url is not None:
            result = self._resolve_git(source)
            if result:
                return result
            raise SkillNotFoundError(f"Could not resolve skill from git URL: {source.git_url}")

        if source.package is not None:
            result = self._resolve_package(source)
            if result:
                return result
            raise SkillNotFoundError(f"Could not resolve skill from package: {source.package}")

        raise SkillNotFoundError("AgentSkillSource has no resolvable source specified")

    def load_sync(self, source: "AgentSkillSource") -> "SkillManifest":
        """Synchronous wrapper around load()."""
        skill_dir = self._resolve_sync(source)
        cache_key = str(Path(skill_dir).resolve())

        if self._enable_cache and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        manifest = _build_manifest(skill_dir, skill_md_path)

        if self._enable_cache:
            self._memory_cache[cache_key] = manifest

        return manifest

    def catalog(self, search_paths: Optional[list[str]] = None) -> list[tuple[str, str]]:
        """
        Return (name, description) for all skills found in search_paths.

        Implements tier-1 progressive disclosure: catalog without loading full instructions.
        """
        from .models.agent_skill import AgentSkillSource

        results: list[tuple[str, str]] = []
        seen: set[str] = set()

        paths = self._get_search_paths(search_paths)
        for raw_path in paths:
            expanded = os.path.expanduser(raw_path)
            if not os.path.isabs(expanded):
                expanded = os.path.abspath(expanded)
            if not os.path.isdir(expanded):
                continue
            for entry in sorted(os.listdir(expanded)):
                entry_path = os.path.join(expanded, entry)
                skill_md = os.path.join(entry_path, "SKILL.md")
                if os.path.isdir(entry_path) and os.path.isfile(skill_md):
                    if entry in seen:
                        continue
                    seen.add(entry)
                    try:
                        source = AgentSkillSource(name=entry, search_paths=[expanded])
                        manifest = self.load_sync(source)
                        results.append((manifest.name, manifest.description))
                    except Exception:
                        pass

        return results

    def catalog_from_agent_skills(self) -> list[tuple[str, str]]:
        """
        Return (name, description) for skills discoverable via the agent-skills
        SkillDirectory library (if installed).

        This supplements the SKILL.md-based catalog() with any Python-module-based
        skills registered through the agent-skills ecosystem.

        Returns empty list if agent-skills is not installed.
        """
        try:
            from agent_skills import SkillDirectory  # type: ignore
        except ImportError:
            return []

        results: list[tuple[str, str]] = []
        seen: set[str] = set()

        for search_path in self.DEFAULT_SEARCH_PATHS:
            expanded = os.path.expanduser(search_path)
            if not os.path.isdir(expanded):
                continue
            try:
                sd = SkillDirectory(expanded)
                for skill_file in sd.list():
                    name = getattr(skill_file, "name", None) or str(skill_file)
                    desc = getattr(skill_file, "description", "") or ""
                    if name and name not in seen:
                        seen.add(name)
                        results.append((str(name), str(desc)))
            except Exception:
                continue

        return results

    def catalog_all(
        self,
        search_paths: Optional[list[str]] = None,
        *,
        payloads: bool = False,
    ) -> "list[tuple[str, str]] | list[dict]":
        """
        Return combined catalog from both SKILL.md-based and agent-skills sources.

        Deduplicates by name; SKILL.md entries take priority.

        Args:
            search_paths: Optional list of paths to scan for SKILL.md
                directories. Defaults to ``self.DEFAULT_SEARCH_PATHS``.
            payloads: When ``False`` (default) returns the legacy
                ``list[tuple[name, description]]`` shape. When ``True``
, returns
                ``list[dict]`` where each entry is the full
                gigaevo-memory ``agent_skill`` payload from
                :meth:`SkillManifest.to_memory_payload` — used by CARE
                to bulk-ingest every locally-installed skill on first
                run.

        Returns:
            ``list[tuple[str, str]]`` when ``payloads=False`` (sorted
            by skill name), ``list[dict]`` when ``payloads=True``
            (sorted by skill ``name`` key).
        """
        if not payloads:
            skill_md_entries = self.catalog(search_paths)
            agent_entries = self.catalog_from_agent_skills()

            seen = {name for name, _ in skill_md_entries}
            combined = list(skill_md_entries)
            for name, desc in agent_entries:
                if name not in seen:
                    seen.add(name)
                    combined.append((name, desc))
            return sorted(combined, key=lambda x: x[0])

        # payloads=True: return full memory-card-shaped dicts. Reuses
        # the manifest-loading code path from ``catalog`` so disk-only
        # skills (SKILL.md directories) are covered; agent-skills
        # library entries are added only when present.
        from .models.agent_skill import AgentSkillSource  # noqa: PLC0415

        payload_results: list[dict] = []
        seen_names: set[str] = set()

        paths = self._get_search_paths(search_paths)
        for raw_path in paths:
            expanded = os.path.expanduser(raw_path)
            if not os.path.isabs(expanded):
                expanded = os.path.abspath(expanded)
            if not os.path.isdir(expanded):
                continue
            for entry in sorted(os.listdir(expanded)):
                entry_path = os.path.join(expanded, entry)
                skill_md = os.path.join(entry_path, "SKILL.md")
                if not (os.path.isdir(entry_path) and os.path.isfile(skill_md)):
                    continue
                if entry in seen_names:
                    continue
                seen_names.add(entry)
                try:
                    source = AgentSkillSource(
                        name=entry, search_paths=[expanded],
                    )
                    manifest = self.load_sync(source)
                except Exception:
                    continue
                payload_results.append(
                    manifest.to_memory_payload(uri=f"local://{entry_path}"),
                )

        return sorted(payload_results, key=lambda d: d.get("name", ""))

    def clear_cache(self) -> None:
        """Clear in-memory skill cache."""
        self._memory_cache.clear()
