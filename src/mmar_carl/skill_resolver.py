"""
SkillResolver — URI-based skill resolution for CARL.

Resolves AgentSkill directories from various sources using URI syntax:
  - local:///abs/path/to/skill  (or plain filesystem paths / skill names)
  - github://owner/repo[/subpath][@ref]  — tarball fetch, SHA-verified cache
  - https://host/skill.tar.gz[#sha256=<hex>]  — generic tarball
  - module://python.pkg.name  — importlib-based package lookup

All resolvers cache results locally so repeat executions are fast.
GitHub tarballs are cached by (owner/repo/subpath@ref) hash.

Examples:
    from mmar_carl.skill_resolver import resolve_skill

    # Official Anthropic PDF skill
    skill = resolve_skill("github://anthropics/skills/skills/pdf@main")

    # Local skill dir
    skill = resolve_skill("local:///home/user/my-skills/summarize")

    # Generic HTTPS tarball with integrity pin
    skill = resolve_skill(
        "https://example.com/skill.tar.gz#sha256=abc123...",
        trust_policy="sha_pinned",
    )
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SkillResolveError(Exception):
    """Raised when a skill URI cannot be resolved."""


class SkillIntegrityError(Exception):
    """Raised when a skill fails SHA256 verification."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SkillFrontmatterData:
    """Parsed SKILL.md frontmatter fields (lightweight version for ResolvedSkill)."""

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ResolvedSkill:
    """
    A skill that has been resolved to a local directory.

    ``local_root`` is always an absolute path where ``SKILL.md`` lives.
    ``sha256`` is the hex digest of the SKILL.md content — useful for
    cache keying and trust verification at the individual file level.
    ``resolved_version`` records where the skill came from (git ref, URL, "local").
    """

    name: str
    local_root: Path
    skill_md_path: Path
    sha256: str               # SHA256 of SKILL.md content
    resolved_version: str     # "local", "owner/repo@ref", URL, etc.
    frontmatter: SkillFrontmatterData
    instructions: str
    scripts: list[Path] = field(default_factory=list)
    references: list[Path] = field(default_factory=list)
    assets: list[Path] = field(default_factory=list)
    source_uri: Optional[str] = None  # original URI passed to resolve_skill, if known
    tarball_url: Optional[str] = None  # github:// origin info
    tarball_sha256: Optional[str] = None

    # ------------------------------------------------------------------
    # gigaevo-memory payload helper
    # ------------------------------------------------------------------

    def to_memory_payload(self) -> dict[str, Any]:
        """Compact dict suitable for a gigaevo-memory ``agent_skill`` card.

        Mirrors :meth:`SkillManifest.to_memory_payload` but pulls
        manifest data from ``self.frontmatter`` and seeds ``uri``,
        ``sha256``, and tarball fields from the resolver state.
        """
        from .models.agent_skill import _sanitize_description  # noqa: PLC0415

        fm = self.frontmatter
        raw_tags = fm.metadata.get("tags") if fm.metadata else None
        tags: list[str]
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags if t is not None]
        elif isinstance(raw_tags, str) and raw_tags.strip():
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        else:
            tags = []

        # Parse `allowed-tools` from the raw frontmatter string the same
        # way SkillManifest.get_allowed_tools does.
        allowed_tools: list[str] = (
            fm.allowed_tools.split() if fm.allowed_tools else []
        )

        return {
            "name": fm.name or self.name,
            "description": _sanitize_description(fm.description),
            "uri": self.source_uri,
            "sha256": self.sha256,
            "manifest": dict(fm.metadata) if fm.metadata else {},
            "instructions": self.instructions,
            "allowed_tools": allowed_tools,
            "tags": tags,
            "compatibility": fm.compatibility,
            "tarball_url": self.tarball_url,
            "tarball_sha256": self.tarball_sha256,
        }


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------


def _parse_github_uri(uri: str) -> tuple[str, str, str, str]:
    """
    Parse ``github://owner/repo[/subpath][@ref]`` into
    ``(owner, repo, subpath, ref)``.

    Examples
    --------
    ``github://anthropics/skills/skills/pdf``
        → ``("anthropics", "skills", "skills/pdf", "main")``
    ``github://anthropics/skills/skills/pptx@v2``
        → ``("anthropics", "skills", "skills/pptx", "v2")``
    ``github://badlogic/pi-skills/brave-search@HEAD``
        → ``("badlogic", "pi-skills", "brave-search", "HEAD")``
    """
    path = uri.removeprefix("github://")

    ref = "main"
    if "@" in path:
        path, ref = path.rsplit("@", 1)

    parts = path.split("/", 2)
    if len(parts) < 2:
        raise SkillResolveError(
            f"Invalid github:// URI — need at least 'github://owner/repo': {uri}"
        )

    owner = parts[0]
    repo = parts[1]
    subpath = parts[2] if len(parts) > 2 else ""

    return owner, repo, subpath, ref


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


class LocalResolver:
    """Resolve a skill from a local filesystem path or skill name."""

    scheme = "local"

    def resolve(self, uri_or_path: str, *, sha256: Optional[str] = None) -> ResolvedSkill:
        if uri_or_path.startswith("local://"):
            path_str = uri_or_path.removeprefix("local://")
            # "local:///abs/path" → "/abs/path"
            if path_str.startswith("//"):
                path_str = path_str[1:]  # keep one leading slash
        else:
            path_str = uri_or_path

        skill_dir = Path(os.path.expanduser(path_str)).resolve()
        if not (skill_dir / "SKILL.md").is_file():
            raise SkillResolveError(f"No SKILL.md found at {skill_dir}")

        return _build_resolved_skill(skill_dir, resolved_version="local", expected_sha256=sha256)


class GithubResolver:
    """
    Resolve a skill from a GitHub repository via tarball download.

    Tarball URL: ``https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}``

    Cache layout: ``cache_dir/github/<key16>/<subpath-or-root>/``
    where ``key16 = SHA256(owner/repo/subpath@ref)[:16]``.

    The cache is permanent (no TTL) — pass ``force_refresh=True`` to re-download.
    """

    scheme = "github"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = (cache_dir or Path("~/.cache/mmar_carl/skills")).expanduser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        uri: str,
        *,
        sha256: Optional[str] = None,
        trust_policy: str = "any",
        force_refresh: bool = False,
    ) -> ResolvedSkill:
        """
        Resolve a ``github://`` URI, downloading and caching the tarball.

        Parameters
        ----------
        uri:
            ``github://owner/repo[/subpath][@ref]``
        sha256:
            Expected SHA256 of the SKILL.md file after extraction.
            Required when ``trust_policy="sha_pinned"``.
        trust_policy:
            ``"any"``        — no verification (default).
            ``"sha_pinned"`` — ``sha256`` must be provided and must match.
        force_refresh:
            Re-download even if already cached.
        """
        owner, repo, subpath, ref = _parse_github_uri(uri)

        # Stable cache key for this (owner, repo, subpath, ref) combo
        cache_key = hashlib.sha256(
            f"{owner}/{repo}/{subpath}@{ref}".encode()
        ).hexdigest()[:16]
        cache_base = self.cache_dir / "github" / cache_key

        # Fast-path: already cached
        if not force_refresh and cache_base.is_dir():
            skill_dir = cache_base / subpath if subpath else cache_base
            if (skill_dir / "SKILL.md").is_file():
                return _build_resolved_skill(
                    skill_dir,
                    resolved_version=f"{owner}/{repo}@{ref}",
                    expected_sha256=sha256 if trust_policy == "sha_pinned" else None,
                )

        # Download
        tarball_bytes = self._download_tarball(owner, repo, ref)

        # Note: sha256 is verified at the SKILL.md level after extraction
        # (see _build_resolved_skill), not at the tarball level. This means
        # sha256 refers to the SKILL.md digest, not the tarball digest.

        # Extract
        cache_base.mkdir(parents=True, exist_ok=True)
        self._extract_tarball(tarball_bytes, cache_base, subpath)

        skill_dir = cache_base / subpath if subpath else cache_base
        if not (skill_dir / "SKILL.md").is_file():
            raise SkillResolveError(
                f"SKILL.md not found at '{skill_dir}' after extracting {uri}. "
                f"Is the subpath '{subpath}' correct?"
            )

        return _build_resolved_skill(
            skill_dir,
            resolved_version=f"{owner}/{repo}@{ref}",
            expected_sha256=sha256 if trust_policy == "sha_pinned" else None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_tarball(self, owner: str, repo: str, ref: str) -> bytes:
        try:
            import httpx
        except ImportError as exc:
            raise SkillResolveError(
                "httpx is required for the GitHub resolver. "
                "Install with: pip install httpx"
            ) from exc

        primary_url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}"
        fallback_url = f"https://github.com/{owner}/{repo}/archive/{ref}.tar.gz"

        with httpx.Client(follow_redirects=True, timeout=120) as client:
            for url in (primary_url, fallback_url):
                try:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return resp.content
                except httpx.RequestError:
                    pass

        raise SkillResolveError(
            f"Could not download tarball for {owner}/{repo}@{ref}. "
            f"Tried: {primary_url}, {fallback_url}"
        )

    def _extract_tarball(self, tarball_bytes: bytes, dest: Path, subpath: str) -> None:
        """
        Extract tarball into ``dest``, stripping the GitHub top-level wrapper dir.

        GitHub tarballs wrap everything in ``{owner}-{repo}-{sha}/``, e.g.
        ``anthropics-skills-abc123/skills/pdf/SKILL.md``. We strip that prefix
        so the content lands directly at ``dest/``.
        """
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            members = tar.getmembers()
            if not members:
                raise SkillResolveError("Downloaded tarball is empty")

            # Determine common wrapper prefix (first path component)
            prefix = members[0].name.split("/")[0] + "/"

            for member in members:
                if not member.name.startswith(prefix):
                    continue
                rel = member.name[len(prefix):]
                if not rel:
                    continue
                # If a subpath is requested, skip unrelated files
                if subpath and not rel.startswith(subpath) and rel != subpath:
                    continue

                target = dest / rel
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    f = tar.extractfile(member)
                    if f is not None:
                        target.write_bytes(f.read())


class HttpsTarballResolver:
    """
    Resolve a skill from a generic HTTPS tarball URL.

    URI format: ``https://host/skill.tar.gz`` or
    ``https://host/skill.tar.gz#sha256=<hex>``
    """

    scheme = "https"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = (cache_dir or Path("~/.cache/mmar_carl/skills")).expanduser()

    def resolve(
        self,
        url: str,
        *,
        sha256: Optional[str] = None,
        trust_policy: str = "any",
    ) -> ResolvedSkill:
        # Extract sha256 from URL fragment if present
        if "#sha256=" in url:
            url, frag_sha256 = url.split("#sha256=", 1)
            sha256 = sha256 or frag_sha256

        if trust_policy == "sha_pinned" and not sha256:
            raise SkillIntegrityError(
                f"trust_policy='sha_pinned' requires skill_sha256 for HTTPS URL: {url}"
            )

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_dir = self.cache_dir / "https" / url_hash

        if cache_dir.is_dir() and (cache_dir / "SKILL.md").is_file():
            return _build_resolved_skill(
                cache_dir,
                resolved_version=url,
                expected_sha256=sha256 if trust_policy == "sha_pinned" else None,
            )

        try:
            import httpx
        except ImportError as exc:
            raise SkillResolveError(
                "httpx is required for the HTTPS tarball resolver: pip install httpx"
            ) from exc

        with httpx.Client(follow_redirects=True, timeout=120) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise SkillResolveError(f"Failed to download {url} (HTTP {resp.status_code})")
            content = resp.content

        actual_sha256 = hashlib.sha256(content).hexdigest()
        if sha256 and trust_policy == "sha_pinned" and actual_sha256 != sha256:
            raise SkillIntegrityError(
                f"Tarball SHA256 mismatch for {url}: expected {sha256}, got {actual_sha256}"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(str(cache_dir))
        else:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
                tar.extractall(str(cache_dir))

        # SKILL.md might be nested — find it
        skill_md = next(cache_dir.rglob("SKILL.md"), None)
        if skill_md is None:
            raise SkillResolveError(f"SKILL.md not found in downloaded archive from {url}")

        return _build_resolved_skill(
            skill_md.parent,
            resolved_version=url,
            expected_sha256=sha256 if trust_policy == "sha_pinned" else None,
        )


class ModuleResolver:
    """Resolve a skill embedded in an installed Python package."""

    scheme = "module"

    def resolve(self, uri: str, *, sha256: Optional[str] = None) -> ResolvedSkill:
        import importlib.util

        pkg_name = uri.removeprefix("module://")

        spec = importlib.util.find_spec(pkg_name)
        if spec is None or spec.origin is None:
            raise SkillResolveError(f"Python package '{pkg_name}' is not installed")

        pkg_dir = Path(os.path.dirname(spec.origin))

        for candidate in [
            pkg_dir,
            pkg_dir / "skills" / pkg_name.split(".")[-1],
        ]:
            if (candidate / "SKILL.md").is_file():
                return _build_resolved_skill(
                    candidate, resolved_version=f"module:{pkg_name}", expected_sha256=sha256
                )

        raise SkillResolveError(
            f"SKILL.md not found in package '{pkg_name}' at {pkg_dir}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class SkillResolverRegistry:
    """
    Dispatches URI strings to the appropriate resolver by scheme.

    Built-in scheme map::

        local:// or plain paths  → LocalResolver
        github://                → GithubResolver
        https:// / http://       → HttpsTarballResolver
        module://                → ModuleResolver
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        _cache = (cache_dir or Path("~/.cache/mmar_carl/skills")).expanduser()
        self._local = LocalResolver()
        self._github = GithubResolver(cache_dir=_cache)
        self._https = HttpsTarballResolver(cache_dir=_cache)
        self._module = ModuleResolver()

    def resolve(
        self,
        uri: str,
        *,
        sha256: Optional[str] = None,
        trust_policy: str = "any",
        force_refresh: bool = False,
    ) -> ResolvedSkill:
        """
        Resolve ``uri`` to a :class:`ResolvedSkill`.

        Dispatches to the correct resolver based on the URI scheme.
        Plain strings without a recognized scheme are treated as local paths.
        """
        if uri.startswith("github://"):
            return self._github.resolve(
                uri, sha256=sha256, trust_policy=trust_policy, force_refresh=force_refresh
            )
        if uri.startswith("https://") or uri.startswith("http://"):
            return self._https.resolve(uri, sha256=sha256, trust_policy=trust_policy)
        if uri.startswith("local://"):
            return self._local.resolve(uri, sha256=sha256)
        if uri.startswith("module://"):
            return self._module.resolve(uri, sha256=sha256)
        # Plain path or skill name → local
        return self._local.resolve(uri, sha256=sha256)

    # ------------------------------------------------------------------
    # list every skill currently in the resolver cache.
    # ------------------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        """Root directory backing GitHub + HTTPS tarball caches."""
        return self._github.cache_dir

    def list_cached(self) -> list[ResolvedSkill]:
        """Enumerate every skill currently cached on disk.

        Walks ``~/.cache/mmar_carl/skills/{github,https}/`` (or the
        registry's overridden ``cache_dir``) and rebuilds a
        :class:`ResolvedSkill` for each directory that contains a
        ``SKILL.md``. Used by CARE's catalog screen to render the
        resolver's local inventory with sizes and last-used times.

        Returns
        -------
        list[ResolvedSkill]
            One entry per cached skill, sorted by ``name``. Empty when
            the cache directory does not exist or contains no skills.
            Directories with broken / unparseable ``SKILL.md`` are
            silently skipped (same convention as
            :meth:`SkillLoader.catalog_all`).
        """
        cache_root = self.cache_dir
        if not cache_root.is_dir():
            return []

        results: list[ResolvedSkill] = []
        seen: set[Path] = set()
        for skill_md in cache_root.rglob("SKILL.md"):
            skill_dir = skill_md.parent.resolve()
            if skill_dir in seen:
                continue
            seen.add(skill_dir)
            # Tag the resolved version by which scheme subdir we're in.
            try:
                rel = skill_dir.relative_to(cache_root)
            except ValueError:
                rel = None
            scheme = rel.parts[0] if rel and rel.parts else "cache"
            try:
                skill = _build_resolved_skill(
                    skill_dir, resolved_version=f"cache:{scheme}",
                )
            except (SkillResolveError, SkillIntegrityError):
                # Broken cache entry — skip rather than crash the
                # whole listing.
                continue
            results.append(skill)

        results.sort(key=lambda s: s.name.lower())
        return results


# Module-level default registry (lazily created)
_DEFAULT_REGISTRY: Optional[SkillResolverRegistry] = None


def get_default_registry(cache_dir: Optional[Path] = None) -> SkillResolverRegistry:
    """Return the process-wide default :class:`SkillResolverRegistry`."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None or cache_dir is not None:
        _DEFAULT_REGISTRY = SkillResolverRegistry(cache_dir=cache_dir)
    return _DEFAULT_REGISTRY


def list_cached_skills(
    *, cache_dir: Optional[Path] = None,
) -> list[ResolvedSkill]:
    """Top-level convenience: list every skill currently in the resolver cache.

    Thin wrapper over :meth:`SkillResolverRegistry.list_cached`. Pass
    ``cache_dir`` to enumerate a non-default cache root (useful in tests).
    """
    return get_default_registry(cache_dir=cache_dir).list_cached()


def resolve_skill(
    uri: str,
    *,
    sha256: Optional[str] = None,
    trust_policy: str = "any",
    cache_dir: Optional[Path] = None,
    force_refresh: bool = False,
) -> ResolvedSkill:
    """
    Top-level convenience function: resolve a skill URI to a local directory.

    Parameters
    ----------
    uri:
        Skill URI.  Examples::

            "github://anthropics/skills/skills/pdf@main"
            "local:///home/user/skills/summarize"
            "/abs/path/to/skill"
            "https://example.com/my-skill.tar.gz#sha256=abc..."
            "module://my_package.skills.pdf"

    sha256:
        Expected SHA256 of the SKILL.md file.  When ``trust_policy="sha_pinned"``
        this is required and a mismatch raises :class:`SkillIntegrityError`.
    trust_policy:
        ``"any"`` (default) or ``"sha_pinned"``.
    cache_dir:
        Override the default cache directory (``~/.cache/mmar_carl/skills``).
    force_refresh:
        Re-download even if already cached (GitHub / HTTPS resolvers).

    Returns
    -------
    ResolvedSkill
        Struct with ``local_root`` pointing to the skill directory.

    Raises
    ------
    SkillResolveError
        The skill could not be located or downloaded.
    SkillIntegrityError
        The SHA256 digest did not match (only raised when ``trust_policy="sha_pinned"``).
    """
    registry = get_default_registry(cache_dir=cache_dir)
    return registry.resolve(
        uri, sha256=sha256, trust_policy=trust_policy, force_refresh=force_refresh
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_resolved_skill(
    skill_dir: Path,
    *,
    resolved_version: str,
    expected_sha256: Optional[str] = None,
) -> ResolvedSkill:
    """
    Parse a local skill directory and return a :class:`ResolvedSkill`.

    Raises :class:`SkillIntegrityError` if ``expected_sha256`` is provided
    and does not match the actual SHA256 of the SKILL.md file.
    """
    # Lazy import to avoid circular dependency
    from .skill_loader import _parse_skill_md, _discover_resources

    skill_dir = skill_dir.resolve()
    skill_md_path = skill_dir / "SKILL.md"

    if not skill_md_path.is_file():
        raise SkillResolveError(f"SKILL.md not found at {skill_dir}")

    actual_sha256 = _sha256_of_file(skill_md_path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise SkillIntegrityError(
            f"SKILL.md SHA256 mismatch at {skill_dir}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        parsed = _parse_skill_md(str(skill_md_path))
    except Exception as exc:
        raise SkillResolveError(
            f"Failed to parse SKILL.md at {skill_md_path}: {exc}"
        ) from exc

    fm = parsed["frontmatter"]
    known = {"name", "description", "license", "compatibility", "allowed_tools"}
    frontmatter = SkillFrontmatterData(
        name=fm.get("name", skill_dir.name),
        description=fm.get("description", ""),
        license=fm.get("license"),
        compatibility=fm.get("compatibility"),
        allowed_tools=fm.get("allowed_tools"),
        metadata={k: v for k, v in fm.items() if k not in known},
    )

    resources = _discover_resources(str(skill_dir))

    return ResolvedSkill(
        name=frontmatter.name or skill_dir.name,
        local_root=skill_dir,
        skill_md_path=skill_md_path,
        sha256=actual_sha256,
        resolved_version=resolved_version,
        frontmatter=frontmatter,
        instructions=parsed["instructions"],
        scripts=[skill_dir / s for s in resources["scripts"]],
        references=[skill_dir / r for r in resources["references"]],
        assets=[skill_dir / a for a in resources["assets"]],
    )
