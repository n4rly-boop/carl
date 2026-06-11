"""Pluggable execution backends for AgentSkill scripts.

The CARE ecosystem needs to run untrusted AgentSkill
scripts in real sandboxes (Docker / E2B / Firejail) instead of the
default host-level subprocess execution. Today
``AgentSkillStepConfig.runtime`` is a declarative string field that no
code branched on; this module turns it into a real protocol with a
registry so CARE can plug in additional sandboxes without forking CARL.

Reference backends shipped here:

* :class:`LocalSkillRuntime` — wraps the existing
  ``asyncio.create_subprocess_exec`` behaviour on the host (no
  isolation). Always available. Emits a one-time warning on first use
  so callers know they are not sandboxed.

Plus the registration plumbing (:func:`register_skill_runtime`,
:func:`get_skill_runtime`) and the typed error
(:class:`SkillRuntimeError`) raised when the requested runtime can't
be loaded.

DockerSkillRuntime, E2BSkillRuntime, and FirejailSkillRuntime will
land in follow-up patches under ``carl[docker]`` / ``carl[e2b]`` extras
— they implement the same Protocol so callers can swap backends by
flipping :attr:`AgentSkillStepConfig.runtime`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network policy contract
# ---------------------------------------------------------------------------

NetworkPolicy = Literal["none", "allowlist", "host"]
"""Network access policy for a skill runtime invocation.

* ``"none"`` (default) — no egress. Containerised backends drop all
  outbound networking (``--network none`` in Docker); local/firejail
  backends document the restriction but can't enforce it.
* ``"allowlist"`` — backend creates a per-task egress filter that only
  allows traffic to hosts listed in
  ``runtime_config["network_allowlist"]``. Useful when a skill's
  ``SKILL.md`` declares ``allowed-tools: WebFetch(domain:api.x.com)``.
* ``"host"`` — full host networking (escape hatch). Backends MUST log
  a ``UserWarning`` when this is selected so it shows up in CARE's TUI
  as a banner.
"""


_WEBFETCH_DOMAIN_RE = re.compile(
    r"WebFetch\s*\(\s*domain\s*:\s*([^\s)]+)\s*\)",
    re.IGNORECASE,
)


def parse_network_allowlist_from_allowed_tools(
    allowed_tools: list[str] | str | None,
) -> list[str]:
    """Extract egress hosts from a skill's ``allowed-tools`` declaration.

    Tokens of the form ``WebFetch(domain:host.example)`` are recognised
    — the domain is pulled out and returned. Bare ``WebFetch`` (no
    domain constraint) is skipped: an unconstrained network policy is
    a deliberate choice for the chain author, not the skill manifest.

    Args:
        allowed_tools: Either the raw frontmatter string
            (``"Bash(git:*) WebFetch(domain:api.x.com)"``), the
            pre-tokenised list (``["Bash(git:*)", "WebFetch(domain:api.x.com)"]``),
            or ``None`` for skills with no restrictions.

    Returns:
        Sorted unique list of domain hosts. Empty when no
        ``WebFetch(domain:*)`` tokens are present.
    """
    if allowed_tools is None:
        return []
    if isinstance(allowed_tools, str):
        text = allowed_tools
    else:
        text = " ".join(allowed_tools)
    matches = _WEBFETCH_DOMAIN_RE.findall(text)
    return sorted({m.strip() for m in matches if m.strip()})


def resolve_network_policy(
    runtime_config: dict[str, Any] | None,
    *,
    manifest_allowed_tools: list[str] | str | None = None,
    default: NetworkPolicy = "none",
) -> tuple[NetworkPolicy, list[str]]:
    """Normalise the runtime's network policy + allowlist.

    Returns ``(policy, allowlist)`` where:

    * ``policy`` is one of ``"none"`` / ``"allowlist"`` / ``"host"`` —
      validated; an unknown value raises :class:`SkillRuntimeError`.
    * ``allowlist`` is the merged allowlist when ``policy == "allowlist"``.
      Hosts come from ``runtime_config["network_allowlist"]`` plus the
      manifest's ``WebFetch(domain:*)`` declarations (skill manifest
      becomes the source of truth for what egress the skill needs).

    Args:
        runtime_config: ``AgentSkillStepConfig.runtime_config`` (or
            ``None`` for the default ``"none"`` policy).
        manifest_allowed_tools: The skill's allowed-tools tokens; only
            consulted when policy is ``"allowlist"``.
        default: Policy when ``runtime_config`` is ``None`` or doesn't
            set a ``"network"`` key. Defaults to ``"none"`` — fail
            closed.

    Raises:
        SkillRuntimeError: When the policy string isn't one of the
            three recognised values.
    """
    cfg = runtime_config or {}
    raw = cfg.get("network", default)
    if raw not in ("none", "allowlist", "host"):
        raise SkillRuntimeError(
            f"Unknown network policy {raw!r}. "
            f"Valid options: 'none', 'allowlist', 'host'."
        )
    policy: NetworkPolicy = raw  # type: ignore[assignment]

    if policy != "allowlist":
        return policy, []

    explicit = cfg.get("network_allowlist") or []
    if not isinstance(explicit, (list, tuple, set)):
        raise SkillRuntimeError(
            "runtime_config['network_allowlist'] must be a list of host strings."
        )
    explicit_hosts = [str(h).strip() for h in explicit if str(h).strip()]

    manifest_hosts = parse_network_allowlist_from_allowed_tools(
        manifest_allowed_tools,
    )

    # De-dup while preserving deterministic order (sorted).
    merged = sorted({*explicit_hosts, *manifest_hosts})
    return policy, merged


class SkillRuntimeError(RuntimeError):
    """Raised when a skill runtime fails to set up or execute.

    Distinct from a skill-script error (which surfaces through the
    return value's exit code / stderr): a ``SkillRuntimeError`` means
    the sandbox itself broke — missing docker daemon, unknown runtime
    name, container failed to start, etc. CARE renders this
    differently from "skill failed" in its TUI.
    """


@dataclass
class RuntimeRunResult:
    """Outcome of a single command execution inside a skill runtime."""

    stdout: bytes
    stderr: bytes
    exit_code: int
    duration_s: float


@dataclass
class SkillRuntimeHandle:
    """Opaque handle returned by :meth:`SkillRuntime.prepare`.

    Carries the workspace root (always present — even containerised
    runtimes mount a host-side dir for I/O) and a free-form ``backend``
    dict that each runtime uses to stash whatever bookkeeping it needs
    (container id, sandbox session token, …).
    """

    workspace_root: Path
    workspace_in: Path
    workspace_out: Path
    backend: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SkillRuntime(Protocol):
    """Protocol every skill execution backend must satisfy.

    Methods are all async so backends with network setup (Docker
    daemon, E2B API) don't block the chain's event loop. ``prepare``
    builds a workspace and returns a :class:`SkillRuntimeHandle`; the
    same handle threads through ``run`` / ``read_file`` /
    ``write_file`` / ``cleanup`` so the backend can stay stateful
    across calls (e.g. keep a long-lived container).
    """

    name: ClassVar[str]

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        """Create a fresh workspace + any backend state."""
        ...

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> RuntimeRunResult:
        """Execute one command inside the runtime."""
        ...

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        """Read a file path *relative to ``workspace_root``*."""
        ...

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        """Write ``data`` to a path relative to ``workspace_root``."""
        ...

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        """Tear down workspace + any backend state. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# LocalSkillRuntime — the always-available reference backend
# ---------------------------------------------------------------------------


class LocalSkillRuntime:
    """Run scripts as host subprocesses with no isolation.

    Wraps the long-standing ``asyncio.create_subprocess_exec`` path
    that the AgentSkill executor used before the runtime protocol was
    introduced. Always available — no extras required — but **not a
    sandbox**: scripts inherit the caller's env, credentials, and
    filesystem access.

    On first use per process, emits a single ``UserWarning`` so users
    who flip to this runtime (default) know they are running unsafe
    code on the host. CARE's TUI promotes this into a visible banner.
    """

    name: ClassVar[str] = "local"
    _unsafe_warned: ClassVar[bool] = False

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        if not LocalSkillRuntime._unsafe_warned:
            warnings.warn(
                "LocalSkillRuntime executes skill scripts directly on the host "
                "with no sandboxing. For untrusted skills, install the "
                "`carl[docker]` extra and set `runtime='docker'`. This warning "
                "fires once per process.",
                UserWarning,
                stacklevel=3,
            )
            LocalSkillRuntime._unsafe_warned = True

        # resolve the network policy contract even though
        # the local backend can't enforce it. We validate the policy
        # string (caught typos surface as a SkillRuntimeError up-front
        # instead of silently failing in the Docker rewire), stash the
        # resolved policy + allowlist on the handle so callers /
        # introspection tests can read them, and emit a one-time warning
        # for the `host` escape hatch.
        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )
        if policy == "host":
            warnings.warn(
                "Skill runtime selected network='host' — full host networking "
                "is enabled. Use 'none' or 'allowlist' for untrusted skills.",
                UserWarning,
                stacklevel=3,
            )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_"))
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_in = workspace / "in"
        workspace_out = workspace / "out"
        workspace_in.mkdir(exist_ok=True)
        workspace_out.mkdir(exist_ok=True)
        return SkillRuntimeHandle(
            workspace_root=workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "isolation": "none",
                "network_policy": policy,
                "network_allowlist": allowlist,
                # LocalSkillRuntime can't actually enforce any of this —
                # surface a single canonical flag so CARE's TUI knows
                # the policy is advisory rather than enforced.
                "network_enforced": False,
            },
        )

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> RuntimeRunResult:
        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                cwd=cwd,
                env=env,
            )
        except OSError as exc:
            # Distinguish "exec failed to start at all" from "script
            # crashed" — the former is a runtime-level failure.
            raise SkillRuntimeError(
                f"LocalSkillRuntime: failed to launch {cmd[0]!r}: {exc}"
            ) from exc

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=stdin), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            duration = asyncio.get_event_loop().time() - loop_start
            return RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,  # standard timeout exit code
                duration_s=duration,
            )

        duration = asyncio.get_event_loop().time() - loop_start
        return RuntimeRunResult(
            stdout=stdout_b,
            stderr=stderr_b,
            exit_code=proc.returncode or 0,
            duration_s=duration,
        )

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        target = self._resolve_path(handle, path)
        return await asyncio.to_thread(target.read_bytes)

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        target = self._resolve_path(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        if handle.backend.get("persisted"):
            return
        await asyncio.to_thread(
            shutil.rmtree, handle.workspace_root, True,
        )

    @staticmethod
    def _resolve_path(handle: SkillRuntimeHandle, path: str) -> Path:
        """Resolve a workspace-relative path safely.

        Absolute paths are rejected — they could escape the workspace,
        which defeats the (already-thin) safety story of LocalSkillRuntime.
        ``..`` segments are tolerated by ``Path.resolve()`` but the
        result must still sit inside ``workspace_root``.
        """
        if os.path.isabs(path):
            raise SkillRuntimeError(
                f"workspace paths must be relative: got {path!r}"
            )
        target = (handle.workspace_root / path).resolve()
        root = handle.workspace_root.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SkillRuntimeError(
                f"path {path!r} escapes workspace {root}"
            ) from exc
        return target


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SKILL_RUNTIME_REGISTRY: dict[str, type] = {}


def register_skill_runtime(name: str, cls: type) -> None:
    """Register a :class:`SkillRuntime` implementation under ``name``.

    The name is what users put in ``AgentSkillStepConfig.runtime``.
    Re-registering an existing name overwrites silently (lets test
    harnesses inject stub backends without juggling fixture order).
    """
    if not isinstance(cls, type):
        raise TypeError(f"runtime must be a class, got {type(cls).__name__}")
    SKILL_RUNTIME_REGISTRY[name] = cls


def get_skill_runtime(name: str) -> "SkillRuntime":
    """Look up + instantiate the runtime registered under ``name``.

    Raises :class:`SkillRuntimeError` when the name is unknown — CARE
    catches this and presents a friendly "install extras" message.
    """
    try:
        cls = SKILL_RUNTIME_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SKILL_RUNTIME_REGISTRY)) or "<none>"
        raise SkillRuntimeError(
            f"Unknown skill runtime {name!r}. "
            f"Available: {available}. "
            f"Install the matching extra (e.g. `pip install mmar-carl[docker]`) "
            f"or call register_skill_runtime() with a custom backend."
        ) from exc
    return cls()  # type: ignore[no-any-return]


def list_skill_runtimes() -> list[str]:
    """Return the sorted list of registered runtime names."""
    return sorted(SKILL_RUNTIME_REGISTRY)


# Pre-register the always-available local backend on import. Other
# backends self-register via their own modules (`docker_skill_runtime.py`,
# etc.) and gate on their optional dependency being importable.
register_skill_runtime(LocalSkillRuntime.name, LocalSkillRuntime)


__all__ = [
    "RuntimeRunResult",
    "SkillRuntime",
    "SkillRuntimeError",
    "SkillRuntimeHandle",
    "LocalSkillRuntime",
    "NetworkPolicy",
    "parse_network_allowlist_from_allowed_tools",
    "resolve_network_policy",
    "SKILL_RUNTIME_REGISTRY",
    "register_skill_runtime",
    "get_skill_runtime",
    "list_skill_runtimes",
]
