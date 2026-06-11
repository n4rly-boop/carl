"""FirejailSkillRuntime — lightweight Linux-only sandbox for AgentSkills.

Sister to :class:`DockerSkillRuntime`, but trades the
container overhead for a process-level sandbox from
`Firejail <https://firejail.wordpress.com/>`_. Same SkillRuntime
protocol; same network-policy contract; same workspace-mount convention
— but the cost per call is a fork/exec instead of a container start.

Implementation note
-------------------
We shell out to the ``firejail`` CLI via subprocess. Self-registers as
``"firejail"`` in the shared
:data:`~mmar_carl.skill_runtime.SKILL_RUNTIME_REGISTRY` on module import.
Hermetic tests stub ``asyncio.create_subprocess_exec`` so they don't
need firejail installed.

``runtime_config`` recognised keys
----------------------------------

* ``profile`` — Optional firejail profile name (e.g. ``"python"``,
  ``"default"``). When ``None`` (default) we pass ``--noprofile`` so
  behaviour is predictable across distros that disagree on
  ``/etc/firejail/<x>.profile`` content.
* ``network`` — ``"none"`` / ``"allowlist"`` / ``"host"``.
  Mapping:
  - ``"none"`` → ``--net=none`` (no networking).
  - ``"host"`` → no ``--net=*`` flag (firejail leaves host networking
    intact).
  - ``"allowlist"`` → ``--net=<bridge>`` where the bridge name comes
    from ``network_allowlist[0]`` (CARE configures the per-task bridge
    out of band).
* ``rlimit_as_mb`` — virtual-address-space cap in MiB. Default 2048
  (matches the Docker backend's 2 GiB ``--memory`` default).
* ``rlimit_cpu_s`` — CPU-time cap in seconds. ``None`` (default) means
  no rlimit; firejail doesn't enforce wall-clock time by default.
* ``extra_args`` — list of additional firejail CLI flags appended to
  every invocation. Escape hatch.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar, Optional

from .skill_runtime import (
    RuntimeRunResult,
    SkillRuntimeError,
    SkillRuntimeHandle,
    register_skill_runtime,
    resolve_network_policy,
)

_log = logging.getLogger(__name__)


_DEFAULT_RLIMIT_AS_MB = 2048


class FirejailSkillRuntime:
    """Run skill scripts inside a per-call firejail sandbox.

    Defaults: ``--noprofile`` (predictable), ``--quiet`` (no firejail
    chatter on stderr), ``--private=<workspace>`` (private home rooted
    at the workspace), ``--net=none`` (no network), and an
    ``--rlimit-as`` virtual-memory cap.

    The workspace lives on the host filesystem; firejail's ``--private``
    flag points the sandbox's home directory at it. ``read_file`` /
    ``write_file`` operate on the host workspace directly (no sandbox
    round-trip needed since the host has full access to its own dir).

    Network policy is enforced by firejail's ``--net=*`` flag —
    ``network_enforced=True`` on the handle, distinct from
    ``LocalSkillRuntime``'s advisory ``False``.
    """

    name: ClassVar[str] = "firejail"

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        # Up-front check: firejail must be on PATH. Surface a clear
        # `SkillRuntimeError` rather than failing inside `run` later.
        if shutil.which("firejail") is None:
            raise SkillRuntimeError(
                "FirejailSkillRuntime requires the `firejail` CLI on PATH. "
                "Install via `apt install firejail` / `dnf install firejail` "
                "on Linux, or pick a different runtime via "
                "AgentSkillStepConfig(runtime=...)."
            )

        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_firejail_"))
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
                "isolation": "firejail",
                "profile": config.get("profile"),  # None → --noprofile
                "rlimit_as_mb": int(
                    config.get("rlimit_as_mb", _DEFAULT_RLIMIT_AS_MB),
                ),
                "rlimit_cpu_s": (
                    int(config["rlimit_cpu_s"])
                    if config.get("rlimit_cpu_s") is not None
                    else None
                ),
                "extra_args": list(config.get("extra_args") or []),
                "network_policy": policy,
                "network_allowlist": allowlist,
                # Firejail enforces network + memory limits for real.
                "network_enforced": True,
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
        firejail_cmd = self._build_firejail_cmd(
            handle, cmd, cwd=cwd,
        )

        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *firejail_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                env=env,
            )
        except OSError as exc:
            raise SkillRuntimeError(
                f"FirejailSkillRuntime: failed to launch firejail CLI: {exc}"
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
                exit_code=124,
                duration_s=duration,
            )

        duration = asyncio.get_event_loop().time() - loop_start
        return RuntimeRunResult(
            stdout=stdout_b,
            stderr=stderr_b,
            exit_code=proc.returncode or 0,
            duration_s=duration,
        )

    async def read_file(
        self, handle: SkillRuntimeHandle, path: str,
    ) -> bytes:
        target = self._resolve_workspace_path(handle, path)
        return await asyncio.to_thread(target.read_bytes)

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        target = self._resolve_workspace_path(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        if handle.backend.get("persisted"):
            return
        await asyncio.to_thread(
            shutil.rmtree, handle.workspace_root, True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_firejail_cmd(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        cwd: Optional[str],
    ) -> list[str]:
        """Assemble the ``firejail [opts...] -- cmd args...`` argv."""
        backend = handle.backend
        firejail: list[str] = ["firejail", "--quiet"]

        # Profile control.
        profile = backend.get("profile")
        if profile:
            firejail.append(f"--profile={profile}")
        else:
            firejail.append("--noprofile")

        # Network policy.
        policy = backend.get("network_policy", "none")
        allowlist = backend.get("network_allowlist") or []
        if policy == "none":
            firejail.append("--net=none")
        elif policy == "host":
            # Firejail leaves host networking intact when no --net flag
            # is provided. We pass nothing — documented behaviour.
            pass
        elif policy == "allowlist":
            # CARE pre-configures a bridge name; the first listed host
            # doubles as the bridge.
            if allowlist:
                firejail.append(f"--net={allowlist[0]}")
            else:
                firejail.append("--net=none")

        # Private home rooted at the workspace.
        firejail.append(f"--private={handle.workspace_root}")

        # Resource limits.
        rlimit_as_mb = backend.get("rlimit_as_mb", _DEFAULT_RLIMIT_AS_MB)
        # firejail --rlimit-as takes bytes.
        firejail.append(f"--rlimit-as={rlimit_as_mb * 1024 * 1024}")
        rlimit_cpu_s = backend.get("rlimit_cpu_s")
        if rlimit_cpu_s is not None:
            firejail.append(f"--rlimit-cpu={rlimit_cpu_s}")

        if cwd is not None:
            firejail.append(f"--chdir={cwd}")

        # Caller-supplied passthrough flags.
        extra = backend.get("extra_args") or []
        firejail += list(extra)

        # ``--`` separates firejail's own flags from the command.
        firejail.append("--")
        firejail += cmd
        return firejail

    @staticmethod
    def _resolve_workspace_path(
        handle: SkillRuntimeHandle, path: str,
    ) -> Path:
        """Same safety contract as ``LocalSkillRuntime`` — reject
        absolute paths and ``..`` segments that escape the workspace.
        """
        import os as _os
        if _os.path.isabs(path):
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


# Self-register on import.
register_skill_runtime(FirejailSkillRuntime.name, FirejailSkillRuntime)


__all__ = ["FirejailSkillRuntime"]
