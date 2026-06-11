"""DockerSkillRuntime — sandbox AgentSkill execution inside a Docker container.

Default profile: ``python:3.12-slim`` image, ``--network none``,
``--cpus=1``, ``--memory=2g``, ``--pids-limit=128``, ``--read-only`` rootfs
with an ephemeral ``--tmpfs /tmp``. The host workspace is mounted so
``read_file`` / ``write_file`` use the existing :class:`LocalSkillRuntime`
path — they don't go through Docker exec.

Implementation note
-------------------
We shell out to the ``docker`` CLI rather than depending on the
``docker`` Python SDK. Reasons: (a) no new optional dependency to declare,
(b) the CLI is what users actually have installed, and (c) the call surface
is small enough that we don't need the SDK's higher-level container objects.
Hermetic tests stub ``asyncio.create_subprocess_exec`` so they don't need a
live daemon.

The runtime self-registers as ``"docker"`` in the shared
:data:`~mmar_carl.skill_runtime.SKILL_RUNTIME_REGISTRY` on module import, so
``AgentSkillStepConfig(runtime="docker")`` works without any extra wiring.

``runtime_config`` recognised keys
----------------------------------

* ``image`` — Docker image reference. Default ``"python:3.12-slim"``.
* ``network`` — ``"none"`` / ``"allowlist"`` / ``"host"``.
  When ``"allowlist"`` is selected we map to a named Docker network whose
  name comes from ``network_allowlist[0]`` (the first listed host doubles
  as the Docker network name). CARE supplies the per-task network out of
  band; CARL just wires the ``--network <name>`` flag.
* ``mem_limit`` — Docker ``--memory`` value. Default ``"2g"``.
* ``cpu_limit`` — Docker ``--cpus`` value. Default ``"1"``.
* ``pids_limit`` — Docker ``--pids-limit`` value. Default ``128``.
* ``extra_args`` — list of additional CLI flags appended to every
  ``docker run`` invocation. Escape hatch for unusual options.
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


_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_MEM_LIMIT = "2g"
_DEFAULT_CPU_LIMIT = "1"
_DEFAULT_PIDS_LIMIT = 128


class DockerSkillRuntime:
    """Run skill scripts inside a fresh Docker container per call.

    Containers are ephemeral (``--rm``). The host workspace is mounted
    into ``/workspace`` so the in/out subdirs are visible from both
    sides; ``in/`` is mounted read-only (skill scripts shouldn't mutate
    their inputs) while ``out/`` is read-write. The container root is
    read-only with an ephemeral tmpfs at ``/tmp`` so a misbehaving
    skill can't leave files on the host disk.

    Network access is governed by :func:`resolve_network_policy` —
    backend supports the full ``none`` / ``allowlist`` / ``host``
    contract.
    """

    name: ClassVar[str] = "docker"

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        # Up-front check: docker CLI must be on PATH. Surface a clear
        # `SkillRuntimeError` rather than failing inside `run` later.
        if shutil.which("docker") is None:
            raise SkillRuntimeError(
                "DockerSkillRuntime requires the `docker` CLI on PATH. "
                "Install Docker Desktop / Docker Engine, or pick a "
                "different runtime via AgentSkillStepConfig(runtime=...)."
            )

        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_docker_"))
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
                "isolation": "docker",
                "image": config.get("image", _DEFAULT_IMAGE),
                "mem_limit": str(config.get("mem_limit", _DEFAULT_MEM_LIMIT)),
                "cpu_limit": str(config.get("cpu_limit", _DEFAULT_CPU_LIMIT)),
                "pids_limit": int(config.get("pids_limit", _DEFAULT_PIDS_LIMIT)),
                "extra_args": list(config.get("extra_args") or []),
                "network_policy": policy,
                "network_allowlist": allowlist,
                # Docker DOES enforce these — distinct from LocalSkillRuntime
                # which only validates the contract.
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
        docker_cmd = self._build_docker_cmd(
            handle, cmd, env=env, timeout=timeout, cwd=cwd,
        )

        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            )
        except OSError as exc:
            raise SkillRuntimeError(
                f"DockerSkillRuntime: failed to launch docker CLI: {exc}"
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

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
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

    def _build_docker_cmd(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]],
        timeout: Optional[float],
        cwd: Optional[str],
    ) -> list[str]:
        """Assemble the full ``docker run`` argv for one ``cmd``.

        Defaults: ``--rm`` ephemeral container, ``--read-only`` rootfs,
        ``--tmpfs /tmp`` ephemeral writable scratch, workspace mounted
        with ``in/`` read-only and ``out/`` read-write. Resource limits
        + network policy come from ``handle.backend``.
        """
        backend = handle.backend
        docker: list[str] = [
            "docker", "run", "--rm",
            "--memory", str(backend.get("mem_limit", _DEFAULT_MEM_LIMIT)),
            "--cpus", str(backend.get("cpu_limit", _DEFAULT_CPU_LIMIT)),
            "--pids-limit", str(backend.get("pids_limit", _DEFAULT_PIDS_LIMIT)),
            "--read-only",
            "--tmpfs", "/tmp",
        ]

        # Network policy — Docker enforces this for real.
        policy = backend.get("network_policy", "none")
        allowlist = backend.get("network_allowlist") or []
        if policy == "none":
            docker += ["--network", "none"]
        elif policy == "host":
            docker += ["--network", "host"]
        elif policy == "allowlist":
            # CARE supplies the per-task network out of band; the first
            # listed host doubles as the Docker network name. Empty
            # allowlist + "allowlist" policy is effectively "none" since
            # we have no network to attach to.
            if allowlist:
                docker += ["--network", str(allowlist[0])]
            else:
                docker += ["--network", "none"]

        # Workspace mount. ``in/`` read-only, ``out/`` read-write,
        # ``workspace_root`` itself read-write so scripts can scratch.
        docker += [
            "-v", f"{handle.workspace_in}:/workspace/in:ro",
            "-v", f"{handle.workspace_out}:/workspace/out",
        ]

        if cwd is not None:
            docker += ["-w", cwd]

        if env:
            for k, v in env.items():
                docker += ["-e", f"{k}={v}"]

        # Pass timeout into the container via a metadata env var so the
        # script can self-impose deadlines if it wants to. The actual
        # timeout enforcement happens at the asyncio.wait_for level.
        if timeout is not None:
            docker += ["-e", f"CARL_TIMEOUT_S={timeout}"]

        # Caller-supplied passthrough flags.
        extra = backend.get("extra_args") or []
        docker += list(extra)

        # Image + the actual command to run inside the container.
        docker.append(str(backend.get("image", _DEFAULT_IMAGE)))
        docker += cmd
        return docker

    @staticmethod
    def _resolve_workspace_path(
        handle: SkillRuntimeHandle, path: str,
    ) -> Path:
        """Same safety contract as LocalSkillRuntime — reject absolute
        paths and ``..`` segments that escape the workspace.
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
register_skill_runtime(DockerSkillRuntime.name, DockerSkillRuntime)


__all__ = ["DockerSkillRuntime"]
