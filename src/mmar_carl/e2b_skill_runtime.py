"""E2BSkillRuntime — remote SaaS sandbox for AgentSkills via e2b.dev.

Sister to :class:`DockerSkillRuntime` /
:class:`FirejailSkillRuntime`, but the sandbox is a hosted micro-VM
managed by `E2B <https://e2b.dev/>`_ instead of a local container or
Linux process. Per-call latency is higher (network round-trip), but
isolation is the strongest of the three reference backends — sandboxes
are full VMs with no host filesystem access.

Implementation note
-------------------
The runtime self-registers as ``"e2b"`` on import. The actual ``e2b``
SDK is imported *lazily* inside :meth:`prepare`, so ``mmar-carl`` users
without the ``carl[e2b]`` extra never pay the import cost. Missing SDK
→ ``SkillRuntimeError`` with the canonical install hint.

A ``sandbox_factory`` kwarg lets tests inject a mock sandbox class
without having to install the SDK or hit the e2b network. Production
callers leave it as the default ``None``, in which case the lazy
import path runs.

Sandbox protocol
----------------
The runtime expects the injected / SDK-provided sandbox object to
support (informally — duck-typed):

* ``await Sandbox.create(template=..., api_key=...)`` — async factory.
* ``await sandbox.commands.run(cmd, *, envs=None, timeout=None,
  cwd=None)`` — returns an object with ``.stdout``, ``.stderr``, and
  ``.exit_code`` attributes (or the string-formatted equivalents).
* ``await sandbox.files.write(path, content)`` — write a bytes/str
  file at an absolute path inside the sandbox.
* ``await sandbox.files.read(path)`` — read a bytes/str file.
* ``await sandbox.kill()`` — terminate.

These are the names used by the current ``e2b`` Python SDK (`v1+`).
The runtime is forgiving about return-value shapes: a result with
``.stdout`` (bytes/str) is accepted whether ``.exit_code`` is on the
result or buried in ``.error``.

``runtime_config`` recognised keys
----------------------------------

* ``template`` — e2b sandbox template id. Default ``"base"``.
* ``api_key_env`` — env var name that holds the e2b API key. Default
  ``"E2B_API_KEY"``. The runtime reads ``os.environ[api_key_env]`` and
  passes the value to ``Sandbox.create``.
* ``extra_create_kwargs`` — dict of additional kwargs forwarded to
  ``Sandbox.create`` (e.g. ``timeout``, ``metadata``).
* ``workspace_root_in_sandbox`` — absolute path inside the sandbox
  that maps to the host workspace. Default ``"/workspace"``. The
  runtime mirrors host files at this path via the SDK's ``files``
  API on each ``write_file`` / ``read_file``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, ClassVar, Optional

from .skill_runtime import (
    RuntimeRunResult,
    SkillRuntimeError,
    SkillRuntimeHandle,
    register_skill_runtime,
    resolve_network_policy,
)

_log = logging.getLogger(__name__)


_DEFAULT_TEMPLATE = "base"
_DEFAULT_API_KEY_ENV = "E2B_API_KEY"
_DEFAULT_WS_IN_SANDBOX = "/workspace"


class E2BSkillRuntime:
    """Run skill scripts inside an e2b.dev hosted micro-VM sandbox.

    The sandbox lives for the duration of the handle (created in
    :meth:`prepare`, killed in :meth:`cleanup`). Every ``run`` reuses
    the same sandbox — distinct from the per-call container model of
    :class:`DockerSkillRuntime`. E2B sandboxes have full network access
    by default; the network-policy contract is honoured at
    ``handle.backend["network_policy"]`` for callers / CARE TUI but is
    NOT enforced by this backend (``network_enforced=False``).

    Parameters
    ----------
    sandbox_factory:
        Optional injection hook. When supplied, ``prepare()`` calls
        ``await sandbox_factory(**create_kwargs)`` instead of importing
        the e2b SDK. Used by tests to swap in a mock sandbox; do NOT
        use in production.
    """

    name: ClassVar[str] = "e2b"

    def __init__(
        self,
        *,
        sandbox_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_e2b_"))
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_in = workspace / "in"
        workspace_out = workspace / "out"
        workspace_in.mkdir(exist_ok=True)
        workspace_out.mkdir(exist_ok=True)

        template = config.get("template", _DEFAULT_TEMPLATE)
        api_key_env = config.get("api_key_env", _DEFAULT_API_KEY_ENV)
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise SkillRuntimeError(
                f"E2BSkillRuntime requires an API key in the "
                f"{api_key_env!r} environment variable. Sign up at "
                f"https://e2b.dev and export the key, or override "
                f"`runtime_config['api_key_env']`."
            )

        ws_in_sandbox = config.get(
            "workspace_root_in_sandbox", _DEFAULT_WS_IN_SANDBOX,
        )
        extra_create_kwargs = dict(config.get("extra_create_kwargs") or {})

        factory = self._sandbox_factory or self._default_sandbox_factory
        try:
            sandbox = await factory(
                template=template, api_key=api_key, **extra_create_kwargs,
            )
        except SkillRuntimeError:
            raise
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: failed to start sandbox "
                f"(template={template!r}): {exc}"
            ) from exc

        return SkillRuntimeHandle(
            workspace_root=workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "isolation": "e2b",
                "template": template,
                "api_key_env": api_key_env,
                "workspace_root_in_sandbox": ws_in_sandbox,
                "sandbox": sandbox,
                "network_policy": policy,
                "network_allowlist": allowlist,
                # E2B sandboxes have full network by design; the policy
                # is recorded but not enforced by this backend.
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
        sandbox = handle.backend.get("sandbox")
        if sandbox is None:
            raise SkillRuntimeError(
                "E2BSkillRuntime: handle has no live sandbox. Did "
                "`prepare()` complete successfully?"
            )

        cmd_str = self._shell_join(cmd)
        loop_start = asyncio.get_event_loop().time()
        try:
            result = await sandbox.commands.run(
                cmd_str,
                envs=env or None,
                timeout=timeout,
                cwd=cwd,
            )
        except asyncio.TimeoutError:
            duration = asyncio.get_event_loop().time() - loop_start
            return RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,
                duration_s=duration,
            )
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: sandbox command failed: {exc}"
            ) from exc

        duration = asyncio.get_event_loop().time() - loop_start
        return self._coerce_result(result, duration_s=duration)

    async def read_file(
        self, handle: SkillRuntimeHandle, path: str,
    ) -> bytes:
        # workspace paths are relative to the host workspace; we mirror
        # them via the SDK's files API at <ws_in_sandbox>/<rel_path>.
        self._reject_unsafe_relative_path(path)
        sandbox = self._require_sandbox(handle)
        sandbox_path = self._sandbox_path(handle, path)
        try:
            content = await sandbox.files.read(sandbox_path)
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: read_file({path!r}) failed: {exc}"
            ) from exc
        return content if isinstance(content, bytes) else str(content).encode()

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        self._reject_unsafe_relative_path(path)
        sandbox = self._require_sandbox(handle)
        sandbox_path = self._sandbox_path(handle, path)
        try:
            await sandbox.files.write(sandbox_path, data)
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: write_file({path!r}) failed: {exc}"
            ) from exc

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        sandbox = handle.backend.get("sandbox")
        if sandbox is not None and not handle.backend.get("persisted"):
            try:
                await sandbox.kill()
            except Exception:  # pragma: no cover - best-effort teardown
                _log.warning(
                    "E2BSkillRuntime: sandbox.kill() failed during cleanup",
                    exc_info=True,
                )
            handle.backend["sandbox"] = None
        # Local workspace dir cleanup mirrors the other backends.
        if not handle.backend.get("persisted"):
            import shutil as _shutil
            await asyncio.to_thread(
                _shutil.rmtree, handle.workspace_root, True,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    async def _default_sandbox_factory(**kwargs: Any) -> Any:
        """Lazy import of the real e2b SDK.

        Raises :class:`SkillRuntimeError` with the install hint if the
        SDK isn't available.
        """
        try:
            from e2b import AsyncSandbox  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SkillRuntimeError(
                "E2BSkillRuntime requires the e2b SDK. Install with "
                "`pip install mmar-carl[e2b]` (or `pip install e2b`)."
            ) from exc

        # The exact e2b SDK API has shifted across versions; we use
        # `AsyncSandbox.create` which is the v1+ shape.
        return await AsyncSandbox.create(**kwargs)

    @staticmethod
    def _require_sandbox(handle: SkillRuntimeHandle) -> Any:
        sandbox = handle.backend.get("sandbox")
        if sandbox is None:
            raise SkillRuntimeError(
                "E2BSkillRuntime: handle has no live sandbox."
            )
        return sandbox

    @staticmethod
    def _sandbox_path(handle: SkillRuntimeHandle, rel: str) -> str:
        """Translate a workspace-relative path to an absolute path
        inside the sandbox."""
        root = handle.backend.get(
            "workspace_root_in_sandbox", _DEFAULT_WS_IN_SANDBOX,
        )
        # Use posix style joining since the sandbox is Linux.
        if root.endswith("/"):
            return f"{root}{rel}"
        return f"{root}/{rel}"

    @staticmethod
    def _reject_unsafe_relative_path(path: str) -> None:
        """Reject absolute paths and ``..`` segments — same contract
        as the other reference backends."""
        if os.path.isabs(path):
            raise SkillRuntimeError(
                f"workspace paths must be relative: got {path!r}"
            )
        # Reject any segment that's exactly `..` (escape attempt).
        parts = path.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise SkillRuntimeError(
                f"path {path!r} escapes workspace"
            )

    @staticmethod
    def _shell_join(cmd: list[str]) -> str:
        """Join argv into a shell command string.

        E2B's ``commands.run`` accepts either a string or an argv list
        depending on SDK version. We pass a string built with
        ``shlex.join`` to maximise compatibility — it works for both.
        """
        import shlex
        return shlex.join(cmd)

    @staticmethod
    def _coerce_result(result: Any, *, duration_s: float) -> RuntimeRunResult:
        """Normalise the SDK's result object into ``RuntimeRunResult``."""
        stdout = getattr(result, "stdout", b"") or b""
        stderr = getattr(result, "stderr", b"") or b""
        exit_code = getattr(result, "exit_code", None)
        if exit_code is None:
            exit_code = getattr(result, "exitCode", 0)
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8", errors="replace")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8", errors="replace")
        return RuntimeRunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=int(exit_code) if exit_code is not None else 0,
            duration_s=duration_s,
        )


# Self-register on import.
register_skill_runtime(E2BSkillRuntime.name, E2BSkillRuntime)


__all__ = ["E2BSkillRuntime"]
