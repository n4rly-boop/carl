""":class:`FirejailSkillRuntime`.

Linux-only lightweight sandbox sister to :class:`DockerSkillRuntime`.
Same hermetic test strategy: stub ``shutil.which`` so the tests don't
need firejail installed, and stub ``asyncio.create_subprocess_exec`` so
they don't actually run a sandbox — they assert the right ``firejail``
flags are emitted.

Defaults under test:
- ``--quiet`` + ``--noprofile`` (no surprise profile from /etc/firejail/).
- ``--net=none`` (fail closed — fits the spec'd "P2" lightweight model).
- ``--private=<workspace>`` (private home points at the workspace).
- ``--rlimit-as`` virtual-memory cap (default 2 GiB).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl import (
    FirejailSkillRuntime,
    SkillRuntimeError,
    SkillRuntimeHandle,
    get_skill_runtime,
    list_skill_runtimes,
)


# ---------------------------------------------------------------------------
# Subprocess stubbing
# ---------------------------------------------------------------------------


def _fake_proc(
    *, stdout: bytes = b"ok", stderr: bytes = b"",
    returncode: int = 0, hang: bool = False,
) -> MagicMock:
    proc = MagicMock()
    if hang:
        async def never_returns(*a: Any, **k: Any) -> Any:
            await asyncio.sleep(10)
            return (b"", b"")
        proc.communicate = never_returns
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.returncode = returncode
    return proc


@pytest.fixture
def firejail_on_path():
    """Pretend firejail CLI is installed."""
    with patch(
        "mmar_carl.firejail_skill_runtime.shutil.which",
        return_value="/usr/bin/firejail",
    ):
        yield


@pytest.fixture
def captured_calls():
    return []


@pytest.fixture
def fake_subprocess(captured_calls):
    async def _fake(*cmd: str, **kwargs: Any) -> MagicMock:
        captured_calls.append(list(cmd))
        return _fake_proc()
    with patch(
        "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
        new=_fake,
    ):
        yield


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_firejail_runtime_pre_registered(self) -> None:
        assert "firejail" in list_skill_runtimes()
        instance = get_skill_runtime("firejail")
        assert isinstance(instance, FirejailSkillRuntime)

    def test_name_attr(self) -> None:
        assert FirejailSkillRuntime.name == "firejail"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    @pytest.mark.asyncio
    async def test_missing_firejail_cli_raises(self, tmp_path: Path) -> None:
        with patch(
            "mmar_carl.firejail_skill_runtime.shutil.which",
            return_value=None,
        ):
            runtime = FirejailSkillRuntime()
            with pytest.raises(
                SkillRuntimeError, match="`firejail` CLI on PATH",
            ):
                await runtime.prepare(
                    skill=None, workspace=tmp_path / "ws", config={},
                )

    @pytest.mark.asyncio
    async def test_defaults_stamped_on_handle(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        assert handle.backend["isolation"] == "firejail"
        assert handle.backend["profile"] is None
        assert handle.backend["rlimit_as_mb"] == 2048
        assert handle.backend["rlimit_cpu_s"] is None
        assert handle.backend["extra_args"] == []
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_enforced"] is True
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()

    @pytest.mark.asyncio
    async def test_custom_runtime_config(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "profile": "python",
                "rlimit_as_mb": 512,
                "rlimit_cpu_s": 30,
                "extra_args": ["--seccomp"],
            },
        )
        assert handle.backend["profile"] == "python"
        assert handle.backend["rlimit_as_mb"] == 512
        assert handle.backend["rlimit_cpu_s"] == 30
        assert handle.backend["extra_args"] == ["--seccomp"]

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )


# ---------------------------------------------------------------------------
# _build_firejail_cmd — flag composition
# ---------------------------------------------------------------------------


class TestBuildFirejailCmd:
    def _make_handle(
        self,
        tmp_path: Path,
        *,
        profile: Optional[str] = None,
        network_policy: str = "none",
        network_allowlist: list[str] | None = None,
        rlimit_as_mb: int = 2048,
        rlimit_cpu_s: Optional[int] = None,
        extra_args: list[str] | None = None,
    ) -> SkillRuntimeHandle:
        ws = tmp_path / "ws"
        ws_in = ws / "in"
        ws_out = ws / "out"
        ws.mkdir()
        ws_in.mkdir()
        ws_out.mkdir()
        return SkillRuntimeHandle(
            workspace_root=ws, workspace_in=ws_in, workspace_out=ws_out,
            backend={
                "profile": profile,
                "rlimit_as_mb": rlimit_as_mb,
                "rlimit_cpu_s": rlimit_cpu_s,
                "extra_args": extra_args or [],
                "network_policy": network_policy,
                "network_allowlist": network_allowlist or [],
            },
        )

    def test_default_flags(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(
            handle, ["python", "-c", "print(1)"], cwd=None,
        )
        assert cmd[0] == "firejail"
        assert "--quiet" in cmd
        assert "--noprofile" in cmd
        assert "--net=none" in cmd
        # rlimit-as in bytes
        assert any(c == f"--rlimit-as={2048 * 1024 * 1024}" for c in cmd)
        # private= points at workspace
        assert any(
            c.startswith("--private=") and str(handle.workspace_root) in c
            for c in cmd
        )
        # `--` separator before the user cmd
        sep = cmd.index("--")
        assert cmd[sep + 1 :] == ["python", "-c", "print(1)"]

    def test_profile_flag(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, profile="python")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--profile=python" in cmd
        assert "--noprofile" not in cmd

    def test_network_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="none")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=none" in cmd

    def test_network_host_omits_net_flag(self, tmp_path: Path) -> None:
        """Host policy → no ``--net=*`` flag (firejail's documented
        default behaviour)."""
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="host")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert not any(c.startswith("--net=") for c in cmd)

    def test_network_allowlist_uses_first_host_as_bridge(
        self, tmp_path: Path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist",
            network_allowlist=["br-care0", "extra"],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=br-care0" in cmd

    def test_empty_allowlist_falls_back_to_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist", network_allowlist=[],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=none" in cmd

    def test_rlimit_as_in_bytes(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_as_mb=512)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert f"--rlimit-as={512 * 1024 * 1024}" in cmd

    def test_rlimit_cpu_optional(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_cpu_s=15)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--rlimit-cpu=15" in cmd

    def test_rlimit_cpu_omitted_when_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_cpu_s=None)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert not any(c.startswith("--rlimit-cpu=") for c in cmd)

    def test_chdir_flag(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd="/work")
        assert "--chdir=/work" in cmd

    def test_extra_args_passthrough(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, extra_args=["--seccomp", "--nogroups"],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--seccomp" in cmd
        assert "--nogroups" in cmd

    def test_user_cmd_after_double_dash(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(
            handle, ["sh", "-c", "echo hi"], cwd=None,
        )
        sep = cmd.index("--")
        assert cmd[sep + 1 :] == ["sh", "-c", "echo hi"]


# ---------------------------------------------------------------------------
# run() — end-to-end with stubbed subprocess
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(
        self, tmp_path: Path, firejail_on_path, fake_subprocess, captured_calls,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["python", "-c", "print(1)"])
        assert result.exit_code == 0
        assert result.stdout == b"ok"
        assert captured_calls
        assert captured_calls[0][0] == "firejail"

    @pytest.mark.asyncio
    async def test_run_timeout_returns_exit_124(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        async def _hang(*cmd, **kwargs):
            return _fake_proc(hang=True)
        with patch(
            "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
            new=_hang,
        ):
            runtime = FirejailSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            result = await runtime.run(handle, ["sleep", "100"], timeout=0.05)
        assert result.exit_code == 124
        assert b"[timeout after" in result.stderr

    @pytest.mark.asyncio
    async def test_oserror_on_launch_wrapped(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        async def _fail(*cmd, **kwargs):
            raise OSError("permission denied")
        with patch(
            "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
            new=_fail,
        ):
            runtime = FirejailSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(SkillRuntimeError, match="failed to launch firejail"):
                await runtime.run(handle, ["echo"])

    @pytest.mark.asyncio
    async def test_runtime_config_flows_into_firejail_args(
        self, tmp_path: Path, firejail_on_path, fake_subprocess,
        captured_calls,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "profile": "python",
                "rlimit_as_mb": 256,
                "network": "host",
            },
        )
        await runtime.run(handle, ["true"])
        cmd = captured_calls[0]
        assert "--profile=python" in cmd
        assert f"--rlimit-as={256 * 1024 * 1024}" in cmd
        # host policy → no --net flag
        assert not any(c.startswith("--net=") for c in cmd)


# ---------------------------------------------------------------------------
# File ops — same contract as Local + Docker
# ---------------------------------------------------------------------------


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.write_file(handle, "out/result.txt", b"hello")
        data = await runtime.read_file(handle, "out/result.txt")
        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await runtime.write_file(handle, "/etc/passwd", b"x")

    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await runtime.write_file(handle, "../escape.txt", b"x")

    @pytest.mark.asyncio
    async def test_cleanup_removes_workspace(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.cleanup(handle)
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_cleanup_persisted_keeps_workspace(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        handle.backend["persisted"] = True
        await runtime.cleanup(handle)
        assert handle.workspace_root.exists()


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------


def test_firejail_runtime_importable_from_top_level() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "FirejailSkillRuntime")
    assert mmar_carl.FirejailSkillRuntime is FirejailSkillRuntime
