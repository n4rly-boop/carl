""":class:`DockerSkillRuntime`.

The runtime shells out to the ``docker`` CLI (no Python SDK dep). These
tests stub ``shutil.which`` (so they don't need a real docker install)
and ``asyncio.create_subprocess_exec`` (so they don't need a daemon)
and assert the right ``docker run`` flags are emitted for every
documented runtime_config option.

Defaults under test:
- image: ``python:3.12-slim``
- network: ``--network none`` (fail closed)
- ``--memory 2g``, ``--cpus 1``, ``--pids-limit 128``
- ``--read-only`` rootfs + ``--tmpfs /tmp``
- workspace mount: ``in/`` ro, ``out/`` rw
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl import (
    DockerSkillRuntime,
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
    """Build a fake asyncio subprocess result."""
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
def docker_on_path():
    """Pretend docker CLI is installed."""
    with patch(
        "mmar_carl.docker_skill_runtime.shutil.which",
        return_value="/usr/local/bin/docker",
    ):
        yield


@pytest.fixture
def captured_calls():
    """List that records every captured docker_cmd from
    create_subprocess_exec.
    """
    return []


@pytest.fixture
def fake_subprocess(captured_calls):
    """Patch create_subprocess_exec to record the docker_cmd argv."""
    async def _fake(*cmd: str, **kwargs: Any) -> MagicMock:
        captured_calls.append(list(cmd))
        return _fake_proc()
    with patch(
        "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
        new=_fake,
    ):
        yield


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_docker_runtime_pre_registered(self) -> None:
        assert "docker" in list_skill_runtimes()
        cls = get_skill_runtime("docker")
        assert isinstance(cls, DockerSkillRuntime)

    def test_name_attr(self) -> None:
        assert DockerSkillRuntime.name == "docker"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    @pytest.mark.asyncio
    async def test_missing_docker_cli_raises(self, tmp_path: Path) -> None:
        with patch(
            "mmar_carl.docker_skill_runtime.shutil.which", return_value=None,
        ):
            runtime = DockerSkillRuntime()
            with pytest.raises(SkillRuntimeError, match="`docker` CLI on PATH"):
                await runtime.prepare(
                    skill=None, workspace=tmp_path / "ws", config={},
                )

    @pytest.mark.asyncio
    async def test_defaults_stamped_on_handle(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        assert handle.backend["isolation"] == "docker"
        assert handle.backend["image"] == "python:3.12-slim"
        assert handle.backend["mem_limit"] == "2g"
        assert handle.backend["cpu_limit"] == "1"
        assert handle.backend["pids_limit"] == 128
        assert handle.backend["extra_args"] == []
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_enforced"] is True
        # Workspace dirs created.
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()

    @pytest.mark.asyncio
    async def test_custom_runtime_config_recognised_keys(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "image": "alpine:3.19",
                "mem_limit": "512m",
                "cpu_limit": "0.5",
                "pids_limit": 64,
                "extra_args": ["--cap-drop", "ALL"],
            },
        )
        assert handle.backend["image"] == "alpine:3.19"
        assert handle.backend["mem_limit"] == "512m"
        assert handle.backend["cpu_limit"] == "0.5"
        assert handle.backend["pids_limit"] == 64
        assert handle.backend["extra_args"] == ["--cap-drop", "ALL"]

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )


# ---------------------------------------------------------------------------
# _build_docker_cmd — flag composition
# ---------------------------------------------------------------------------


class TestBuildDockerCmd:
    def _make_handle(
        self,
        tmp_path: Path,
        *,
        image: str = "python:3.12-slim",
        mem_limit: str = "2g",
        cpu_limit: str = "1",
        pids_limit: int = 128,
        network_policy: str = "none",
        network_allowlist: list[str] | None = None,
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
                "image": image,
                "mem_limit": mem_limit,
                "cpu_limit": cpu_limit,
                "pids_limit": pids_limit,
                "network_policy": network_policy,
                "network_allowlist": network_allowlist or [],
                "extra_args": extra_args or [],
            },
        )

    def test_default_flags(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["python", "-c", "print(1)"],
            env=None, timeout=None, cwd=None,
        )
        assert cmd[:2] == ["docker", "run"]
        assert "--rm" in cmd
        assert "--read-only" in cmd
        assert cmd[cmd.index("--memory") + 1] == "2g"
        assert cmd[cmd.index("--cpus") + 1] == "1"
        assert cmd[cmd.index("--pids-limit") + 1] == "128"
        # tmpfs at /tmp
        assert "--tmpfs" in cmd
        assert "/tmp" in cmd
        # Image immediately precedes the user cmd.
        img_idx = cmd.index("python:3.12-slim")
        assert cmd[img_idx + 1 :] == ["python", "-c", "print(1)"]

    def test_network_none(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="none")
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_network_host(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="host")
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "host"

    def test_network_allowlist_uses_first_host_as_network_name(
        self, tmp_path: Path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist",
            network_allowlist=["my-net", "extra-host"],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "my-net"

    def test_empty_allowlist_falls_back_to_none(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist", network_allowlist=[],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_workspace_mounts(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        # Two -v entries: in/ ro, out/ rw.
        mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-v"]
        assert any(m.endswith(":/workspace/in:ro") for m in mounts)
        assert any(m.endswith(":/workspace/out") for m in mounts)

    def test_env_flags(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env={"FOO": "bar", "BAZ": "qux"},
            timeout=None, cwd=None,
        )
        env_pairs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
        assert "FOO=bar" in env_pairs
        assert "BAZ=qux" in env_pairs

    def test_cwd_flag(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd="/workspace/out",
        )
        assert "-w" in cmd
        assert cmd[cmd.index("-w") + 1] == "/workspace/out"

    def test_timeout_env_var(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=30.0, cwd=None,
        )
        env_pairs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
        assert any(p.startswith("CARL_TIMEOUT_S=") for p in env_pairs)

    def test_extra_args_passthrough(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path,
            extra_args=["--cap-drop", "ALL", "--user", "1000:1000"],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        assert "--cap-drop" in cmd
        assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
        assert "--user" in cmd
        assert cmd[cmd.index("--user") + 1] == "1000:1000"

    def test_user_cmd_appended_after_image(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, image="alpine:3.19")
        cmd = runtime._build_docker_cmd(
            handle, ["sh", "-c", "echo hi"], env=None, timeout=None, cwd=None,
        )
        img_idx = cmd.index("alpine:3.19")
        assert cmd[img_idx + 1 :] == ["sh", "-c", "echo hi"]


# ---------------------------------------------------------------------------
# run() — end-to-end with stubbed subprocess
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(
        self, tmp_path: Path, docker_on_path, fake_subprocess, captured_calls,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["python", "-c", "print(1)"])
        assert result.exit_code == 0
        assert result.stdout == b"ok"
        assert captured_calls  # one docker run dispatched
        assert captured_calls[0][:2] == ["docker", "run"]

    @pytest.mark.asyncio
    async def test_run_timeout_returns_exit_124(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        async def _hang(*cmd, **kwargs):
            return _fake_proc(hang=True)
        with patch(
            "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
            new=_hang,
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            result = await runtime.run(handle, ["sleep", "100"], timeout=0.05)
        assert result.exit_code == 124
        assert b"[timeout after" in result.stderr

    @pytest.mark.asyncio
    async def test_oserror_on_launch_wrapped(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        async def _fail(*cmd, **kwargs):
            raise OSError("docker daemon not running")
        with patch(
            "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
            new=_fail,
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(SkillRuntimeError, match="failed to launch docker"):
                await runtime.run(handle, ["echo"])

    @pytest.mark.asyncio
    async def test_runtime_config_flows_into_docker_args(
        self, tmp_path: Path, docker_on_path, fake_subprocess, captured_calls,
    ) -> None:
        """Custom mem/cpu/image from runtime_config must reach the
        emitted docker run argv."""
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "image": "alpine:3.19",
                "mem_limit": "512m",
                "cpu_limit": "0.5",
                "network": "host",
            },
        )
        await runtime.run(handle, ["true"])
        cmd = captured_calls[0]
        assert cmd[cmd.index("--memory") + 1] == "512m"
        assert cmd[cmd.index("--cpus") + 1] == "0.5"
        assert cmd[cmd.index("--network") + 1] == "host"
        assert "alpine:3.19" in cmd


# ---------------------------------------------------------------------------
# read_file / write_file / cleanup — same contract as LocalSkillRuntime
# ---------------------------------------------------------------------------


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.write_file(handle, "out/result.txt", b"hello")
        data = await runtime.read_file(handle, "out/result.txt")
        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await runtime.write_file(handle, "/etc/passwd", b"x")

    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await runtime.write_file(handle, "../escape.txt", b"x")

    @pytest.mark.asyncio
    async def test_cleanup_removes_workspace(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.cleanup(handle)
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_cleanup_persisted_keeps_workspace(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        handle.backend["persisted"] = True
        await runtime.cleanup(handle)
        assert handle.workspace_root.exists()


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------


def test_docker_runtime_importable_from_top_level() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "DockerSkillRuntime")
    assert mmar_carl.DockerSkillRuntime is DockerSkillRuntime
