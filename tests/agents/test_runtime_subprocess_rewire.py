"""verify every ``create_subprocess_exec`` site in
the AgentSkill executor now routes through ``self._skill_runtime.run``.

A custom :class:`SkillRuntime` records every ``run`` invocation so the
test can assert the executor reaches the runtime — not
``asyncio.create_subprocess_exec`` directly — for all three call sites:

- ``_install_extra_pip`` (pip install for ``extra_pip`` packages)
- ``_execute_script_mode`` (SCRIPT / HYBRID / SUBAGENT modes)
- LLM_AGENT ``run_script`` tool

A future Docker / E2B backend can intercept all skill subprocess
execution by registering as the runtime — no further executor changes
needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Optional

import pytest

from mmar_carl import (
    RuntimeRunResult,
    SkillRuntimeHandle,
    register_skill_runtime,
)
from mmar_carl.step_executors import AgentSkillStepExecutor


# ---------------------------------------------------------------------------
# Recording runtime — captures every run() call without spawning a real
# subprocess. Returns canned stdout for callers that look at the output.
# ---------------------------------------------------------------------------


class _RecordingRuntime:
    """Records every ``run`` call into a class-level list so tests can
    assert the executor reached the runtime."""

    name: ClassVar[str] = "test_recording"
    calls: ClassVar[list[dict[str, Any]]] = []

    async def prepare(
        self, skill: Any, workspace: Optional[Path], config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        ws = workspace or Path("/tmp")
        return SkillRuntimeHandle(
            workspace_root=ws, workspace_in=ws / "in", workspace_out=ws / "out",
            backend={"isolation": "test"},
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
        type(self).calls.append({
            "cmd": list(cmd), "env": env, "cwd": cwd, "timeout": timeout,
        })
        return RuntimeRunResult(
            stdout=b"recorded-output", stderr=b"", exit_code=0, duration_s=0.0,
        )

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        return b""

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        pass

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    _RecordingRuntime.calls.clear()


@pytest.fixture
def _register_recording() -> None:
    register_skill_runtime("test_recording", _RecordingRuntime)


# ---------------------------------------------------------------------------
# _execute_script_mode now routes through the runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_script_mode_uses_runtime(tmp_path: Path) -> None:
    """The SCRIPT-mode subprocess call no longer hits
    ``asyncio.create_subprocess_exec`` directly — it goes through
    ``self._skill_runtime.run`` so backends can intercept.
    """
    from mmar_carl.models.agent_skill import AgentSkillStepConfig, SkillManifest

    # Build a minimal skill on disk with one .py script.
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: rec\ndescription: testing rewire\n---\nGo.\n",
        encoding="utf-8",
    )
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "go.py").write_text("print('hi')", encoding="utf-8")

    manifest = SkillManifest(
        name="rec", description="d",
        instructions="Go.",
        skill_dir=str(skill_dir),
        skill_md_path=str(skill_dir / "SKILL.md"),
        scripts=["scripts/go.py"],
    )
    config = AgentSkillStepConfig(skill="rec", task="t")
    executor = AgentSkillStepExecutor()
    # Install the recording runtime on the executor directly (mirrors
    # what execute() does after resolving the runtime).
    executor._skill_runtime = _RecordingRuntime()

    stdout, stderr, rc = await executor._execute_script_mode(
        config, manifest, resolved_inputs={},
    )
    assert rc == 0
    assert stdout == "recorded-output"
    # Exactly one runtime.run call.
    assert len(_RecordingRuntime.calls) == 1
    call = _RecordingRuntime.calls[0]
    assert call["cmd"][0].endswith("python") or "python" in call["cmd"][0]
    # The script absolute path made it into the cmd.
    assert any("go.py" in arg for arg in call["cmd"])


# ---------------------------------------------------------------------------
# _install_extra_pip now routes through the runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_extra_pip_uses_runtime(tmp_path: Path) -> None:
    """The pip-install subprocess no longer runs directly on the host —
    it routes through the runtime so Docker / E2B can run it inside
    the sandbox."""
    executor = AgentSkillStepExecutor()
    executor._skill_runtime = _RecordingRuntime()
    target = tmp_path / "overlay"
    await executor._install_extra_pip(["mypkg"], target)
    assert len(_RecordingRuntime.calls) == 1
    cmd = _RecordingRuntime.calls[0]["cmd"]
    assert cmd[1:4] == ["-m", "pip", "install"]
    assert "--target" in cmd
    assert str(target) in cmd
    assert "mypkg" in cmd


@pytest.mark.asyncio
async def test_install_extra_pip_raises_on_nonzero(tmp_path: Path) -> None:
    """Non-zero pip exit code still surfaces as RuntimeError after the
    rewire."""
    class _FailingRuntime(_RecordingRuntime):
        async def run(self, handle, cmd, **kwargs):  # type: ignore[override]
            return RuntimeRunResult(
                stdout=b"", stderr=b"oh no", exit_code=1, duration_s=0.0,
            )
    executor = AgentSkillStepExecutor()
    executor._skill_runtime = _FailingRuntime()
    with pytest.raises(RuntimeError, match="pip install failed"):
        await executor._install_extra_pip(["mypkg"], tmp_path / "overlay")


# ---------------------------------------------------------------------------
# Lazy-init behaviour: tests/callers that don't pre-set _skill_runtime
# get a LocalSkillRuntime by default so legacy direct-mode tests keep
# working.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_run_lazy_inits_local_when_missing(tmp_path: Path) -> None:
    """If ``self._skill_runtime`` was never set (test calls
    ``_execute_script_mode`` directly), ``_runtime_run`` falls back to
    LocalSkillRuntime so legacy code paths keep working."""
    from mmar_carl.skill_runtime import LocalSkillRuntime

    executor = AgentSkillStepExecutor()
    assert not hasattr(executor, "_skill_runtime") or executor._skill_runtime is None
    # Run a quick echo through the lazy-init path.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, err, rc = await executor._runtime_run(
            ["/bin/echo", "hello"], timeout=5.0,
        )
    assert rc == 0
    assert "hello" in out
    assert isinstance(executor._skill_runtime, LocalSkillRuntime)


# ---------------------------------------------------------------------------
# Direct-source-grep sanity: the rewire removed every previous direct
# ``create_subprocess_exec`` call inside the AgentSkill executor's
# subprocess sites. We grep the source so any future regression
# (someone adding a fourth subprocess call) surfaces here.
# ---------------------------------------------------------------------------


def test_no_direct_subprocess_in_agent_skill_executor() -> None:
    """The three rewired sites (_install_extra_pip, _execute_script_mode,
    LLM_AGENT run_script) must NOT call ``create_subprocess_exec``
    directly anymore — they go through ``self._runtime_run``.

    The ``LocalSkillRuntime.run`` and ``AsyncToolWrapper`` paths still
    use it (and SHOULD, since they wrap subprocess execution natively).
    """
    src = Path(__file__).resolve().parents[2] / "src" / "mmar_carl" / "step_executors.py"
    text = src.read_text(encoding="utf-8")
    # Hand-coded boundaries of the AgentSkill executor section.
    # The class starts at `class AgentSkillStepExecutor` and ends at
    # the next `class ` declaration.
    start = text.index("class AgentSkillStepExecutor")
    end_marker = text.index("class ", start + len("class AgentSkillStepExecutor"))
    agent_section = text[start:end_marker]
    # Look for actual call sites — `await asyncio.create_subprocess_exec(` —
    # not docstring references like ``asyncio.create_subprocess_exec``.
    assert "await asyncio.create_subprocess_exec(" not in agent_section, (
        "Regression: a `create_subprocess_exec` call leaked back into "
        "AgentSkillStepExecutor. Route it through self._runtime_run."
    )
