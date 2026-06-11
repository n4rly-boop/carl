"""Live-daemon smoke tests for the sandbox skill runtimes.

CARE-M1 (Docker / Firejail) and CARE-M2 (E2B) — the unit suite proves
each runtime emits the right flags / dispatches the right SDK calls.
This live suite proves the same backends actually *work* end-to-end
against the real daemon / CLI / SaaS endpoint when available.

Marker
------
Tests are decorated with ``@pytest.mark.skill_runtime_live``. The
default ``pytest`` run deselects them so the hermetic-mock suite stays
fast. Opt in with::

    pytest -m skill_runtime_live tests/agents/

Each test gracefully **skips** (not fails) when its backend isn't
available — missing CLI, missing API key, etc. This keeps the live
suite safe to run on any host: docker-installed hosts run the docker
test; Linux+firejail hosts run the firejail test; hosts with
``E2B_API_KEY`` set run the e2b test.

Coverage scope
--------------
Each test spawns the cheapest possible real sandbox call (an ``echo``
or trivial Python ``print``) and asserts the canonical exit-code 0 +
expected stdout. We're verifying the *plumbing* end-to-end, not
benchmarking — anything more substantial belongs in CARE's own
integration suite.
"""

from __future__ import annotations

import os
import shutil

import pytest

from mmar_carl import (
    DockerSkillRuntime,
    E2BSkillRuntime,
    FirejailSkillRuntime,
)


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_docker_runtime_runs_echo(tmp_path) -> None:
    """Run ``echo hello`` inside ``python:3.12-slim`` (or whatever image
    happens to be cached) via the real ``docker`` CLI.

    Skips when:
    - The ``docker`` CLI isn't on PATH.
    - The daemon isn't running (caught as a non-zero exit + clear
      stderr).
    """
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not on PATH")

    runtime = DockerSkillRuntime()
    handle = await runtime.prepare(
        skill=None, workspace=tmp_path / "ws",
        config={"image": "python:3.12-slim"},
    )
    try:
        result = await runtime.run(handle, ["echo", "hello-from-docker"])
        if result.exit_code != 0 and b"docker daemon" in result.stderr.lower():
            pytest.skip(
                f"docker daemon not running (stderr: {result.stderr.decode()[:200]})"
            )
        assert result.exit_code == 0, (
            f"docker run failed (stderr: {result.stderr.decode()[:200]})"
        )
        assert b"hello-from-docker" in result.stdout
    finally:
        await runtime.cleanup(handle)


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_docker_runtime_enforces_network_none(tmp_path) -> None:
    """Verify ``--network none`` actually drops outbound networking
    inside the container. The exit code from ``ping`` is non-zero
    when the network is unreachable.

    Note: we use python's ``socket.gethostbyname`` to avoid relying on
    ``ping`` being present in the slim image.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker CLI not on PATH")

    runtime = DockerSkillRuntime()
    handle = await runtime.prepare(
        skill=None, workspace=tmp_path / "ws",
        config={"image": "python:3.12-slim", "network": "none"},
    )
    try:
        # Python's socket lookup should fail when there's no network.
        result = await runtime.run(handle, [
            "python", "-c",
            "import socket; socket.gethostbyname('example.com')",
        ])
        if result.exit_code == 124:
            pytest.skip("docker pull or initial run timed out")
        if b"docker daemon" in result.stderr.lower():
            pytest.skip("docker daemon not running")
        # Network=none → DNS resolution fails → non-zero exit.
        assert result.exit_code != 0, (
            f"`--network none` was not enforced; stdout={result.stdout!r}"
        )
    finally:
        await runtime.cleanup(handle)


# ---------------------------------------------------------------------------
# Firejail (Linux only)
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_firejail_runtime_runs_echo(tmp_path) -> None:
    """Run ``echo hello`` via the real ``firejail`` CLI."""
    if shutil.which("firejail") is None:
        pytest.skip("firejail CLI not on PATH (Linux-only)")

    runtime = FirejailSkillRuntime()
    handle = await runtime.prepare(
        skill=None, workspace=tmp_path / "ws", config={},
    )
    try:
        result = await runtime.run(handle, ["echo", "hello-from-firejail"])
        assert result.exit_code == 0, (
            f"firejail run failed (stderr: {result.stderr.decode()[:200]})"
        )
        assert b"hello-from-firejail" in result.stdout
    finally:
        await runtime.cleanup(handle)


# ---------------------------------------------------------------------------
# E2B (requires API key)
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_e2b_runtime_runs_echo(tmp_path) -> None:
    """Spin up a real e2b sandbox and run ``echo``.

    Skips when:
    - ``E2B_API_KEY`` isn't set in the environment.
    - The ``e2b`` Python SDK isn't installed.
    """
    if not os.environ.get("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY not set — e2b sandbox unreachable")
    try:
        import e2b  # noqa: F401
    except ImportError:
        pytest.skip("e2b SDK not installed (`pip install e2b`)")

    runtime = E2BSkillRuntime()
    handle = await runtime.prepare(
        skill=None, workspace=tmp_path / "ws", config={},
    )
    try:
        result = await runtime.run(handle, ["echo", "hello-from-e2b"])
        assert result.exit_code == 0, (
            f"e2b sandbox run failed (stderr: {result.stderr.decode()[:200]})"
        )
        assert b"hello-from-e2b" in result.stdout
    finally:
        await runtime.cleanup(handle)


# ---------------------------------------------------------------------------
# Hermetic guards — these run in the default suite and assert the
# marker plumbing is intact so a contributor who drops the marker from
# a test surfaces it here.
# ---------------------------------------------------------------------------


def test_skill_runtime_live_marker_declared_in_pyproject() -> None:
    from pathlib import Path

    pyproject = (
        Path(__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "skill_runtime_live:" in pyproject, (
        "The `skill_runtime_live` marker must be declared in "
        "pyproject.toml's [tool.pytest.ini_options] markers."
    )
    assert "not skill_runtime_live" in pyproject, (
        "The default pytest addopts must deselect `skill_runtime_live` "
        "so the hermetic suite stays fast."
    )


def test_every_live_test_carries_marker() -> None:
    """Grep this file's source: every async test function that hits a
    real runtime must carry the ``@pytest.mark.skill_runtime_live``
    decorator. A missing decorator would make the test run in the
    default suite and likely hang on a daemon / API call.
    """
    from pathlib import Path

    text = Path(__file__).read_text(encoding="utf-8")
    test_funcs = [
        "test_docker_runtime_runs_echo",
        "test_docker_runtime_enforces_network_none",
        "test_firejail_runtime_runs_echo",
        "test_e2b_runtime_runs_echo",
    ]
    for fn in test_funcs:
        idx = text.index(f"async def {fn}(")
        prefix = text[max(0, idx - 200):idx]
        assert "@pytest.mark.skill_runtime_live" in prefix, (
            f"{fn} hits a real backend but lacks the "
            f"@pytest.mark.skill_runtime_live decorator — it would run "
            f"unexpectedly in the hermetic suite."
        )
