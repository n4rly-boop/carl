"""Legacy chain JSON fixture round-trip + typed
``ChainFormatNewerError`` exception.

Every JSON file dropped under ``tests/fixtures/legacy_chains/`` must
survive ``ReasoningChain.migrate`` + ``from_dict`` without raising. The
test is intentionally parametrised over the directory so CARE
contributors can drop new versioned fixtures here as the format
evolves — running the suite catches missed migration rungs.

Plus: ``from_dict`` raises a typed :class:`ChainFormatNewerError` when
the saved ``format_version`` is greater than the current library's
``FORMAT_VERSION`` — CARE catches it and prompts the user to upgrade
``mmar-carl``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmar_carl import ChainFormatNewerError, ReasoningChain


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "legacy_chains"


# ---------------------------------------------------------------------------
# Parametrised fixture round-trip
# ---------------------------------------------------------------------------


def _fixture_paths() -> list[Path]:
    """Return every JSON file under ``tests/fixtures/legacy_chains/``.

    Sorted so test ids are stable across runs.
    """
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "fixture_path",
    _fixture_paths(),
    ids=lambda p: p.name,
)
def test_legacy_chain_fixture_round_trips(fixture_path: Path) -> None:
    """Each ``.json`` fixture under ``tests/fixtures/legacy_chains/``
    survives migration + reconstruction without raising, and the
    rebuilt chain re-serialises into something whose ``to_dict()``
    round-trips back through ``from_dict``.

    This is the CI guard the spec asks for: when a step-config field is
    renamed or removed, the migration ladder MUST cover the legacy
    fixture or this test fires.
    """
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    migrated = ReasoningChain.migrate(raw)
    chain = ReasoningChain.from_dict(migrated)

    # Basic sanity: at least one step and step numbers are positive.
    assert chain.steps, f"{fixture_path.name}: rebuilt chain has no steps"
    for step in chain.steps:
        assert step.number >= 1

    # Re-serialise and round-trip through migrate + from_dict again.
    second = ReasoningChain.from_dict(
        ReasoningChain.migrate(chain.to_dict()),
    )
    assert len(second.steps) == len(chain.steps)
    # Each step's title + number survives the second pass.
    for orig, rt in zip(chain.steps, second.steps):
        assert rt.number == orig.number
        assert rt.title == orig.title


def test_fixtures_directory_is_present() -> None:
    """If the fixtures directory disappears (e.g. accidentally
    deleted), fail loudly so contributors notice — the parametrised
    test would otherwise silently collect zero cases.
    """
    assert FIXTURES_DIR.is_dir(), (
        f"Expected fixtures dir at {FIXTURES_DIR} — has it been moved?"
    )
    assert _fixture_paths(), (
        "tests/fixtures/legacy_chains/ exists but contains no .json files. "
        "Add at least one fixture so the regression test has something to "
        "exercise."
    )


def test_v0_pre_format_version_fixture_migrates_to_current() -> None:
    """The hand-written ``v0_pre_format_version.json`` fixture uses the
    pre-format_version layout (carries a `version` string key, no
    `format_version` integer). The migrate ladder must strip the old
    `version` key and stamp ``format_version=1``.

    This documents the migration contract: a legacy on-disk chain
    saved by an older mmar-carl release must keep loading.
    """
    raw = json.loads(
        (FIXTURES_DIR / "v0_pre_format_version.json").read_text(encoding="utf-8"),
    )
    assert "version" in raw  # pre-migration shape
    assert "format_version" not in raw

    migrated = ReasoningChain.migrate(raw)
    assert migrated["format_version"] == ReasoningChain.FORMAT_VERSION
    assert "version" not in migrated  # the migration rung dropped it

    chain = ReasoningChain.from_dict(migrated)
    assert len(chain.steps) == 1
    assert chain.steps[0].title == "Pre-format-version step"


# ---------------------------------------------------------------------------
# Newer-format → typed error
# ---------------------------------------------------------------------------


class TestChainFormatNewerError:
    """A chain serialised by a *future* mmar-carl release must raise
    :class:`ChainFormatNewerError` instead of silently parsing on a
    best-effort basis. CARE catches this and surfaces an "upgrade
    mmar-carl" prompt.
    """

    def test_future_format_version_raises_typed_error(self) -> None:
        future = {
            "format_version": ReasoningChain.FORMAT_VERSION + 99,
            "carl_version": "99.0.0",
            "max_workers": 3,
            "enable_progress": False,
            "metadata": {},
            "timeout": None,
            "replan_policy": None,
            "search_config": None,
            "steps": [
                {
                    "number": 1,
                    "title": "x",
                    "step_type": "llm",
                    "aim": "x",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        with pytest.raises(ChainFormatNewerError) as exc_info:
            ReasoningChain.from_dict(future)
        err = exc_info.value
        assert err.required_version == ReasoningChain.FORMAT_VERSION + 99
        assert err.this_version == ReasoningChain.FORMAT_VERSION
        # Message mentions both versions + the upgrade hint.
        msg = str(err)
        assert str(err.required_version) in msg
        assert str(err.this_version) in msg
        assert "upgrade mmar-carl" in msg.lower()

    def test_current_format_version_loads_normally(self) -> None:
        data = {
            "format_version": ReasoningChain.FORMAT_VERSION,
            "carl_version": "unknown",
            "max_workers": 3,
            "enable_progress": False,
            "metadata": {},
            "timeout": None,
            "replan_policy": None,
            "search_config": None,
            "steps": [
                {
                    "number": 1,
                    "title": "ok",
                    "step_type": "llm",
                    "aim": "ok",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        chain = ReasoningChain.from_dict(data)
        assert chain.steps[0].title == "ok"

    def test_missing_format_version_loads_via_migrate(self) -> None:
        """A pre-format_version chain (no ``format_version`` key at all)
        loads after migrate() applies the v0 → v1 rung — no
        ChainFormatNewerError raised.
        """
        legacy = {
            "version": "old-string-version",
            "max_workers": 3,
            "metadata": {},
            "steps": [
                {
                    "number": 1,
                    "title": "legacy",
                    "step_type": "llm",
                    "aim": "x",
                    "dependencies": [],
                    "checkpoint": False,
                    "checkpoint_name": None,
                    "replan_enabled": None,
                }
            ],
        }
        migrated = ReasoningChain.migrate(legacy)
        chain = ReasoningChain.from_dict(migrated)
        assert chain.steps[0].title == "legacy"

    def test_exported_from_top_level(self) -> None:
        """``ChainFormatNewerError`` is importable as
        ``mmar_carl.ChainFormatNewerError`` for CARE's `except` clause."""
        from mmar_carl import ChainFormatNewerError as TopLevel
        assert TopLevel is ChainFormatNewerError
