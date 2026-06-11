"""Tests for the ``mmar-carl[viz]`` optional-extras gate.

Centralizes the matplotlib install hint that
``format_token_pie`` and ``format_score_evolution``
previously inlined, and registers ``viz`` as a first-class feature group
alongside ``vector-search`` / ``mcp`` / ``langfuse``.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from mmar_carl import _optional_deps as deps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matplotlib_missing() -> bool:
    try:
        import matplotlib  # noqa: F401

        return False
    except ImportError:
        return True


# ---------------------------------------------------------------------------
# Registration in feature groups
# ---------------------------------------------------------------------------


class TestVizFeatureGroupRegistered:
    def test_matplotlib_listed_in_optional_imports(self) -> None:
        assert "matplotlib" in deps._optional_imports

    def test_viz_listed_in_feature_groups(self) -> None:
        assert "viz" in deps._feature_groups
        assert "matplotlib" in deps._feature_groups["viz"]


# ---------------------------------------------------------------------------
# check_viz_available
# ---------------------------------------------------------------------------


class TestCheckVizAvailable:
    def test_returns_false_when_matplotlib_missing(self) -> None:
        if not _matplotlib_missing():
            pytest.skip("matplotlib is installed; mock test only valid otherwise")
        assert deps.check_viz_available() is False

    def test_returns_true_when_matplotlib_installed(self) -> None:
        if _matplotlib_missing():
            pytest.skip("matplotlib not installed")
        assert deps.check_viz_available() is True

    def test_returns_false_when_matplotlib_unimportable(self) -> None:
        """Simulate a missing matplotlib by blocking the import."""
        # Save and restore sys.modules so we don't pollute the session.
        saved = sys.modules.pop("matplotlib", None)
        try:
            # Inject a None entry so `import matplotlib` raises ImportError
            sys.modules["matplotlib"] = None  # type: ignore[assignment]
            assert deps.check_viz_available() is False
        finally:
            if saved is not None:
                sys.modules["matplotlib"] = saved
            else:
                sys.modules.pop("matplotlib", None)


# ---------------------------------------------------------------------------
# require_matplotlib
# ---------------------------------------------------------------------------


class TestRequireMatplotlib:
    def test_raises_with_clear_install_hint_when_missing(self) -> None:
        if not _matplotlib_missing():
            pytest.skip("matplotlib is installed; install-hint test only valid otherwise")
        with pytest.raises(ImportError) as exc_info:
            deps.require_matplotlib()
        msg = str(exc_info.value)
        assert "matplotlib" in msg
        assert "mmar-carl[viz]" in msg
        assert "pip install" in msg

    def test_returns_pyplot_when_installed(self) -> None:
        if _matplotlib_missing():
            pytest.skip("matplotlib not installed")
        plt = deps.require_matplotlib()
        # The returned object should be matplotlib.pyplot
        assert hasattr(plt, "subplots")
        assert hasattr(plt, "savefig")

    def test_sets_agg_backend_for_headless(self) -> None:
        """Calling require_matplotlib should force the Agg backend so PNG
        writes work in CI / headless environments."""
        if _matplotlib_missing():
            pytest.skip("matplotlib not installed")
        deps.require_matplotlib()
        import matplotlib

        # `Agg` (case-insensitive — matplotlib normalises)
        assert matplotlib.get_backend().lower() == "agg"

    def test_chains_from_none_for_clean_traceback(self) -> None:
        """The raised ImportError should use ``from None`` so the traceback
        doesn't show the underlying matplotlib ImportError noise."""
        if not _matplotlib_missing():
            pytest.skip("matplotlib is installed; chained-exception test only valid otherwise")
        with pytest.raises(ImportError) as exc_info:
            deps.require_matplotlib()
        # `from None` means __cause__ is explicitly None and __suppress_context__
        # is True (which suppresses the implicit __context__ in tracebacks).
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__suppress_context__ is True


# ---------------------------------------------------------------------------
# Existing PNG paths route through the new helper
# ---------------------------------------------------------------------------


class TestPngPathsRouteThroughHelper:
    def test_format_token_pie_uses_require_matplotlib(self) -> None:
        """When format='png' is invoked and matplotlib is missing, the new
        require_matplotlib() helper should be the source of the ImportError
        (not an inline try/import in results.py)."""
        from mmar_carl.models.enums import StepType
        from mmar_carl.models.results import ReasoningResult, StepExecutionResult

        r = ReasoningResult(
            success=True,
            history=[],
            step_results=[
                StepExecutionResult(
                    step_number=1,
                    step_title="s",
                    step_type=StepType.LLM,
                    result="",
                    success=True,
                    token_usage={"prompt": 100, "completion": 50, "total": 150},
                )
            ],
        )
        # Patch require_matplotlib to track that it's the dispatch path.
        with patch(
            "mmar_carl._optional_deps.require_matplotlib",
            side_effect=ImportError("custom test message"),
        ) as mock:
            with pytest.raises(ImportError, match="custom test message"):
                r.format_token_pie(format="png", png_path="/tmp/test.png")
            mock.assert_called_once()

    def test_format_score_evolution_uses_require_matplotlib(self) -> None:
        from mmar_carl.chain_evolution import EvolutionResult, GenerationStats

        result = EvolutionResult(
            best_chain_spec={},
            best_score=0.5,
            best_generation=0,
            history=[
                GenerationStats(
                    generation=0,
                    best_score=0.5,
                    mean_score=0.5,
                    population_scores=[0.5],
                ),
            ],
        )
        with patch(
            "mmar_carl._optional_deps.require_matplotlib",
            side_effect=ImportError("custom test message"),
        ) as mock:
            with pytest.raises(ImportError, match="custom test message"):
                result.format_score_evolution(format="png", png_path="/tmp/test.png")
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# pyproject.toml has the [viz] extra registered
# ---------------------------------------------------------------------------


def test_pyproject_declares_viz_extra() -> None:
    """Sanity-check: the [viz] extras line is wired in pyproject.toml so
    `pip install 'mmar-carl[viz]'` actually resolves."""
    from pathlib import Path

    pyproject = (
        Path(__file__).parent.parent.parent / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "viz = [" in pyproject
    assert "matplotlib" in pyproject
    # And the `all` extra rolls it up
    assert '"mmar-carl[viz]"' in pyproject
