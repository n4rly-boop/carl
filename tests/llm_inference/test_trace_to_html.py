"""Tests for ``ExecutionTrace.to_html``.

Stand-alone HTML/JS playback export. Verifies the rendered HTML
contains the expected scaffolding (head, scripts, styles), bakes the
events into a JSON payload that JS can hydrate, and persists to disk
when ``path`` is provided.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mmar_carl.execution_trace import ExecutionTrace, TraceEvent
from mmar_carl.models.enums import StepType


def _event(
    num: int,
    *,
    title: str = "Step",
    step_type: StepType = StepType.LLM,
    success: bool = True,
    skipped: bool = False,
    execution_time: float | None = 0.1,
    result: str = "ok",
    error_message: str | None = None,
    token_usage: dict[str, int] | None = None,
    batch_index: int = 0,
    inputs: dict | None = None,
    result_data=None,
) -> TraceEvent:
    return TraceEvent(
        step_number=num, step_title=title, step_type=step_type,
        success=success, skipped=skipped, execution_time=execution_time,
        result=result, error_message=error_message,
        token_usage=token_usage or {}, batch_index=batch_index,
        inputs=inputs or {}, result_data=result_data,
    )


def _trace(*events: TraceEvent, **kwargs) -> ExecutionTrace:
    return ExecutionTrace(
        chain_title=kwargs.get("title", "T"),
        success=kwargs.get("success", True),
        total_execution_time=kwargs.get("total", sum(e.execution_time or 0 for e in events)),
        events=list(events),
    )


# ---------------------------------------------------------------------------
# Return value + file persistence
# ---------------------------------------------------------------------------


class TestReturnAndPersistence:
    def test_returns_html_string(self) -> None:
        t = _trace(_event(1))
        html = t.to_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_path_none_does_not_write_file(self, tmp_path: Path) -> None:
        t = _trace(_event(1))
        before = list(tmp_path.iterdir())
        t.to_html(None)
        after = list(tmp_path.iterdir())
        assert before == after

    def test_path_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "playback.html"
        t = _trace(_event(1))
        result = t.to_html(out)
        assert out.exists()
        # Same content returned and written
        assert out.read_text(encoding="utf-8") == result

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "playback.html"
        _trace(_event(1)).to_html(out)
        assert out.exists()


# ---------------------------------------------------------------------------
# HTML structure
# ---------------------------------------------------------------------------


class TestStructure:
    def test_contains_chain_title(self) -> None:
        html = _trace(_event(1), title="My Pipeline").to_html()
        assert "My Pipeline" in html

    def test_unnamed_chain_placeholder(self) -> None:
        html = _trace(_event(1), title="").to_html()
        assert "(unnamed chain)" in html

    def test_status_banner_success(self) -> None:
        html = _trace(_event(1), success=True).to_html()
        assert "✅ success" in html

    def test_status_banner_failure(self) -> None:
        html = _trace(_event(1, success=False), success=False).to_html()
        assert "❌ failed" in html

    def test_total_elapsed_time_shown(self) -> None:
        html = _trace(_event(1, execution_time=1.5), total=1.5).to_html()
        assert "1.50s" in html

    def test_step_count_shown_in_header(self) -> None:
        html = _trace(_event(1), _event(2), _event(3)).to_html()
        assert "3 step(s)" in html

    def test_inline_style_and_script_blocks(self) -> None:
        html = _trace(_event(1)).to_html()
        # Standalone — both <style> and <script> are inline
        assert "<style>" in html
        assert "</style>" in html
        assert "<script>" in html
        assert "</script>" in html
        # No external dependencies
        assert "src=\"http" not in html
        assert "href=\"http" not in html

    def test_play_pause_step_reset_controls_present(self) -> None:
        html = _trace(_event(1)).to_html()
        for btn in ("Play", "Pause", "Step", "Reset"):
            assert btn in html


# ---------------------------------------------------------------------------
# Embedded EVENTS payload
# ---------------------------------------------------------------------------


def _extract_events_json(html: str) -> list[dict]:
    """Pull the EVENTS = [...] payload out of the embedded JS."""
    # Match the JSON array — both empty ``[]`` and multi-line forms.
    m = re.search(r"const EVENTS = (\[[\s\S]*?\]);", html)
    assert m, "EVENTS payload not found"
    return json.loads(m.group(1))


class TestEmbeddedPayload:
    def test_events_serialised_to_json(self) -> None:
        t = _trace(
            _event(1, title="A", batch_index=0),
            _event(2, title="B", batch_index=1),
        )
        events = _extract_events_json(t.to_html())
        assert len(events) == 2
        assert events[0]["step_number"] == 1
        assert events[1]["step_number"] == 2

    def test_token_usage_round_trips(self) -> None:
        t = _trace(_event(1, token_usage={"prompt": 100, "completion": 50, "total": 150}))
        events = _extract_events_json(t.to_html())
        assert events[0]["token_usage"] == {"prompt": 100, "completion": 50, "total": 150}

    def test_error_message_round_trips(self) -> None:
        t = _trace(_event(1, success=False, error_message="boom"))
        events = _extract_events_json(t.to_html())
        assert events[0]["error_message"] == "boom"

    def test_step_type_serialised_as_string(self) -> None:
        t = _trace(_event(1, step_type=StepType.TOOL))
        events = _extract_events_json(t.to_html())
        # to_dict converts StepType to str() form ("StepType.TOOL" via __str__)
        assert "TOOL" in events[0]["step_type"] or "tool" in events[0]["step_type"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_events_renders_valid_html(self) -> None:
        t = ExecutionTrace(chain_title="empty", success=True, events=[])
        html = t.to_html()
        assert "<!DOCTYPE html>" in html
        # Empty EVENTS array still present
        events = _extract_events_json(html)
        assert events == []

    def test_unicode_in_title_preserved(self) -> None:
        html = _trace(_event(1), title="Демонстрация 🎉").to_html()
        assert "Демонстрация" in html
        assert "🎉" in html

    def test_close_tag_in_title_escaped(self) -> None:
        """The title is sliced into the page <head> via str.format; a
        raw ``</script>`` would break the parser. We pre-escape it."""
        html = _trace(_event(1), title="bad</script>title").to_html()
        # The literal close-tag must not appear in the rendered HTML
        assert "bad</script>title" not in html
        # But the title is still recognisable
        assert "bad" in html


# ---------------------------------------------------------------------------
# Method existence
# ---------------------------------------------------------------------------


class TestExposure:
    def test_method_available(self) -> None:
        assert callable(ExecutionTrace.to_html)
