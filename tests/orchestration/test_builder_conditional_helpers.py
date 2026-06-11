"""
Tests for ChainBuilder.add_if_else and add_switch convenience helpers.

Covers:
- add_if_else: basic routing, condition_step auto-dependency, default branch
- add_switch: multi-way routing, "default" key, condition_step dependency
- Both integrate correctly with ConditionalStepExecutor
- Branch steps that lose the condition are skipped
"""

import pytest
from mmar_carl import ChainBuilder, ChainTestHarness


def _tool(name: str) -> dict:
    """Shorthand for a ToolStepConfig dict args."""
    return {"tool_name": name, "input_mapping": {}}


# ---------------------------------------------------------------------------
# add_if_else structure tests
# ---------------------------------------------------------------------------


class TestAddIfElseStructure:
    def test_builds_without_error(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_if_else(2, "Route", condition="'A' in value", if_true=3, if_false=4, condition_step=1)
            .add_tool_step(3, "Branch A", **_tool("branch_a"))
            .add_tool_step(4, "Branch B", **_tool("branch_b"))
            .build()
        )
        assert len(chain.steps) == 4

    def test_condition_step_added_to_dependencies(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Producer", **_tool("produce"))
            .add_if_else(2, "Route", condition="nonempty", if_true=3, if_false=4, condition_step=1)
            .add_tool_step(3, "A", **_tool("a"))
            .add_tool_step(4, "B", **_tool("b"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 2)
        assert 1 in route_step.dependencies

    def test_no_condition_step_means_no_auto_dependency(self):
        chain = (
            ChainBuilder()
            .add_if_else(1, "Route", condition="nonempty", if_true=2, if_false=3)
            .add_tool_step(2, "A", **_tool("a"))
            .add_tool_step(3, "B", **_tool("b"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 1)
        assert route_step.dependencies == []

    def test_if_else_creates_single_branch_plus_default(self):
        chain = (
            ChainBuilder()
            .add_if_else(1, "Route", condition="nonempty", if_true=2, if_false=3)
            .add_tool_step(2, "A", **_tool("a"))
            .add_tool_step(3, "B", **_tool("b"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 1)
        cfg = route_step.step_config
        assert len(cfg.branches) == 1
        assert cfg.default_step == 3


# ---------------------------------------------------------------------------
# add_if_else execution tests
# ---------------------------------------------------------------------------


class TestAddIfElseExecution:
    @pytest.mark.asyncio
    async def test_if_true_branch_runs_when_condition_matches(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_if_else(
                2, "Route",
                condition="'yes' in value",
                if_true=3,
                if_false=4,
                condition_step=1,
            )
            .add_tool_step(3, "Yes branch", dependencies=[2], **_tool("yes_branch"))
            .add_tool_step(4, "No branch", dependencies=[2], **_tool("no_branch"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "yes please")
        harness.set_tool_response("yes_branch", "yes output")
        await harness.run("ctx")
        harness.assert_step_called(3)
        harness.assert_step_not_called(4)

    @pytest.mark.asyncio
    async def test_if_false_branch_runs_when_condition_does_not_match(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_if_else(
                2, "Route",
                condition="'yes' in value",
                if_true=3,
                if_false=4,
                condition_step=1,
            )
            .add_tool_step(3, "Yes branch", dependencies=[2], **_tool("yes_branch"))
            .add_tool_step(4, "No branch", dependencies=[2], **_tool("no_branch"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "no way")
        harness.set_tool_response("no_branch", "no output")
        await harness.run("ctx")
        harness.assert_step_called(4)
        harness.assert_step_not_called(3)

    @pytest.mark.asyncio
    async def test_if_else_uses_history_context_key_by_default(self):
        """condition_context_key="$history[-1]" reads previous history entry."""
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_if_else(
                2, "Route",
                condition="'URGENT' in value",
                if_true=3,
                if_false=4,
                condition_step=1,
            )
            .add_tool_step(3, "Urgent", dependencies=[2], **_tool("urgent"))
            .add_tool_step(4, "Normal", dependencies=[2], **_tool("normal"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "URGENT request")
        harness.set_tool_response("urgent", "handled urgent")
        await harness.run("")
        harness.assert_step_called(3)
        harness.assert_step_not_called(4)


# ---------------------------------------------------------------------------
# add_switch structure tests
# ---------------------------------------------------------------------------


class TestAddSwitchStructure:
    def test_builds_without_error(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Source", **_tool("source"))
            .add_switch(
                2, "Route",
                branches={
                    "contains:urgent": 3,
                    "contains:billing": 4,
                    "default": 5,
                },
                condition_step=1,
            )
            .add_tool_step(3, "Urgent", **_tool("urgent"))
            .add_tool_step(4, "Billing", **_tool("billing"))
            .add_tool_step(5, "General", **_tool("general"))
            .build()
        )
        assert len(chain.steps) == 5

    def test_condition_step_auto_dependency(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Source", **_tool("source"))
            .add_switch(2, "Route", branches={"default": 3}, condition_step=1)
            .add_tool_step(3, "Next", **_tool("next"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 2)
        assert 1 in route_step.dependencies

    def test_default_key_becomes_default_step(self):
        chain = (
            ChainBuilder()
            .add_switch(
                1, "Route",
                branches={"contains:foo": 2, "default": 3},
            )
            .add_tool_step(2, "Foo", **_tool("foo"))
            .add_tool_step(3, "Other", **_tool("other"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 1)
        cfg = route_step.step_config
        assert cfg.default_step == 3
        assert len(cfg.branches) == 1
        assert cfg.branches[0].condition == "contains:foo"

    def test_no_default_key_means_no_default_step(self):
        chain = (
            ChainBuilder()
            .add_switch(
                1, "Route",
                branches={"contains:foo": 2, "contains:bar": 3},
            )
            .add_tool_step(2, "Foo", **_tool("foo"))
            .add_tool_step(3, "Bar", **_tool("bar"))
            .build()
        )
        route_step = next(s for s in chain.steps if getattr(s, "number", None) == 1)
        assert route_step.step_config.default_step is None
        assert len(route_step.step_config.branches) == 2


# ---------------------------------------------------------------------------
# add_switch execution tests
# ---------------------------------------------------------------------------


class TestAddSwitchExecution:
    @pytest.mark.asyncio
    async def test_first_matching_branch_wins(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_switch(
                2, "Route",
                branches={
                    "contains:urgent": 3,
                    "contains:billing": 4,
                    "default": 5,
                },
                condition_step=1,
            )
            .add_tool_step(3, "Urgent handler", dependencies=[2], **_tool("urgent"))
            .add_tool_step(4, "Billing handler", dependencies=[2], **_tool("billing"))
            .add_tool_step(5, "General handler", dependencies=[2], **_tool("general"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "billing issue")
        harness.set_tool_response("billing", "billing result")
        await harness.run("")
        harness.assert_step_called(4)
        harness.assert_step_not_called(3)
        harness.assert_step_not_called(5)

    @pytest.mark.asyncio
    async def test_default_branch_runs_when_no_condition_matches(self):
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_switch(
                2, "Route",
                branches={
                    "contains:urgent": 3,
                    "default": 4,
                },
                condition_step=1,
            )
            .add_tool_step(3, "Urgent", dependencies=[2], **_tool("urgent"))
            .add_tool_step(4, "Fallback", dependencies=[2], **_tool("fallback"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "routine request")
        harness.set_tool_response("fallback", "fallback result")
        await harness.run("")
        harness.assert_step_called(4)
        harness.assert_step_not_called(3)

    @pytest.mark.asyncio
    async def test_no_match_no_default_skips_branches(self):
        """Without default, no-match routes to None, branch steps are skipped."""
        chain = (
            ChainBuilder()
            .add_tool_step(1, "Classify", **_tool("classify"))
            .add_switch(
                2, "Route",
                branches={"contains:urgent": 3},
                condition_step=1,
            )
            .add_tool_step(3, "Urgent", dependencies=[2], **_tool("urgent"))
            .build()
        )
        harness = ChainTestHarness(chain)
        harness.set_tool_response("classify", "nothing matched")
        harness.set_tool_response("urgent", "urgent result")
        await harness.run("")
        harness.assert_step_not_called(3)
