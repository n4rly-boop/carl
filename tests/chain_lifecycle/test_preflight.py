"""Tests for Tool registry & pre-flight introspection.

`ReasoningChain.required_tools` / `required_mcp_servers` /
`required_skills` walk the chain's step configs to surface what the
chain will try to talk to. `preflight(context)` compares those against
the supplied context's tool registry and returns a `PreflightReport`.
"""

from __future__ import annotations

from mmar_carl import (
    AgentSkillSource,
    AgentSkillStepConfig,
    AgentSkillStepDescription,
    LLMStepDescription,
    MCPResourceStepConfig,
    MCPResourceStepDescription,
    MCPServerConfig,
    MCPStepConfig,
    MCPStepDescription,
    PreflightReport,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.models.llm_client_base import LLMClientBase


class _FakeClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "ok"


def _tool_step(n: int, name: str) -> ToolStepDescription:
    return ToolStepDescription(
        number=n, title=f"step_{n}",
        config=ToolStepConfig(tool_name=name),
    )


def _mcp_step(n: int, server: str) -> MCPStepDescription:
    return MCPStepDescription(
        number=n, title=f"mcp_{n}",
        config=MCPStepConfig(
            server=MCPServerConfig(
                server_name=server, transport="stdio", command="srv",
            ),
            tool_name="some_remote_tool",
        ),
    )


def _mcp_resource_step(n: int, server: str) -> MCPResourceStepDescription:
    return MCPResourceStepDescription(
        number=n, title=f"mcp_res_{n}",
        config=MCPResourceStepConfig(
            server=MCPServerConfig(
                server_name=server, transport="sse", url="https://x.test",
            ),
            resource_uri="docs://foo",
        ),
    )


def _skill_step(n: int, *, skill="github://x/y@main") -> AgentSkillStepDescription:
    return AgentSkillStepDescription(
        number=n, title=f"skill_{n}",
        config=AgentSkillStepConfig(skill=skill, task="do"),
    )


# ---------------------------------------------------------------------------
# required_tools
# ---------------------------------------------------------------------------


class TestRequiredTools:
    def test_collects_tool_names_from_tool_steps(self) -> None:
        chain = ReasoningChain(steps=[
            _tool_step(1, "fetch"),
            _tool_step(2, "store"),
        ])
        assert chain.required_tools() == ["fetch", "store"]

    def test_deduplicates_preserving_first_seen_order(self) -> None:
        chain = ReasoningChain(steps=[
            _tool_step(1, "fetch"),
            _tool_step(2, "store"),
            _tool_step(3, "fetch"),
        ])
        assert chain.required_tools() == ["fetch", "store"]

    def test_ignores_llm_steps(self) -> None:
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="P", aim="x"),
            LLMStepDescription(number=2, title="S", aim="x", dependencies=[1]),
        ])
        assert chain.required_tools() == []

    def test_does_not_pick_up_mcp_step_tool_name(self) -> None:
        """MCPStepConfig also has a ``tool_name`` field — but that's a
        REMOTE tool. It should NOT appear in required_tools."""
        chain = ReasoningChain(steps=[
            _tool_step(1, "fetch"),
            _mcp_step(2, "memory-server"),  # has tool_name="some_remote_tool"
        ])
        assert chain.required_tools() == ["fetch"]


# ---------------------------------------------------------------------------
# required_mcp_servers
# ---------------------------------------------------------------------------


class TestRequiredMcpServers:
    def test_collects_from_mcp_steps(self) -> None:
        chain = ReasoningChain(steps=[
            _mcp_step(1, "weather"),
            _mcp_step(2, "memory"),
        ])
        assert chain.required_mcp_servers() == ["weather", "memory"]

    def test_collects_from_mcp_resource_steps(self) -> None:
        chain = ReasoningChain(steps=[
            _mcp_resource_step(1, "docs-server"),
        ])
        assert chain.required_mcp_servers() == ["docs-server"]

    def test_deduplicates_across_step_kinds(self) -> None:
        chain = ReasoningChain(steps=[
            _mcp_step(1, "memory"),
            _mcp_resource_step(2, "memory"),
        ])
        assert chain.required_mcp_servers() == ["memory"]

    def test_empty_when_no_mcp_steps(self) -> None:
        chain = ReasoningChain(steps=[_tool_step(1, "fetch")])
        assert chain.required_mcp_servers() == []


# ---------------------------------------------------------------------------
# required_skills
# ---------------------------------------------------------------------------


class TestRequiredSkills:
    def test_uri_string_skill(self) -> None:
        chain = ReasoningChain(steps=[
            _skill_step(1, skill="github://anthropics/skills/skills/pdf@main"),
        ])
        # The AgentSkillSource validator may rewrite URI → git_url etc.
        # The important property is that the result is **non-empty** and
        # encodes enough info to round-trip through resolve_skill.
        skills = chain.required_skills()
        assert len(skills) == 1
        assert "anthropics" in skills[0] or skills[0].startswith("github://")

    def test_agent_skill_source_path(self) -> None:
        chain = ReasoningChain(steps=[
            _skill_step(1, skill=AgentSkillSource(path="/tmp/local-skill")),
        ])
        assert chain.required_skills() == ["/tmp/local-skill"]

    def test_agent_skill_source_name(self) -> None:
        chain = ReasoningChain(steps=[
            _skill_step(1, skill=AgentSkillSource(name="pdf")),
        ])
        assert chain.required_skills() == ["name://pdf"]

    def test_agent_skill_source_package(self) -> None:
        chain = ReasoningChain(steps=[
            _skill_step(1, skill=AgentSkillSource(package="my_pkg.skills.pdf")),
        ])
        assert chain.required_skills() == ["module://my_pkg.skills.pdf"]

    def test_dedup_across_steps(self) -> None:
        chain = ReasoningChain(steps=[
            _skill_step(1, skill=AgentSkillSource(path="/x")),
            _skill_step(2, skill=AgentSkillSource(path="/x")),
        ])
        assert chain.required_skills() == ["/x"]


# ---------------------------------------------------------------------------
# preflight(context)
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_all_present_when_every_tool_registered(self) -> None:
        chain = ReasoningChain(steps=[_tool_step(1, "fetch")])
        ctx = ReasoningContext(outer_context="N/A", api=_FakeClient())
        ctx.register_tool("fetch", lambda: "ok")
        report = chain.preflight(ctx)
        assert isinstance(report, PreflightReport)
        assert report.all_present is True
        assert report.missing_tools == []
        assert report.required_tools == ["fetch"]

    def test_missing_tool_reported(self) -> None:
        chain = ReasoningChain(steps=[
            _tool_step(1, "fetch"),
            _tool_step(2, "missing_tool"),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=_FakeClient())
        ctx.register_tool("fetch", lambda: "ok")
        report = chain.preflight(ctx)
        assert report.all_present is False
        assert report.missing_tools == ["missing_tool"]

    def test_no_context_treats_everything_as_missing(self) -> None:
        chain = ReasoningChain(steps=[_tool_step(1, "fetch")])
        report = chain.preflight(None)
        assert report.missing_tools == ["fetch"]
        assert report.all_present is False

    def test_mcp_and_skills_currently_diagnostic_only(self) -> None:
        chain = ReasoningChain(steps=[
            _mcp_step(1, "memory-server"),
            _skill_step(2, skill=AgentSkillSource(path="/tmp/x")),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=_FakeClient())
        report = chain.preflight(ctx)
        # Required lists populated
        assert report.required_mcp_servers == ["memory-server"]
        assert report.required_skills == ["/tmp/x"]
        # But missing-* always empty for these kinds (diagnostic only)
        assert report.missing_mcp_servers == []
        assert report.missing_skills == []
        # And because nothing is missing → all_present True
        assert report.all_present is True


# ---------------------------------------------------------------------------
# PreflightReport rendering
# ---------------------------------------------------------------------------


class TestReportRendering:
    def test_format_text_ok_path(self) -> None:
        report = PreflightReport(
            required_tools=["a", "b"],
            required_mcp_servers=["x"],
            required_skills=[],
        )
        out = report.format_text()
        assert "preflight: ok" in out
        assert "2 tools" in out
        assert "1 mcp server" in out
        assert "0 skills" in out

    def test_format_text_missing_path(self) -> None:
        report = PreflightReport(
            required_tools=["a", "b"],
            missing_tools=["b"],
        )
        out = report.format_text()
        assert "missing dependencies" in out
        assert "tools: b" in out

    def test_format_text_pluralisation(self) -> None:
        report = PreflightReport(required_tools=["only"])
        out = report.format_text()
        assert "1 tool" in out
        assert "1 tools" not in out  # singular when count==1

    def test_all_present_property_reflects_every_missing_list(self) -> None:
        assert PreflightReport().all_present is True
        assert PreflightReport(missing_tools=["x"]).all_present is False
        assert PreflightReport(missing_skills=["x"]).all_present is False
        assert PreflightReport(missing_mcp_servers=["x"]).all_present is False


# ---------------------------------------------------------------------------
# Mixed-step end-to-end
# ---------------------------------------------------------------------------


class TestMixedChain:
    def test_mixed_chain_full_report(self) -> None:
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="P", aim="x"),
            _tool_step(2, "fetch"),
            _tool_step(3, "fetch"),  # dup
            _tool_step(4, "store"),
            _mcp_step(5, "weather"),
            _mcp_resource_step(6, "docs"),
            _skill_step(7, skill=AgentSkillSource(path="/x")),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=_FakeClient())
        ctx.register_tool("fetch", lambda: "ok")
        # Note: 'store' is NOT registered
        report = chain.preflight(ctx)
        assert report.required_tools == ["fetch", "store"]
        assert report.required_mcp_servers == ["weather", "docs"]
        assert report.required_skills == ["/x"]
        assert report.missing_tools == ["store"]
        assert report.all_present is False
