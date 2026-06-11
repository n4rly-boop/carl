"""Preflight report — what does this chain need before it can run?

CARE's library re-run flow needs to know what tools / MCP servers /
AgentSkills a saved chain will try to invoke so the user can register
or install the missing pieces before execution. CARL ships static
introspection helpers on :class:`ReasoningChain`
(``required_tools`` / ``required_mcp_servers`` / ``required_skills``)
and a :meth:`ReasoningChain.preflight` that compares those against a
fresh :class:`ReasoningContext`'s registry and returns the gap.

The report is intentionally **structured, not narrated**: CARE's TUI
turns it into a modal dialog, and other callers (CLI, CI, web UI) may
want the raw lists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PreflightReport(BaseModel):
    """Result of :meth:`ReasoningChain.preflight`.

    Fields are grouped into ``required_*`` (everything the chain
    references) and ``missing_*`` (a subset — only the items the
    given context can't satisfy). ``all_present`` is the obvious
    convenience flag.
    """

    required_tools: list[str] = Field(
        default_factory=list,
        description="De-duplicated tool names the chain will try to invoke.",
    )
    required_mcp_servers: list[str] = Field(
        default_factory=list,
        description=(
            "De-duplicated MCP server names referenced by MCP steps. "
            "MCP servers are self-described by their step config (transport "
            "+ command/URL), so this list is diagnostic — CARE's TUI shows "
            "it next to the run button so users can sanity-check what the "
            "chain is about to dial out to."
        ),
    )
    required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "AgentSkill identifiers — URI strings when the user supplied "
            "URI-form skills (``github://...``, ``module://...``, etc.) or "
            "the resolved ``path`` / ``name`` / ``git_url`` / ``package`` "
            "string when an ``AgentSkillSource`` was passed directly."
        ),
    )

    missing_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tools referenced by ``ToolStepDescription`` that are not "
            "registered in the supplied context's tool registry. CARE "
            "shows these as a 'register before running' list."
        ),
    )
    missing_mcp_servers: list[str] = Field(
        default_factory=list,
        description=(
            "Currently always empty — MCP step configs carry their own "
            "transport details, so 'missing' isn't well-defined without a "
            "registry. Reserved for future work."
        ),
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Currently always empty — skill resolution is async + network-"
            "bound, so CARE handles the resolve-and-cache step itself. "
            "Reserved for future work."
        ),
    )

    @property
    def all_present(self) -> bool:
        """``True`` when every required item resolves in the supplied context."""
        return not (
            self.missing_tools or self.missing_mcp_servers or self.missing_skills
        )

    def format_text(self) -> str:
        """Human-readable summary suitable for stdout / CARE's TUI footer.

        Returns a single line when nothing is missing, a multi-line
        breakdown otherwise.
        """
        if self.all_present:
            n_t = len(self.required_tools)
            n_m = len(self.required_mcp_servers)
            n_s = len(self.required_skills)
            return (
                f"preflight: ok ({n_t} tool{'s' if n_t != 1 else ''}, "
                f"{n_m} mcp server{'s' if n_m != 1 else ''}, "
                f"{n_s} skill{'s' if n_s != 1 else ''})"
            )
        lines = ["preflight: missing dependencies"]
        if self.missing_tools:
            lines.append(
                f"  tools: {', '.join(self.missing_tools)}"
            )
        if self.missing_mcp_servers:
            lines.append(
                f"  mcp servers: {', '.join(self.missing_mcp_servers)}"
            )
        if self.missing_skills:
            lines.append(
                f"  skills: {', '.join(self.missing_skills)}"
            )
        return "\n".join(lines)


__all__ = ["PreflightReport"]
