"""Network policy contract for skill runtimes.

Standardises ``runtime_config["network"]`` semantics across backends
(``"none"`` / ``"allowlist"`` / ``"host"``) and the auto-allowlist
derivation from a skill manifest's ``WebFetch(domain:*)`` declarations.

The ``LocalSkillRuntime`` can't enforce any of this (subprocess inherits
host networking), but it validates the policy + stashes the resolved
state on the handle so CARE's TUI can render the contract as an
advisory banner. ``DockerSkillRuntime`` (next CARL-M0 loop) will
actually enforce.
"""

from __future__ import annotations

import warnings

import pytest

from mmar_carl import (
    LocalSkillRuntime,
    SkillRuntimeError,
    parse_network_allowlist_from_allowed_tools,
    resolve_network_policy,
)


# ---------------------------------------------------------------------------
# parse_network_allowlist_from_allowed_tools — manifest → allowlist
# ---------------------------------------------------------------------------


class TestParseAllowlistFromManifest:
    def test_none_returns_empty(self) -> None:
        assert parse_network_allowlist_from_allowed_tools(None) == []

    def test_empty_string_returns_empty(self) -> None:
        assert parse_network_allowlist_from_allowed_tools("") == []

    def test_empty_list_returns_empty(self) -> None:
        assert parse_network_allowlist_from_allowed_tools([]) == []

    def test_extracts_single_webfetch_domain(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            "WebFetch(domain:api.example.com)"
        )
        assert result == ["api.example.com"]

    def test_extracts_multiple_domains(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            "WebFetch(domain:api.x.com) WebFetch(domain:cdn.y.org) Bash(git:*)"
        )
        assert result == ["api.x.com", "cdn.y.org"]

    def test_skips_unconstrained_webfetch(self) -> None:
        """A bare ``WebFetch`` (no `domain:` constraint) is NOT
        auto-added — that's a deliberate "open network" signal which
        the chain author has to put in `runtime_config["network"]`
        themselves."""
        result = parse_network_allowlist_from_allowed_tools(
            "WebFetch Bash(git:*) Read"
        )
        assert result == []

    def test_skips_non_webfetch_tokens(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            "Bash(git:*) Read Write Edit"
        )
        assert result == []

    def test_accepts_pre_tokenised_list(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            ["Bash(git:*)", "WebFetch(domain:api.x.com)", "Read"],
        )
        assert result == ["api.x.com"]

    def test_dedupes_repeated_domains(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            "WebFetch(domain:api.x.com) WebFetch(domain:api.x.com)"
        )
        assert result == ["api.x.com"]

    def test_case_insensitive_webfetch_match(self) -> None:
        result = parse_network_allowlist_from_allowed_tools(
            "webfetch(domain:api.x.com)"
        )
        assert result == ["api.x.com"]


# ---------------------------------------------------------------------------
# resolve_network_policy — runtime_config → (policy, allowlist)
# ---------------------------------------------------------------------------


class TestResolveNetworkPolicy:
    def test_none_runtime_config_defaults_to_none(self) -> None:
        policy, allowlist = resolve_network_policy(None)
        assert policy == "none"
        assert allowlist == []

    def test_empty_runtime_config_defaults_to_none(self) -> None:
        policy, allowlist = resolve_network_policy({})
        assert policy == "none"
        assert allowlist == []

    def test_explicit_none_policy(self) -> None:
        policy, allowlist = resolve_network_policy({"network": "none"})
        assert policy == "none"
        assert allowlist == []

    def test_explicit_host_policy(self) -> None:
        policy, allowlist = resolve_network_policy({"network": "host"})
        assert policy == "host"
        assert allowlist == []

    def test_allowlist_policy_with_explicit_hosts(self) -> None:
        policy, allowlist = resolve_network_policy({
            "network": "allowlist",
            "network_allowlist": ["api.example.com", "cdn.x.com"],
        })
        assert policy == "allowlist"
        assert allowlist == ["api.example.com", "cdn.x.com"]

    def test_allowlist_merges_manifest_webfetch_domains(self) -> None:
        """When policy is ``allowlist``, manifest's
        ``WebFetch(domain:*)`` tokens are merged in — skill manifest
        becomes the source of truth for what egress the skill needs."""
        policy, allowlist = resolve_network_policy(
            {"network": "allowlist", "network_allowlist": ["explicit.com"]},
            manifest_allowed_tools="WebFetch(domain:manifest.com) Bash",
        )
        assert policy == "allowlist"
        assert allowlist == ["explicit.com", "manifest.com"]

    def test_allowlist_dedupes_across_sources(self) -> None:
        policy, allowlist = resolve_network_policy(
            {
                "network": "allowlist",
                "network_allowlist": ["api.x.com", "cdn.y.org"],
            },
            manifest_allowed_tools="WebFetch(domain:cdn.y.org) WebFetch(domain:other.io)",
        )
        assert policy == "allowlist"
        assert allowlist == ["api.x.com", "cdn.y.org", "other.io"]

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            resolve_network_policy({"network": "wide-open"})

    def test_non_list_allowlist_raises(self) -> None:
        with pytest.raises(SkillRuntimeError, match="network_allowlist"):
            resolve_network_policy({
                "network": "allowlist",
                "network_allowlist": "not-a-list.com",
            })

    def test_allowlist_with_no_hosts_is_empty(self) -> None:
        """An ``allowlist`` policy with no hosts is the most restrictive
        setting — effectively the same as ``none``. Backends that need
        to surface a friendly warning can detect this themselves."""
        policy, allowlist = resolve_network_policy({"network": "allowlist"})
        assert policy == "allowlist"
        assert allowlist == []

    def test_blank_hosts_dropped(self) -> None:
        policy, allowlist = resolve_network_policy({
            "network": "allowlist",
            "network_allowlist": ["api.x.com", "", "  "],
        })
        assert allowlist == ["api.x.com"]


# ---------------------------------------------------------------------------
# LocalSkillRuntime threads policy through the handle
# ---------------------------------------------------------------------------


class TestLocalRuntimeNetworkContract:
    @pytest.mark.asyncio
    async def test_default_policy_stamped_on_handle(self, tmp_path) -> None:
        runtime = LocalSkillRuntime()
        # Suppress the unrelated "unsafe local runtime" warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_allowlist"] == []
        # Local backend can't enforce — surface that explicitly.
        assert handle.backend["network_enforced"] is False
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_allowlist_policy_with_manifest_tokens(self, tmp_path) -> None:
        runtime = LocalSkillRuntime()
        cfg = {
            "network": "allowlist",
            "network_allowlist": ["explicit.com"],
            "_manifest_allowed_tools": "WebFetch(domain:from-manifest.io) Bash",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config=cfg,
            )
        assert handle.backend["network_policy"] == "allowlist"
        assert handle.backend["network_allowlist"] == ["explicit.com", "from-manifest.io"]
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_host_policy_emits_warning(self, tmp_path) -> None:
        runtime = LocalSkillRuntime()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "host"},
            )
        host_warnings = [
            w for w in caught
            if "network='host'" in str(w.message)
        ]
        assert host_warnings, "host policy must emit a UserWarning"
        assert handle.backend["network_policy"] == "host"
        await runtime.cleanup(handle)

    @pytest.mark.asyncio
    async def test_unknown_policy_raises_in_prepare(self, tmp_path) -> None:
        runtime = LocalSkillRuntime()
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )


# ---------------------------------------------------------------------------
# Type / export surface
# ---------------------------------------------------------------------------


def test_network_policy_is_a_literal_alias() -> None:
    """``NetworkPolicy`` is a typing Literal — pyright/mypy can narrow
    on it. Verify the runtime-visible string values."""
    # At runtime, Literal aliases don't enforce values, but our
    # functions raise on unknown strings. Spot-check the canonical ones.
    for valid in ("none", "allowlist", "host"):
        policy, _ = resolve_network_policy({"network": valid})
        assert policy == valid


def test_exports() -> None:
    """Every public name is reachable from ``mmar_carl`` top level."""
    import mmar_carl
    for name in (
        "NetworkPolicy",
        "parse_network_allowlist_from_allowed_tools",
        "resolve_network_policy",
    ):
        assert hasattr(mmar_carl, name), f"missing top-level export: {name}"
