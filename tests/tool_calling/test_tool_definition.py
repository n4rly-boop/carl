"""
Tests for the provider-agnostic ``ToolDefinition``.

Verifies round-trip conversion between the canonical model and OpenAI /
Anthropic native tool-call formats, plus extras preservation and error
handling.
"""

import pytest

from mmar_carl import ToolDefinition


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_minimal_construction() -> None:
    t = ToolDefinition(name="x")
    assert t.name == "x"
    assert t.description == ""
    assert t.parameters == {"type": "object", "properties": {}}
    assert t.extras == {}


def test_name_required_non_empty() -> None:
    with pytest.raises(Exception):
        ToolDefinition(name="")


def test_full_construction() -> None:
    t = ToolDefinition(
        name="search",
        description="Web search",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        extras={"strict": True},
    )
    assert t.name == "search"
    assert t.parameters["required"] == ["q"]


# --------------------------------------------------------------------------- #
# OpenAI format
# --------------------------------------------------------------------------- #


def test_to_openai_wraps_in_function() -> None:
    t = ToolDefinition(name="x", description="d", parameters={"type": "object"})
    out = t.to_openai_dict()
    assert out["type"] == "function"
    assert out["function"]["name"] == "x"
    assert out["function"]["description"] == "d"
    assert out["function"]["parameters"] == {"type": "object"}


def test_to_openai_includes_strict_extra() -> None:
    t = ToolDefinition(name="x", extras={"strict": True})
    out = t.to_openai_dict()
    assert out["function"]["strict"] is True


def test_to_openai_omits_strict_when_absent() -> None:
    t = ToolDefinition(name="x")
    out = t.to_openai_dict()
    assert "strict" not in out["function"]


def test_from_openai_wrapped_shape() -> None:
    data = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    t = ToolDefinition.from_openai_dict(data)
    assert t.name == "search"
    assert t.description == "Web search"
    assert t.parameters["properties"]["q"]["type"] == "string"


def test_from_openai_bare_shape() -> None:
    data = {
        "name": "search",
        "description": "Web search",
        "parameters": {"type": "object"},
    }
    t = ToolDefinition.from_openai_dict(data)
    assert t.name == "search"
    assert t.parameters == {"type": "object"}


def test_from_openai_preserves_strict() -> None:
    data = {
        "type": "function",
        "function": {"name": "x", "parameters": {"type": "object"}, "strict": True},
    }
    t = ToolDefinition.from_openai_dict(data)
    assert t.extras["strict"] is True


def test_from_openai_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="missing 'name'"):
        ToolDefinition.from_openai_dict({"type": "function", "function": {"parameters": {}}})


def test_from_openai_non_dict_raises() -> None:
    with pytest.raises(TypeError):
        ToolDefinition.from_openai_dict("not a dict")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Anthropic format
# --------------------------------------------------------------------------- #


def test_to_anthropic_uses_input_schema_key() -> None:
    t = ToolDefinition(
        name="x",
        description="d",
        parameters={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    out = t.to_anthropic_dict()
    assert out["name"] == "x"
    assert out["description"] == "d"
    assert out["input_schema"]["properties"]["q"]["type"] == "string"
    assert "parameters" not in out


def test_to_anthropic_includes_cache_control_extra() -> None:
    t = ToolDefinition(name="x", extras={"cache_control": {"type": "ephemeral"}})
    out = t.to_anthropic_dict()
    assert out["cache_control"] == {"type": "ephemeral"}


def test_from_anthropic_round_trip() -> None:
    data = {
        "name": "search",
        "description": "Web search",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    }
    t = ToolDefinition.from_anthropic_dict(data)
    assert t.name == "search"
    assert t.description == "Web search"
    assert t.parameters["required"] == ["q"]


def test_from_anthropic_preserves_cache_control() -> None:
    data = {
        "name": "x",
        "input_schema": {"type": "object"},
        "cache_control": {"type": "ephemeral"},
    }
    t = ToolDefinition.from_anthropic_dict(data)
    assert t.extras["cache_control"] == {"type": "ephemeral"}


def test_from_anthropic_defaults_input_schema_when_absent() -> None:
    t = ToolDefinition.from_anthropic_dict({"name": "x"})
    assert t.parameters == {"type": "object", "properties": {}}


def test_from_anthropic_missing_name_raises() -> None:
    with pytest.raises(ValueError, match="missing 'name'"):
        ToolDefinition.from_anthropic_dict({"input_schema": {}})


# --------------------------------------------------------------------------- #
# Cross-provider round trips (lossless on the canonical subset)
# --------------------------------------------------------------------------- #


def test_openai_to_canonical_to_openai_is_identity() -> None:
    data = {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Calculator",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
    t = ToolDefinition.from_openai_dict(data)
    assert t.to_openai_dict() == data


def test_anthropic_to_canonical_to_anthropic_is_identity() -> None:
    data = {
        "name": "calc",
        "description": "Calculator",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "required": ["a"],
        },
    }
    t = ToolDefinition.from_anthropic_dict(data)
    assert t.to_anthropic_dict() == data


def test_openai_to_anthropic_cross_conversion() -> None:
    """Converting OpenAI → canonical → Anthropic preserves the schema."""
    openai_data = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
    anthropic_data = ToolDefinition.from_openai_dict(openai_data).to_anthropic_dict()
    assert anthropic_data["name"] == "search"
    assert anthropic_data["description"] == "Search the web"
    assert anthropic_data["input_schema"] == openai_data["function"]["parameters"]


def test_anthropic_to_openai_cross_conversion() -> None:
    anthropic_data = {
        "name": "calc",
        "description": "Calc",
        "input_schema": {"type": "object", "properties": {"a": {"type": "number"}}},
    }
    openai_data = ToolDefinition.from_anthropic_dict(anthropic_data).to_openai_dict()
    assert openai_data["type"] == "function"
    assert openai_data["function"]["name"] == "calc"
    assert openai_data["function"]["parameters"] == anthropic_data["input_schema"]


# --------------------------------------------------------------------------- #
# Provider dispatcher
# --------------------------------------------------------------------------- #


def test_to_provider_dict_openai_alias() -> None:
    t = ToolDefinition(name="x")
    for alias in ("openai", "OpenAI", "openai-compat", "openrouter", "AZURE"):
        out = t.to_provider_dict(alias)
        assert out["type"] == "function"


def test_to_provider_dict_anthropic() -> None:
    t = ToolDefinition(name="x")
    out = t.to_provider_dict("anthropic")
    assert "input_schema" in out


def test_to_provider_dict_unknown_provider_raises() -> None:
    t = ToolDefinition(name="x")
    with pytest.raises(ValueError, match="Unsupported provider"):
        t.to_provider_dict("cohere")


def test_from_provider_dict_dispatches() -> None:
    t_openai = ToolDefinition.from_provider_dict(
        "openai",
        {"type": "function", "function": {"name": "x", "parameters": {}}},
    )
    assert t_openai.name == "x"
    t_anthropic = ToolDefinition.from_provider_dict(
        "anthropic", {"name": "y", "input_schema": {"type": "object"}}
    )
    assert t_anthropic.name == "y"


# --------------------------------------------------------------------------- #
# Immutability of parameters via rendered dicts
# --------------------------------------------------------------------------- #


def test_rendered_openai_dict_is_decoupled_from_source() -> None:
    """Mutating the rendered dict should not change the source ToolDefinition."""
    t = ToolDefinition(name="x", parameters={"type": "object", "properties": {"a": {"type": "string"}}})
    rendered = t.to_openai_dict()
    rendered["function"]["parameters"]["properties"]["a"]["type"] = "integer"
    assert t.parameters["properties"]["a"]["type"] == "string"


def test_rendered_anthropic_dict_is_decoupled_from_source() -> None:
    t = ToolDefinition(name="x", parameters={"type": "object", "properties": {"a": {"type": "string"}}})
    rendered = t.to_anthropic_dict()
    rendered["input_schema"]["properties"]["a"]["type"] = "integer"
    assert t.parameters["properties"]["a"]["type"] == "string"


def test_parsed_definition_is_decoupled_from_source_dict() -> None:
    """from_openai_dict should not share refs with the input dict."""
    data = {
        "type": "function",
        "function": {
            "name": "x",
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        },
    }
    t = ToolDefinition.from_openai_dict(data)
    data["function"]["parameters"]["properties"]["a"]["type"] = "integer"
    assert t.parameters["properties"]["a"]["type"] == "string"


# --------------------------------------------------------------------------- #
# Bulk helpers
# --------------------------------------------------------------------------- #


def test_render_many() -> None:
    tools = [
        ToolDefinition(name="a"),
        ToolDefinition(name="b", description="bee"),
    ]
    rendered = ToolDefinition.render_many(tools, "openai")
    assert len(rendered) == 2
    assert rendered[0]["function"]["name"] == "a"
    assert rendered[1]["function"]["description"] == "bee"


def test_parse_many_round_trip() -> None:
    tools = [
        ToolDefinition(name="a", description="alpha"),
        ToolDefinition(name="b", description="beta"),
    ]
    rendered = ToolDefinition.render_many(tools, "anthropic")
    parsed = ToolDefinition.parse_many("anthropic", rendered)
    assert [t.name for t in parsed] == ["a", "b"]
    assert [t.description for t in parsed] == ["alpha", "beta"]


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_default_parameters_render_correctly_in_both_formats() -> None:
    t = ToolDefinition(name="noargs")
    oai = t.to_openai_dict()
    assert oai["function"]["parameters"] == {"type": "object", "properties": {}}
    ant = t.to_anthropic_dict()
    assert ant["input_schema"] == {"type": "object", "properties": {}}


def test_empty_extras_does_not_leak_keys() -> None:
    t = ToolDefinition(name="x")
    oai = t.to_openai_dict()
    ant = t.to_anthropic_dict()
    assert "strict" not in oai["function"]
    assert "cache_control" not in ant
