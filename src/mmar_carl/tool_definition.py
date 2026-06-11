"""
Provider-agnostic tool definitions for CARL.

``ToolDefinition`` is a thin pydantic wrapper around the
``(name, description, input_schema)`` triple every modern LLM provider needs
for native tool calling. Converters render it into the dict shape each
provider expects, and parsers ingest the same shapes back into the canonical
model — handy for adapter code that has to bridge between providers.

OpenAI shape::

    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search",
            "parameters": {"type": "object", "properties": {...}, "required": [...]},
        },
    }

Anthropic shape::

    {
        "name": "search",
        "description": "Web search",
        "input_schema": {"type": "object", "properties": {...}, "required": [...]},
    }

Round-tripping is lossless for the common subset (name / description / JSON
schema parameters). Provider-specific extensions like OpenAI's
``"strict": true`` flag and Anthropic's ``cache_control`` block are
preserved verbatim in :attr:`ToolDefinition.extras` so callers don't lose
data when re-emitting.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """
    Canonical tool definition that converts to any supported provider format.

    Fields:
        name: Tool name (the LLM uses this to refer to the tool when calling it).
        description: Short docstring shown to the LLM.
        parameters: JSON Schema object describing the tool's input parameters.
            Must be an ``"object"``-typed schema for compatibility with OpenAI.
        extras: Provider-specific extensions preserved across round-trips
            (e.g. OpenAI's ``"strict"``, Anthropic's ``"cache_control"``).
    """

    name: str = Field(..., min_length=1, description="Tool name.")
    description: str = Field(default="", description="Short description shown to the LLM.")
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description=(
            "JSON Schema for the tool's input parameters. Must be an "
            "``'object'``-typed schema."
        ),
    )
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Provider-specific extensions preserved across format conversions "
            "(e.g. OpenAI's ``strict``, Anthropic's ``cache_control``)."
        ),
    )

    # --------------------------------------------------------------- #
    # OpenAI format
    # --------------------------------------------------------------- #

    def to_openai_dict(self) -> dict[str, Any]:
        """Render to OpenAI's ``{"type": "function", "function": {...}}`` shape."""
        function_body: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": _deepcopy_dict(self.parameters),
        }
        # OpenAI's per-function "strict" flag lives inside the function body.
        if "strict" in self.extras:
            function_body["strict"] = self.extras["strict"]
        return {"type": "function", "function": function_body}

    @classmethod
    def from_openai_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Parse an OpenAI-format dict. Accepts both the wrapped and unwrapped shapes.

        Accepts either::

            {"type": "function", "function": {"name": ..., "parameters": ...}}

        or the bare inner ``{"name": ..., "parameters": ...}`` form.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if data.get("type") == "function" and isinstance(data.get("function"), dict):
            body = data["function"]
        else:
            body = data
        name = body.get("name")
        if not name:
            raise ValueError("Tool definition is missing 'name'")
        extras: dict[str, Any] = {}
        if "strict" in body:
            extras["strict"] = body["strict"]
        return cls(
            name=name,
            description=body.get("description", "") or "",
            parameters=_deepcopy_dict(body.get("parameters") or {"type": "object", "properties": {}}),
            extras=extras,
        )

    # --------------------------------------------------------------- #
    # Anthropic format
    # --------------------------------------------------------------- #

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Render to Anthropic's ``{"name": ..., "input_schema": ...}`` shape."""
        out: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": _deepcopy_dict(self.parameters),
        }
        if "cache_control" in self.extras:
            out["cache_control"] = self.extras["cache_control"]
        return out

    @classmethod
    def from_anthropic_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Parse an Anthropic-format dict (``{"name": ..., "input_schema": ...}``)."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        name = data.get("name")
        if not name:
            raise ValueError("Tool definition is missing 'name'")
        schema = data.get("input_schema")
        if schema is None:
            schema = {"type": "object", "properties": {}}
        extras: dict[str, Any] = {}
        if "cache_control" in data:
            extras["cache_control"] = data["cache_control"]
        return cls(
            name=name,
            description=data.get("description", "") or "",
            parameters=_deepcopy_dict(schema),
            extras=extras,
        )

    # --------------------------------------------------------------- #
    # Dispatcher
    # --------------------------------------------------------------- #

    def to_provider_dict(self, provider: str) -> dict[str, Any]:
        """Render to *provider*'s native format.

        Recognised provider names (case-insensitive): ``openai``, ``anthropic``.
        Aliases: ``"openai-compat"`` / ``"openrouter"`` / ``"azure"`` all map
        to the OpenAI shape since they share the same function-calling schema.
        """
        normalized = provider.strip().lower()
        if normalized in ("openai", "openai-compat", "openrouter", "azure"):
            return self.to_openai_dict()
        if normalized == "anthropic":
            return self.to_anthropic_dict()
        raise ValueError(
            f"Unsupported provider {provider!r}; "
            "supported: 'openai' (and compat aliases), 'anthropic'"
        )

    @classmethod
    def from_provider_dict(cls, provider: str, data: dict[str, Any]) -> "ToolDefinition":
        """Parse *data* using *provider*'s native format."""
        normalized = provider.strip().lower()
        if normalized in ("openai", "openai-compat", "openrouter", "azure"):
            return cls.from_openai_dict(data)
        if normalized == "anthropic":
            return cls.from_anthropic_dict(data)
        raise ValueError(
            f"Unsupported provider {provider!r}; "
            "supported: 'openai' (and compat aliases), 'anthropic'"
        )

    # --------------------------------------------------------------- #
    # Bulk helpers
    # --------------------------------------------------------------- #

    @staticmethod
    def render_many(tools: list["ToolDefinition"], provider: str) -> list[dict[str, Any]]:
        """Convert a list of ``ToolDefinition`` objects to provider format."""
        return [t.to_provider_dict(provider) for t in tools]

    @staticmethod
    def parse_many(
        provider: str, data: list[dict[str, Any]]
    ) -> list["ToolDefinition"]:
        """Parse a list of provider-format dicts back into ``ToolDefinition`` objects."""
        return [ToolDefinition.from_provider_dict(provider, d) for d in data]


def _deepcopy_dict(d: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Shallow-ish copy so callers can't mutate ``ToolDefinition.parameters`` via the rendered dict."""
    if d is None:
        return {}
    import copy
    return copy.deepcopy(d)
