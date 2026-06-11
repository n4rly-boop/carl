"""
Optional dependencies with clear error messages.

This module provides utilities for handling optional dependencies gracefully.
Each optional dependency is imported on-demand with helpful error messages
if the dependency is not installed.
"""

from typing import Any


_optional_imports = {
    "faiss": "faiss-cpu",
    "fastembed": "fastembed",
    "numpy": "numpy",
    "mcp": "mcp",
    "openai": "openai",
    "langfuse": "langfuse",
    "matplotlib": "matplotlib",
    "agent_skills": "agent-skills",
    "pypdf": "pypdf",
    "pdfplumber": "pdfplumber",
    "pptx": "python-pptx",
}

_feature_groups = {
    "vector-search": ["faiss", "fastembed", "numpy"],
    "mcp": ["mcp"],
    "openai": ["openai"],
    "langfuse": ["langfuse"],
    "viz": ["matplotlib"],
    "agent-skills": ["agent_skills"],
    "pdf": ["pypdf", "pdfplumber"],
    "pptx": ["pptx"],
}


def require(module_name: str) -> Any:
    """
    Import an optional dependency with helpful error message.

    Args:
        module_name: Name of the module to import

    Returns:
        The imported module

    Raises:
        ImportError: If the module is not installed, with installation instructions
    """
    try:
        return __import__(module_name)
    except ImportError:
        package = _optional_imports.get(module_name, module_name)

        # Find which feature group this belongs to
        feature_group = None
        for group, modules in _feature_groups.items():
            if module_name in modules:
                feature_group = group
                break

        if feature_group:
            raise ImportError(
                f"Optional dependency '{module_name}' not found. "
                f"This is required for {feature_group} features. "
                f"Install with: pip install 'mmar-carl[{feature_group}]'"
            ) from None
        else:
            raise ImportError(
                f"Optional dependency '{module_name}' not found. "
                f"Install with: pip install '{package}'"
            ) from None


def require_faiss():
    """Import faiss with helpful error message if not installed."""
    return require("faiss")


def require_fastembed():
    """Import fastembed with helpful error message if not installed."""
    return require("fastembed")


def require_numpy():
    """Import numpy with helpful error message if not installed."""
    return require("numpy")


def require_mcp():
    """Import mcp with helpful error message if not installed."""
    return require("mcp")


def require_openai():
    """Import openai with helpful error message if not installed."""
    return require("openai")


def require_langfuse():
    """Import langfuse with helpful error message if not installed."""
    return require("langfuse")


def require_agent_skills():
    """Import the ``agent_skills`` library with a helpful error message."""
    return require("agent_skills")


def check_agent_skills_available() -> bool:
    """Check if the optional ``agent-skills`` library is installed."""
    try:
        import agent_skills  # noqa: F401
        return True
    except ImportError:
        return False


def check_vector_search_available() -> bool:
    """Check if vector search dependencies are available."""
    try:
        import faiss  # noqa: F401
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def check_mcp_available() -> bool:
    """Check if MCP dependencies are available."""
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


def check_openai_available() -> bool:
    """Check if OpenAI dependencies are available."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def check_langfuse_available() -> bool:
    """Check if Langfuse dependencies are available."""
    try:
        import langfuse  # noqa: F401
        return True
    except ImportError:
        return False


def require_matplotlib() -> Any:
    """Import matplotlib (configured with the Agg headless backend) with
    a helpful error message pointing at the ``mmar-carl[viz]`` install.

    Returns the ``matplotlib.pyplot`` module so callers can use it
    directly: ``plt = require_matplotlib()``.
    """
    try:
        import matplotlib  # noqa: PLC0415

        # Force the Agg backend so PNG-writing chart methods are safe in
        # headless / CI environments where no display is available.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt
    except ImportError:
        raise ImportError(
            "Optional dependency 'matplotlib' not found. "
            "This is required for viz features (chart PNG export). "
            "Install with: pip install 'mmar-carl[viz]'"
        ) from None


def check_viz_available() -> bool:
    """Check if visualization (matplotlib-backed) dependencies are available."""
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False
