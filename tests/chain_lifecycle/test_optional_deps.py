"""Tests for optional dependency handling."""
import pytest

from mmar_carl._optional_deps import check_vector_search_available

# Skip tests that expect missing vector-search deps when they ARE installed
VECTOR_SEARCH_AVAILABLE = check_vector_search_available()


@pytest.mark.skipif(VECTOR_SEARCH_AVAILABLE, reason="Vector search dependencies are installed")
def test_vector_search_without_dependencies():
    """Test that vector search fails gracefully without deps."""
    from mmar_carl.models.search import ContextSearchConfig

    config = ContextSearchConfig(strategy="vector")

    # Should raise helpful error
    with pytest.raises(ImportError) as exc_info:
        config.get_strategy()

    assert "pip install 'mmar-carl[vector-search]'" in str(exc_info.value)


def test_substring_search_without_dependencies():
    """Test that substring search works without vector deps."""
    from mmar_carl.models.search import ContextSearchConfig, SubstringSearchStrategy

    config = ContextSearchConfig(strategy="substring")
    strategy = config.get_strategy()

    assert isinstance(strategy, SubstringSearchStrategy)


@pytest.mark.skipif(VECTOR_SEARCH_AVAILABLE, reason="Vector search dependencies are installed")
def test_vector_search_strategy_init_without_deps():
    """Test that VectorSearchStrategy raises error without deps."""
    from mmar_carl.models.search import VectorSearchStrategy

    with pytest.raises(ImportError) as exc_info:
        VectorSearchStrategy()

    assert "pip install 'mmar-carl[vector-search]'" in str(exc_info.value)


def test_optional_deps_module():
    """Test the _optional_deps module functions."""
    from mmar_carl._optional_deps import (
        check_vector_search_available,
        check_mcp_available,
        check_openai_available,
        check_langfuse_available,
    )

    # These should return False if deps not installed
    assert isinstance(check_vector_search_available(), bool)
    assert isinstance(check_mcp_available(), bool)
    assert isinstance(check_openai_available(), bool)
    assert isinstance(check_langfuse_available(), bool)


def test_optional_deps_require_function():
    """Test the require function with non-existent module."""
    from mmar_carl._optional_deps import require

    with pytest.raises(ImportError) as exc_info:
        require("nonexistent_module_12345")

    assert "pip install" in str(exc_info.value)


def test_substring_search_basic_functionality():
    """Test that substring search works with basic functionality."""
    from mmar_carl.models.search import ContextSearchConfig

    config = ContextSearchConfig(strategy="substring")
    strategy = config.get_strategy()

    outer_context = """
    Company XYZ Report:
    Revenue: $2.5 million
    Profit: $700,000
    EBITDA: 32%
    """

    queries = ["Revenue", "Profit"]
    result = strategy.extract_context(outer_context, queries)

    assert "Revenue" in result or "Profit" in result
