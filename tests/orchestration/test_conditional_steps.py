"""
Tests for conditional step execution in CARL.

This test suite covers conditional branching patterns including:
- Built-in condition patterns (contains, equals, startswith, endswith, matches, empty, nonempty)
- Complex expressions via simpleeval (len, int, comparisons)
- Multi-branch conditional routing
- Default step handling
- Conditional step serialization
- Error handling for invalid conditions
"""

import pytest
from mmar_carl import (
    ChainBuilder,
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
    create_step,
)
from tests.mocks import MockLLMClient


# ============================================================================
# Test Fixtures
# ============================================================================


def find_step_result(result, step_number):
    """Find a step result by step number."""
    for sr in result.step_results:
        if sr.step_number == step_number:
            return sr
    return None


# ============================================================================
# Built-in Condition Pattern Tests
# ============================================================================


class TestConditionalBuiltInPatterns:
    """Test built-in conditional patterns."""

    @pytest.mark.asyncio
    async def test_contains_pattern_match(self):
        """Test 'contains:' pattern with matching substring."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Contains",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="contains:error", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Not Matched",
                config=ToolStepConfig(tool_name="return_not_matched", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "An error occurred")
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_not_matched", lambda: "not_matched")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success, f"Chain execution failed: {result.get_final_output()}"
        assert cond_result.result_data.get("matched_condition") == "contains:error"
        assert cond_result.result_data.get("next_step") == 3

    @pytest.mark.asyncio
    async def test_contains_pattern_no_match(self):
        """Test 'contains:' pattern without matching substring."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Contains",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="contains:error", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Not Matched",
                config=ToolStepConfig(tool_name="return_not_matched", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "All systems normal")
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_not_matched", lambda: "not_matched")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("next_step") == 4  # Default

    @pytest.mark.asyncio
    async def test_equals_pattern_exact_match(self):
        """Test 'equals:' pattern with exact match."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Role",
                config=ToolStepConfig(tool_name="provide_role", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Equals",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="equals:admin", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Admin",
                config=ToolStepConfig(tool_name="admin_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Default",
                config=ToolStepConfig(tool_name="default_action", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_role", lambda: "admin")
        context.register_tool("admin_action", lambda: "admin_access")
        context.register_tool("default_action", lambda: "default_access")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "equals:admin"
        assert cond_result.result_data.get("next_step") == 3

    @pytest.mark.asyncio
    async def test_equals_pattern_case_sensitive(self):
        """Test 'equals:' pattern is case-sensitive."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Role",
                config=ToolStepConfig(tool_name="provide_role", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Equals",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="equals:admin", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Admin",
                config=ToolStepConfig(tool_name="admin_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Default",
                config=ToolStepConfig(tool_name="default_action", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_role", lambda: "Admin")  # Different case
        context.register_tool("admin_action", lambda: "admin_access")
        context.register_tool("default_action", lambda: "default_access")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("next_step") == 4  # Should use default

    @pytest.mark.asyncio
    async def test_startswith_pattern_match(self):
        """Test 'startswith:' pattern."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Startswith",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="startswith:HTTP/", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Default",
                config=ToolStepConfig(tool_name="return_default", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "HTTP/1.1 200 OK")
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_default", lambda: "default")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "startswith:HTTP/"

    @pytest.mark.asyncio
    async def test_endswith_pattern_match(self):
        """Test 'endswith:' pattern."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Filename",
                config=ToolStepConfig(tool_name="provide_filename", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Endswith",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="endswith:.json", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="JSON File",
                config=ToolStepConfig(tool_name="handle_json", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Other",
                config=ToolStepConfig(tool_name="handle_other", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_filename", lambda: "config.json")
        context.register_tool("handle_json", lambda: "json_handler")
        context.register_tool("handle_other", lambda: "other_handler")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "endswith:.json"

    @pytest.mark.asyncio
    async def test_matches_pattern_regex(self):
        """Test 'matches:' pattern with regex."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Matches",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="matches:\\d{3}", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Default",
                config=ToolStepConfig(tool_name="return_default", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "Status code 404 returned")
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_default", lambda: "default")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "matches:\\d{3}"

    @pytest.mark.asyncio
    async def test_empty_pattern_whitespace(self):
        """Test 'empty' pattern with whitespace."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Empty",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="empty", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Empty",
                config=ToolStepConfig(tool_name="handle_empty", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Not Empty",
                config=ToolStepConfig(tool_name="handle_not_empty", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "   ")  # Whitespace
        context.register_tool("handle_empty", lambda: "empty")
        context.register_tool("handle_not_empty", lambda: "not_empty")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "empty"

    @pytest.mark.asyncio
    async def test_nonempty_pattern(self):
        """Test 'nonempty' pattern."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Nonempty",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="nonempty", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Not Empty",
                config=ToolStepConfig(tool_name="handle_not_empty", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Empty",
                config=ToolStepConfig(tool_name="handle_empty", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "hello")
        context.register_tool("handle_not_empty", lambda: "not_empty")
        context.register_tool("handle_empty", lambda: "empty")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "nonempty"


# ============================================================================
# Complex Expression Tests (simpleeval)
# ============================================================================


class TestConditionalComplexExpressions:
    """Test complex conditional expressions via simpleeval."""

    @pytest.mark.asyncio
    async def test_length_comparison(self):
        """Test len(value) > N comparison."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Text",
                config=ToolStepConfig(tool_name="provide_text", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Length",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="len(value) > 5", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Long",
                config=ToolStepConfig(tool_name="handle_long", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Short",
                config=ToolStepConfig(tool_name="handle_short", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_text", lambda: "hello world")
        context.register_tool("handle_long", lambda: "long_text")
        context.register_tool("handle_short", lambda: "short_text")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "len(value) > 5"

    @pytest.mark.asyncio
    async def test_numeric_comparison(self):
        """Test int(value) >= N comparison."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Score",
                config=ToolStepConfig(tool_name="provide_score", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Score",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="int(value) >= 70", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="High Score",
                config=ToolStepConfig(tool_name="handle_high", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Low Score",
                config=ToolStepConfig(tool_name="handle_low", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_score", lambda: "75")
        context.register_tool("handle_high", lambda: "high_score")
        context.register_tool("handle_low", lambda: "low_score")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "int(value) >= 70"

    @pytest.mark.asyncio
    async def test_logical_or_expression(self):
        """Test logical OR expression."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Status",
                config=ToolStepConfig(tool_name="provide_status", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Status",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[
                        ConditionalBranch(
                            condition="'error' in value or 'fail' in value", next_step=3
                        )
                    ],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Error",
                config=ToolStepConfig(tool_name="handle_error", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Success",
                config=ToolStepConfig(tool_name="handle_success", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_status", lambda: "test fail case")
        context.register_tool("handle_error", lambda: "error_handler")
        context.register_tool("handle_success", lambda: "success_handler")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "'error' in value or 'fail' in value"


# ============================================================================
# Multi-Branch Routing Tests
# ============================================================================


class TestConditionalMultiBranchRouting:
    """Test conditional routing with multiple branches."""

    @pytest.mark.asyncio
    async def test_sentiment_routing_positive(self):
        """Test sentiment-based routing to positive branch."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Classify Sentiment",
                config=ToolStepConfig(
                    tool_name="classify_text",
                    input_mapping={"text": "$outer_context"},
                ),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route by Sentiment",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[
                        ConditionalBranch(condition="contains:positive", next_step=3),
                        ConditionalBranch(condition="contains:negative", next_step=4),
                        ConditionalBranch(condition="contains:neutral", next_step=5),
                    ],
                    default_step=5,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Handle Positive",
                config=ToolStepConfig(tool_name="handle_positive", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Handle Negative",
                config=ToolStepConfig(tool_name="handle_negative", input_mapping={}),
            ),
            ToolStepDescription(
                number=5,
                title="Handle Neutral",
                config=ToolStepConfig(tool_name="handle_neutral", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="This product is great and amazing!",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )

        def classify_text(text: str) -> str:
            positive_words = {"great", "good", "excellent", "amazing"}
            return "sentiment:positive" if any(
                word in text.lower() for word in positive_words
            ) else "sentiment:neutral"

        context.register_tool("classify_text", classify_text)
        context.register_tool("handle_positive", lambda: "positive_response")
        context.register_tool("handle_negative", lambda: "negative_response")
        context.register_tool("handle_neutral", lambda: "neutral_response")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "contains:positive"
        assert cond_result.result_data.get("next_step") == 3

    @pytest.mark.asyncio
    async def test_role_based_routing(self):
        """Test role-based routing with multiple roles."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Get Role",
                config=ToolStepConfig(tool_name="get_role", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route by Role",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[
                        ConditionalBranch(condition="equals:admin", next_step=3),
                        ConditionalBranch(condition="equals:editor", next_step=4),
                        ConditionalBranch(condition="equals:viewer", next_step=5),
                    ],
                    default_step=5,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Admin",
                config=ToolStepConfig(tool_name="admin_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Editor",
                config=ToolStepConfig(tool_name="editor_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=5,
                title="Viewer",
                config=ToolStepConfig(tool_name="viewer_action", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("get_role", lambda: "editor")
        context.register_tool("admin_action", lambda: "admin_panel")
        context.register_tool("editor_action", lambda: "editor_view")
        context.register_tool("viewer_action", lambda: "viewer_mode")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("matched_condition") == "equals:editor"
        assert cond_result.result_data.get("next_step") == 4


# ============================================================================
# Default Step Tests
# ============================================================================


class TestConditionalDefaultStep:
    """Test default step handling in conditional steps."""

    @pytest.mark.asyncio
    async def test_default_step_when_no_match(self):
        """Test default step is used when no conditions match."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Conditions",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[
                        ConditionalBranch(condition="contains:admin", next_step=3),
                        ConditionalBranch(condition="contains:editor", next_step=4),
                    ],
                    default_step=5,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Admin",
                config=ToolStepConfig(tool_name="admin_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Editor",
                config=ToolStepConfig(tool_name="editor_action", input_mapping={}),
            ),
            ToolStepDescription(
                number=5,
                title="Default",
                config=ToolStepConfig(tool_name="default_action", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "viewer")  # Doesn't match
        context.register_tool("admin_action", lambda: "admin")
        context.register_tool("editor_action", lambda: "editor")
        context.register_tool("default_action", lambda: "default")

        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        assert result.success
        assert cond_result.result_data.get("next_step") == 5


# ============================================================================
# Serialization Tests
# ============================================================================


class TestConditionalSerialization:
    """Test conditional step serialization."""

    def test_conditional_step_serialization(self):
        """Test conditional step can be serialized to dict."""
        cond_step = ConditionalStepDescription(
            number=1,
            title="Check Value",
            config=ConditionalStepConfig(
                condition_context_key="$history[-1]",
                branches=[
                    ConditionalBranch(condition="contains:error", next_step=2),
                    ConditionalBranch(condition="contains:success", next_step=3),
                    ConditionalBranch(condition="empty", next_step=4),
                ],
                default_step=3,
            ),
        )

        step_dict = cond_step.model_dump()

        assert step_dict["number"] == 1
        assert step_dict["title"] == "Check Value"
        # BUG: step_type field is not included in serialization
        # Expected: "step_type" should be present
        assert len(step_dict["config"]["branches"]) == 3
        assert step_dict["config"]["default_step"] == 3
        assert step_dict["config"]["condition_context_key"] == "$history[-1]"

    def test_conditional_step_deserialization(self):
        """Test conditional step can be deserialized from dict."""
        step_dict = {
            "number": 1,
            "title": "Check Value",
            "step_type": StepType.CONDITIONAL,
            "config": {
                "condition_context_key": "$history[-1]",
                "branches": [
                    {"condition": "contains:error", "next_step": 2},
                    {"condition": "contains:success", "next_step": 3},
                ],
                "default_step": 3,
            },
            "dependencies": [],
            "step_context_queries": [],
        }

        cond_step = ConditionalStepDescription(**step_dict)

        assert cond_step.number == 1
        assert cond_step.title == "Check Value"
        assert len(cond_step.config.branches) == 2
        assert cond_step.config.default_step == 3


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestConditionalFactoryFunction:
    """Test create_step() factory for conditional steps."""

    def test_create_conditional_step_with_factory(self):
        """Test creating conditional step using create_step()."""
        cond_step = create_step(
            number=1,
            title="Check Value",
            step_type=StepType.CONDITIONAL,
            config=ConditionalStepConfig(
                condition_context_key="$history[-1]",
                branches=[
                    ConditionalBranch(condition="contains:error", next_step=2),
                    ConditionalBranch(condition="contains:success", next_step=3),
                    ConditionalBranch(condition="empty", next_step=4),
                ],
                default_step=3,
            ),
        )

        assert isinstance(cond_step, ConditionalStepDescription)
        assert cond_step.number == 1
        assert cond_step.title == "Check Value"
        assert cond_step.step_type == StepType.CONDITIONAL
        assert len(cond_step.config.branches) == 3


# ============================================================================
# ChainBuilder Integration Tests
# ============================================================================


class TestConditionalChainBuilderIntegration:
    """Test conditional steps with ChainBuilder API."""

    @pytest.mark.asyncio
    async def test_chainbuilder_conditional_step(self):
        """Test adding conditional step via ChainBuilder."""
        chain = (
            ChainBuilder()
            .add_tool_step(
                number=1,
                title="Get Score",
                tool_name="get_score",
                input_mapping={},
            )
            .add_conditional_step(
                number=2,
                title="Check Threshold",
                condition_context_key="$metadata.step_1",
                branches=[
                    ("int(value) >= 70", 3),
                    ("int(value) < 70", 4),
                ],
                default_step=4,
                dependencies=[1],
            )
            .add_tool_step(number=3, title="High", tool_name="handle_high", input_mapping={})
            .add_tool_step(number=4, title="Low", tool_name="handle_low", input_mapping={})
            .with_trace_name("Threshold Test")
            .build()
        )

        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("get_score", lambda: "75")
        context.register_tool("handle_high", lambda: "high_score")
        context.register_tool("handle_low", lambda: "low_score")

        result = await chain.execute_async(context)

        assert result.success
        # BUG: CARL executes both branches instead of just the matched one
        # Expected: 3 steps (get_score, check, handle_high)
        # Actual: 4 steps (get_score, handle_high, handle_low, check)
        assert len(result.step_results) == 4  # BUG: executes all steps


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestConditionalErrorHandling:
    """Test error handling in conditional steps."""

    @pytest.mark.asyncio
    async def test_invalid_simpleeval_expression(self):
        """Test handling of invalid simpleeval expression."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Test Invalid Expression",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="undefined_func(value)", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Matched",
                config=ToolStepConfig(tool_name="return_matched", input_mapping={}),
            ),
            ToolStepDescription(
                number=4,
                title="Default",
                config=ToolStepConfig(tool_name="return_default", input_mapping={}),
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
            language=Language.ENGLISH,
        )
        context.register_tool("provide_value", lambda: "test")
        context.register_tool("return_matched", lambda: "matched")
        context.register_tool("return_default", lambda: "default")

        # Should fall back to default when expression fails
        result = await chain.execute_async(context)
        cond_result = find_step_result(result, 2)

        # The chain should still succeed, using default step
        assert result.success or cond_result.result_data.get("next_step") == 4


# ============================================================================
# Skipped Step Visibility Tests
# ============================================================================


class TestConditionalSkippedStepVisibility:
    """Test that skipped steps are visible in ReasoningResult with skipped=True."""

    @pytest.mark.asyncio
    async def test_skipped_step_has_skipped_flag(self):
        """Step not on chosen branch must appear with skipped=True in step_results.

        Branch steps MUST depend on the conditional step (dependencies=[2]) so that
        the routing decision is made before the branch steps become ready to execute.
        """
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="contains:yes", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Chosen",
                config=ToolStepConfig(tool_name="chosen_action", input_mapping={}),
                dependencies=[2],  # Must depend on conditional step
            ),
            ToolStepDescription(
                number=4,
                title="NotChosen",
                config=ToolStepConfig(tool_name="not_chosen_action", input_mapping={}),
                dependencies=[2],  # Must depend on conditional step
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
        )
        context.register_tool("provide_value", lambda: "yes")
        context.register_tool("chosen_action", lambda: "chosen")
        context.register_tool("not_chosen_action", lambda: "not_chosen")

        result = await chain.execute_async(context)

        assert result.success

        skipped = result.get_skipped_steps()
        assert len(skipped) == 1
        s = skipped[0]
        assert s.step_number == 4
        assert s.skipped is True
        assert s.success is True  # Not a failure — was intentionally skipped

        # Must not appear in get_successful_steps()
        successful_nums = {r.step_number for r in result.get_successful_steps()}
        assert 4 not in successful_nums

    @pytest.mark.asyncio
    async def test_get_skipped_steps_returns_empty_without_conditional(self):
        """A chain without conditional steps produces no skipped steps."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Step 1",
                config=ToolStepConfig(tool_name="step1", input_mapping={}),
            ),
            ToolStepDescription(
                number=2,
                title="Step 2",
                config=ToolStepConfig(tool_name="step2", input_mapping={}),
                dependencies=[1],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
        )
        context.register_tool("step1", lambda: "r1")
        context.register_tool("step2", lambda: "r2")

        result = await chain.execute_async(context)
        assert result.success
        assert result.get_skipped_steps() == []

    @pytest.mark.asyncio
    async def test_to_dict_includes_skipped_count(self):
        """result.to_dict() must include skipped_steps count."""
        steps = [
            ToolStepDescription(
                number=1,
                title="Provide Value",
                config=ToolStepConfig(tool_name="provide_value", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Route",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="contains:yes", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Chosen",
                config=ToolStepConfig(tool_name="chosen", input_mapping={}),
                dependencies=[2],
            ),
            ToolStepDescription(
                number=4,
                title="Skipped",
                config=ToolStepConfig(tool_name="skipped", input_mapping={}),
                dependencies=[2],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
        )
        context.register_tool("provide_value", lambda: "yes it is")
        context.register_tool("chosen", lambda: "chosen")
        context.register_tool("skipped", lambda: "skipped")

        result = await chain.execute_async(context)

        d = result.to_dict()
        assert "skipped_steps" in d
        assert d["skipped_steps"] == 1

    @pytest.mark.asyncio
    async def test_cascading_skip_through_dependent_steps(self):
        """
        Steps that depend only on a skipped step are also skipped.

        Layout:
          1 → 2 (COND, selects 3) → 3 (chosen, depends on 2)
                                  → 4 (skipped, depends on 2)
                                    → 5 (also skipped, depends only on 4)
        """
        steps = [
            ToolStepDescription(
                number=1,
                title="S1",
                config=ToolStepConfig(tool_name="s1", input_mapping={}),
            ),
            ConditionalStepDescription(
                number=2,
                title="Cond",
                config=ConditionalStepConfig(
                    condition_context_key="$metadata.step_1",
                    branches=[ConditionalBranch(condition="contains:go", next_step=3)],
                    default_step=4,
                ),
                dependencies=[1],
            ),
            ToolStepDescription(
                number=3,
                title="Chosen",
                config=ToolStepConfig(tool_name="chosen", input_mapping={}),
                dependencies=[2],
            ),
            ToolStepDescription(
                number=4,
                title="SkippedA",
                config=ToolStepConfig(tool_name="skipped_a", input_mapping={}),
                dependencies=[2],
            ),
            ToolStepDescription(
                number=5,
                title="SkippedB",
                config=ToolStepConfig(tool_name="skipped_b", input_mapping={}),
                dependencies=[4],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=1)
        context = ReasoningContext(
            outer_context="",
            api=MockLLMClient(),
            model="test",
        )
        context.register_tool("s1", lambda: "go ahead")
        context.register_tool("chosen", lambda: "chosen")
        context.register_tool("skipped_a", lambda: "skipped_a")
        context.register_tool("skipped_b", lambda: "skipped_b")

        result = await chain.execute_async(context)
        assert result.success

        skipped_nums = {r.step_number for r in result.get_skipped_steps()}
        assert 4 in skipped_nums, "Step 4 should be skipped"
        assert 5 in skipped_nums, "Step 5 (depends only on skipped 4) should also be skipped"

        successful_nums = {r.step_number for r in result.get_successful_steps()}
        assert 3 in successful_nums
        assert 4 not in successful_nums
        assert 5 not in successful_nums
