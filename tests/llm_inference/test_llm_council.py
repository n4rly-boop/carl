"""
Tests for LLM Council pattern in CARL.

This test suite covers multi-model voting scenarios including:
- Parallel council member execution
- Per-step model overrides via LLMStepConfig
- Vote aggregation logic (unanimous, majority, split)
- Council with varying member counts
- Model-specific temperature/params
- Council synthesis step
"""

import pytest
from typing import Any
from mmar_carl import (
    Language,
    LLMStepConfig,
    LLMStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
)
from tests.mocks import CouncilMockClient, MockLLMClient


# ============================================================================
# Test Fixtures
# ============================================================================


def aggregate_test_votes(vote_1: str, vote_2: str, vote_3: str) -> dict[str, Any]:
    """Aggregate votes from council members for testing."""
    votes = {"vote_1": vote_1, "vote_2": vote_2, "vote_3": vote_3}
    labels = {
        "vote_1": "Member 1",
        "vote_2": "Member 2",
        "vote_3": "Member 3",
    }

    # Extract votes - look for "Option A", "Option B", or "Option C"
    extracted = {}
    tally: dict[str, list[str]] = {}

    for key, response in votes.items():
        # Simple extraction for testing
        if "Option A" in response:
            choice = "Option A"
        elif "Option B" in response:
            choice = "Option B"
        elif "Option C" in response:
            choice = "Option C"
        else:
            choice = "Abstain"

        extracted[key] = {
            "member": labels[key],
            "vote": choice,
            "reasoning_excerpt": response[:100],
        }

        tally.setdefault(choice, [])
        tally[choice].append(labels[key])

    # Determine winner
    winner = max(tally, key=lambda k: len(tally[k])) if tally else "No consensus"
    unanimous = len(tally) == 1 and "Abstain" not in tally

    return {
        "votes": extracted,
        "tally": {k: len(v) for k, v in tally.items()},
        "winner": winner,
        "unanimous": unanimous,
        "consensus_level": "unanimous"
        if unanimous
        else ("majority" if any(len(v) >= 2 for v in tally.values()) else "split"),
    }


def find_step_result(result, step_number):
    """Find a step result by step number."""
    for sr in result.step_results:
        if sr.step_number == step_number:
            return sr
    return None


# ============================================================================
# Parallel Council Execution Tests
# ============================================================================


class TestCouncilParallelExecution:
    """Test parallel execution of council members."""

    @pytest.mark.asyncio
    async def test_parallel_council_members(self):
        """Test that council members execute in parallel when they have no dependencies."""
        steps = []

        # Create 3 council members with no dependencies (should run in parallel)
        for i in range(1, 4):
            step = LLMStepDescription(
                number=i,
                title=f"Council Member {i}",
                aim=f"Evaluate options from perspective {i}",
                reasoning_questions="What is the best option?",
                stage_action="Cast your vote",
                example_reasoning="Option A is best. My vote: Option A",
                llm_config=LLMStepConfig(
                    model=f"model-{i}",
                    temperature=0.5,
                    max_tokens=1000,
                ),
                dependencies=[],  # No dependencies = parallel execution
            )
            steps.append(step)

        # Add aggregation step that depends on all members
        aggregate_step = ToolStepDescription(
            number=4,
            title="Aggregate Votes",
            config=ToolStepConfig(
                tool_name="aggregate_test_votes",
                input_mapping={
                    "vote_1": "$metadata.step_1",
                    "vote_2": "$metadata.step_2",
                    "vote_3": "$metadata.step_3",
                },
            ),
            dependencies=[1, 2, 3],  # Wait for all council members
        )
        steps.append(aggregate_step)

        chain = ReasoningChain(
            steps=steps,
            max_workers=3,  # Should allow parallel execution
            trace_name="Council Parallel Test",
        )

        # Create mock clients for each member
        mock_client_1 = CouncilMockClient("member_1", "Option A", "neutral")

        # Use the first mock client as the base
        context = ReasoningContext(
            outer_context="Technology decision scenario",
            api=mock_client_1,
            model="base-model",
            language=Language.ENGLISH,
        )

        context.register_tool("aggregate_test_votes", aggregate_test_votes)

        result = await chain.execute_async(context)

        assert result.success, f"Chain execution failed: {result.get_final_output()}"
        assert len(result.step_results) == 4  # 3 members + 1 aggregator

        # Verify aggregation worked
        aggregate_result = find_step_result(result, 4)
        assert aggregate_result.success

    @pytest.mark.asyncio
    async def test_council_member_count_variation(self):
        """Test council with different member counts."""
        # Test with 2 members
        steps = [
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate option 1",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-1", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate option 2",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-2", temperature=0.5),
                dependencies=[],
            ),
            ToolStepDescription(
                number=3,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_two_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                    },
                ),
                dependencies=[1, 2],
            ),
        ]

        def aggregate_two_votes(vote_1: str, vote_2: str) -> dict[str, Any]:
            """Aggregate 2 votes."""
            return {
                "votes": {"vote_1": vote_1, "vote_2": vote_2},
                "tally": {"Option A": 2} if "Option A" in vote_1 and "Option A" in vote_2 else {},
                "winner": "Option A",
                "unanimous": True,
                "consensus_level": "unanimous",
            }

        chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Two Member Council")

        context = ReasoningContext(
            outer_context="Decision scenario",
            api=CouncilMockClient("member_1", "Option A", "neutral"),
            model="base-model",
            language=Language.ENGLISH,
        )
        context.register_tool("aggregate_two_votes", aggregate_two_votes)

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 3  # 2 members + 1 aggregator


# ============================================================================
# Per-Step Model Override Tests
# ============================================================================


class TestCouncilModelOverrides:
    """Test per-step model configuration in council scenarios."""

    @pytest.mark.asyncio
    async def test_per_step_model_overrides(self):
        """Test that each council member can use a different model."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A",
                llm_config=LLMStepConfig(
                    model="gpt-4",  # High-end model
                    temperature=0.3,
                ),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option B",
                llm_config=LLMStepConfig(
                    model="claude-3-opus",  # Different high-end model
                    temperature=0.5,
                ),
                dependencies=[],
            ),
            LLMStepDescription(
                number=3,
                title="Member 3",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option C",
                llm_config=LLMStepConfig(
                    model="llama-3-70b",  # Open-source model
                    temperature=0.7,
                ),
                dependencies=[],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Model Override Test")

        context = ReasoningContext(
            outer_context="Decision",
            api=MockLLMClient(),
            model="base-model",  # Should be overridden
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 3

        # Each step should have executed with its config
        for step_result in result.step_results:
            assert step_result.success

    @pytest.mark.asyncio
    async def test_model_specific_temperature(self):
        """Test that each council member can have different temperature settings."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Conservative Member",
                aim="Evaluate carefully",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A",
                llm_config=LLMStepConfig(
                    model="model-1",
                    temperature=0.1,  # Very conservative
                    max_tokens=500,
                ),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Balanced Member",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option B",
                llm_config=LLMStepConfig(
                    model="model-2",
                    temperature=0.5,  # Balanced
                    max_tokens=1000,
                ),
                dependencies=[],
            ),
            LLMStepDescription(
                number=3,
                title="Creative Member",
                aim="Evaluate creatively",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option C",
                llm_config=LLMStepConfig(
                    model="model-3",
                    temperature=0.9,  # Very creative
                    max_tokens=1500,
                ),
                dependencies=[],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Temperature Test")

        context = ReasoningContext(
            outer_context="Decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 3


# ============================================================================
# Vote Aggregation Tests
# ============================================================================


class TestCouncilVoteAggregation:
    """Test vote aggregation logic for council decisions."""

    @pytest.mark.asyncio
    async def test_unanimous_decision(self):
        """Test unanimous council decision."""
        steps = [
            LLMStepDescription(
                number=i,
                title=f"Member {i}",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote Option A",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model=f"model-{i}", temperature=0.5),
                dependencies=[],
            )
            for i in range(1, 4)
        ]

        # Add aggregation step
        steps.append(
            ToolStepDescription(
                number=4,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_test_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                        "vote_3": "$metadata.step_3",
                    },
                ),
                dependencies=[1, 2, 3],
            )
        )

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Unanimous Test")

        # All members vote for Option A
        context = ReasoningContext(
            outer_context="Decision",
            api=CouncilMockClient("unanimous", "Option A", "neutral"),
            model="base-model",
            language=Language.ENGLISH,
        )
        context.register_tool("aggregate_test_votes", aggregate_test_votes)

        result = await chain.execute_async(context)

        assert result.success
        aggregate_result = find_step_result(result, 4)
        assert aggregate_result.success

        # Check aggregation result
        if isinstance(aggregate_result.result_data, dict):
            assert aggregate_result.result_data.get("winner") == "Option A"
            assert aggregate_result.result_data.get("unanimous") is True
            assert aggregate_result.result_data.get("consensus_level") == "unanimous"

    @pytest.mark.asyncio
    async def test_majority_decision(self):
        """Test majority council decision."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-1", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-2", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=3,
                title="Member 3",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option B. My vote: Option B",
                llm_config=LLMStepConfig(model="model-3", temperature=0.5),
                dependencies=[],
            ),
            ToolStepDescription(
                number=4,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_test_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                        "vote_3": "$metadata.step_3",
                    },
                ),
                dependencies=[1, 2, 3],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Majority Test")

        # Create a wrapper function that provides the mock responses
        def mock_majority_aggregator(vote_1: str, vote_2: str, vote_3: str) -> dict[str, Any]:
            """Mock aggregator that simulates 2 votes for A, 1 vote for B."""
            return aggregate_test_votes(
                "After careful analysis, Member 1 votes for: Option A",
                "After review, Member 2 votes for: Option A",
                "After consideration, Member 3 votes for: Option B"
            )

        context = ReasoningContext(
            outer_context="Decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )

        context.register_tool("aggregate_test_votes", mock_majority_aggregator)

        result = await chain.execute_async(context)

        assert result.success
        aggregate_result = find_step_result(result, 4)
        assert aggregate_result.success

        # Check aggregation result
        if isinstance(aggregate_result.result_data, dict):
            assert aggregate_result.result_data.get("winner") == "Option A"
            assert aggregate_result.result_data.get("unanimous") is False
            assert aggregate_result.result_data.get("consensus_level") == "majority"

    @pytest.mark.asyncio
    async def test_split_decision(self):
        """Test split council decision (no majority)."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-1", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option B. My vote: Option B",
                llm_config=LLMStepConfig(model="model-2", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=3,
                title="Member 3",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option C. My vote: Option C",
                llm_config=LLMStepConfig(model="model-3", temperature=0.5),
                dependencies=[],
            ),
            ToolStepDescription(
                number=4,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_test_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                        "vote_3": "$metadata.step_3",
                    },
                ),
                dependencies=[1, 2, 3],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Split Test")

        # Create a wrapper function that provides the mock responses
        def mock_split_aggregator(vote_1: str, vote_2: str, vote_3: str) -> dict[str, Any]:
            """Mock aggregator that simulates split votes (A, B, C)."""
            return aggregate_test_votes(
                "After analysis, Member 1 votes for: Option A",
                "After review, Member 2 votes for: Option B",
                "After consideration, Member 3 votes for: Option C"
            )

        context = ReasoningContext(
            outer_context="Decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )

        context.register_tool("aggregate_test_votes", mock_split_aggregator)

        result = await chain.execute_async(context)

        assert result.success
        aggregate_result = find_step_result(result, 4)
        assert aggregate_result.success

        # Check aggregation result
        if isinstance(aggregate_result.result_data, dict):
            assert aggregate_result.result_data.get("unanimous") is False
            assert aggregate_result.result_data.get("consensus_level") == "split"


# ============================================================================
# Council Synthesis Tests
# ============================================================================


class TestCouncilSynthesis:
    """Test council synthesis step."""

    @pytest.mark.asyncio
    async def test_council_synthesis_step(self):
        """Test final synthesis step after council voting."""
        steps = [
            # Council members
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-1", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-2", temperature=0.5),
                dependencies=[],
            ),
            # Aggregation
            ToolStepDescription(
                number=3,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_test_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                    },
                ),
                dependencies=[1, 2],
            ),
            # Synthesis step
            LLMStepDescription(
                number=4,
                title="Council Verdict",
                aim="Synthesize council votes into final verdict",
                reasoning_questions="What was the council's decision?",
                stage_action="Produce final verdict",
                example_reasoning="The council voted unanimously for Option A",
                llm_config=LLMStepConfig(model="synthesis-model", temperature=0.3),
                dependencies=[3],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=2, trace_name="Council Synthesis")

        def aggregate_two_votes(vote_1: str, vote_2: str) -> dict[str, Any]:
            """Aggregate 2 votes for synthesis."""
            return {
                "votes": {"vote_1": vote_1, "vote_2": vote_2},
                "winner": "Option A",
                "unanimous": True,
                "consensus_level": "unanimous",
            }

        context = ReasoningContext(
            outer_context="Technology decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )
        context.register_tool("aggregate_test_votes", aggregate_two_votes)

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 4  # 2 members + 1 aggregator + 1 synthesis

        # Verify synthesis step executed
        synthesis_result = find_step_result(result, 4)
        assert synthesis_result.success
        assert synthesis_result.step_type == StepType.LLM


# ============================================================================
# Complex Council Scenarios
# ============================================================================


class TestCouncilComplexScenarios:
    """Test complex council scenarios."""

    @pytest.mark.asyncio
    async def test_large_council_five_members(self):
        """Test council with 5+ members."""
        steps = []

        # Create 5 council members
        for i in range(1, 6):
            step = LLMStepDescription(
                number=i,
                title=f"Council Member {i}",
                aim=f"Evaluate from perspective {i}",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model=f"model-{i}", temperature=0.5),
                dependencies=[],
            )
            steps.append(step)

        # Aggregation step
        steps.append(
            ToolStepDescription(
                number=6,
                title="Aggregate Five Votes",
                config=ToolStepConfig(
                    tool_name="aggregate_five_votes",
                    input_mapping={
                        f"vote_{i}": f"$metadata.step_{i}" for i in range(1, 6)
                    },
                ),
                dependencies=[1, 2, 3, 4, 5],
            )
        )

        def aggregate_five_votes(**kwargs) -> dict[str, Any]:
            """Aggregate 5 votes."""
            return {
                "votes": kwargs,
                "winner": "Option A",
                "unanimous": True,
                "consensus_level": "unanimous",
            }

        chain = ReasoningChain(steps=steps, max_workers=5, trace_name="Large Council")

        context = ReasoningContext(
            outer_context="Complex decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )
        context.register_tool("aggregate_five_votes", aggregate_five_votes)

        result = await chain.execute_async(context)

        assert result.success
        assert len(result.step_results) == 6  # 5 members + 1 aggregator

    @pytest.mark.asyncio
    async def test_council_with_abstentions(self):
        """Test council handling of abstentions."""
        steps = [
            LLMStepDescription(
                number=1,
                title="Member 1",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-1", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=2,
                title="Member 2",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Abstain from voting",
                example_reasoning="Unable to decide. Abstaining",
                llm_config=LLMStepConfig(model="model-2", temperature=0.5),
                dependencies=[],
            ),
            LLMStepDescription(
                number=3,
                title="Member 3",
                aim="Evaluate",
                reasoning_questions="Best option?",
                stage_action="Vote",
                example_reasoning="Option A. My vote: Option A",
                llm_config=LLMStepConfig(model="model-3", temperature=0.5),
                dependencies=[],
            ),
            ToolStepDescription(
                number=4,
                title="Aggregate",
                config=ToolStepConfig(
                    tool_name="aggregate_test_votes",
                    input_mapping={
                        "vote_1": "$metadata.step_1",
                        "vote_2": "$metadata.step_2",
                        "vote_3": "$metadata.step_3",
                    },
                ),
                dependencies=[1, 2, 3],
            ),
        ]

        chain = ReasoningChain(steps=steps, max_workers=3, trace_name="Abstention Test")

        context = ReasoningContext(
            outer_context="Decision",
            api=MockLLMClient(),
            model="base-model",
            language=Language.ENGLISH,
        )
        context.register_tool("aggregate_test_votes", aggregate_test_votes)

        result = await chain.execute_async(context)

        assert result.success
        aggregate_result = find_step_result(result, 4)
        assert aggregate_result.success

        # Should handle abstentions gracefully
        if isinstance(aggregate_result.result_data, dict):
            assert "winner" in aggregate_result.result_data
