"""
Centralized mock infrastructure for CARL testing.

This module provides reusable mock LLM clients for testing various CARL patterns
including conditional steps, LLM council, tool steps, structured output, and RE-PLAN scenarios.
"""

from typing import Any, Optional
from mmar_carl import LLMClientBase


class MockLLMClient(LLMClientBase):
    """Basic mock LLM client for general testing."""

    def __init__(self, response_text: str = "Mock response"):
        self.response_text = response_text
        self.call_count = 0
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        """Mock async method that returns predefined response."""
        self.call_count += 1
        self.prompts.append(prompt)
        return f"Response to: {prompt[:50]}..."

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock async method with retries."""
        self.call_count += 1
        self.prompts.append(prompt)
        return f"Response to: {prompt[:50]}... (retries={retries})"


class ConditionalMockClient(LLMClientBase):
    """Mock client for conditional step testing.

    Returns predictable responses based on condition patterns
    to test routing logic without requiring actual LLM calls.
    """

    def __init__(self, response_patterns: Optional[dict[str, str]] = None):
        """Initialize with optional response patterns.

        Args:
            response_patterns: Dict mapping condition patterns to responses
        """
        self.response_patterns = response_patterns or {}
        self.prompts: list[str] = []
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        """Return response based on prompt patterns."""
        self.call_count += 1
        self.prompts.append(prompt)

        # Check for known patterns
        for pattern, response in self.response_patterns.items():
            if pattern in prompt:
                return response

        # Default response
        return "Conditional step executed successfully"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class CouncilMockClient(LLMClientBase):
    """Mock client for LLM council testing.

    Simulates different model personas for council voting scenarios.
    Each instance represents a different council member with specific voting patterns.
    """

    def __init__(self, member_id: str, vote_pattern: str, persona: str = "neutral"):
        """Initialize council member mock.

        Args:
            member_id: Unique identifier for this council member
            vote_pattern: Voting pattern ('Option A', 'Option B', 'Option C', etc.)
            persona: Member personality (affects response style)
        """
        self.member_id = member_id
        self.vote_pattern = vote_pattern
        self.persona = persona
        self.prompts: list[str] = []
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        """Return response reflecting this council member's vote."""
        self.call_count += 1
        self.prompts.append(prompt)

        persona_prefixes = {
            "conservative": "Carefully considering the options, ",
            "aggressive": "Quick analysis shows ",
            "neutral": "After review, ",
            "detailed": "Looking at all the details, ",
        }

        prefix = persona_prefixes.get(self.persona, "")
        return f"{prefix}Member {self.member_id} votes for: {self.vote_pattern}"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class ToolTrackingMockClient(LLMClientBase):
    """Mock client that tracks tool execution and data flow.

    Records all tool calls, data transformations, and execution order
    for validation in tests.
    """

    def __init__(self):
        self.prompts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.data_flow: list[dict[str, Any]] = []
        self.execution_order: list[str] = []
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        """Track tool execution in prompt."""
        self.call_count += 1
        self.prompts.append(prompt)

        # Record tool execution if detected
        if "tool:" in prompt.lower():
            self.execution_order.append(f"tool_call_{self.call_count}")
            self.tool_calls.append({
                "call_number": self.call_count,
                "prompt": prompt[:100],
            })

        return "Tool execution tracked successfully"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class StructuredOutputMockClient(LLMClientBase):
    """Mock client for structured output testing.

    Returns JSON responses with various validity states to test
    schema validation, error handling, and recovery mechanisms.
    """

    def __init__(self, response_type: str = "valid"):
        """Initialize structured output mock.

        Args:
            response_type: Type of response to generate
                - 'valid': Returns valid JSON matching schema
                - 'invalid_json': Returns malformed JSON
                - 'missing_fields': Returns valid JSON missing required fields
                - 'wrong_types': Returns JSON with incorrect field types
                - 'nested': Returns complex nested structures
        """
        self.response_type = response_type
        self.prompts: list[str] = []
        self.call_count = 0

        self.responses = {
            "valid": '{"status": "success", "value": 42, "message": "Valid output"}',
            "invalid_json": '{status: success, value: 42}',  # Missing quotes
            "missing_fields": '{"value": 42}',  # Missing required 'status' field
            "wrong_types": '{"status": 123, "value": "not_a_number"}',  # Wrong types
            "nested": '{"user": {"name": "Alice", "age": 30, "address": {"city": "NYC"}}}',
        }

    async def get_response(self, prompt: str) -> str:
        """Return structured response based on type."""
        self.call_count += 1
        self.prompts.append(prompt)

        return self.responses.get(
            self.response_type,
            self.responses["valid"]
        )

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class ReplanScenarioMockClient(LLMClientBase):
    """Mock client for RE-PLAN scenario testing.

    Simulates various RE-PLAN scenarios including loops, flaky responses,
    gradual improvement, and degradation.
    """

    def __init__(self, scenario: str = "improving"):
        """Initialize RE-PLAN scenario mock.

        Args:
            scenario: Type of scenario to simulate
                - 'loop': Always fails to trigger budget limits
                - 'flaky': Alternates between success and failure
                - 'improving': Gradually improves quality over attempts
                - 'degrading': Gets worse over attempts
                - 'immediate_success': Succeeds on first try
        """
        self.scenario = scenario
        self.attempt_count = 0
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        """Return response based on scenario and attempt count."""
        self.attempt_count += 1
        self.prompts.append(prompt)

        scenarios = {
            "loop": self._loop_scenario,
            "flaky": self._flaky_scenario,
            "improving": self._improving_scenario,
            "degrading": self._degrading_scenario,
            "immediate_success": self._immediate_success_scenario,
        }

        handler = scenarios.get(self.scenario, self._immediate_success_scenario)
        return handler()

    def _loop_scenario(self) -> str:
        """Always fail to trigger budget limits."""
        return f"Attempt {self.attempt_count}: Incomplete, needs another try"

    def _flaky_scenario(self) -> str:
        """Alternate between success and failure."""
        if self.attempt_count % 2 == 0:
            return f"Attempt {self.attempt_count}: Complete and correct"
        return f"Attempt {self.attempt_count}: Incomplete, needs revision"

    def _improving_scenario(self) -> str:
        """Gradually improve quality."""
        quality_levels = [
            "Very rough draft with major issues",
            "Better draft but still has problems",
            "Good response with minor issues",
            "Complete and correct response",
        ]

        index = min(self.attempt_count - 1, len(quality_levels) - 1)
        return f"Attempt {self.attempt_count}: {quality_levels[index]}"

    def _degrading_scenario(self) -> str:
        """Get worse over attempts."""
        return f"Attempt {self.attempt_count}: Response quality decreasing"

    def _immediate_success_scenario(self) -> str:
        """Succeed immediately."""
        return "Complete and correct response"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class ExecutionModeMockClient(LLMClientBase):
    """Mock client for execution mode testing.

    Simulates FAST vs SELF_CRITIC behavior with different response patterns
    for draft generation and evaluation phases.
    """

    def __init__(self, mode: str = "FAST", revision_pattern: str = "approve"):
        """Initialize execution mode mock.

        Args:
            mode: Execution mode ('FAST' or 'SELF_CRITIC')
            revision_pattern: How to respond to SELF_CRITIC evaluation
                - 'approve': Always approve the draft
                - 'improve': Request revision (then approve)
                - 'reject': Reject the draft
        """
        self.mode = mode
        self.revision_pattern = revision_pattern
        self.prompts: list[str] = []
        self.draft_count = 0
        self.evaluation_count = 0

    async def get_response(self, prompt: str) -> str:
        """Return response based on execution mode."""
        self.prompts.append(prompt)

        # Detect if this is a draft or evaluation request
        is_evaluation = any(
            keyword in prompt.lower()
            for keyword in ["evaluate", "review", "critic", "assess", "approve"]
        )

        if is_evaluation:
            self.evaluation_count += 1
            return self._get_evaluation_response()
        else:
            self.draft_count += 1
            return self._get_draft_response()

    def _get_draft_response(self) -> str:
        """Return draft response."""
        return f"Draft {self.draft_count}: This is the generated content"

    def _get_evaluation_response(self) -> str:
        """Return evaluation response based on pattern."""
        patterns = {
            "approve": '{"decision": "approve", "feedback": "Looks good"}',
            "improve": '{"decision": "revise", "feedback": "Needs improvement"}',
            "reject": '{"decision": "reject", "feedback": "Not acceptable"}',
        }

        return patterns.get(self.revision_pattern, patterns["approve"])

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class ChainBuilderMockClient(LLMClientBase):
    """Mock client for ChainBuilder pattern testing.

    Validates chain construction parameters and provides predictable responses
    for testing complex chain building scenarios.
    """

    def __init__(self, validate_chains: bool = True):
        """Initialize ChainBuilder mock.

        Args:
            validate_chains: Whether to validate chain construction
        """
        self.validate_chains = validate_chains
        self.prompts: list[str] = []
        self.chains_built: list[dict[str, Any]] = []
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        """Return response for chain building."""
        self.call_count += 1
        self.prompts.append(prompt)

        # Extract chain info from prompt for validation
        if "step:" in prompt.lower():
            chain_info = {
                "step_number": self.call_count,
                "prompt_length": len(prompt),
            }
            self.chains_built.append(chain_info)

        return f"Chain step {self.call_count} executed successfully"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class SerializationMockClient(LLMClientBase):
    """Mock client for serialization and tracing testing.

    Provides standard mock responses for testing chain serialization,
    deserialization, and LangFuse tracing features.
    """

    def __init__(self):
        self.prompts: list[str] = []
        self.call_count = 0
        self.trace_data: list[dict[str, Any]] = []

    async def get_response(self, prompt: str) -> str:
        """Return response for serialization tests."""
        self.call_count += 1
        self.prompts.append(prompt)

        # Record trace-like data
        self.trace_data.append({
            "call_number": self.call_count,
            "prompt_length": len(prompt),
            "timestamp": self.call_count,  # Simplified timestamp
        })

        return f"Response {self.call_count}: Serialized and traced"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        """Mock with retry tracking."""
        return await self.get_response(prompt)


class ModeAwareMockLLMClient(LLMClientBase):
    """Mock LLM client for FAST + SELF_CRITIC execution mode tests.

    Differentiates between draft generation and evaluator requests,
    returning appropriate responses for each phase.
    """

    def __init__(self):
        self.prompts: list[str] = []
        self._generation_round = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.prompts.append(prompt)

        # Check if this is an evaluator request
        is_evaluator = any(
            keyword in prompt.lower()
            for keyword in ["evaluate", "review", "critique", "assess"]
        )

        if is_evaluator:
            # Evaluator response (JSON decision)
            if self._generation_round == 0:
                self._generation_round += 1
                return '{"decision": "revise", "feedback": "Initial draft needs improvement"}'
            else:
                return '{"decision": "approve", "feedback": "Response is good"}'
        else:
            # Draft generation response
            return f"Draft round {self._generation_round}: Generated content"


class InvalidJsonCriticMockLLMClient(LLMClientBase):
    """Mock client that returns invalid JSON for testing error handling."""

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        # Return invalid JSON to test error handling
        return "{invalid json that will fail parsing"


class SequenceMockLLMClient(LLMClientBase):
    """Simple deterministic mock for step generations.

    Returns pre-configured responses in sequence for predictable testing.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt, retries=1)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.prompts.append(prompt)

        if self.calls < len(self.responses):
            response = self.responses[self.calls]
            self.calls += 1
            return response

        return "Default response"


class SimpleMockClient(LLMClientBase):
    """Returns a fixed, deterministic string for every request."""

    def __init__(self, response: str = "hello world mock response with some words"):
        self.response = response
        self.call_count = 0

    async def get_response(self, prompt: str) -> str:
        return await self.get_response_with_retries(prompt)

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        self.call_count += 1
        return self.response
