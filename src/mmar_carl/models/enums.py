"""
Enums for CARL reasoning system.
"""

from enum import StrEnum


class StepType(StrEnum):
    """Types of steps supported in reasoning chains."""

    LLM = "llm"  # Standard LLM reasoning step (default)
    TOOL = "tool"  # External tool/function call
    MCP = "mcp"  # Model Context Protocol server call
    MEMORY = "memory"  # Memory read/write operation
    TRANSFORM = "transform"  # Data transformation (no LLM)
    CONDITIONAL = "conditional"  # Conditional branching
    STRUCTURED_OUTPUT = "structured_output"  # LLM step with schema-constrained JSON output
    AGENT_SKILL = "agent_skill"  # AgentSkills format skill execution
    EVALUATION = "evaluation"  # Inline quality gate — evaluates another step's output
    AGENT_HANDOFF = "agent_handoff"  # Delegates execution to a complete sub-chain
    PARALLEL_SAMPLING = "parallel_sampling"  # Sample N independent LLM responses, vote on best
    TOOL_DISCOVERY = "tool_discovery"  # Discover and register tools at runtime
    HUMAN_INPUT = "human_input"  # Pause execution and wait for human input
    SUPERVISOR = "supervisor"  # LLM routes a task to one of N registered sub-chains
    DEBATE = "debate"  # Round-robin multi-agent debate with a judge synthesis
    MCP_RESOURCE = "mcp_resource"  # Read a named MCP resource into memory/history


class MemoryOperation(StrEnum):
    """Types of memory operations."""

    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    LIST = "list"


class Language(StrEnum):
    """Supported languages."""

    RUSSIAN = "ru"
    ENGLISH = "en"
