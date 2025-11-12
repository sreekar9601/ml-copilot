"""LangGraph state definitions for agent orchestration."""

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the agent orchestrator.
    
    The 'messages' key uses the add_messages reducer to append new messages
    while preserving the conversation history.
    """
    
    # Conversation history (automatically managed by add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Metadata
    conversation_id: str
    iteration_count: int
    total_cost: float
    
    # Current execution context
    current_tool: Optional[str]
    tool_results: list[dict]
    
    # Self-reflection
    reflection: Optional[str]
    needs_improvement: bool
    
    # User context
    detected_frameworks: list[str]
    query_intent: str  # "how-to", "debug", "conceptual", "comparison"


class ToolCallState(TypedDict):
    """State for individual tool executions."""
    tool_name: str
    tool_input: dict
    tool_output: str
    success: bool
    error: Optional[str]
    execution_time_ms: float
    cost: float

