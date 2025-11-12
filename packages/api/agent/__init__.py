"""
Agentic System for ML Documentation Copilot v3.0

This package contains the core agentic orchestration system built with LangGraph.

Components:
- orchestrator: Main agent coordination with LangGraph
- tools: RAG, web search, and code execution tools
- agents: Specialized agents (debug, tutorial)
- memory: Conversation history and semantic memory
- utils: Cost tracking and error handling
- evaluation: Testing and metrics
"""

from .orchestrator import orchestrator, create_orchestrator
from .config import config, get_config
from .state import AgentState, ToolCallState

__version__ = "3.0.0"

__all__ = [
    "orchestrator",
    "create_orchestrator",
    "config",
    "get_config",
    "AgentState",
    "ToolCallState",
]

