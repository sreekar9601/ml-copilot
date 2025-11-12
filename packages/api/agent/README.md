# Agent System - ML Documentation Copilot v3.0

> Agentic orchestration system built with LangGraph for intelligent ML documentation assistance

## Overview

This package transforms the ML Documentation Copilot from a static RAG pipeline into a dynamic agentic system that can:

- **Reason** about complex queries using LLM-powered decision making
- **Use tools** to search documentation, web, and execute code
- **Track costs** and enforce budget limits
- **Remember** conversation history across sessions
- **Self-reflect** on response quality (Phase 3)

## Architecture

```
┌─────────────────────────────────────────────────┐
│              User Query                         │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Agent Orchestrator (LangGraph)          │
│  • System prompt with guidelines                │
│  • Conversation history                         │
│  • Budget & iteration tracking                  │
└────────┬────────────────────┬───────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐   ┌─────────────────┐
│   Call Tools?   │   │  Final Answer?  │
└────────┬────────┘   └────────┬────────┘
         │                     │
         ▼                     ▼
┌─────────────────┐          END
│   Tool Node     │
│  • Doc Search   │
│  • Web Search   │
│  • Code Exec    │
└────────┬────────┘
         │
         └──────► Back to Agent
```

## Core Components

### 1. Orchestrator (`orchestrator.py`)

The main LangGraph workflow that coordinates agent behavior:

```python
from agent import orchestrator
from langchain_core.messages import HumanMessage

result = await orchestrator.ainvoke({
    "messages": [HumanMessage(content="How to use PyTorch DataLoader?")],
    "conversation_id": "session_123",
    "iteration_count": 0,
    "total_cost": 0.0,
    "current_tool": None,
    "tool_results": [],
    "reflection": None,
    "needs_improvement": False,
    "detected_frameworks": [],
    "query_intent": "unknown"
})

print(result["messages"][-1].content)
```

### 2. Tools (`tools/`)

#### Documentation Search (`retrieval_tools.py`)
```python
from agent.tools import hybrid_doc_search

result = hybrid_doc_search(
    query="PyTorch DataLoader batch size",
    frameworks=["pytorch"],
    top_k=5
)
# Returns: {chunks, frameworks_found, confidence, source_urls}
```

#### Web Search (`web_tools.py`)
```python
from agent.tools import web_search

result = web_search(
    query="PyTorch 2.5 breaking changes",
    max_results=3
)
# Returns: {results, query, num_results}
```

#### Code Execution (`code_executor_tool.py`)
```python
from agent.tools import execute_python_code

result = execute_python_code(
    code="import torch; print(torch.__version__)",
    timeout=30
)
# Returns: {success, stdout, stderr, error}
```

### 3. Memory (`memory/`)

Conversation storage and retrieval:

```python
from agent.memory import EnhancedChatHistory

memory = EnhancedChatHistory("session_123")
memory.add_messages(messages, tokens_used=100, cost=0.002)

# Get history
history = memory.get_last_n_messages(n=10)

# Get stats
stats = memory.get_conversation_stats()
```

### 4. Configuration (`config.py`)

Environment-based settings with Pydantic:

```python
from agent.config import config

# Access settings
print(config.orchestrator_model)  # "gemini-2.0-flash-exp"
print(config.max_iterations)       # 10
print(config.max_cost_per_session) # 0.50
```

### 5. State Management (`state.py`)

LangGraph state definitions:

```python
from agent.state import AgentState

# State is managed automatically by LangGraph
# Includes: messages, costs, tool results, metadata
```

## Configuration

### Environment Variables

Required:
```bash
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
```

Optional (with graceful fallback):
```bash
# Agent behavior
AGENT_ORCHESTRATOR_MODEL=gemini-2.0-flash-exp
AGENT_MAX_ITERATIONS=10
AGENT_MAX_COST_PER_SESSION=0.50

# Tools
TAVILY_API_KEY=your_tavily_key          # Web search
E2B_API_KEY=your_e2b_key                # Code execution

# Observability
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
```

### Pydantic Settings

All settings can be overridden via environment variables with `AGENT_` prefix:

```bash
# Override in .env
AGENT_TEMPERATURE=0.5
AGENT_ENABLE_CODE_EXECUTION=false
AGENT_MEMORY_BACKEND=redis
AGENT_REDIS_URL=redis://localhost:6379
```

## Usage Examples

### Basic Query

```python
from agent import orchestrator
from langchain_core.messages import HumanMessage
import asyncio

async def ask_question(query: str):
    result = await orchestrator.ainvoke({
        "messages": [HumanMessage(content=query)],
        "conversation_id": "demo",
        "iteration_count": 0,
        "total_cost": 0.0,
        "current_tool": None,
        "tool_results": [],
        "reflection": None,
        "needs_improvement": False,
        "detected_frameworks": [],
        "query_intent": "unknown"
    })
    return result["messages"][-1].content

# Run
answer = asyncio.run(ask_question("How to use PyTorch DataLoader?"))
print(answer)
```

### Multi-Turn Conversation

```python
from agent import orchestrator
from agent.memory import EnhancedChatHistory
from langchain_core.messages import HumanMessage

memory = EnhancedChatHistory("conv_001")

async def chat(query: str):
    # Get history
    history = memory.get_last_n_messages(n=10)
    
    # Add new query
    state = {
        "messages": history + [HumanMessage(content=query)],
        "conversation_id": "conv_001",
        # ... other state fields
    }
    
    # Run orchestrator
    result = await orchestrator.ainvoke(state)
    
    # Save to memory
    memory.add_messages(
        [HumanMessage(content=query), result["messages"][-1]],
        cost=result["total_cost"]
    )
    
    return result["messages"][-1].content

# Multi-turn conversation
await chat("What is PyTorch?")
await chat("Show me an example")  # Remembers context
await chat("How do I install it?") # Remembers context
```

### Cost Tracking

```python
from agent.utils import CostTracker

# Calculate cost
cost = CostTracker.calculate_cost(
    input_tokens=1000,
    output_tokens=500
)
print(f"Cost: ${cost:.4f}")

# Check budget
try:
    CostTracker.check_budget("session_123", current_cost=0.45)
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
```

## Tool Development

### Adding a New Tool

1. Create tool function with `@tool` decorator:

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param1: str = Field(description="Parameter description")
    param2: int = Field(default=10, description="Optional parameter")

@tool("my_tool_name", args_schema=MyToolInput)
def my_tool(param1: str, param2: int = 10) -> dict:
    """
    Tool description for the LLM.
    
    Args:
        param1: What this does
        param2: What this does
    
    Returns:
        Dictionary with results
    """
    # Tool implementation
    return {"result": "success"}
```

2. Add to orchestrator's tool list:

```python
# In orchestrator.py
from .tools.my_new_tool import my_tool

tools = [
    hybrid_doc_search,
    web_search,
    my_tool,  # Add here
    # ...
]
```

## Testing

### Unit Tests

```bash
# Test orchestrator
python test_orchestrator.py

# Test individual tools
python -c "from agent.tools import hybrid_doc_search; print(hybrid_doc_search('PyTorch DataLoader'))"
```

### Integration Tests

```python
import pytest
from agent import orchestrator

@pytest.mark.asyncio
async def test_orchestrator_basic():
    result = await orchestrator.ainvoke({
        "messages": [HumanMessage(content="What is PyTorch?")],
        # ... minimal state
    })
    assert result["messages"][-1].content
    assert result["total_cost"] > 0
```

## Performance

### Typical Latencies

- **Simple query (doc search only)**: 2-4 seconds
- **Complex query (doc + web search)**: 5-8 seconds
- **With code execution**: 8-12 seconds

### Cost Estimates

- **Gemini Flash** (default):
  - Input: $0.075 per 1M tokens
  - Output: $0.30 per 1M tokens
  - Typical query: $0.001 - $0.01

### Memory Usage

- SQLite database grows ~1KB per message
- In-memory state ~5MB per active conversation

## Troubleshooting

### Common Issues

**"No module named 'langchain_google_vertexai'"**
```bash
pip install -r requirements-api.txt
```

**"Web search unavailable"**
- Set `TAVILY_API_KEY` in `.env`
- Or disable: `AGENT_ENABLE_WEB_SEARCH=false`

**"Budget exceeded"**
- Increase limit: `AGENT_MAX_COST_PER_SESSION=1.0`
- Or clear conversation and start new one

**"Maximum iterations reached"**
- Increase limit: `AGENT_MAX_ITERATIONS=20`
- Or simplify query

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent")
```

## Roadmap

### Phase 2 (Weeks 5-8) - In Progress
- [ ] Checkpointing for state persistence
- [ ] FastAPI endpoints
- [ ] SSE streaming
- [ ] Frontend integration

### Phase 3 (Weeks 9-12) - Planned
- [ ] Debug agent (specialized for errors)
- [ ] Tutorial agent (step-by-step guides)
- [ ] Self-reflection for quality
- [ ] Comprehensive evaluation suite

## Contributing

When adding features:

1. Follow existing patterns (tools, agents, etc.)
2. Add type hints and docstrings
3. Handle errors gracefully
4. Add tests
5. Update this README

## License

Same as parent project.

