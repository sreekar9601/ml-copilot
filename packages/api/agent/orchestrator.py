"""Main agent orchestrator using LangGraph."""

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import sys
import os
import vertexai

# Add api directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .state import AgentState
from .config import config
from .tools.retrieval_tools import hybrid_doc_search, get_specific_documentation
from .tools.web_tools import web_search
from .tools.code_executor_tool import execute_python_code, validate_code_syntax
from .utils.cost_tracker import CostTracker, BudgetExceededError
from .memory.chat_history import EnhancedChatHistory

# System prompt for the orchestrator
ORCHESTRATOR_PROMPT = """You are an expert ML/AI assistant specializing in PyTorch, TensorFlow, Scikit-learn, MLflow, Ray, and Weights & Biases.

## Your Role
Help ML engineers and data scientists with:
- API references and implementation details
- Best practices and design patterns
- Debugging and error resolution
- Code examples and tutorials
- Framework comparisons

## Available Tools
You have access to these tools:
1. **hybrid_doc_search**: Search official ML framework documentation (PRIMARY TOOL - use this first)
2. **web_search**: Find recent information, GitHub issues, Stack Overflow (use for recent updates)
3. **execute_python_code**: Run code in a sandbox to validate examples (use to test fixes)
4. **validate_code_syntax**: Quick syntax check without execution

## Guidelines
1. **Choose the right tool first**: 
   - For "known issues", "bug reports", "GitHub issues", "recent updates" → Use **web_search** FIRST
   - For API references, tutorials, general docs → Use **hybrid_doc_search** FIRST
2. **Search once or twice, then answer**: Don't repeatedly search for the same topic
3. **Maximum 2-3 tool calls per response**: Avoid excessive searching
4. **Be specific**: Cite exact function names, parameters, and code examples when available
5. **Explain tradeoffs**: When multiple approaches exist, explain pros/cons
6. **Know when to stop**: If you have enough information after 1-2 searches, generate the answer

## Response Format
- Start with a direct answer to the question
- Include working code examples when relevant
- Use numbered citations [1], [2], [3] to reference sources (DO NOT include URLs in your response text)
- Suggest related topics or next steps
- The citation numbers will be automatically linked to documentation sources in a References section

## When to Use Each Tool
- **hybrid_doc_search**: API refs, tutorials, concepts, general documentation (use 1-2 times maximum)
- **web_search**: ALWAYS use for:
  * Questions about "known issues", "bug reports", "GitHub issues"
  * Recent version updates, breaking changes, or "latest" information
  * Community discussions, Stack Overflow, blog posts
  * Troubleshooting specific errors with version numbers
  * Questions asking "are there" or "what are" about recent problems
- **execute_python_code**: Validate code examples if needed, when user asks to "test" or "verify"
- **validate_code_syntax**: Quick checks before suggesting code

## IMPORTANT: Stopping Criteria
- After calling tools 2-3 times, STOP and generate your answer
- If searches return 0 results, use your general knowledge and stop searching
- Do NOT keep searching repeatedly - provide the answer with available information
- Quality over quantity: A good answer with 1-2 searches is better than endless searching

Remember: You are helping real engineers. Be decisive and efficient."""


def create_orchestrator():
    """
    Create the main orchestrator agent graph.
    
    Flow:
    1. User query → Agent (reasoning)
    2. Agent decides to call tools OR provide final answer
    3. If tools called → ToolNode executes them → back to Agent
    4. Loop until Agent provides final answer
    """
    
    # Initialize Vertex AI using existing configuration
    from api.config import settings as api_settings
    
    # Initialize Vertex AI with project and location from existing settings
    vertexai.init(
        project=api_settings.google_cloud_project,
        location=api_settings.google_cloud_location
    )
    
    # Initialize LLM
    llm = ChatVertexAI(
        model_name=config.orchestrator_model,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        project=api_settings.google_cloud_project,
        location=api_settings.google_cloud_location
    )
    
    # Available tools
    tools = [
        hybrid_doc_search,
        get_specific_documentation,
        web_search,
        execute_python_code,
        validate_code_syntax,
    ]
    
    # Bind tools to LLM (enables function calling)
    llm_with_tools = llm.bind_tools(tools)
    
    # Create tool execution node
    tool_node = ToolNode(tools)
    
    def capture_tool_results(state: AgentState) -> AgentState:
        """
        Extract tool results from ToolMessage objects and save to state.
        This runs after tool execution to capture outputs.
        """
        import json
        import logging
        
        logger = logging.getLogger(__name__)
        
        messages = state["messages"]
        existing_results = state.get("tool_results", [])
        
        # Find the most recent AIMessage with tool_calls and following ToolMessages
        # Start from the end to find the latest tool execution
        new_results = []
        
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Look for corresponding ToolMessages after this AIMessage
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    tool_id = tool_call.get("id", "")
                    
                    # Find the corresponding ToolMessage
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], ToolMessage):
                            if hasattr(messages[j], 'tool_call_id') and messages[j].tool_call_id == tool_id:
                                # Found the result!
                                tool_output = messages[j].content
                                
                                # If content is a string, try to parse as JSON
                                if isinstance(tool_output, str):
                                    try:
                                        tool_output = json.loads(tool_output)
                                    except (json.JSONDecodeError, ValueError):
                                        # Keep as string if not valid JSON
                                        pass
                                
                                new_results.append({
                                    "tool_name": tool_name,
                                    "tool_input": tool_call.get("args", {}),
                                    "output": tool_output
                                })
                                break
                
                # Only process the most recent AIMessage with tool_calls
                break
        
        # Merge with existing results (accumulate over multiple iterations)
        all_results = existing_results + new_results
        
        return {
            "tool_results": all_results
        }
    
    def agent_node(state: AgentState) -> AgentState:
        """
        Main agent reasoning node.
        
        Takes the current state, formats the prompt with history,
        calls the LLM, and returns updated state.
        """
        
        # Budget check
        try:
            CostTracker.check_budget(
                state["conversation_id"],
                state["total_cost"]
            )
        except BudgetExceededError as e:
            return {
                "messages": [AIMessage(content=f"⚠️ {str(e)}\nPlease start a new conversation.")],
                "needs_improvement": False
            }
        
        # Iteration limit check
        if state["iteration_count"] >= config.max_iterations:
            return {
                "messages": [AIMessage(content="⚠️ Maximum iterations reached. Providing best answer with available information...")],
                "needs_improvement": False
            }
        
        # Build messages for LLM
        messages = [
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            *state["messages"]
        ]
        
        # Call LLM with tool binding
        response = llm_with_tools.invoke(messages)
        
        # Calculate cost (rough estimate based on token counts)
        # In production, get actual token counts from API response
        input_tokens = sum(CostTracker.estimate_tokens(str(m.content)) for m in messages)
        output_tokens = CostTracker.estimate_tokens(str(response.content))
        cost = CostTracker.calculate_cost(int(input_tokens), int(output_tokens))
        
        # Extract tool calls if any
        tool_calls_info = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls_info.append({
                    "tool_name": tc.get("name", "unknown"),
                    "tool_input": tc.get("args", {})
                })
        
        return {
            "messages": [response],
            "iteration_count": state["iteration_count"] + 1,
            "total_cost": state["total_cost"] + cost,
            "current_tool": tool_calls_info[0]["tool_name"] if tool_calls_info else None,
            "tool_results": state.get("tool_results", []) + tool_calls_info
        }
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """
        Determine next step based on agent's output.
        
        If the last message has tool_calls, route to ToolNode.
        Otherwise, we have a final answer, so end.
        """
        last_message = state["messages"][-1]
        
        # If LLM called tools, execute them
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Otherwise, we have a final answer
        return "end"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("capture_results", capture_tool_results)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools execute, capture results, then go back to agent
    workflow.add_edge("tools", "capture_results")
    workflow.add_edge("capture_results", "agent")
    
    # Compile graph without checkpointer for now
    # Note: Checkpointing will be added in a future update
    # The agent will still work but conversations won't persist across restarts
    return workflow.compile()


# Create global orchestrator instance
orchestrator = create_orchestrator()

