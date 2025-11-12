"""
FastAPI endpoints for agentic system v3.0.

This provides the unified /agent/invoke endpoint that replaces
the old /ask, /ask-advanced, /howto endpoints with intelligent routing.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.orchestrator import orchestrator
from agent.memory.chat_history import EnhancedChatHistory
from agent.config import config

# Create router
router = APIRouter(prefix="/agent", tags=["agent"])


class AgentInvokeRequest(BaseModel):
    """Request model for agent invocation."""
    query: str = Field(..., description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn")
    stream: bool = Field(False, description="Enable streaming response")


class AgentInvokeResponse(BaseModel):
    """Response model for agent invocation."""
    response: str
    conversation_id: str
    metadata: dict


class ConversationStatsResponse(BaseModel):
    """Response model for conversation statistics."""
    conversation_id: str
    message_count: int
    total_tokens: int
    total_cost: float
    started_at: Optional[str]
    last_activity: Optional[str]


@router.post("/invoke", response_model=AgentInvokeResponse)
async def agent_invoke(request: AgentInvokeRequest):
    """
    Unified endpoint for all agent interactions.
    
    This replaces /ask, /ask-advanced, /howto, /multi-source endpoints.
    The agent automatically determines the best approach based on the query.
    
    Example:
        ```python
        response = requests.post("/agent/invoke", json={
            "query": "How to create a PyTorch DataLoader?",
            "conversation_id": "optional-session-id"
        })
        ```
    """
    
    # Generate or use existing conversation ID
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Get memory for this conversation
    memory = EnhancedChatHistory(conversation_id, db_path=config.memory_db_path)
    
    # Get conversation history (last N messages)
    history = memory.get_last_n_messages(n=config.max_history_messages)
    
    # Prepare initial state
    initial_state = {
        "messages": [
            *history,  # Include conversation history
            HumanMessage(content=request.query)
        ],
        "conversation_id": conversation_id,
        "iteration_count": 0,
        "total_cost": 0.0,
        "current_tool": None,
        "tool_results": [],
        "reflection": None,
        "needs_improvement": False,
        "detected_frameworks": [],
        "query_intent": "unknown"
    }
    
    # Configuration for this invocation
    config_dict = {
        "configurable": {
            "thread_id": conversation_id  # Enables checkpointing per conversation
        },
        "recursion_limit": config.max_iterations * 4  # Each iteration = 3-4 graph steps (agent -> tools -> capture -> agent)
    }
    
    try:
        # Invoke the orchestrator
        result = await orchestrator.ainvoke(initial_state, config=config_dict)
        
        # Extract final response
        final_message = result["messages"][-1]
        response_text = final_message.content
        
        # Save to memory
        memory.add_messages(
            [HumanMessage(content=request.query), final_message],
            tokens_used=0,  # TODO: Get actual token count from response metadata
            cost=result["total_cost"],
            tool_calls=[{"tool_name": tr.get("tool_name", "unknown")} for tr in result.get("tool_results", [])]
        )
        
        # Prepare metadata
        metadata = {
            "iterations": result["iteration_count"],
            "total_cost": result["total_cost"],
            "tools_used": [tr.get("tool_name") for tr in result.get("tool_results", [])],
            "frameworks_detected": result.get("detected_frameworks", [])
        }
        
        return AgentInvokeResponse(
            response=response_text,
            conversation_id=conversation_id,
            metadata=metadata
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent execution failed: {str(e)}"
        )


@router.get("/conversations/{conversation_id}", response_model=dict)
async def get_conversation(conversation_id: str):
    """
    Get conversation history and statistics.
    
    Returns all messages and metadata for a conversation.
    """
    try:
        memory = EnhancedChatHistory(conversation_id, db_path=config.memory_db_path)
        
        messages = memory.get_messages()
        stats = memory.get_conversation_stats()
        
        return {
            "conversation_id": conversation_id,
            "messages": [
                {
                    "type": m.__class__.__name__.replace("Message", "").lower(),
                    "content": m.content
                }
                for m in messages
            ],
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """
    Clear conversation history.
    
    Deletes all messages for the specified conversation.
    """
    try:
        memory = EnhancedChatHistory(conversation_id, db_path=config.memory_db_path)
        memory.clear()
        return {
            "status": "cleared",
            "conversation_id": conversation_id,
            "message": "Conversation history has been deleted"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation: {str(e)}"
        )


@router.post("/invoke/stream")
async def agent_invoke_stream(request: AgentInvokeRequest):
    """
    Streaming version of agent invocation using Server-Sent Events (SSE).
    
    Yields real-time events:
    - thought: Agent is thinking
    - tool_call: Tool is being called
    - tool_result: Tool execution completed
    - token: Token-by-token response
    - done: Final completion with metadata
    - error: Error occurred
    
    Example:
        ```javascript
        const eventSource = new EventSource('/agent/invoke/stream');
        eventSource.addEventListener('token', (e) => {
            const data = JSON.parse(e.data);
            console.log(data.content);
        });
        ```
    """
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    memory = EnhancedChatHistory(conversation_id, db_path=config.memory_db_path)
    history = memory.get_last_n_messages(n=config.max_history_messages)
    
    initial_state = {
        "messages": [
            *history,
            HumanMessage(content=request.query)
        ],
        "conversation_id": conversation_id,
        "iteration_count": 0,
        "total_cost": 0.0,
        "current_tool": None,
        "tool_results": [],
        "reflection": None,
        "needs_improvement": False,
        "detected_frameworks": [],
        "query_intent": "unknown"
    }
    
    config_dict = {
        "configurable": {"thread_id": conversation_id},
        "recursion_limit": config.max_iterations * 4  # Each iteration = 3-4 graph steps
    }
    
    async def event_generator():
        """Generate SSE events for streaming."""
        # Ensure json module is available (imported at module level)
        import json as json_module
        
        try:
            # Track state for final save
            final_state = None
            response_chunks = []
            
            # Stream events from the graph
            async for event in orchestrator.astream_events(
                initial_state,
                config=config_dict,
                version="v2"
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                
                # Agent is starting to think
                if event_type == "on_chat_model_start":
                    yield f"event: thought\ndata: {json_module.dumps({'content': '🧠 Analyzing your question...'})}\n\n"
                
                # Tool is being called
                elif event_type == "on_tool_start":
                    tool_name = event_name
                    tool_input = event.get("data", {}).get("input", {})
                    
                    # Map internal tool names to our tool names
                    tool_name_mapping = {
                        "tavily_search_results_json": "web_search",
                        "tavily_search_results": "web_search",
                        "tavily": "web_search",
                    }
                    
                    # Normalize tool name
                    normalized_tool_name = tool_name_mapping.get(tool_name, tool_name)
                    
                    # Format tool name for display
                    tool_display = {
                        "hybrid_doc_search": "🔍 Searching documentation",
                        "web_search": "🌐 Searching the web",
                        "execute_python_code": "⚡ Executing code",
                        "validate_code_syntax": "✅ Validating syntax",
                        "get_specific_documentation": "📚 Looking up specific docs"
                    }.get(normalized_tool_name, f"🔧 Using {normalized_tool_name}")
                    
                    yield f"event: tool_call\ndata: {json_module.dumps({'tool': normalized_tool_name, 'display': tool_display, 'status': 'running'})}\n\n"
                
                # Tool execution completed
                elif event_type == "on_tool_end":
                    tool_name = event_name
                    output = event.get("data", {}).get("output", {})
                    
                    # Create preview of result
                    if isinstance(output, dict):
                        if "num_results" in output:
                            preview = f"Found {output.get('num_results', 0)} results"
                        elif "success" in output:
                            preview = "✅ Success" if output["success"] else "❌ Failed"
                        else:
                            preview = "Completed"
                    else:
                        preview = str(output)[:100] if output else "Completed"
                    
                    yield f"event: tool_result\ndata: {json_module.dumps({'tool': tool_name, 'preview': preview, 'status': 'complete'})}\n\n"
                
                # Token from final response (streaming from LLM)
                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        response_chunks.append(token)
                        yield f"event: token\ndata: {json_module.dumps({'content': token})}\n\n"
                
                # Graph execution completed
                elif event_type == "on_chain_end" and event_name == "LangGraph":
                    final_state = event.get("data", {}).get("output")
                    break
            
            # Process final state
            if final_state:
                final_message = final_state["messages"][-1]
                full_response = final_message.content
                
                # If we didn't get tokens via streaming, send full response
                if not response_chunks:
                    # Send response in chunks for better UX
                    words = full_response.split()
                    for i in range(0, len(words), 5):  # 5 words at a time
                        chunk = " ".join(words[i:i+5]) + " "
                        yield f"event: token\ndata: {json_module.dumps({'content': chunk})}\n\n"
                        await asyncio.sleep(0.05)  # Small delay for visual effect
                
                # Save to memory
                memory.add_messages(
                    [HumanMessage(content=request.query), final_message],
                    tokens_used=0,
                    cost=final_state.get("total_cost", 0.0),
                    tool_calls=[{"tool_name": tr.get("tool_name")} for tr in final_state.get("tool_results", [])]
                )
                
                # Extract sources from tool results
                sources = []
                tool_results = final_state.get("tool_results", [])
                
                logger.info(f"Processing {len(tool_results)} tool results for source extraction")
                
                for tr in tool_results:
                    tool_name = tr.get("tool_name", "")
                    tool_output = tr.get("output", {})
                    
                    # Parse tool_output if it's a string (JSON)
                    if isinstance(tool_output, str):
                        try:
                            tool_output = json_module.loads(tool_output)
                        except (json_module.JSONDecodeError, ValueError):
                            logger.warning(f"Could not parse tool_output as JSON for {tool_name}")
                            continue
                    
                    # Debug logging
                    logger.info(f"Processing tool: {tool_name}, output type: {type(tool_output)}")
                    if isinstance(tool_output, dict):
                        logger.info(f"Tool output keys: {list(tool_output.keys())}")
                    
                    # Extract from hybrid_doc_search or get_specific_documentation
                    if tool_name in ["hybrid_doc_search", "get_specific_documentation"]:
                        if isinstance(tool_output, dict) and "chunks" in tool_output:
                            chunks = tool_output.get("chunks", [])
                            logger.info(f"Tool {tool_name}: found {len(chunks)} chunks")
                            
                            for chunk in chunks:
                                source_url = chunk.get("source", "")
                                if source_url and source_url not in [s["url"] for s in sources]:
                                    sources.append({
                                        "chunk_id": chunk.get("chunk_id", ""),
                                        "title": chunk.get("vendor", "Documentation"),
                                        "url": source_url,
                                        "heading_path": chunk.get("heading", ""),
                                        "anchor_link": "",
                                        "relevance_score": chunk.get("score", 0.0),
                                        "vendor": chunk.get("vendor", "unknown")
                                    })
                    
                    # Extract from web_search
                    elif tool_name == "web_search":
                        if isinstance(tool_output, dict) and "results" in tool_output:
                            web_results = tool_output.get("results", [])
                            logger.info(f"Tool {tool_name}: found {len(web_results)} web results")
                            
                            for result in web_results:
                                source_url = result.get("url", "")
                                if source_url and source_url not in [s["url"] for s in sources]:
                                    sources.append({
                                        "chunk_id": "",  # Web results don't have chunk IDs
                                        "title": result.get("title", "Web Result"),
                                        "url": source_url,
                                        "heading_path": "",
                                        "anchor_link": "",
                                        "relevance_score": result.get("score", 0.0),
                                        "vendor": "Web Search"
                                    })
                
                logger.info(f"Extracted {len(sources)} total sources for frontend")
                if sources:
                    logger.info(f"First source: {sources[0]}")
                
                # Send completion event
                metadata = {
                    "iterations": final_state.get("iteration_count", 0),
                    "cost": final_state.get("total_cost", 0.0),
                    "tools_used": [tr.get("tool_name") for tr in final_state.get("tool_results", [])],
                    "frameworks_detected": final_state.get("detected_frameworks", []),
                    "sources": sources  # Add sources for citations
                }
                
                yield f"event: done\ndata: {json_module.dumps({'conversation_id': conversation_id, 'metadata': metadata})}\n\n"
        
        except Exception as e:
            # Use json_module to avoid scoping issues
            yield f"event: error\ndata: {json_module.dumps({'error': str(e), 'type': type(e).__name__})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for the agent system.
    
    Returns system status and configuration.
    """
    return {
        "status": "healthy",
        "version": "3.0.0",
        "agent_model": config.orchestrator_model,
        "features": {
            "code_execution": config.enable_code_execution,
            "web_search": config.enable_web_search,
            "self_reflection": config.enable_self_reflection
        },
        "limits": {
            "max_iterations": config.max_iterations,
            "max_cost_per_session": config.max_cost_per_session
        }
    }

