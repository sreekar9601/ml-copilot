"""Web search tool for finding recent information."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import os

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    from pathlib import Path
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Check if Tavily is available (check at runtime, not import time)
def _check_tavily_available():
    """Check if Tavily is available at runtime."""
    if not os.getenv("TAVILY_API_KEY"):
        return False, None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        return True, TavilySearchResults
    except ImportError:
        return False, None


class WebSearchInput(BaseModel):
    """Input for web search."""
    query: str = Field(description="Search query for recent information")
    max_results: int = Field(default=3, description="Maximum number of results")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str, max_results: int = 3) -> dict:
    """
    Search the web for recent information not available in documentation.
    
    **USE THIS TOOL FIRST** when the query asks about:
    - "known issues", "bug reports", "GitHub issues" (ALWAYS use for these!)
    - Recent version updates, breaking changes, or "latest" information
    - Questions like "are there known issues with X" or "what are the problems with Y"
    - Troubleshooting specific errors with version numbers mentioned
    - Stack Overflow discussions and community solutions
    - Blog posts about best practices or recent changes
    - Breaking news about ML frameworks
    
    Args:
        query: What to search for
        max_results: Number of results to return (default 3)
    
    Returns:
        Dictionary with search results including titles, URLs, and snippets
    
    Examples:
        - "PyTorch 2.5 breaking changes"
        - "CUDA out of memory error solutions"
        - "latest MLflow model registry features"
    """
    
    # Check availability at runtime
    tavily_available, TavilySearchResults = _check_tavily_available()
    
    if not tavily_available:
        return {
            "error": "Web search unavailable - TAVILY_API_KEY not configured or tavily-python not installed",
            "results": [],
            "query": query,
            "suggestion": "Configure TAVILY_API_KEY in .env and install langchain-community to enable web search"
        }
    
    try:
        tavily = TavilySearchResults(max_results=max_results)
        results = tavily.invoke(query)
        
        # Format results for better readability
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", "N/A"),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            })
        
        return {
            "results": formatted_results,
            "query": query,
            "num_results": len(formatted_results),
            "source": "tavily_web_search"
        }
    
    except Exception as e:
        return {
            "error": f"Web search failed: {str(e)}",
            "results": [],
            "query": query
        }

