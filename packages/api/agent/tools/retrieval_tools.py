"""Refactored RAG pipelines as LangChain tools."""

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import sys
import os

# Add parent directory to path to import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from api.retrieval import HybridRetriever

# Global retriever instance (singleton)
_retriever_instance = None


def get_retriever() -> HybridRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


class HybridSearchInput(BaseModel):
    """Input schema for hybrid documentation search."""
    query: str = Field(description="The search query for ML documentation")
    frameworks: Optional[list[str]] = Field(
        default=None,
        description="Optional list of ML frameworks to filter by: ['pytorch', 'tensorflow', 'sklearn', 'mlflow', 'ray', 'wandb']"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return (default 5)"
    )


@tool("hybrid_doc_search", args_schema=HybridSearchInput)
def hybrid_doc_search(
    query: str,
    frameworks: Optional[list[str]] = None,
    top_k: int = 5
) -> dict:
    """
    Search ML framework documentation using hybrid vector + keyword search.
    
    This tool searches across PyTorch, TensorFlow, Scikit-learn, MLflow, Ray,
    and Weights & Biases documentation using both semantic similarity (vector)
    and exact keyword matching (BM25), then fuses results with RRF.
    
    Use this tool when you need to:
    - Find API references or implementation details
    - Look up best practices or usage patterns
    - Understand how to use specific ML frameworks
    - Get code examples from official documentation
    
    Args:
        query: Natural language search query
        frameworks: Filter results to specific frameworks (e.g., ['pytorch', 'tensorflow'])
        top_k: Number of results to return
    
    Returns:
        Dictionary with retrieved chunks, metadata, and confidence score
    
    Examples:
        - "How to create a DataLoader in PyTorch"
        - "MLflow model registry best practices"
        - "Ray Tune hyperparameter tuning examples"
    """
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Get the retriever (reuses existing retrieval.py logic)
    retriever = get_retriever()
    
    # Perform hybrid search (using the retrieve method which does vector + keyword + RRF)
    results = retriever.retrieve(
        query=query,
        top_k=top_k * 2,  # Get more, then filter
        expand_context=False  # We'll limit results ourselves
    )
    
    logger.info(f"Retrieved {len(results)} results from retriever for query: {query}")
    if results:
        logger.info(f"First result has vendor: '{results[0].metadata.get('vendor', 'EMPTY')}'")
    
    # DISABLED: Framework filtering causes too many false negatives
    # The semantic search already handles framework relevance via the query
    # Explicit filtering by vendor metadata is too strict and filters out relevant results
    if frameworks:
        logger.info(f"Framework filter requested: {frameworks}, but skipping strict filtering (relying on semantic search)")
    
    # Limit to top_k
    results = results[:top_k]
    
    # Detect frameworks in results
    frameworks_found = list(set(
        r.metadata.get('vendor', 'unknown') for r in results
    ))
    
    # Calculate confidence (based on scores)
    if results:
        confidence = sum(r.score for r in results) / len(results)
        # Normalize confidence to 0-1 range (assuming scores are cosine similarity -1 to 1)
        confidence = (confidence + 1) / 2
    else:
        confidence = 0.0
    
    # Extract source URLs
    source_urls = list(set(
        r.metadata.get('source_url', '') for r in results
        if r.metadata.get('source_url')
    ))
    
    # Format output
    chunks = [
        {
            "chunk_id": r.chunk_id,  # Include chunk_id for frontend
            "content": r.content,
            "source": r.metadata.get('source_url', ''),
            "vendor": r.metadata.get('vendor', 'unknown'),
            "heading": r.metadata.get('heading_path', ''),
            "score": r.score
        }
        for r in results
    ]
    
    if not chunks and results:
        logger.warning(f"No chunks created despite having {len(results)} results - check metadata!")
    
    return {
        "chunks": chunks,
        "frameworks_found": frameworks_found,
        "confidence": confidence,
        "source_urls": source_urls,
        "query": query,
        "num_results": len(chunks)
    }


class SpecificDocInput(BaseModel):
    """Input for getting specific documentation."""
    framework: str = Field(description="One of 'pytorch', 'tensorflow', 'sklearn', 'mlflow', 'ray', 'wandb'")
    topic: str = Field(description="Specific topic or API to look up")


@tool("get_specific_documentation", args_schema=SpecificDocInput)
def get_specific_documentation(framework: str, topic: str) -> str:
    """
    Get focused documentation for a specific framework and topic.
    
    Args:
        framework: One of 'pytorch', 'tensorflow', 'sklearn', 'mlflow', 'ray', 'wandb'
        topic: Specific topic or API to look up
    
    Returns:
        Relevant documentation content
    """
    # Construct targeted query
    query = f"{framework} {topic}"
    
    # Search with framework filter
    result = hybrid_doc_search(
        query=query,
        frameworks=[framework],
        top_k=3
    )
    
    # Combine chunks into cohesive text
    if not result["chunks"]:
        return f"No documentation found for {framework} on topic: {topic}"
    
    combined = f"Documentation for {framework} - {topic}:\n\n"
    for i, chunk in enumerate(result["chunks"], 1):
        combined += f"--- Section {i} ({chunk['source']}) ---\n"
        combined += chunk["content"] + "\n\n"
    
    return combined

