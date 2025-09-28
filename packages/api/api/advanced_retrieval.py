"""Advanced retrieval system with query expansion and re-ranking capabilities."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .retrieval import HybridRetriever, RetrievalResult
from .clients import get_client

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries to help with expansion strategy."""
    ARCHITECTURAL = "architectural"
    IMPLEMENTATION = "implementation"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    TUTORIAL = "tutorial"


@dataclass
class ExpandedQuery:
    """Represents an expanded query with metadata."""
    query: str
    query_type: QueryType
    weight: float
    metadata: Dict[str, Any]


class QueryExpander:
    """Expands user queries into multiple focused sub-queries."""
    
    def __init__(self):
        self.client = get_client()
        
    def expand_query(self, original_query: str) -> List[ExpandedQuery]:
        """Expand a single query into multiple focused sub-queries."""
        try:
            expanded_queries = []
            
            # Always include the original query
            expanded_queries.append(ExpandedQuery(
                query=original_query,
                query_type=self._classify_query(original_query),
                weight=1.0,
                metadata={"type": "original"}
            ))
        
        # Generate architectural sub-queries
        if self._is_architectural_query(original_query):
            expanded_queries.extend(self._generate_architectural_queries(original_query))
        
        # Generate implementation sub-queries
        if self._is_implementation_query(original_query):
            expanded_queries.extend(self._generate_implementation_queries(original_query))
        
        # Generate comparison sub-queries
        if self._is_comparison_query(original_query):
            expanded_queries.extend(self._generate_comparison_queries(original_query))
        
        # Generate best practices queries
        expanded_queries.extend(self._generate_best_practices_queries(original_query))
        
        # Generate troubleshooting queries
        expanded_queries.extend(self._generate_troubleshooting_queries(original_query))
        
        return expanded_queries
        
        except Exception as e:
            logger.error(f"Error expanding query '{original_query}': {e}")
            # Return just the original query if expansion fails
            return [ExpandedQuery(
                query=original_query,
                query_type=QueryType.TUTORIAL,
                weight=1.0,
                metadata={"type": "original", "error": str(e)}
            )]
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["architecture", "design", "system", "infrastructure", "deployment"]):
            return QueryType.ARCHITECTURAL
        elif any(term in query_lower for term in ["how to", "implement", "code", "example", "tutorial"]):
            return QueryType.IMPLEMENTATION
        elif any(term in query_lower for term in ["compare", "vs", "difference", "alternative"]):
            return QueryType.COMPARISON
        elif any(term in query_lower for term in ["error", "issue", "problem", "debug", "fix"]):
            return QueryType.TROUBLESHOOTING
        else:
            return QueryType.TUTORIAL
    
    def _is_architectural_query(self, query: str) -> bool:
        """Check if query is about system architecture."""
        architectural_terms = [
            "architecture", "design", "system", "infrastructure", "deployment",
            "scalability", "performance", "monitoring", "logging", "security",
            "microservices", "distributed", "cluster", "kubernetes", "docker"
        ]
        return any(term in query.lower() for term in architectural_terms)
    
    def _is_implementation_query(self, query: str) -> bool:
        """Check if query is about implementation details."""
        implementation_terms = [
            "how to", "implement", "code", "example", "tutorial", "step by step",
            "api", "function", "method", "class", "library", "package"
        ]
        return any(term in query.lower() for term in implementation_terms)
    
    def _is_comparison_query(self, query: str) -> bool:
        """Check if query is about comparing technologies."""
        comparison_terms = [
            "compare", "vs", "versus", "difference", "alternative", "choice",
            "better", "best", "pros", "cons", "trade-off"
        ]
        return any(term in query.lower() for term in comparison_terms)
    
    def _generate_architectural_queries(self, original_query: str) -> List[ExpandedQuery]:
        """Generate architectural sub-queries."""
        queries = []
        
        # Extract key technologies from original query
        technologies = self._extract_technologies(original_query)
        
        for tech in technologies:
            queries.append(ExpandedQuery(
                query=f"{tech} architecture best practices",
                query_type=QueryType.ARCHITECTURAL,
                weight=0.8,
                metadata={"technology": tech, "type": "architecture"}
            ))
            
            queries.append(ExpandedQuery(
                query=f"{tech} deployment patterns",
                query_type=QueryType.ARCHITECTURAL,
                weight=0.7,
                metadata={"technology": tech, "type": "deployment"}
            ))
        
        # Add general architectural queries
        queries.append(ExpandedQuery(
            query="ML system architecture patterns",
            query_type=QueryType.ARCHITECTURAL,
            weight=0.6,
            metadata={"type": "general_architecture"}
        ))
        
        return queries
    
    def _generate_implementation_queries(self, original_query: str) -> List[ExpandedQuery]:
        """Generate implementation sub-queries."""
        queries = []
        
        technologies = self._extract_technologies(original_query)
        
        for tech in technologies:
            queries.append(ExpandedQuery(
                query=f"{tech} implementation examples",
                query_type=QueryType.IMPLEMENTATION,
                weight=0.8,
                metadata={"technology": tech, "type": "examples"}
            ))
            
            queries.append(ExpandedQuery(
                query=f"{tech} API reference",
                query_type=QueryType.IMPLEMENTATION,
                weight=0.7,
                metadata={"technology": tech, "type": "api"}
            ))
        
        return queries
    
    def _generate_comparison_queries(self, original_query: str) -> List[ExpandedQuery]:
        """Generate comparison sub-queries."""
        queries = []
        
        technologies = self._extract_technologies(original_query)
        
        if len(technologies) >= 2:
            for i, tech1 in enumerate(technologies):
                for tech2 in technologies[i+1:]:
                    queries.append(ExpandedQuery(
                        query=f"{tech1} vs {tech2} comparison",
                        query_type=QueryType.COMPARISON,
                        weight=0.9,
                        metadata={"tech1": tech1, "tech2": tech2, "type": "comparison"}
                    ))
        
        return queries
    
    def _generate_best_practices_queries(self, original_query: str) -> List[ExpandedQuery]:
        """Generate best practices sub-queries."""
        queries = []
        
        technologies = self._extract_technologies(original_query)
        
        for tech in technologies:
            queries.append(ExpandedQuery(
                query=f"{tech} best practices",
                query_type=QueryType.TUTORIAL,
                weight=0.6,
                metadata={"technology": tech, "type": "best_practices"}
            ))
        
        return queries
    
    def _generate_troubleshooting_queries(self, original_query: str) -> List[ExpandedQuery]:
        """Generate troubleshooting sub-queries."""
        queries = []
        
        technologies = self._extract_technologies(original_query)
        
        for tech in technologies:
            queries.append(ExpandedQuery(
                query=f"{tech} common issues troubleshooting",
                query_type=QueryType.TROUBLESHOOTING,
                weight=0.5,
                metadata={"technology": tech, "type": "troubleshooting"}
            ))
        
        return queries
    
    def _extract_technologies(self, query: str) -> List[str]:
        """Extract technology names from the query."""
        technologies = []
        
        # Common ML/AI technologies
        tech_patterns = {
            "pytorch": ["pytorch", "torch"],
            "tensorflow": ["tensorflow", "tf"],
            "mlflow": ["mlflow"],
            "ray": ["ray", "ray serve"],
            "kserve": ["kserve", "k serve"],
            "kubernetes": ["kubernetes", "k8s"],
            "docker": ["docker"],
            "dvc": ["dvc"],
            "github actions": ["github actions", "ci/cd"],
            "aws": ["aws", "amazon web services"],
            "gcp": ["gcp", "google cloud", "vertex ai"],
            "azure": ["azure", "microsoft azure"]
        }
        
        query_lower = query.lower()
        for tech, patterns in tech_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                technologies.append(tech)
        
        return technologies


class CrossEncoderReranker:
    """Re-ranks results using a cross-encoder model for better relevance."""
    
    def __init__(self):
        self.client = get_client()
        
    def rerank(self, query: str, results: List[RetrievalResult], 
               top_k: int = 10) -> List[RetrievalResult]:
        """Re-rank results using cross-encoder scoring."""
        if not results:
            return results
        
        # For now, use a simple scoring approach
        # In production, you would use a proper cross-encoder model
        reranked_results = self._simple_rerank(query, results)
        
        # Update ranks
        for i, result in enumerate(reranked_results[:top_k]):
            result.rank = i + 1
        
        return reranked_results[:top_k]
    
    def _simple_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Simple re-ranking based on query-term matching."""
        query_terms = set(query.lower().split())
        
        for result in results:
            content_terms = set(result.content.lower().split())
            title_terms = set((result.metadata.get('title', '')).lower().split())
            
            # Calculate term overlap scores
            content_overlap = len(query_terms.intersection(content_terms)) / len(query_terms)
            title_overlap = len(query_terms.intersection(title_terms)) / len(query_terms)
            
            # Boost score based on term overlap
            term_boost = (content_overlap * 0.7 + title_overlap * 0.3) * 0.1
            result.score += term_boost
        
        # Sort by updated scores
        return sorted(results, key=lambda x: x.score, reverse=True)


class AdvancedRetriever:
    """Advanced retrieval system with query expansion and re-ranking."""
    
    def __init__(self):
        self.base_retriever = HybridRetriever()
        self.query_expander = QueryExpander()
        self.reranker = CrossEncoderReranker()
        
    def retrieve(self, query: str, top_k: int = 10, 
                 use_expansion: bool = True, use_reranking: bool = True) -> List[RetrievalResult]:
        """Advanced retrieval with query expansion and re-ranking."""
        
        try:
            logger.info(f"Advanced retrieval for query: {query}")
            
            if not use_expansion:
                # Use simple retrieval
                results = self.base_retriever.retrieve(query, top_k)
                if use_reranking:
                    results = self.reranker.rerank(query, results, top_k)
                return results
        
        # Expand query into multiple sub-queries
        expanded_queries = self.query_expander.expand_query(query)
        logger.info(f"Generated {len(expanded_queries)} expanded queries")
        
        # Retrieve documents for each expanded query
        all_results = {}
        
        for expanded_query in expanded_queries:
            logger.debug(f"Processing expanded query: {expanded_query.query}")
            
            # Retrieve documents for this sub-query
            sub_results = self.base_retriever.retrieve(
                expanded_query.query, 
                top_k=min(20, top_k * 2)  # Get more results for sub-queries
            )
            
            # Weight results by query weight
            for result in sub_results:
                result.score *= expanded_query.weight
                
                # Add to combined results
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # Keep the higher score
                    if result.score > all_results[result.chunk_id].score:
                        all_results[result.chunk_id] = result
        
        # Convert to list and sort by score
        combined_results = list(all_results.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Combined {len(combined_results)} unique results from expanded queries")
        
        # Re-rank if requested
        if use_reranking:
            combined_results = self.reranker.rerank(query, combined_results, top_k)
        
        return combined_results[:top_k]
        
        except Exception as e:
            logger.error(f"Error in advanced retrieval for query '{query}': {e}")
            # Fallback to simple retrieval
            try:
                logger.info("Falling back to simple retrieval")
                results = self.base_retriever.retrieve(query, top_k)
                return results
            except Exception as fallback_error:
                logger.error(f"Fallback retrieval also failed: {fallback_error}")
                return []
    
    def close(self):
        """Close the underlying retriever."""
        try:
            self.base_retriever.close()
        except Exception as e:
            logger.error(f"Error closing advanced retriever: {e}")


# Global advanced retriever instance
_advanced_retriever = None


def get_advanced_retriever() -> AdvancedRetriever:
    """Get or create global advanced retriever instance."""
    global _advanced_retriever
    if _advanced_retriever is None:
        _advanced_retriever = AdvancedRetriever()
    return _advanced_retriever


def advanced_retrieve_documents(query: str, top_k: int = 10, 
                                use_expansion: bool = True, 
                                use_reranking: bool = True) -> List[RetrievalResult]:
    """High-level function for advanced document retrieval."""
    retriever = get_advanced_retriever()
    return retriever.retrieve(query, top_k, use_expansion, use_reranking)


if __name__ == "__main__":
    # Test the advanced retriever
    logging.basicConfig(level=logging.INFO)
    
    # Test queries
    test_queries = [
        "Design a machine learning system for fraud detection",
        "How to implement PyTorch data loading with MLflow tracking",
        "Compare Ray Serve vs KServe for model deployment",
        "MLflow model registry best practices"
    ]
    
    retriever = AdvancedRetriever()
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Test with expansion and re-ranking
        results = retriever.retrieve(query, top_k=5, use_expansion=True, use_reranking=True)
        
        print(f"Retrieved {len(results)} results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result.metadata.get('title', 'Unknown')} (score: {result.score:.3f})")
            print(f"   Source: {result.metadata.get('source_url', 'Unknown')}")
            print(f"   Content: {result.content[:150]}...")
            print()
    
    retriever.close()
