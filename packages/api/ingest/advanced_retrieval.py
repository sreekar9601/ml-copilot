"""Advanced retrieval system with query expansion and re-ranking for architectural decisions."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ExpandedQuery:
    """Represents an expanded query with context."""
    original_query: str
    expanded_queries: List[str]
    query_type: str  # 'architectural', 'implementation', 'comparison', 'troubleshooting'
    focus_areas: List[str]
    priority_topics: List[str]

class QueryExpander:
    """Expands user queries into multiple focused sub-queries for better retrieval."""
    
    def __init__(self):
        # Query expansion patterns for different types of architectural questions
        self.expansion_patterns = {
            'architectural': {
                'scalability': ['scaling', 'performance', 'throughput', 'latency'],
                'reliability': ['fault tolerance', 'error handling', 'monitoring', 'alerting'],
                'maintainability': ['code organization', 'testing', 'documentation', 'versioning'],
                'security': ['authentication', 'authorization', 'encryption', 'compliance']
            },
            'implementation': {
                'setup': ['installation', 'configuration', 'initialization'],
                'integration': ['api', 'sdk', 'client', 'connection'],
                'optimization': ['performance tuning', 'resource management', 'caching']
            },
            'comparison': {
                'alternatives': ['vs', 'compared to', 'alternative', 'instead of'],
                'trade_offs': ['pros and cons', 'advantages', 'disadvantages', 'limitations']
            }
        }
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """Expand a user query into multiple focused sub-queries."""
        
        query_lower = query.lower()
        
        # Determine query type
        query_type = self._classify_query_type(query_lower)
        
        # Extract focus areas
        focus_areas = self._extract_focus_areas(query_lower)
        
        # Generate expanded queries
        expanded_queries = self._generate_expanded_queries(query, query_type, focus_areas)
        
        # Extract priority topics
        priority_topics = self._extract_priority_topics(query_lower)
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded_queries,
            query_type=query_type,
            focus_areas=focus_areas,
            priority_topics=priority_topics
        )
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        
        if any(term in query for term in ['design', 'architecture', 'system', 'infrastructure']):
            return 'architectural'
        elif any(term in query for term in ['how to', 'implement', 'setup', 'configure']):
            return 'implementation'
        elif any(term in query for term in ['vs', 'compare', 'alternative', 'better']):
            return 'comparison'
        elif any(term in query for term in ['error', 'problem', 'issue', 'debug']):
            return 'troubleshooting'
        else:
            return 'general'
    
    def _extract_focus_areas(self, query: str) -> List[str]:
        """Extract focus areas from the query."""
        
        focus_areas = []
        
        # Technology focus
        if 'pytorch' in query:
            focus_areas.append('pytorch')
        if 'mlflow' in query:
            focus_areas.append('mlflow')
        if 'ray' in query:
            focus_areas.append('ray')
        if 'kubernetes' in query or 'k8s' in query:
            focus_areas.append('kubernetes')
        if 'docker' in query:
            focus_areas.append('docker')
        
        # Functional focus
        if any(term in query for term in ['serving', 'deployment', 'inference']):
            focus_areas.append('model_serving')
        if any(term in query for term in ['training', 'learning', 'optimization']):
            focus_areas.append('training')
        if any(term in query for term in ['monitoring', 'logging', 'metrics']):
            focus_areas.append('monitoring')
        if any(term in query for term in ['data', 'pipeline', 'etl']):
            focus_areas.append('data_management')
        
        return focus_areas
    
    def _generate_expanded_queries(self, original_query: str, query_type: str, focus_areas: List[str]) -> List[str]:
        """Generate expanded queries based on the original query."""
        
        expanded = [original_query]
        
        # Add technology-specific variations
        for area in focus_areas:
            if area == 'pytorch':
                expanded.extend([
                    f"{original_query} PyTorch distributed training",
                    f"{original_query} PyTorch model serving",
                    f"{original_query} PyTorch performance optimization"
                ])
            elif area == 'mlflow':
                expanded.extend([
                    f"{original_query} MLflow model registry",
                    f"{original_query} MLflow experiment tracking",
                    f"{original_query} MLflow deployment"
                ])
            elif area == 'ray':
                expanded.extend([
                    f"{original_query} Ray Serve architecture",
                    f"{original_query} Ray distributed computing",
                    f"{original_query} Ray hyperparameter tuning"
                ])
        
        # Add architectural considerations
        if query_type == 'architectural':
            expanded.extend([
                f"{original_query} scalability considerations",
                f"{original_query} reliability patterns",
                f"{original_query} best practices",
                f"{original_query} trade-offs"
            ])
        
        # Add implementation details
        if query_type == 'implementation':
            expanded.extend([
                f"{original_query} setup configuration",
                f"{original_query} integration examples",
                f"{original_query} troubleshooting"
            ])
        
        return list(set(expanded))  # Remove duplicates
    
    def _extract_priority_topics(self, query: str) -> List[str]:
        """Extract priority topics for ranking."""
        
        priority_topics = []
        
        # High priority architectural topics
        if any(term in query for term in ['scalability', 'performance', 'throughput']):
            priority_topics.append('performance')
        if any(term in query for term in ['reliability', 'fault tolerance', 'monitoring']):
            priority_topics.append('reliability')
        if any(term in query for term in ['security', 'authentication', 'authorization']):
            priority_topics.append('security')
        if any(term in query for term in ['cost', 'efficiency', 'optimization']):
            priority_topics.append('efficiency')
        
        return priority_topics

class MetadataFilter:
    """Filters results based on rich metadata for architectural relevance."""
    
    def __init__(self):
        self.relevance_weights = {
            'chunk_type': {
                'architecture': 1.0,
                'best_practice': 0.9,
                'decision_guide': 0.95,
                'tutorial': 0.7,
                'api_reference': 0.6,
                'concept': 0.8
            },
            'priority': {
                'high': 1.0,
                'medium': 0.7,
                'low': 0.4
            },
            'architectural_relevance': {
                'high': 1.0,
                'medium': 0.6,
                'low': 0.3
            }
        }
    
    def filter_and_rank(self, results: List[Dict], expanded_query: ExpandedQuery) -> List[Dict]:
        """Filter and rank results based on architectural relevance."""
        
        filtered_results = []
        
        for result in results:
            metadata = result.get('metadata', {})
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result, expanded_query)
            
            # Apply filters
            if self._passes_filters(result, expanded_query):
                result['architectural_relevance_score'] = relevance_score
                filtered_results.append(result)
        
        # Sort by relevance score
        filtered_results.sort(key=lambda x: x['architectural_relevance_score'], reverse=True)
        
        return filtered_results
    
    def _calculate_relevance_score(self, result: Dict, expanded_query: ExpandedQuery) -> float:
        """Calculate architectural relevance score for a result."""
        
        metadata = result.get('metadata', {})
        content = result.get('content', '').lower()
        
        score = 0.0
        
        # Base score from chunk type
        chunk_type = metadata.get('chunk_type', 'concept')
        score += self.relevance_weights['chunk_type'].get(chunk_type, 0.5)
        
        # Priority boost
        priority = metadata.get('priority', 'medium')
        score += self.relevance_weights['priority'].get(priority, 0.7)
        
        # Architectural relevance boost
        arch_relevance = metadata.get('architectural_relevance', 0.5)
        score += arch_relevance * 0.3
        
        # Decision content boost
        if metadata.get('decision_content', False):
            score += 0.2
        
        # Implementation content boost for implementation queries
        if expanded_query.query_type == 'implementation' and metadata.get('implementation_content', False):
            score += 0.15
        
        # Topic matching boost
        topics = metadata.get('topics', [])
        for priority_topic in expanded_query.priority_topics:
            if priority_topic in topics:
                score += 0.1
        
        # Focus area matching boost
        for focus_area in expanded_query.focus_areas:
            if focus_area in content or focus_area in topics:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _passes_filters(self, result: Dict, expanded_query: ExpandedQuery) -> bool:
        """Check if result passes the relevance filters."""
        
        metadata = result.get('metadata', {})
        
        # Minimum relevance threshold
        if metadata.get('architectural_relevance', 0) < 0.3:
            return False
        
        # For architectural queries, prefer architectural content
        if expanded_query.query_type == 'architectural':
            chunk_type = metadata.get('chunk_type', '')
            if chunk_type not in ['architecture', 'best_practice', 'decision_guide']:
                return False
        
        return True

class AdvancedRetriever:
    """Advanced retrieval system combining query expansion, metadata filtering, and re-ranking."""
    
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.query_expander = QueryExpander()
        self.metadata_filter = MetadataFilter()
    
    def retrieve_architectural_knowledge(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve architectural knowledge using advanced techniques."""
        
        logger.info(f"Advanced retrieval for query: {query}")
        
        # Step 1: Expand the query
        expanded_query = self.query_expander.expand_query(query)
        logger.info(f"Expanded to {len(expanded_query.expanded_queries)} sub-queries")
        
        # Step 2: Retrieve documents for each expanded query
        all_results = []
        for sub_query in expanded_query.expanded_queries:
            try:
                results = self.base_retriever.retrieve(sub_query, top_k=top_k//2)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to retrieve for sub-query '{sub_query}': {e}")
        
        # Step 3: Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Step 4: Filter and rank by architectural relevance
        filtered_results = self.metadata_filter.filter_and_rank(unique_results, expanded_query)
        
        # Step 5: Return top-k results
        return filtered_results[:top_k]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on chunk_id."""
        
        seen_ids = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.get('chunk_id')
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
