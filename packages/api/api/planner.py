"""Advanced planning and multi-step retrieval for tutorial generation."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np

from .clients import get_client, GENERATION_MODEL_NAME
from .retrieval import RetrievalResult, get_retriever

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors using numpy.
    Lightweight alternative to sklearn.metrics.pairwise.cosine_similarity
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    
    # Compute dot product
    dot_product = np.dot(a, b.T)
    
    # Compute norms
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    
    # Avoid division by zero
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_a * norm_b.T)
    
    return similarity


class Step(BaseModel):
    """A single step in a tutorial plan."""
    title: str
    goal: str
    query: str
    vendor_scope: Optional[str] = None


class Plan(BaseModel):
    """A complete tutorial plan with steps."""
    task: str
    intent: str = "howto"  # howto, conceptual, navigational, troubleshooting
    prereqs: List[str] = Field(default_factory=list)
    steps: List[Step]


class Citation(BaseModel):
    """A citation with source information."""
    title: str
    url: str
    anchor_link: Optional[str] = None
    quote: str  # ≤220 chars
    source_vendor: Optional[str] = None


class TutorialStep(BaseModel):
    """A completed tutorial step with citations."""
    title: str
    explanation: str
    commands: List[str] = Field(default_factory=list)
    citations: List[Citation]
    vendor_scope: Optional[str] = None


class Tutorial(BaseModel):
    """A complete tutorial with all steps."""
    title: str
    intent: str
    prereqs: List[str]
    steps: List[TutorialStep]
    total_citations: int


class AdvancedPlanner:
    """Enhanced planner for multi-step tutorials."""
    
    def __init__(self):
        self.client = get_client()
        self.retriever = get_retriever()
    
    def classify_intent(self, query: str) -> str:
        """Classify query intent to guide retrieval strategy."""
        try:
            prompt = f"""
Classify this query into one of these intents:
- howto: Step-by-step instructions, implementation, setup
- conceptual: Understanding concepts, theory, comparisons
- navigational: Finding specific docs, references
- troubleshooting: Debugging, error fixing, problems

Query: "{query}"

Return only the intent name (lowercase).
"""
            
            response = self.client.models.generate_content(
                model=GENERATION_MODEL_NAME,
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            
            intent = response.text.strip().lower()
            if intent in ["howto", "conceptual", "navigational", "troubleshooting"]:
                return intent
            return "howto"  # default
            
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return "howto"
    
    def create_plan(self, query: str) -> Plan:
        """Create a plan by breaking down the query into steps."""
        try:
            intent = self.classify_intent(query)
            
            prompt = f"""
Create a step-by-step plan for: "{query}"

Requirements:
- Break into logical, sequential steps
- Each step should be focused on ONE vendor/tool when possible
- Include prerequisites if needed
- Make steps actionable and specific

Return JSON with this structure:
{{
  "task": "brief description",
  "intent": "{intent}",
  "prereqs": ["prerequisite 1", "prerequisite 2"],
  "steps": [
    {{
      "title": "Step name",
      "goal": "What this step achieves",
      "query": "Search query for this step",
      "vendor_scope": "main vendor/tool for this step (optional)"
    }}
  ]
}}

Focus on practical implementation steps.
"""
            
            response = self.client.models.generate_content(
                model=GENERATION_MODEL_NAME,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"response_mime_type": "application/json"}
            )
            
            plan_data = json.loads(response.text)
            return Plan(**plan_data)
            
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            # Fallback: single step
            return Plan(
                task=query,
                intent="howto",
                steps=[Step(title="Complete Task", goal=query, query=query)]
            )


class EnhancedRetriever:
    """Enhanced retrieval with adaptive weights, MMR, and normalization."""
    
    def __init__(self):
        self.retriever = get_retriever()
        
        # Intent-based hybrid weights (BM25, vector)
        self.intent_weights = {
            "howto": (0.55, 0.45),
            "conceptual": (0.35, 0.65),
            "navigational": (0.6, 0.4),
            "troubleshooting": (0.5, 0.5)
        }
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0,1] range."""
        if not scores or len(scores) == 1:
            return scores
        
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def extract_vendor(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract vendor from source URL or metadata."""
        source_url = metadata.get('source_url', '')
        
        # Simple vendor detection from URL patterns
        if 'pytorch.org' in source_url:
            return 'pytorch'
        elif 'mlflow.org' in source_url:
            return 'mlflow'
        elif 'ray.io' in source_url:
            return 'ray'
        elif 'kserve' in source_url:
            return 'kserve'
        elif 'aws' in source_url:
            return 'aws'
        elif 'tensorflow.org' in source_url:
            return 'tensorflow'
        
        return None
    
    def enhanced_fusion(self, vector_results: List[RetrievalResult], 
                       keyword_results: List[RetrievalResult],
                       intent: str = "howto", k: int = 60) -> List[RetrievalResult]:
        """Enhanced RRF with normalization and boosting."""
        
        # Get adaptive weights
        bm25_weight, vector_weight = self.intent_weights.get(intent, (0.5, 0.5))
        
        # Normalize scores within each list
        if vector_results:
            vector_scores = [r.score for r in vector_results]
            norm_vector_scores = self.normalize_scores(vector_scores)
            for i, result in enumerate(vector_results):
                result.normalized_score = norm_vector_scores[i] * vector_weight
        
        if keyword_results:
            keyword_scores = [r.score for r in keyword_results]
            norm_keyword_scores = self.normalize_scores(keyword_scores)
            for i, result in enumerate(keyword_results):
                result.normalized_score = norm_keyword_scores[i] * bm25_weight
        
        # Create lookup for fusion
        all_results = {}
        
        # Add vector results with RRF
        for rank, result in enumerate(vector_results, 1):
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
                result.vector_rank = rank
                result.keyword_rank = float('inf')
            else:
                all_results[result.chunk_id].vector_rank = rank
        
        # Add keyword results with RRF
        for rank, result in enumerate(keyword_results, 1):
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
                result.keyword_rank = rank
                result.vector_rank = float('inf')
            else:
                all_results[result.chunk_id].keyword_rank = rank
        
        # Calculate enhanced RRF scores with boosting
        for chunk_id, result in all_results.items():
            # Base RRF score
            rrf_score = 0
            if result.vector_rank != float('inf'):
                rrf_score += 1 / (k + result.vector_rank)
            if result.keyword_rank != float('inf'):
                rrf_score += 1 / (k + result.keyword_rank)
            
            # Add boosting factors
            boost = 0.0
            metadata = result.metadata or {}
            
            # Boost for heading path matches (simple heuristic)
            heading = metadata.get('heading_path', '').lower()
            if any(term in heading for term in ['install', 'setup', 'getting started']):
                boost += 0.05
            
            # Boost for authority (official docs)
            source_url = metadata.get('source_url', '')
            if any(domain in source_url for domain in ['.org', 'docs.', 'github.com']):
                boost += 0.02
            
            result.score = rrf_score + boost
        
        # Sort by enhanced score
        fused_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    def mmr_diversify(self, results: List[RetrievalResult], 
                     lambda_param: float = 0.7, max_results: int = 8) -> List[RetrievalResult]:
        """Apply Maximal Marginal Relevance for diversity."""
        if len(results) <= max_results:
            return results
        
        selected = [results[0]]  # Start with highest scoring
        remaining = results[1:]
        
        while len(selected) < max_results and remaining:
            best_score = -float('inf')
            best_candidate = None
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score (from fusion)
                relevance = candidate.score
                
                # Calculate max similarity to already selected
                max_similarity = 0.0
                for selected_result in selected:
                    # Simple content similarity (could use embeddings)
                    content_sim = self._simple_content_similarity(
                        candidate.content, selected_result.content
                    )
                    max_similarity = max(max_similarity, content_sim)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = i
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)
            else:
                break
        
        return selected
    
    def _simple_content_similarity(self, content1: str, content2: str) -> float:
        """Simple content similarity based on word overlap."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve_for_step(self, step: Step, intent: str = "howto") -> List[RetrievalResult]:
        """Enhanced retrieval for a specific step."""
        try:
            # Get vector and keyword results
            vector_results = self.retriever.vector_search(step.query, top_k=20)
            keyword_results = self.retriever.keyword_search(step.query, top_k=20)
            
            # Enhanced fusion with intent-based weights
            fused_results = self.enhanced_fusion(vector_results, keyword_results, intent)
            
            # Apply MMR for diversity
            diversified_results = self.mmr_diversify(fused_results[:15], max_results=8)
            
            # Filter by vendor scope if specified
            if step.vendor_scope:
                vendor_filtered = []
                for result in diversified_results:
                    vendor = self.extract_vendor(result.metadata or {})
                    if vendor == step.vendor_scope:
                        vendor_filtered.append(result)
                
                # If vendor filtering gives us too few results, fall back to all
                if len(vendor_filtered) >= 2:
                    return vendor_filtered
            
            return diversified_results
            
        except Exception as e:
            logger.error(f"Error in enhanced retrieval: {e}")
            return []


def create_tutorial_from_plan(plan: Plan) -> Tutorial:
    """Create a complete tutorial from a plan."""
    planner = AdvancedPlanner()
    retriever = EnhancedRetriever()
    
    tutorial_steps = []
    total_citations = 0
    
    for step in plan.steps:
        try:
            # Retrieve for this step
            results = retriever.retrieve_for_step(step, plan.intent)
            
            if not results:
                logger.warning(f"No results for step: {step.title}")
                continue
            
            # Synthesize step with citations
            tutorial_step = synthesize_step(planner.client, step, results)
            tutorial_steps.append(tutorial_step)
            total_citations += len(tutorial_step.citations)
            
        except Exception as e:
            logger.error(f"Error processing step {step.title}: {e}")
            continue
    
    return Tutorial(
        title=plan.task,
        intent=plan.intent,
        prereqs=plan.prereqs,
        steps=tutorial_steps,
        total_citations=total_citations
    )


def synthesize_step(client, step: Step, results: List[RetrievalResult]) -> TutorialStep:
    """Synthesize a tutorial step from retrieval results."""
    try:
        # Determine majority vendor for scoping
        retriever = EnhancedRetriever()
        vendors = [
            retriever.extract_vendor(r.metadata or {}) 
            for r in results
        ]
        vendor_counts = {}
        for v in vendors:
            if v:
                vendor_counts[v] = vendor_counts.get(v, 0) + 1
        
        majority_vendor = max(vendor_counts.items(), key=lambda x: x[1])[0] if vendor_counts else None
        
        # Filter to single vendor if possible
        if majority_vendor:
            scoped_results = [
                r for r in results 
                if retriever.extract_vendor(r.metadata or {}) == majority_vendor
            ]
            if len(scoped_results) >= 2:
                results = scoped_results[:5]
        
        # Format context
        context_parts = []
        for i, result in enumerate(results[:5]):
            metadata = result.metadata or {}
            title = metadata.get('title', 'Unknown')
            url = metadata.get('source_url', '')
            anchor = metadata.get('anchor_link', '')
            
            context_parts.append(f"""
[{i+1}] {title}
URL: {url}{anchor}
Content: {result.content[:400]}...
""")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
You are writing ONE step of a tutorial: "{step.title}"
Goal: {step.goal}

Use ONLY the sources below. Every claim must include an inline citation like [^1].
Extract specific commands/code when available.

Return JSON with this structure:
{{
  "title": "{step.title}",
  "explanation": "Clear explanation with citations [^1] [^2]",
  "commands": ["command1", "command2"],
  "citations": [
    {{
      "title": "Source title",
      "url": "full url",
      "anchor_link": "anchor if available",
      "quote": "exact quote under 220 chars",
      "source_vendor": "vendor name if known"
    }}
  ]
}}

SOURCES:
{context}
"""
        
        response = client.models.generate_content(
            model=GENERATION_MODEL_NAME,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"response_mime_type": "application/json"}
        )
        
        step_data = json.loads(response.text)
        return TutorialStep(**step_data, vendor_scope=majority_vendor)
        
    except Exception as e:
        logger.error(f"Error synthesizing step: {e}")
        # Fallback step
        return TutorialStep(
            title=step.title,
            explanation=f"Unable to generate detailed instructions for: {step.goal}",
            commands=[],
            citations=[]
        )
