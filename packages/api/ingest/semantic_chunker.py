"""Semantic chunker for architectural knowledge extraction."""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ArchitecturalChunk:
    """Represents a semantically meaningful chunk of architectural knowledge."""
    content: str
    chunk_id: str
    source_url: str
    title: str
    heading_path: str
    chunk_type: str  # 'concept', 'tutorial', 'api_reference', 'best_practice', 'architecture'
    topics: List[str]
    priority: str
    metadata: Dict[str, Any]
    token_count: int
    prev_id: Optional[str] = None
    next_id: Optional[str] = None

class SemanticChunker:
    """Semantic chunker that preserves architectural context and decision-making information."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Patterns for different types of content
        self.architectural_patterns = {
            'decision_points': [
                r'when to use',
                r'choose between',
                r'trade-offs',
                r'considerations',
                r'best practices',
                r'recommendations'
            ],
            'implementation_patterns': [
                r'implementation',
                r'example',
                r'code sample',
                r'configuration',
                r'setup'
            ],
            'conceptual_explanations': [
                r'overview',
                r'introduction',
                r'concepts',
                r'architecture',
                r'design'
            ]
        }
    
    def chunk_document(self, document: Dict[str, Any]) -> List[ArchitecturalChunk]:
        """Chunk a document semantically, preserving architectural context."""
        
        content = document.get('content', '')
        source_url = document.get('url', '')
        title = document.get('title', '')
        
        # Determine document type and topics
        doc_type = self._classify_document_type(content, source_url)
        topics = self._extract_topics(content, source_url)
        priority = self._determine_priority(source_url, topics)
        
        # Split by semantic boundaries
        semantic_sections = self._split_by_semantic_boundaries(content)
        
        chunks = []
        for i, section in enumerate(semantic_sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            chunk = ArchitecturalChunk(
                content=section,
                chunk_id=f"{document.get('id', 'unknown')}_{i}",
                source_url=source_url,
                title=title,
                heading_path=self._extract_heading_path(section),
                chunk_type=doc_type,
                topics=topics,
                priority=priority,
                metadata={
                    'section_index': i,
                    'total_sections': len(semantic_sections),
                    'architectural_relevance': self._assess_architectural_relevance(section),
                    'decision_content': self._contains_decision_content(section),
                    'implementation_content': self._contains_implementation_content(section)
                },
                token_count=len(section.split())
            )
            chunks.append(chunk)
        
        # Link chunks together
        for i in range(len(chunks) - 1):
            chunks[i].next_id = chunks[i + 1].chunk_id
            chunks[i + 1].prev_id = chunks[i].chunk_id
        
        return chunks
    
    def _classify_document_type(self, content: str, source_url: str) -> str:
        """Classify the document type based on content and URL."""
        
        url_lower = source_url.lower()
        content_lower = content.lower()
        
        if 'api' in url_lower or 'reference' in url_lower:
            return 'api_reference'
        elif 'tutorial' in url_lower or 'getting-started' in url_lower:
            return 'tutorial'
        elif 'architecture' in url_lower or 'design' in url_lower:
            return 'architecture'
        elif 'best-practices' in url_lower or 'guide' in url_lower:
            return 'best_practice'
        elif any(term in content_lower for term in ['decision', 'trade-off', 'consideration']):
            return 'decision_guide'
        else:
            return 'concept'
    
    def _extract_topics(self, content: str, source_url: str) -> List[str]:
        """Extract relevant topics from content."""
        
        topics = []
        content_lower = content.lower()
        
        # Technology topics
        tech_topics = {
            'pytorch': ['pytorch', 'torch', 'dataloader', 'ddp'],
            'mlflow': ['mlflow', 'experiment', 'model_registry'],
            'ray': ['ray', 'serve', 'tune', 'distributed'],
            'kubernetes': ['kubernetes', 'k8s', 'pod', 'deployment'],
            'docker': ['docker', 'container', 'image'],
            'aws': ['aws', 'sagemaker', 'ec2', 's3'],
            'gcp': ['gcp', 'vertex', 'google_cloud'],
            'azure': ['azure', 'ml_studio', 'microsoft']
        }
        
        for topic, keywords in tech_topics.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        # Functional topics
        functional_topics = {
            'model_serving': ['serving', 'inference', 'deployment'],
            'training': ['training', 'learning', 'optimization'],
            'monitoring': ['monitoring', 'logging', 'metrics'],
            'data_management': ['data', 'pipeline', 'versioning'],
            'mlops': ['mlops', 'ci_cd', 'automation']
        }
        
        for topic, keywords in functional_topics.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return list(set(topics))
    
    def _determine_priority(self, source_url: str, topics: List[str]) -> str:
        """Determine the priority of this content for architectural decisions."""
        
        # High priority for core architectural concepts
        high_priority_indicators = [
            'architecture', 'design', 'best-practices', 'decision',
            'trade-off', 'comparison', 'recommendation'
        ]
        
        if any(indicator in source_url.lower() for indicator in high_priority_indicators):
            return 'high'
        
        # High priority for core technologies
        core_technologies = ['pytorch', 'mlflow', 'ray', 'kubernetes']
        if any(tech in topics for tech in core_technologies):
            return 'high'
        
        return 'medium'
    
    def _split_by_semantic_boundaries(self, content: str) -> List[str]:
        """Split content by semantic boundaries (headings, paragraphs, code blocks)."""
        
        # Split by markdown headings
        sections = re.split(r'\n#{1,6}\s+', content)
        
        # Further split long sections by paragraphs
        final_sections = []
        for section in sections:
            if len(section) <= self.max_chunk_size:
                final_sections.append(section)
            else:
                # Split by paragraphs
                paragraphs = section.split('\n\n')
                current_chunk = ""
                
                for para in paragraphs:
                    if len(current_chunk + para) <= self.max_chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            final_sections.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                
                if current_chunk:
                    final_sections.append(current_chunk.strip())
        
        return [s.strip() for s in final_sections if s.strip()]
    
    def _extract_heading_path(self, content: str) -> str:
        """Extract the heading path for context."""
        lines = content.split('\n')
        headings = []
        
        for line in lines[:5]:  # Check first 5 lines for headings
            if line.startswith('#'):
                heading = line.strip('#').strip()
                headings.append(heading)
        
        return ' > '.join(headings) if headings else ''
    
    def _assess_architectural_relevance(self, content: str) -> float:
        """Assess how relevant this content is for architectural decisions (0-1)."""
        
        content_lower = content.lower()
        
        # High relevance indicators
        high_relevance_terms = [
            'architecture', 'design', 'pattern', 'best practice',
            'trade-off', 'consideration', 'recommendation', 'decision',
            'scalability', 'performance', 'reliability', 'maintainability'
        ]
        
        relevance_score = 0.0
        for term in high_relevance_terms:
            if term in content_lower:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _contains_decision_content(self, content: str) -> bool:
        """Check if content contains decision-making information."""
        
        decision_indicators = [
            'when to use', 'choose between', 'consider', 'recommend',
            'trade-off', 'pros and cons', 'advantages', 'disadvantages'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in decision_indicators)
    
    def _contains_implementation_content(self, content: str) -> bool:
        """Check if content contains implementation details."""
        
        implementation_indicators = [
            'code', 'example', 'implementation', 'configuration',
            'setup', 'install', 'run', 'execute'
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in implementation_indicators)
