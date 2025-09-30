#!/usr/bin/env python3
"""
Production-grade semantic chunker with quality controls and rich metadata.
Focuses on preserving context and ensuring high-quality chunks.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProductionChunk:
    """High-quality chunk with comprehensive metadata."""
    chunk_id: str
    content: str
    source_url: str
    title: str
    heading_path: str
    chunk_type: str  # 'concept', 'tutorial', 'api_reference', 'code_example', 'best_practice'
    vendor: str
    topics: List[str]
    priority: str
    quality_score: float
    word_count: int
    has_code: bool
    has_examples: bool
    technical_depth: str  # 'beginner', 'intermediate', 'advanced'
    metadata: Dict[str, Any]
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None

class ProductionChunker:
    """Production-grade semantic chunker with quality controls."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Content type patterns
        self.tutorial_patterns = [
            r'tutorial', r'guide', r'how.?to', r'getting.?started',
            r'walkthrough', r'step.?by.?step', r'example'
        ]
        
        self.api_patterns = [
            r'api', r'reference', r'function', r'method', r'class',
            r'parameter', r'return', r'interface'
        ]
        
        self.code_patterns = [
            r'```', r'`[^`]+`', r'def\s+', r'class\s+', r'import\s+',
            r'function\s+', r'const\s+', r'var\s+', r'<code>'
        ]
        
        # Technical depth indicators
        self.beginner_indicators = [
            'introduction', 'basics', 'getting started', 'simple',
            'easy', 'beginner', 'tutorial', 'example'
        ]
        
        self.advanced_indicators = [
            'advanced', 'optimization', 'performance', 'scaling',
            'production', 'enterprise', 'architecture', 'design'
        ]
    
    def chunk_document(self, document: Dict[str, Any]) -> List[ProductionChunk]:
        """Chunk a document with production-grade quality controls."""
        chunks = []
        
        # Extract document metadata
        vendor = document.get('source_vendor', 'unknown')
        doc_type = document.get('doc_type', 'unknown')
        topics = document.get('topics', [])
        priority = document.get('priority', 'medium')
        
        # Classify document type
        chunk_type = self._classify_document_type(document['content'], document['url'])
        
        # Assess technical depth
        technical_depth = self._assess_technical_depth(document['content'])
        
        # Split content into semantic chunks
        content_chunks = self._split_content_semantically(document['content'])
        
        # Create chunks with rich metadata
        for i, chunk_content in enumerate(content_chunks):
            if not chunk_content.strip():
                continue
            
            # Quality assessment
            quality_score = self._assess_chunk_quality(chunk_content)
            if quality_score < 0.4:  # Skip low-quality chunks
                continue
            
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(document['url'], i)
            
            # Extract heading path
            heading_path = self._extract_heading_path(chunk_content, document.get('heading_structure', []))
            
            # Analyze chunk characteristics
            has_code = self._has_code_examples(chunk_content)
            has_examples = self._has_practical_examples(chunk_content)
            word_count = len(chunk_content.split())
            
            # Create chunk
            chunk = ProductionChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                source_url=document['url'],
                title=document['title'],
                heading_path=heading_path,
                chunk_type=chunk_type,
                vendor=vendor,
                topics=topics,
                priority=priority,
                quality_score=quality_score,
                word_count=word_count,
                has_code=has_code,
                has_examples=has_examples,
                technical_depth=technical_depth,
                metadata={
                    'original_doc_type': doc_type,
                    'chunk_index': i,
                    'total_chunks': len(content_chunks),
                    'scraped_at': document.get('metadata', {}).get('scraped_at'),
                    'source_quality': document.get('quality_score', 0.0),
                    'has_code_examples': has_code,
                    'has_practical_examples': has_examples,
                    'content_length': len(chunk_content),
                    'heading_structure': document.get('heading_structure', [])
                }
            )
            
            chunks.append(chunk)
        
        # Link chunks together
        self._link_chunks(chunks)
        
        logger.info(f"Created {len(chunks)} high-quality chunks from {document['title']}")
        return chunks
    
    def _split_content_semantically(self, content: str) -> List[str]:
        """Split content into semantic chunks preserving context."""
        # Split by headings first
        heading_splits = re.split(r'\n(#{1,6}\s+.+)\n', content)
        
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(heading_splits):
            if part.strip().startswith('#'):
                # This is a heading
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + "\n"
            else:
                # This is content
                current_chunk += part
                
                # Check if chunk is getting too long
                if len(current_chunk.split()) > self.max_chunk_size:
                    # Split by paragraphs
                    para_chunks = self._split_by_paragraphs(current_chunk)
                    chunks.extend(para_chunks[:-1])  # Add all but last
                    current_chunk = para_chunks[-1] if para_chunks else ""
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Further split large chunks
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs."""
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len((current_chunk + para).split()) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_large_chunk(self, content: str) -> List[str]:
        """Split large chunks by sentences while preserving context."""
        sentences = re.split(r'(?<=[.!?])\s+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len((current_chunk + sentence).split()) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _classify_document_type(self, content: str, url: str) -> str:
        """Classify the type of content."""
        content_lower = content.lower()
        url_lower = url.lower()
        
        # Check for tutorial patterns
        if any(re.search(pattern, content_lower) for pattern in self.tutorial_patterns):
            return 'tutorial'
        
        # Check for API reference patterns
        if any(re.search(pattern, content_lower) for pattern in self.api_patterns):
            return 'api_reference'
        
        # Check for code examples
        if any(re.search(pattern, content_lower) for pattern in self.code_patterns):
            return 'code_example'
        
        # Check URL patterns
        if '/tutorial' in url_lower or '/guide' in url_lower:
            return 'tutorial'
        elif '/api' in url_lower or '/reference' in url_lower:
            return 'api_reference'
        
        return 'concept'
    
    def _assess_technical_depth(self, content: str) -> str:
        """Assess the technical depth of content."""
        content_lower = content.lower()
        
        beginner_count = sum(1 for indicator in self.beginner_indicators if indicator in content_lower)
        advanced_count = sum(1 for indicator in self.advanced_indicators if indicator in content_lower)
        
        if advanced_count > beginner_count:
            return 'advanced'
        elif beginner_count > 0:
            return 'beginner'
        else:
            return 'intermediate'
    
    def _assess_chunk_quality(self, content: str) -> float:
        """Assess the quality of a chunk."""
        score = 0.0
        
        # Length factor
        word_count = len(content.split())
        if word_count > 100:
            score += 0.3
        elif word_count > 50:
            score += 0.2
        
        # Code examples factor
        if self._has_code_examples(content):
            score += 0.3
        
        # Structure factor
        if re.search(r'#{1,6}\s+', content):  # Has headings
            score += 0.2
        
        # Technical content factor
        technical_terms = ['function', 'class', 'method', 'api', 'config', 'example']
        technical_count = sum(1 for term in technical_terms if term in content.lower())
        if technical_count > 2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _has_code_examples(self, content: str) -> bool:
        """Check if content has code examples."""
        return any(re.search(pattern, content) for pattern in self.code_patterns)
    
    def _has_practical_examples(self, content: str) -> bool:
        """Check if content has practical examples."""
        example_indicators = ['example', 'for instance', 'such as', 'e.g.', 'like']
        return any(indicator in content.lower() for indicator in example_indicators)
    
    def _extract_heading_path(self, content: str, heading_structure: List[str]) -> str:
        """Extract the heading path for a chunk."""
        # Find the most relevant heading from the structure
        for heading in reversed(heading_structure):
            if heading.lower() in content.lower():
                return heading
        
        # Fallback to first heading in content
        heading_match = re.search(r'#{1,6}\s+(.+)', content)
        if heading_match:
            return heading_match.group(1).strip()
        
        return ""
    
    def _generate_chunk_id(self, url: str, index: int) -> str:
        """Generate a unique chunk ID."""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{url_hash}_{index}"
    
    def _link_chunks(self, chunks: List[ProductionChunk]):
        """Link chunks together for context preservation."""
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].prev_chunk_id = chunks[i-1].chunk_id
            if i < len(chunks) - 1:
                chunks[i].next_chunk_id = chunks[i+1].chunk_id

def main():
    """Test the production chunker."""
    import json
    from pathlib import Path
    
    # Load a sample document
    sample_doc = {
        'url': 'https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html',
        'title': 'PyTorch Quickstart Tutorial',
        'content': '''
# PyTorch Quickstart Tutorial

This tutorial will walk you through the basics of PyTorch.

## Tensors

Tensors are the fundamental data structure in PyTorch.

```python
import torch
x = torch.tensor([1, 2, 3, 4])
print(x)
```

## Data Loading

PyTorch provides efficient data loading utilities.

```python
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Model Training

Here's how to train a simple model:

```python
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters())
```

This completes our quickstart tutorial.
        ''',
        'source_vendor': 'pytorch',
        'doc_type': 'tutorial',
        'topics': ['pytorch', 'tutorial', 'beginner'],
        'priority': 'high',
        'quality_score': 0.8,
        'heading_structure': ['PyTorch Quickstart Tutorial', 'Tensors', 'Data Loading', 'Model Training']
    }
    
    chunker = ProductionChunker()
    chunks = chunker.chunk_document(sample_doc)
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Quality: {chunk.quality_score:.2f}")
        print(f"  Has Code: {chunk.has_code}")
        print(f"  Technical Depth: {chunk.technical_depth}")
        print(f"  Content: {chunk.content[:100]}...")

if __name__ == "__main__":
    main()
