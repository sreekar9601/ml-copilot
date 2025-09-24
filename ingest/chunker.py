"""Text chunking module for splitting documents into semantic chunks."""

import re
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    chunk_id: str
    content: str
    source_url: str
    title: str
    heading_path: str
    anchor_link: str
    token_count: int
    prev_id: Optional[str] = None
    next_id: Optional[str] = None


class SemanticChunker:
    """Splits documents into semantic chunks with context linking."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, doc: Dict[str, str]) -> List[DocumentChunk]:
        """Split a document into semantic chunks."""
        content = doc['content']
        url = doc['url']
        title = doc['title']
        
        # Split into sections based on headings
        sections = self._split_by_headings(content)
        
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, url, title)
            chunks.extend(section_chunks)
        
        # Link chunks with prev/next relationships
        self._link_chunks(chunks)
        
        return chunks
    
    def _split_by_headings(self, content: str) -> List[Dict[str, str]]:
        """Split content into sections based on markdown headings."""
        lines = content.split('\n')
        sections = []
        current_section = {'heading': '', 'content': [], 'level': 0}
        heading_stack = []
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if heading_match:
                # Save previous section if it has content
                if current_section['content']:
                    sections.append({
                        'heading_path': self._build_heading_path(heading_stack),
                        'anchor': self._create_anchor(current_section['heading']),
                        'content': '\n'.join(current_section['content']).strip()
                    })
                
                # Start new section
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()
                
                # Update heading stack
                heading_stack = heading_stack[:level-1] + [heading_text]
                
                current_section = {
                    'heading': heading_text,
                    'content': [line],
                    'level': level
                }
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            sections.append({
                'heading_path': self._build_heading_path(heading_stack),
                'anchor': self._create_anchor(current_section['heading']),
                'content': '\n'.join(current_section['content']).strip()
            })
        
        return sections
    
    def _build_heading_path(self, heading_stack: List[str]) -> str:
        """Build a hierarchical heading path."""
        if not heading_stack:
            return "Introduction"
        return " > ".join(heading_stack)
    
    def _create_anchor(self, heading: str) -> str:
        """Create an anchor link from heading text."""
        if not heading:
            return ""
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', heading.lower())
        anchor = re.sub(r'[\s-]+', '-', anchor).strip('-')
        return f"#{anchor}"
    
    def _chunk_section(self, section: Dict[str, str], url: str, title: str) -> List[DocumentChunk]:
        """Split a section into appropriately sized chunks."""
        content = section['content']
        heading_path = section['heading_path']
        anchor = section['anchor']
        
        if not content.strip():
            return []
        
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        token_count = len(content) // 4
        
        if token_count <= self.chunk_size:
            # Section fits in one chunk
            return [DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=content,
                source_url=url,
                title=title,
                heading_path=heading_path,
                anchor_link=urljoin(url, anchor),
                token_count=token_count
            )]
        
        # Split into multiple chunks
        chunks = []
        sentences = self._split_sentences(content)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_content = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=chunk_content,
                    source_url=url,
                    title=title,
                    heading_path=heading_path,
                    anchor_link=urljoin(url, anchor),
                    token_count=current_tokens
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(s) // 4 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_content,
                source_url=url,
                title=title,
                heading_path=heading_path,
                anchor_link=urljoin(url, anchor),
                token_count=current_tokens
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving markdown structure."""
        # Simple sentence splitting that respects markdown
        sentences = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # If line is a heading, code block, or list item, treat as single unit
            if (line.startswith('#') or 
                line.startswith('```') or 
                line.startswith('-') or 
                line.startswith('*') or 
                line.startswith('+')):
                sentences.append(line)
            else:
                # Split on sentence boundaries
                sentence_splits = re.split(r'(?<=[.!?])\s+', line)
                sentences.extend(s.strip() for s in sentence_splits if s.strip())
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap with next chunk."""
        if not sentences:
            return []
        
        overlap_tokens = 0
        overlap_sentences = []
        
        # Take sentences from the end until we reach overlap limit
        for sentence in reversed(sentences):
            sentence_tokens = len(sentence) // 4
            if overlap_tokens + sentence_tokens > self.overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        
        return overlap_sentences
    
    def _link_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Link chunks with prev/next relationships."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_id = chunks[i + 1].chunk_id


def chunk_documents(documents: List[Dict[str, str]], 
                   chunk_size: int = 500, 
                   overlap: int = 50) -> List[DocumentChunk]:
    """Chunk a list of documents into semantic chunks."""
    chunker = SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    
    all_chunks = []
    for doc in documents:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error chunking document {doc.get('title', 'Unknown')}: {e}")
    
    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


if __name__ == "__main__":
    # Test the chunker
    test_doc = {
        'url': 'https://example.com/test',
        'title': 'Test Document',
        'content': '''# Introduction

This is a test document.

## Section 1

This is the first section with some content. It has multiple sentences. This should be split properly.

### Subsection 1.1

More detailed content here. This subsection has even more content to test chunking.

## Section 2

Another section with different content. This tests the heading hierarchy.'''
    }
    
    chunker = SemanticChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_document(test_doc)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Heading: {chunk.heading_path}")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Content: {chunk.content[:100]}...")
        print(f"  Prev: {chunk.prev_id}")
        print(f"  Next: {chunk.next_id}")
        print("---")

