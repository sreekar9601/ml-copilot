#!/usr/bin/env python3
"""Create SQLite database with curated content for keyword search."""

import sqlite3
import hashlib
import re
from pathlib import Path

def create_database():
    """Create SQLite database with curated content."""
    
    # Create database
    conn = sqlite3.connect('/app/data/bm25.db')
    cursor = conn.cursor()

    # Create FTS5 table
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
        USING fts5(
            chunk_id UNINDEXED,
            content,
            title,
            heading_path,
            source_url UNINDEXED,
            anchor_link UNINDEXED,
            tokenize = 'porter'
        )
    ''')

    # Create metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            source_url TEXT,
            title TEXT,
            heading_path TEXT,
            anchor_link TEXT,
            token_count INTEGER,
            prev_id TEXT,
            next_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Curated content about Ray Serve vs KServe and MLOps
    content = '''
# Architecture Decision Records (ADRs)

## ADR-001: Model Serving Platform Selection

**Status:** Accepted
**Date:** 2024-01-15

### Context
We need to choose between Ray Serve and KServe for model serving in our ML platform.

### Decision
Use Ray Serve for high-throughput serving and KServe for Kubernetes-native deployments.

### Rationale
- Ray Serve provides better performance for high-throughput scenarios
- KServe offers better Kubernetes integration and standardization
- Both can coexist in the same architecture for different use cases

### Consequences
- Need to maintain expertise in both platforms
- Different deployment patterns for different use cases
- More complex monitoring and observability setup

## MLOps Best Practices for Production ML Systems

### Architecture Patterns

#### Model Serving Architecture

**When to use Ray Serve:**
- Need for high-throughput, low-latency serving
- Complex model pipelines with preprocessing/postprocessing
- A/B testing and canary deployments
- Integration with existing Ray ecosystem

**When to use KServe:**
- Kubernetes-native deployment requirements
- Need for standardized inference protocols (Seldon, TensorFlow Serving)
- Multi-cloud or hybrid cloud deployments
- Compliance and governance requirements

#### Data Pipeline Architecture

**PyTorch DataLoader Best Practices:**
- Use `num_workers > 0` for I/O bound workloads
- Pin memory with `pin_memory=True` for GPU training
- Use `persistent_workers=True` to avoid worker recreation overhead
- Set appropriate `batch_size` based on available memory
- Use `DataLoader2` for advanced features like automatic optimization

**MLflow Model Registry:**
- Version all models with semantic versioning
- Use tags for model categorization and metadata
- Implement model validation before promotion
- Set up automated model deployment pipelines
- Monitor model performance in production

**Ray Serve Deployment:**
- Use async deployment for better throughput
- Implement health checks and readiness probes
- Set up proper resource limits and scaling policies
- Use Ray's built-in monitoring and observability
- Implement graceful shutdown handling

**KServe InferenceService:**
- Use standard Kubernetes deployment patterns
- Implement proper resource requests and limits
- Set up horizontal pod autoscaling (HPA)
- Use ConfigMaps and Secrets for configuration
- Implement proper logging and monitoring
'''

    # Split content into chunks
    paragraphs = re.split(r'\n\s*\n', content)
    chunks = []

    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 50:
            continue
            
        chunk_id = hashlib.md5(f'curated_{i}'.encode()).hexdigest()
        clean_content = re.sub(r'#+\s*', '', paragraph)
        clean_content = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_content)
        clean_content = re.sub(r'\*(.*?)\*', r'\1', clean_content)
        clean_content = clean_content.strip()
        
        if len(clean_content) < 50:
            continue
            
        chunks.append({
            'chunk_id': chunk_id,
            'content': clean_content,
            'title': 'MLOps Best Practices',
            'heading_path': f'MLOps > Section {i+1}',
            'source_url': 'file://curated_content.md',
            'anchor_link': f'#section-{i+1}',
            'token_count': len(clean_content.split()),
            'prev_id': chunks[-1]['chunk_id'] if chunks else None,
            'next_id': None
        })
        
        if len(chunks) > 1:
            chunks[-2]['next_id'] = chunk_id

    # Insert chunks
    for chunk in chunks:
        cursor.execute('''
            INSERT INTO documents_fts 
            (chunk_id, content, title, heading_path, source_url, anchor_link)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            chunk['chunk_id'],
            chunk['content'],
            chunk['title'],
            chunk['heading_path'],
            chunk['source_url'],
            chunk['anchor_link']
        ))
        
        cursor.execute('''
            INSERT INTO chunk_metadata 
            (chunk_id, source_url, title, heading_path, anchor_link, 
             token_count, prev_id, next_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chunk['chunk_id'],
            chunk['source_url'],
            chunk['title'],
            chunk['heading_path'],
            chunk['anchor_link'],
            chunk['token_count'],
            chunk['prev_id'],
            chunk['next_id']
        ))

    conn.commit()
    conn.close()
    print(f'Created SQLite database with {len(chunks)} chunks')

if __name__ == '__main__':
    create_database()
