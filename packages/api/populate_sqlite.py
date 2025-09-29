#!/usr/bin/env python3
"""Populate SQLite database with curated content for testing."""

import sqlite3
import logging
from pathlib import Path
import hashlib
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_chunks_from_markdown(content: str, title: str, source_url: str) -> list:
    """Split markdown content into chunks."""
    # Remove markdown headers and split by paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    chunks = []
    
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 50:  # Skip very short paragraphs
            continue
            
        # Create chunk ID
        chunk_id = hashlib.md5(f"{source_url}_{i}".encode()).hexdigest()
        
        # Clean up the content
        clean_content = re.sub(r'#+\s*', '', paragraph)  # Remove markdown headers
        clean_content = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_content)  # Remove bold
        clean_content = re.sub(r'\*(.*?)\*', r'\1', clean_content)  # Remove italic
        clean_content = clean_content.strip()
        
        if len(clean_content) < 50:
            continue
            
        chunks.append({
            'chunk_id': chunk_id,
            'content': clean_content,
            'title': title,
            'heading_path': f"{title} > Section {i+1}",
            'source_url': source_url,
            'anchor_link': f"#section-{i+1}",
            'token_count': len(clean_content.split()),
            'prev_id': chunks[-1]['chunk_id'] if chunks else None,
            'next_id': None
        })
        
        # Set next_id for previous chunk
        if len(chunks) > 1:
            chunks[-2]['next_id'] = chunk_id
    
    return chunks

def populate_sqlite():
    """Populate SQLite database with curated content."""
    
    # Database path
    db_path = Path('./data/bm25.db')
    db_path.parent.mkdir(exist_ok=True)
    
    # Connect to SQLite
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
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
    """)
    
    cursor.execute("""
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
    """)
    
    # Clear existing data
    cursor.execute("DELETE FROM documents_fts")
    cursor.execute("DELETE FROM chunk_metadata")
    
    # Load curated content
    curated_dir = Path('./data/curated')
    
    if not curated_dir.exists():
        logger.error("Curated directory not found")
        return
    
    total_chunks = 0
    
    for md_file in curated_dir.glob('*.md'):
        logger.info(f"Processing {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        title = md_file.stem.replace('_', ' ').title()
        source_url = f"file://{md_file}"
        
        # Create chunks
        chunks = create_chunks_from_markdown(content, title, source_url)
        
        logger.info(f"Created {len(chunks)} chunks from {md_file.name}")
        
        # Insert chunks
        for chunk in chunks:
            # Insert into FTS table
            cursor.execute("""
                INSERT INTO documents_fts 
                (chunk_id, content, title, heading_path, source_url, anchor_link)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                chunk['chunk_id'],
                chunk['content'],
                chunk['title'],
                chunk['heading_path'],
                chunk['source_url'],
                chunk['anchor_link']
            ))
            
            # Insert into metadata table
            cursor.execute("""
                INSERT INTO chunk_metadata 
                (chunk_id, source_url, title, heading_path, anchor_link, 
                 token_count, prev_id, next_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk['chunk_id'],
                chunk['source_url'],
                chunk['title'],
                chunk['heading_path'],
                chunk['anchor_link'],
                chunk['token_count'],
                chunk['prev_id'],
                chunk['next_id']
            ))
        
        total_chunks += len(chunks)
    
    conn.commit()
    
    # Get final counts
    cursor.execute("SELECT COUNT(*) FROM documents_fts")
    fts_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
    metadata_count = cursor.fetchone()[0]
    
    logger.info(f"✅ SQLite database populated successfully!")
    logger.info(f"📊 FTS documents: {fts_count}")
    logger.info(f"📊 Metadata records: {metadata_count}")
    logger.info(f"📊 Total chunks: {total_chunks}")
    
    conn.close()

if __name__ == '__main__':
    populate_sqlite()
