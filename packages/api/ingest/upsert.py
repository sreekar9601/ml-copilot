"""Database upsert module for storing chunks in ChromaDB and SQLite FTS5."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import numpy as np

from .chunker import DocumentChunk
from .embedder import CachedEmbedder

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages vector DB (Qdrant or ChromaDB) and SQLite for the RAG system."""
    
    def __init__(self, data_dir: Path, collection_name: str = "ml_docs", 
                 sqlite_db: str = "bm25.db"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.sqlite_path = self.data_dir / sqlite_db
        self.chroma_path = self.data_dir / "chroma"
        
        # Initialize databases
        self.chroma_client = None
        self.collection = None
        self.sqlite_conn = None
        
        self.embedder = CachedEmbedder()
    
    def initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            logger.info(f"Initializing ChromaDB at {self.chroma_path}")
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ML documentation chunks"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise

    def initialize_qdrant(self, url: str, api_key: str, collection_name: str, vector_size: int) -> QdrantClient:
        """Initialize Qdrant Cloud client and ensure collection exists."""
        client = QdrantClient(url=url, api_key=api_key)
        try:
            client.get_collection(collection_name)
        except Exception:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )
        return client
    
    def initialize_sqlite(self) -> None:
        """Initialize SQLite database with FTS5 table."""
        try:
            logger.info(f"Initializing SQLite at {self.sqlite_path}")
            
            self.sqlite_conn = sqlite3.connect(self.sqlite_path)
            cursor = self.sqlite_conn.cursor()
            
            # Create FTS5 table for keyword search
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
            
            # Create metadata table for chunk relationships
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
            
            self.sqlite_conn.commit()
            logger.info("SQLite tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite: {e}")
            raise
    
    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> None:
        """Upsert chunks into both ChromaDB and SQLite."""
        if not chunks:
            logger.warning("No chunks to upsert")
            return
        
        logger.info(f"Upserting {len(chunks)} chunks...")
        
        # Initialize databases if not already done
        if self.collection is None:
            self.initialize_chromadb()
        if self.sqlite_conn is None:
            self.initialize_sqlite()
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._upsert_batch(batch)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        logger.info("All chunks upserted successfully")
    
    def _upsert_batch(self, chunks: List[DocumentChunk]) -> None:
        """Upsert a batch of chunks."""
        # Prepare data common to vector DB
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        contents = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            metadata = {
                "source_url": chunk.source_url,
                "title": chunk.title,
                "heading_path": chunk.heading_path,
                "anchor_link": chunk.anchor_link,
                "token_count": chunk.token_count,
                "prev_id": chunk.prev_id or "",
                "next_id": chunk.next_id or ""
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        logger.debug(f"Generating embeddings for batch of {len(chunks)} chunks")
        embeddings = self.embedder.encode_batch(contents)
        
        # Upsert to vector DB (prefer Qdrant if configured via env)
        use_qdrant = bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"))
        if use_qdrant:
            try:
                client = self.initialize_qdrant(
                    url=os.getenv("QDRANT_URL"),
                    api_key=os.getenv("QDRANT_API_KEY"),
                    collection_name=os.getenv("QDRANT_COLLECTION", self.collection_name),
                    vector_size=len(embeddings[0]) if len(embeddings) > 0 else 768,
                )
                points = [
                    qmodels.PointStruct(
                        id=chunk_ids[i],
                        vector=embeddings[i].tolist(),
                        payload={
                            "text": contents[i],
                            "metadata": metadatas[i],
                        },
                    )
                    for i in range(len(chunk_ids))
                ]
                client.upsert(collection_name=os.getenv("QDRANT_COLLECTION", self.collection_name), points=points, wait=True)
                logger.debug(f"Upserted {len(chunks)} chunks to Qdrant")
            except Exception as e:
                logger.error(f"Error upserting to Qdrant: {e}")
                raise
        else:
            # Upsert to ChromaDB
            try:
                self.collection.upsert(
                    ids=chunk_ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    documents=contents
                )
                logger.debug(f"Upserted {len(chunks)} chunks to ChromaDB")
            except Exception as e:
                logger.error(f"Error upserting to ChromaDB: {e}")
                raise
        
        # Upsert to SQLite
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Insert into FTS table
            fts_data = [
                (chunk.chunk_id, chunk.content, chunk.title, 
                 chunk.heading_path, chunk.source_url, chunk.anchor_link)
                for chunk in chunks
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO documents_fts 
                (chunk_id, content, title, heading_path, source_url, anchor_link)
                VALUES (?, ?, ?, ?, ?, ?)
            """, fts_data)
            
            # Insert into metadata table
            metadata_data = [
                (chunk.chunk_id, chunk.source_url, chunk.title, 
                 chunk.heading_path, chunk.anchor_link, chunk.token_count,
                 chunk.prev_id, chunk.next_id)
                for chunk in chunks
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO chunk_metadata 
                (chunk_id, source_url, title, heading_path, anchor_link, 
                 token_count, prev_id, next_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, metadata_data)
            
            self.sqlite_conn.commit()
            logger.debug(f"Upserted {len(chunks)} chunks to SQLite")
            
        except Exception as e:
            logger.error(f"Error upserting to SQLite: {e}")
            self.sqlite_conn.rollback()
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collections."""
        stats = {}
        
        try:
            if self.collection:
                chroma_count = self.collection.count()
                stats["chromadb_count"] = chroma_count
            
            if self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM documents_fts")
                fts_count = cursor.fetchone()[0]
                stats["sqlite_fts_count"] = fts_count
                
                cursor.execute("SELECT COUNT(*) FROM chunk_metadata")
                metadata_count = cursor.fetchone()[0]
                stats["sqlite_metadata_count"] = metadata_count
                
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def clear_collections(self) -> None:
        """Clear all data from both databases."""
        logger.warning("Clearing all data from databases")
        
        try:
            if self.collection:
                # Delete and recreate ChromaDB collection
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "ML documentation chunks"}
                )
                logger.info("ChromaDB collection cleared")
            
            if self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute("DELETE FROM documents_fts")
                cursor.execute("DELETE FROM chunk_metadata")
                self.sqlite_conn.commit()
                logger.info("SQLite tables cleared")
                
        except Exception as e:
            logger.error(f"Error clearing collections: {e}")
            raise
    
    def close(self) -> None:
        """Close database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None
        
        # ChromaDB client doesn't need explicit closing
        self.chroma_client = None
        self.collection = None


def upsert_document_chunks(chunks: List[DocumentChunk], 
                          data_dir: Path,
                          collection_name: str = "ml_docs",
                          sqlite_db: str = "bm25.db") -> Dict[str, Any]:
    """High-level function to upsert chunks into the databases."""
    db_manager = DatabaseManager(data_dir, collection_name, sqlite_db)
    
    try:
        db_manager.upsert_chunks(chunks)
        stats = db_manager.get_collection_stats()
        return stats
    finally:
        db_manager.close()


if __name__ == "__main__":
    # Test the database manager
    logging.basicConfig(level=logging.INFO)
    
    from pathlib import Path
    
    # Create test chunks
    test_chunks = [
        DocumentChunk(
            chunk_id="test-1",
            content="This is a test chunk about PyTorch DataLoader.",
            source_url="https://pytorch.org/docs/stable/data.html",
            title="Data Loading in PyTorch",
            heading_path="Introduction > DataLoader",
            anchor_link="https://pytorch.org/docs/stable/data.html#dataloader",
            token_count=50
        ),
        DocumentChunk(
            chunk_id="test-2",
            content="This is another test chunk about MLflow tracking.",
            source_url="https://mlflow.org/docs/latest/tracking.html",
            title="MLflow Tracking",
            heading_path="Concepts > Tracking",
            anchor_link="https://mlflow.org/docs/latest/tracking.html#concepts",
            token_count=45
        )
    ]
    
    # Link chunks
    test_chunks[0].next_id = test_chunks[1].chunk_id
    test_chunks[1].prev_id = test_chunks[0].chunk_id
    
    # Test upsert
    data_dir = Path("./test_data")
    stats = upsert_document_chunks(test_chunks, data_dir)
    print(f"Upsert completed. Stats: {stats}")
    
    # Cleanup
    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir)

