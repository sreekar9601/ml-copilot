"""Hybrid retrieval system combining vector search and keyword search with RRF."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings

from .config import settings

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Represents a single retrieval result with metadata."""
    
    def __init__(self, chunk_id: str, content: str, metadata: Dict[str, Any], 
                 score: float, rank: int):
        self.chunk_id = chunk_id
        self.content = content
        self.metadata = metadata
        self.score = score
        self.rank = rank
    
    def __repr__(self):
        return f"RetrievalResult(id={self.chunk_id}, score={self.score:.3f}, rank={self.rank})"


class HybridRetriever:
    """Hybrid retrieval system using ChromaDB for vector search and SQLite for keyword search."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.data_dir
        self.chroma_client = None
        self.collection = None
        self.sqlite_conn = None
        self.embedder = None
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self) -> None:
        """Initialize database connections."""
        try:
            # Initialize ChromaDB
            chroma_path = self.data_dir / "chroma"
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection(settings.chroma_collection)
                logger.info(f"Connected to ChromaDB collection: {settings.chroma_collection}")
            except Exception:
                logger.warning(f"ChromaDB collection {settings.chroma_collection} not found")
                self.collection = None
            
            # Initialize SQLite
            sqlite_path = self.data_dir / settings.sqlite_db
            if sqlite_path.exists():
                self.sqlite_conn = sqlite3.connect(str(sqlite_path))
                self.sqlite_conn.row_factory = sqlite3.Row  # Enable dict-like access
                logger.info(f"Connected to SQLite database: {sqlite_path}")
            else:
                logger.warning(f"SQLite database not found: {sqlite_path}")
                self.sqlite_conn = None
            
            # Initialize embedder lazily
            self.embedder = None
            
        except Exception as e:
            logger.error(f"Error initializing retrieval connections: {e}")
            raise
    
    def _get_embedder(self):
        """Lazy initialization of embedder."""
        if self.embedder is None:
            from ingest.embedder import CachedEmbedder
            self.embedder = CachedEmbedder(settings.embedding_model)
        return self.embedder
    
    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform vector similarity search using ChromaDB."""
        if not self.collection:
            logger.warning("ChromaDB collection not available for vector search")
            return []
        
        try:
            # Generate query embedding
            embedder = self._get_embedder()
            query_embedding = embedder.encode_query(query)
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            if results['ids'] and results['ids'][0]:
                for i, (chunk_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity = 1 / (1 + distance)
                    
                    retrieval_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        content=document,
                        metadata=metadata,
                        score=similarity,
                        rank=i + 1
                    ))
            
            logger.debug(f"Vector search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform keyword search using SQLite FTS5."""
        if not self.sqlite_conn:
            logger.warning("SQLite connection not available for keyword search")
            return []
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Prepare FTS5 query - escape special characters and add wildcards
            fts_query = self._prepare_fts_query(query)
            
            # Search with ranking (bm25 available only in ORDER BY/select). Avoid window functions.
            cursor.execute(
                """
                SELECT 
                    chunk_id,
                    content,
                    title,
                    heading_path,
                    source_url,
                    anchor_link,
                    bm25(documents_fts) AS score
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            
            results = cursor.fetchall()
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for idx, row in enumerate(results):
                metadata = {
                    'title': row['title'],
                    'heading_path': row['heading_path'],
                    'source_url': row['source_url'],
                    'anchor_link': row['anchor_link']
                }
                
                # BM25 scores are negative (lower is better), convert to positive similarity
                score = abs(row['score']) if row['score'] else 0
                
                retrieval_results.append(RetrievalResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    metadata=metadata,
                    score=score,
                    rank=idx + 1
                ))
            
            logger.debug(f"Keyword search returned {len(retrieval_results)} results")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query for FTS5 search."""
        # Remove special FTS5 characters and split into terms
        import re
        
        # Normalize and remove special characters that break MATCH
        cleaned = query.lower()
        cleaned = re.sub(r'[\p{P}\p{S}]', ' ', cleaned) if hasattr(re, 'UNICODE') else re.sub(r'[^\w\s-]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Split into terms and add prefix matching for meaningful tokens
        terms = [t for t in cleaned.split(' ') if t]
        
        if not terms:
            return '""'  # Return empty query for FTS5
        
        # Require all terms (AND). Use prefix for length>=3 to broaden recall.
        fts_terms = []
        for term in terms:
            safe = re.sub(r'[^a-z0-9_-]', '', term)
            if not safe:
                continue
            if len(safe) >= 3:
                fts_terms.append(f'{safe}*')
            else:
                fts_terms.append(safe)
        
        return ' AND '.join(fts_terms) if fts_terms else '""'
    
    def reciprocal_rank_fusion(self, 
                              vector_results: List[RetrievalResult],
                              keyword_results: List[RetrievalResult],
                              k: int = 60) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        
        # Create lookup for results by chunk_id
        all_results = {}
        
        # Add vector results
        for result in vector_results:
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
            all_results[result.chunk_id].vector_rank = result.rank
            all_results[result.chunk_id].vector_score = result.score
        
        # Add keyword results
        for result in keyword_results:
            if result.chunk_id not in all_results:
                all_results[result.chunk_id] = result
            all_results[result.chunk_id].keyword_rank = getattr(result, 'rank', float('inf'))
            all_results[result.chunk_id].keyword_score = result.score
        
        # Calculate RRF scores
        for chunk_id, result in all_results.items():
            vector_rank = getattr(result, 'vector_rank', float('inf'))
            keyword_rank = getattr(result, 'keyword_rank', float('inf'))
            
            # RRF formula: score = 1/(k + rank_vector) + 1/(k + rank_keyword)
            rrf_score = 0
            if vector_rank != float('inf'):
                rrf_score += 1 / (k + vector_rank)
            if keyword_rank != float('inf'):
                rrf_score += 1 / (k + keyword_rank)
            
            result.score = rrf_score
        
        # Sort by RRF score (descending)
        fused_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        logger.debug(f"RRF fusion combined {len(vector_results)} vector + {len(keyword_results)} keyword = {len(fused_results)} total results")
        
        return fused_results
    
    def expand_context(self, results: List[RetrievalResult], 
                      max_expansions: int = 2) -> List[RetrievalResult]:
        """Expand context by fetching neighboring chunks."""
        if not self.sqlite_conn:
            return results
        
        expanded_results = []
        processed_ids = set()
        
        for result in results:
            # Add the original result
            if result.chunk_id not in processed_ids:
                expanded_results.append(result)
                processed_ids.add(result.chunk_id)
            
            # Get neighboring chunks
            try:
                cursor = self.sqlite_conn.cursor()
                
                # Get chunk metadata
                cursor.execute("""
                    SELECT prev_id, next_id FROM chunk_metadata 
                    WHERE chunk_id = ?
                """, (result.chunk_id,))
                
                row = cursor.fetchone()
                if not row:
                    continue
                
                prev_id, next_id = row['prev_id'], row['next_id']
                
                # Fetch previous chunks
                current_prev = prev_id
                for _ in range(max_expansions):
                    if not current_prev or current_prev in processed_ids:
                        break
                    
                    prev_chunk = self._fetch_chunk_by_id(current_prev)
                    if prev_chunk:
                        # Insert before current result in expanded_results
                        insert_idx = next(i for i, r in enumerate(expanded_results) 
                                        if r.chunk_id == result.chunk_id)
                        expanded_results.insert(insert_idx, prev_chunk)
                        processed_ids.add(current_prev)
                        
                        # Get next previous chunk
                        cursor.execute("""
                            SELECT prev_id FROM chunk_metadata WHERE chunk_id = ?
                        """, (current_prev,))
                        prev_row = cursor.fetchone()
                        current_prev = prev_row['prev_id'] if prev_row else None
                
                # Fetch next chunks
                current_next = next_id
                for _ in range(max_expansions):
                    if not current_next or current_next in processed_ids:
                        break
                    
                    next_chunk = self._fetch_chunk_by_id(current_next)
                    if next_chunk:
                        expanded_results.append(next_chunk)
                        processed_ids.add(current_next)
                        
                        # Get next next chunk
                        cursor.execute("""
                            SELECT next_id FROM chunk_metadata WHERE chunk_id = ?
                        """, (current_next,))
                        next_row = cursor.fetchone()
                        current_next = next_row['next_id'] if next_row else None
                        
            except Exception as e:
                logger.error(f"Error expanding context for {result.chunk_id}: {e}")
                continue
        
        logger.debug(f"Context expansion: {len(results)} -> {len(expanded_results)} chunks")
        return expanded_results
    
    def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[RetrievalResult]:
        """Fetch a chunk by its ID from SQLite."""
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT chunk_id, content, title, heading_path, source_url, anchor_link
                FROM documents_fts
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cursor.fetchone()
            if row:
                metadata = {
                    'title': row['title'],
                    'heading_path': row['heading_path'],
                    'source_url': row['source_url'],
                    'anchor_link': row['anchor_link']
                }
                
                return RetrievalResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    metadata=metadata,
                    score=0.0,  # Context expansion doesn't have relevance score
                    rank=0
                )
            
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_id}: {e}")
        
        return None
    
    def retrieve(self, query: str, top_k: int = 5, 
                expand_context: bool = True) -> List[RetrievalResult]:
        """Main retrieval method combining vector and keyword search with RRF."""
        
        logger.info(f"Retrieving documents for query: {query}")
        
        # Perform parallel searches
        vector_results = self.vector_search(query, settings.top_k_vector)
        keyword_results = self.keyword_search(query, settings.top_k_keyword)
        
        # Fuse results using RRF
        fused_results = self.reciprocal_rank_fusion(
            vector_results, keyword_results, settings.rrf_k
        )
        
        # Optional: domain-aware boosting based on query hints
        lowered = query.lower()
        boost_py = any(term in lowered for term in ["pytorch", "torch", "dataloader"])
        boost_mlflow = "mlflow" in lowered
        if boost_py or boost_mlflow:
            for r in fused_results:
                src = (r.metadata or {}).get('source_url', '')
                if boost_py and 'pytorch-docs' in src:
                    r.score += 0.02
                if boost_mlflow and 'mlflow-docs' in src:
                    r.score += 0.02
            fused_results = sorted(fused_results, key=lambda x: x.score, reverse=True)

        # Take top-k results after boosting
        top_results = fused_results[:top_k]
        
        # Expand context if requested
        if expand_context:
            final_results = self.expand_context(top_results)
        else:
            final_results = top_results
        
        logger.info(f"Retrieved {len(final_results)} total chunks ({len(top_results)} primary + context)")
        
        return final_results
    
    def close(self) -> None:
        """Close database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None
        
        # ChromaDB doesn't need explicit closing
        self.chroma_client = None
        self.collection = None


# Global retriever instance
_retriever = None


def get_retriever() -> HybridRetriever:
    """Get or create global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve_documents(query: str, top_k: int = 5) -> List[RetrievalResult]:
    """High-level function for document retrieval."""
    retriever = get_retriever()
    return retriever.retrieve(query, top_k)


if __name__ == "__main__":
    # Test the retriever
    logging.basicConfig(level=logging.INFO)
    
    # Test queries
    test_queries = [
        "How to use PyTorch DataLoader?",
        "MLflow model registry",
        "KServe inference service",
        "Ray Serve deployment configuration"
    ]
    
    retriever = HybridRetriever()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result.metadata.get('title', 'Unknown')} (score: {result.score:.3f})")
            print(f"   {result.content[:100]}...")
    
    retriever.close()

