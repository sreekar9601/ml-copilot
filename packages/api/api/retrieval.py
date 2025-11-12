"""Hybrid retrieval system combining vector search and keyword search with RRF."""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter as QdrantFilter  # noqa: F401 (placeholder)

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
            # Prefer Qdrant Cloud if configured; else fall back to Chroma local
            self.qdrant_client = None
            self.use_qdrant = bool(settings.qdrant_url and settings.qdrant_api_key)

            if self.use_qdrant:
                try:
                    self.qdrant_client = QdrantClient(
                        url=settings.qdrant_url,
                        api_key=settings.qdrant_api_key,
                    )
                    logger.info("Connected to Qdrant Cloud")
                except Exception as e:
                    logger.error(f"Failed to connect to Qdrant: {e}")
                    self.use_qdrant = False

            if not self.use_qdrant:
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
                # Create empty SQLite database with proper schema for API service
                try:
                    self.sqlite_conn = sqlite3.connect(str(sqlite_path))
                    self.sqlite_conn.row_factory = sqlite3.Row
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
                    logger.info(f"Created empty SQLite database: {sqlite_path}")
                except Exception as e:
                    logger.error(f"Failed to create SQLite database: {e}")
                    self.sqlite_conn = None
            
            # Initialize embedder lazily
            self.embedder = None
            
        except Exception as e:
            logger.error(f"Error initializing retrieval connections: {e}")
            raise
    
    def _get_embedder(self):
        """Lazy initialization of embedder."""
        if self.embedder is None:
            try:
                # Try lightweight embedder first (for API service)
                from .lightweight_embedder import LightweightEmbedder
                self.embedder = LightweightEmbedder("text-embedding-004")
                logger.info("Using lightweight Google Cloud Vertex AI embedder")
            except ImportError:
                # Fallback to Vertex AI embedder (matches ingestion embeddings)
                from ingest.vertex_embedder import CachedVertexEmbedder
                self.embedder = CachedVertexEmbedder()
                logger.info("Using Vertex AI embedder (fallback)")
        return self.embedder
    
    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform vector similarity search using Qdrant (if configured) or ChromaDB."""
        try:
            # Generate query embedding
            embedder = self._get_embedder()
            query_embedding = embedder.encode_query(query)

            if self.use_qdrant and self.qdrant_client is not None:
                # Qdrant search
                hits = self.qdrant_client.search(
                    collection_name=settings.qdrant_collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
                retrieval_results: List[RetrievalResult] = []
                for idx, hit in enumerate(hits):
                    payload = hit.payload or {}
                    # Get metadata - try both nested and flat structures
                    metadata = payload.get("metadata", {})
                    
                    # If metadata is empty, payload itself might contain the metadata fields
                    if not metadata:
                        metadata = {
                            'source_url': payload.get('source_url', ''),
                            'vendor': payload.get('vendor', ''),
                            'heading_path': payload.get('heading_path', ''),
                            'title': payload.get('title', '')
                        }
                    
                    content = payload.get("text", payload.get("content", ""))
                    chunk_id = str(hit.id)
                    retrieval_results.append(
                        RetrievalResult(
                            chunk_id=chunk_id,
                            content=content,
                            metadata=metadata,
                            score=float(hit.score),
                            rank=idx + 1,
                        )
                    )
                return retrieval_results

            # Fallback: ChromaDB local
            if not self.collection:
                logger.warning("ChromaDB collection not available for vector search")
                return []

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            retrieval_results = []
            if results['ids'] and results['ids'][0]:
                for i, (chunk_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity = 1 / (1 + distance)
                    retrieval_results.append(RetrievalResult(
                        chunk_id=chunk_id,
                        content=document,
                        metadata=metadata,
                        score=similarity,
                        rank=i + 1
                    ))
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
            
            # First, check what columns exist in the FTS table
            cursor.execute("PRAGMA table_info(documents_fts)")
            columns_info = cursor.fetchall()
            available_columns = {col['name'] for col in columns_info}
            
            # Build SELECT clause based on available columns
            select_columns = ['chunk_id', 'content']
            optional_columns = ['title', 'heading_path', 'source_url', 'anchor_link']
            for col in optional_columns:
                if col in available_columns:
                    select_columns.append(col)
            
            select_clause = ', '.join(select_columns)
            
            # Search with ranking
            cursor.execute(
                f"""
                SELECT 
                    {select_clause},
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
                # Build metadata from available columns
                metadata = {}
                for col in optional_columns:
                    if col in available_columns:
                        metadata[col] = row[col]
                
                # Get vendor from metadata if we have source_url
                if 'source_url' in metadata:
                    source_url = metadata['source_url']
                    # Extract vendor from URL
                    for vendor in ['pytorch', 'tensorflow', 'sklearn', 'mlflow', 'ray', 'wandb']:
                        if vendor in source_url.lower():
                            metadata['vendor'] = vendor
                            break
                
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
            
            # If SQLite results don't have source_url, enrich from Qdrant
            if retrieval_results and not retrieval_results[0].metadata.get('source_url'):
                retrieval_results = self._enrich_metadata_from_qdrant(retrieval_results)
            
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _enrich_metadata_from_qdrant(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Enrich results with metadata from Qdrant when SQLite doesn't have it."""
        if not self.qdrant_client:
            return results
        
        try:
            # Batch fetch points from Qdrant
            # Convert chunk_ids back to integers (they were stored as ints in Qdrant)
            chunk_ids = []
            for r in results:
                try:
                    # Try to convert to int (if it's a string representation of an int)
                    chunk_ids.append(int(r.chunk_id))
                except (ValueError, TypeError):
                    # If it's already an int or a UUID string, use as-is
                    chunk_ids.append(r.chunk_id)
            
            points = self.qdrant_client.retrieve(
                collection_name=settings.qdrant_collection_name,
                ids=chunk_ids,
                with_payload=True,
                with_vectors=False
            )
            
            # Create a mapping of chunk_id to metadata (convert to string for matching)
            metadata_map = {
                str(point.id): point.payload
                for point in points
            }
            
            # Enrich results
            enriched_results = []
            for result in results:
                chunk_id_str = str(result.chunk_id)
                if chunk_id_str in metadata_map:
                    payload = metadata_map[chunk_id_str]
                    # Update metadata with Qdrant payload
                    result.metadata.update({
                        'source_url': payload.get('source_url', ''),
                        'vendor': payload.get('vendor', 'unknown'),
                        'heading_path': payload.get('heading_path', ''),
                        'title': payload.get('title', '')
                    })
                enriched_results.append(result)
            
            return enriched_results
        except Exception as e:
            logger.warning(f"Failed to enrich metadata from Qdrant: {e}")
            return results
    
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query for FTS5 search."""
        # Remove special FTS5 characters and split into terms
        import re
        
        # Normalize and remove special characters that break MATCH
        cleaned = query.lower()
        # Use simple regex that works in all Python versions
        cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
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

        # Keep raw RRF scores - don't normalize artificially
        # The raw RRF scores are more meaningful than normalized percentages
        # Only ensure scores are positive and reasonable
        for result in fused_results:
            # Ensure score is positive and not too high
            result.score = max(0.0, min(result.score, 1.0))
        
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
                
                # Reset expansion index for each primary result
                expansion_distance = 0
                
                # Fetch previous chunks
                current_prev = prev_id
                for _ in range(max_expansions):
                    if not current_prev or current_prev in processed_ids:
                        break
                    
                    prev_chunk = self._fetch_chunk_by_id(current_prev)
                    if prev_chunk:
                        # Give context chunks a much lower score than primary chunks
                        # Score decreases with distance from primary chunk
                        expansion_distance += 1
                        # Context chunks get only 5-15% of the primary chunk's score
                        prev_chunk.score = result.score * 0.1 / expansion_distance
                        
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
                        # Give context chunks a much lower score than primary chunks
                        expansion_distance += 1
                        # Context chunks get only 5-15% of the primary chunk's score
                        next_chunk.score = result.score * 0.1 / expansion_distance
                        
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
                    score=0.01,  # Low initial score, will be updated by expand_context
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

