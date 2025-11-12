"""Vertex AI embedder for ingestion - matches the query embedder for consistency."""

import logging
import numpy as np
from typing import List
import sys
import os

# Add parent directory to path to import from api package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import clients

logger = logging.getLogger(__name__)


class VertexAIEmbedder:
    """
    Vertex AI embedder using Google's text-embedding-004 model.
    This matches the LightweightEmbedder used for queries, ensuring
    embeddings are in the same vector space.
    """
    
    def __init__(self, model_name: str = clients.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Initialized VertexAIEmbedder with model: {model_name}")
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Encode a batch of texts into embeddings.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once (for compatibility, 
                       actual batching is handled by the API client)
        
        Returns:
            numpy array of embeddings with shape (len(texts), 768)
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Encoding batch of {len(texts)} texts with Vertex AI")
            
            # Use RETRIEVAL_DOCUMENT task type for ingestion
            embeddings = clients.embed_content(
                texts=texts, 
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            logger.debug(f"Generated embeddings with shape: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), 768))
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding."""
        return self.encode_batch([text])[0]
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query text with RETRIEVAL_QUERY task type.
        Note: For ingestion, you should use encode_batch or encode_document.
        """
        try:
            embeddings = clients.embed_content(
                texts=[query], 
                task_type="RETRIEVAL_QUERY"
            )
            return np.array(embeddings[0])
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            return np.zeros(768)
    
    def encode_document(self, text: str) -> np.ndarray:
        """Encode a document text with RETRIEVAL_DOCUMENT task type."""
        return self.encode_single(text)


class CachedVertexEmbedder:
    """Vertex AI embedder with caching capabilities."""
    
    def __init__(self, model_name: str = clients.EMBEDDING_MODEL_NAME, cache_size: int = 1000):
        self.embedder = VertexAIEmbedder(model_name)
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []
        logger.info(f"Initialized CachedVertexEmbedder with cache size {cache_size}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hash(text)
    
    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode batch with caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append((i, self.cache[cache_key]))
                # Update access order
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)
                self.access_order.append(cache_key)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            new_embeddings = self.embedder.encode_batch(uncached_texts, batch_size)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                self.access_order.append(cache_key)
                
                # Evict oldest if cache is full
                if len(self.cache) > self.cache_size:
                    oldest_key = self.access_order.pop(0)
                    del self.cache[oldest_key]
            
            # Add to results
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))
        
        # Sort by original order and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings])
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text with caching."""
        return self.encode_batch([text])[0]
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with caching."""
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        embedding = self.embedder.encode_query(query)
        self.cache[cache_key] = embedding
        self.access_order.append(cache_key)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        return embedding


if __name__ == "__main__":
    # Test the embedder
    logging.basicConfig(level=logging.INFO)
    
    embedder = VertexAIEmbedder()
    
    # Test single embedding
    test_text = "PyTorch is a popular machine learning framework"
    embedding = embedder.encode_single(test_text)
    print(f"Single embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch embedding
    test_texts = [
        "PyTorch DataLoader handles batching",
        "MLflow tracks experiments",
        "TensorFlow is another ML framework"
    ]
    embeddings = embedder.encode_batch(test_texts)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    print(f"Successfully embedded {len(test_texts)} texts")


