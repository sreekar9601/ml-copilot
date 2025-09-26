"""Lightweight embedder using Google Generative AI for the API service."""

import logging
import numpy as np
from typing import List, Union
from . import clients  # This ensures clients.py is imported and configured
from .config import settings

logger = logging.getLogger(__name__)


class LightweightEmbedder:
    """Lightweight embedder using Google Generative AI embeddings."""
    
    def __init__(self, model_name: str = clients.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        # DO NOT initialize a new client here.
        # The genai module is already configured in clients.py
        logger.info(f"Initialized LightweightEmbedder with Google AI model: {model_name}")
    
    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query text into embedding vector."""
        try:
            embeddings = clients.embed_content(texts=[text], task_type="RETRIEVAL_QUERY")
            return np.array(embeddings[0])
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            # Return a zero vector as fallback (gemini-embedding-001 has 3072 dims)
            return np.zeros(3072)
    
    def encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple document texts into embedding vectors."""
        try:
            embeddings = clients.embed_content(texts=texts, task_type="RETRIEVAL_DOCUMENT")
            return [np.array(e) for e in embeddings]
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            # Return zero vectors as fallback (gemini-embedding-001 has 3072 dims)
            return [np.zeros(3072) for _ in texts]
    
    def encode(self, text: str) -> np.ndarray:
        """Generic encode method for backward compatibility."""
        return self.encode_query(text)
