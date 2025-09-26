"""Lightweight embedder using Google Generative AI for the API service."""

import logging
import numpy as np
from typing import List, Union
import google.generativeai as genai
from .config import settings

logger = logging.getLogger(__name__)


class LightweightEmbedder:
    """Lightweight embedder using Google Generative AI embeddings."""
    
    def __init__(self, model_name: str = "gemini-embedding-001"):
        self.model_name = model_name
        # Configure API key
        genai.configure(api_key=settings.google_api_key)
        logger.info(f"Initialized LightweightEmbedder with Google AI model: {model_name}")
    
    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query text into embedding vector."""
        try:
            # Use Google's embedding model
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'])
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            # Return a zero vector as fallback
            return np.zeros(768)
    
    def encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple document texts into embedding vectors."""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(np.array(result['embedding']))
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            # Return zero vectors as fallback
            return [np.zeros(768) for _ in texts]
    
    def encode(self, text: str) -> np.ndarray:
        """Generic encode method for backward compatibility."""
        return self.encode_query(text)
