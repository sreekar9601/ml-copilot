"""Centralized Google AI client configuration for Vertex AI.

This module uses the new google-genai SDK with Vertex AI authentication.
"""

import logging
import os
from google import genai

logger = logging.getLogger(__name__)

logger.info("--- Initializing Full Vertex AI Client via Environment ---")

# Define model names centrally  
# Use current stable model names for Vertex AI
GENERATION_MODEL_NAME = "gemini-2.5-flash"  # Current stable model
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# Create a shared client instance
_client = None

def get_client():
    """Get or create the shared client instance."""
    global _client
    if _client is None:
        _client = genai.Client()
        logger.info("✅ Vertex AI client created")
    return _client

def get_generation_model():
    """Returns the client for generation tasks via Vertex AI."""
    # Since the SDK forces Vertex AI when environment variables are present,
    # use the shared Vertex AI client for consistency
    client = get_client()
    logger.info("✅ Using Vertex AI for generation")
    return client

def embed_content(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", title: str = None) -> list[list[float]]:
    """A wrapper for the embed_content API call."""
    logger.info(f"Embedding {len(texts)} chunks using model {EMBEDDING_MODEL_NAME} for task: {task_type}")
    
    client = get_client()
    
    # Import the types module for configuration
    from google.genai import types
    
    # Use the client's models.embed_content method with output_dimensionality=768
    # This maintains compatibility with existing Qdrant collection
    response = client.models.embed_content(
        model=EMBEDDING_MODEL_NAME,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768  # Maintain 768 dimensions for compatibility
        )
    )
    
    # Extract embeddings from response
    return [embedding.values for embedding in response.embeddings]

logger.info(f"✅ Generation Model ('{GENERATION_MODEL_NAME}') is ready to be used.")
logger.info(f"✅ Embedding Model ('{EMBEDDING_MODEL_NAME}') is ready to be used.")