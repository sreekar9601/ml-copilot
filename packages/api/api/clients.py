"""Centralized Google AI client configuration for Vertex AI.

This module uses the new google-genai SDK with Vertex AI authentication.
"""

import logging
import os
import json
import google.genai as genai
from .config import settings

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
        # Check if we should use Vertex AI
        use_vertexai = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'
        
        if use_vertexai:
            # Use Vertex AI configuration from settings
            project = settings.google_cloud_project
            location = settings.google_cloud_location
            
            logger.info(f"Project: {project}, Location: {location}")
            
            if not project:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
            
            # Set up credentials from environment variable
            credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if credentials_json:
                # Parse the JSON credentials
                credentials = json.loads(credentials_json)
                # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
                credentials_file = os.path.join(os.getcwd(), 'gcp-credentials.json')
                with open(credentials_file, 'w') as f:
                    json.dump(credentials, f)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_file
                logger.info("✅ Credentials file created and set")
            
            _client = genai.Client(
                vertexai=True,
                project=project,
                location=location
            )
            logger.info("✅ Vertex AI client created")
        else:
            # Use API key for Google AI Studio
            api_key = settings.google_api_key
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Google AI Studio")
            
            _client = genai.Client(api_key=api_key)
            logger.info("✅ Google AI client created with API key")
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
    
    # Use the new SDK's embed_content method
    embeddings = []
    for text in texts:
        try:
            # Use the client.models.embed_content method with correct parameters
            response = client.models.embed_content(
                model=EMBEDDING_MODEL_NAME,
                contents=text,  # Use 'contents' parameter (plural)
                task_type=task_type,
                title=title,
                output_dimensionality=768
            )
            embeddings.append(response.embedding)
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            # Return zero vector as fallback
            embeddings.append([0.0] * 768)
    
    return embeddings

logger.info(f"✅ Generation Model ('{GENERATION_MODEL_NAME}') is ready to be used.")
logger.info(f"✅ Embedding Model ('{EMBEDDING_MODEL_NAME}') is ready to be used.")