"""Centralized Google AI client configuration and model instances.

"""

import google.generativeai as genai
import os
import logging

logger = logging.getLogger(__name__)

# This code runs ONLY ONCE when this module is first imported.
# This guarantees configuration happens before any client is used.

logger.info("--- Centralized Client Initialization ---")
logger.info("--- Configuring Google AI Client with REST transport ---")

# Perform the crucial configuration here
genai.configure(
    api_key=os.environ["GOOGLE_API_KEY"],
    transport='rest'  # Force REST API to prevent Vertex AI routing
)


# Instance for Text Generation
GENERATION_MODEL = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config={
        "temperature": 0.1,
        "top_p": 0.9,
        "max_output_tokens": 2048,
    }
)

# Instance for Embeddings
# The new SDK uses genai.embed_content, not a separate model object
# But let's keep the model name for clarity if your embedder uses it.
EMBEDDING_MODEL_NAME = "gemini-embedding-001"

logger.info(f"✅ Generation Model ('{GENERATION_MODEL.model_name}') is ready.")
logger.info(f"✅ Embedding Model ('{EMBEDDING_MODEL_NAME}') is configured.")
logger.info("--- Google AI Client Configuration Complete ---")
