#!/usr/bin/env python3
"""Debug metadata extraction from Qdrant."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize clients
client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv('QDRANT_API_KEY')
)

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

# Test query
query = "How to deploy models with AWS SageMaker"
query_embedding = model.encode(query).tolist()

print(f"Query: {query}")
print("=" * 50)

# Search Qdrant
results = client.search(
    collection_name="ml-docs-copilot",
    query_vector=query_embedding,
    limit=3,
    with_payload=True
)

print(f"Found {len(results)} results")
print()

for i, hit in enumerate(results):
    print(f"Result {i+1}:")
    print(f"  ID: {hit.id}")
    print(f"  Score: {hit.score}")
    print(f"  Payload keys: {list(hit.payload.keys())}")
    print(f"  Text preview: {hit.payload.get('text', '')[:100]}...")
    print(f"  Metadata: {hit.payload.get('metadata', {})}")
    print()
