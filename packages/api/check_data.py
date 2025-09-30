#!/usr/bin/env python3
"""Check what data is in Qdrant."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv('QDRANT_API_KEY')
)

# Get collection info
info = client.get_collection('ml-docs-copilot')
print(f"Collection points: {info.points_count}")

# Sample some data
result = client.search('ml-docs-copilot', query_vector=[0.1] * 768, limit=5)
print("\nSample chunks:")
for i, r in enumerate(result):
    text = r.payload.get('text', '')
    metadata = r.payload.get('metadata', {})
    print(f"\n{i+1}. Text: {text[:150]}...")
    print(f"   Source: {metadata.get('source_url', 'Unknown')}")
    print(f"   Title: {metadata.get('title', 'Unknown')}")
