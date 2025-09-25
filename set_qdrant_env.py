"""Set Qdrant environment variables directly."""

import os

# Set the Qdrant environment variables directly
os.environ['QDRANT_URL'] = 'https://67a47e0d-9430-4684-b4b7-e146043f29b9.eu-central-1-0.aws.cloud.qdrant.io'
os.environ['QDRANT_API_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.QijIjgvEos53KcGZ2wDpsHQMXpBb-S-arCN1kOJa-Uk'
os.environ['QDRANT_COLLECTION'] = 'ml-docs-copilot'

print("✅ Qdrant environment variables set:")
print(f"QDRANT_URL: {os.environ['QDRANT_URL']}")
print(f"QDRANT_API_KEY: {os.environ['QDRANT_API_KEY'][:20]}...")
print(f"QDRANT_COLLECTION: {os.environ['QDRANT_COLLECTION']}")

# Test the connection
from qdrant_client import QdrantClient

try:
    client = QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ['QDRANT_API_KEY'])
    collections = client.get_collections()
    print(f"✅ Connected to Qdrant Cloud")
    print(f"📋 Available collections: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
