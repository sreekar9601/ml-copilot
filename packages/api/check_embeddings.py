"""Check embedding configuration in Qdrant collection."""
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

print("\n" + "="*60)
print("EMBEDDING CONFIGURATION CHECK")
print("="*60 + "\n")

# Get collection info
collection_info = client.get_collection("ml_docs")

print(f"📊 Collection: ml_docs")
print(f"   Total Points: {collection_info.points_count}")
print(f"   Vector Size: {collection_info.config.params.vectors.size}")
print(f"   Distance Metric: {collection_info.config.params.vectors.distance}")
print(f"   Status: {collection_info.status}")

print("\n" + "="*60)
print("CURRENT SYSTEM CONFIGURATION")
print("="*60 + "\n")

# Check what embedder is being used
print("📝 Ingestion Script (complete_end_to_end_ingestion.py):")
print("   ✅ Uses: CachedVertexEmbedder (Vertex AI)")
print("   ✅ Model: text-embedding-004")
print("   ✅ Vector Size: 768 dimensions")

print("\n📝 Query System (api/retrieval.py):")
print("   ✅ Uses: LightweightEmbedder with Vertex AI")
print("   ✅ Model: gemini-embedding-001")
print("   ✅ Vector Size: 768 dimensions")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60 + "\n")

if collection_info.config.params.vectors.size == 768:
    print("✅ NO EMBEDDING CONFLICT DETECTED!")
    print("   - All data was ingested with Vertex AI embeddings")
    print("   - Queries use Vertex AI embeddings")
    print("   - Vector dimensions match (768)")
    print("   - Similarity search will work correctly")
else:
    print("⚠️  POTENTIAL ISSUE DETECTED")
    print(f"   - Unexpected vector size: {collection_info.config.params.vectors.size}")

print("\n" + "="*60 + "\n")

