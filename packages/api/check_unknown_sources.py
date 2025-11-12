"""Check why some chunks show as 'Unknown' in citations."""
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

print("\n=== Checking Qdrant Metadata ===\n")

# Get a sample of points
results, _ = client.scroll(
    collection_name='ml_docs',
    limit=10,
    with_payload=True,
    with_vectors=False
)

print(f"Analyzing {len(results)} sample chunks:\n")

unknown_count = 0
for i, point in enumerate(results):
    title = point.payload.get('title', 'MISSING')
    source = point.payload.get('source', 'MISSING')
    vendor = point.payload.get('vendor', 'MISSING')
    
    if title == 'MISSING' or title == '' or title == 'Unknown':
        unknown_count += 1
        print(f"❌ Chunk {i+1}:")
        print(f"   Title: {title}")
        print(f"   Source: {source[:80] if source != 'MISSING' else 'MISSING'}...")
        print(f"   Vendor: {vendor}")
        print(f"   Text preview: {point.payload.get('text', '')[:100]}...")
        print()

print(f"\n📊 Summary: {unknown_count}/{len(results)} chunks have missing/empty titles")


