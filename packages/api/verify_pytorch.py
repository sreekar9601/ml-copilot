from qdrant_client import QdrantClient
import os

client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv('QDRANT_API_KEY')
)

# Check for PyTorch chunks
results = client.scroll(
    collection_name='ml_docs',
    scroll_filter={'must': [{'key': 'vendor', 'match': {'value': 'PyTorch'}}]},
    limit=10
)

print(f"\n{'='*60}")
print(f"PyTorch Chunks Found: {len(results[0])}")
print(f"{'='*60}\n")

for i, point in enumerate(results[0][:10], 1):
    title = point.payload.get('title', 'No title')
    print(f"{i}. {title[:70]}")

# Check if embeddings are real (not zeros)
if results[0]:
    first_vector = results[0][0].vector
    if isinstance(first_vector, list):
        is_zero = all(v == 0 for v in first_vector[:10])
        print(f"\n{'='*60}")
        print(f"Embeddings Status: {'ZERO VECTORS (BROKEN)' if is_zero else 'REAL VECTORS (WORKING!)'}")
        print(f"Vector sample: {first_vector[:5]}")
        print(f"{'='*60}\n")


