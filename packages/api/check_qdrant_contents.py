#!/usr/bin/env python3
"""Check what's actually in Qdrant."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from collections import defaultdict

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

print("\n" + "="*60)
print("QDRANT COLLECTION CONTENTS")
print("="*60)

# Get collection info
collection = client.get_collection("ml_docs")
print(f"\nTotal Points: {collection.points_count}")
print(f"Status: {collection.status}")

# Sample some points to see what we have
print("\n" + "="*60)
print("SAMPLING 50 POINTS TO ANALYZE VENDORS")
print("="*60)

# Scroll through points
offset = None
vendors = defaultdict(int)
tiers = defaultdict(int)
sources = defaultdict(int)
sample_chunks = []

for _ in range(5):  # Get 5 batches of 10 = 50 points
    result = client.scroll(
        collection_name="ml_docs",
        limit=10,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    points, next_offset = result
    
    if not points:
        break
    
    for point in points:
        metadata = point.payload.get('metadata', {})
        vendor = metadata.get('vendor', 'unknown')
        tier = metadata.get('tier', 'unknown')
        source_url = metadata.get('source_url', 'unknown')
        
        vendors[vendor] += 1
        tiers[tier] += 1
        sources[source_url] += 1
        
        if len(sample_chunks) < 5:
            sample_chunks.append({
                'vendor': vendor,
                'title': metadata.get('title', 'N/A'),
                'content': point.payload.get('text', '')[:200],
                'size': len(point.payload.get('text', ''))
            })
    
    offset = next_offset
    if not next_offset:
        break

print("\nVENDOR DISTRIBUTION (from sample):")
print("-" * 60)
for vendor, count in sorted(vendors.items(), key=lambda x: x[1], reverse=True):
    print(f"  {vendor}: {count} chunks")

print("\nTIER DISTRIBUTION (from sample):")
print("-" * 60)
for tier, count in sorted(tiers.items(), key=lambda x: x[1], reverse=True):
    print(f"  {tier}: {count} chunks")

print("\nTOP SOURCE URLs (from sample):")
print("-" * 60)
for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {count}x: {source[:80]}...")

print("\n" + "="*60)
print("SAMPLE CHUNKS (Quality Check)")
print("="*60)

for i, chunk in enumerate(sample_chunks, 1):
    print(f"\nChunk {i}:")
    print(f"  Vendor: {chunk['vendor']}")
    print(f"  Title: {chunk['title'][:60]}...")
    print(f"  Size: {chunk['size']} chars")
    print(f"  Content Preview: {chunk['content']}...")
    print()

# Test search for PyTorch
print("\n" + "="*60)
print("TEST SEARCH: 'PyTorch training loop'")
print("="*60)

# Create a dummy embedding for search (we need actual embeddings in production)
# For now, just scroll to find PyTorch content
pytorch_found = False
offset = None

for _ in range(10):  # Check up to 100 points
    result = client.scroll(
        collection_name="ml_docs",
        limit=10,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    points, next_offset = result
    
    if not points:
        break
    
    for point in points:
        metadata = point.payload.get('metadata', {})
        if 'pytorch' in metadata.get('vendor', '').lower():
            pytorch_found = True
            print(f"\nFound PyTorch chunk:")
            print(f"  Title: {metadata.get('title', 'N/A')}")
            print(f"  URL: {metadata.get('source_url', 'N/A')}")
            print(f"  Content: {point.payload.get('text', '')[:300]}...")
            break
    
    if pytorch_found:
        break
    
    offset = next_offset
    if not next_offset:
        break

if not pytorch_found:
    print("\n❌ NO PYTORCH CONTENT FOUND IN FIRST 100 POINTS!")
    print("This confirms PyTorch ingestion failed.")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)


