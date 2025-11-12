#!/usr/bin/env python3
"""Analyze chunk quality in detail."""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import json

load_dotenv()

client = QdrantClient(
    url=os.getenv('QDRANT_URL'),
    api_key=os.getenv('QDRANT_API_KEY')
)

print("\n" + "="*60)
print("CHUNK QUALITY ANALYSIS")
print("="*60)

# Get sample chunks
offset = None
issues = {
    'too_short': [],
    'too_long': [],
    'html_noise': [],
    'rss_feed': [],
    'low_content': [],
    'good': []
}

for _ in range(30):  # Check 300 chunks
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
        text = point.payload.get('text', '')
        metadata = point.payload.get('metadata', {})
        vendor = metadata.get('vendor', 'unknown')
        url = metadata.get('source_url', 'unknown')
        
        chunk_info = {
            'vendor': vendor,
            'url': url[:80],
            'size': len(text),
            'preview': text[:150]
        }
        
        # Check for issues
        if len(text) < 500:
            issues['too_short'].append(chunk_info)
        elif len(text) > 10000:
            issues['too_long'].append(chunk_info)
        elif 'rss.xml' in url or '<rss' in text[:200]:
            issues['rss_feed'].append(chunk_info)
        elif text.count('<') > 10 and text.count('>') > 10:
            issues['html_noise'].append(chunk_info)
        elif text.count('Skip to') > 2 or text.count('Navigation') > 2:
            issues['low_content'].append(chunk_info)
        else:
            issues['good'].append(chunk_info)
    
    offset = next_offset
    if not next_offset:
        break

# Print analysis
print("\n📊 QUALITY DISTRIBUTION:")
print("-" * 60)
total = sum(len(v) for v in issues.values())
for issue_type, chunks in issues.items():
    pct = (len(chunks) / total * 100) if total > 0 else 0
    print(f"  {issue_type}: {len(chunks)} ({pct:.1f}%)")

# Show examples of each issue
print("\n" + "="*60)
print("ISSUE EXAMPLES")
print("="*60)

for issue_type, chunks in issues.items():
    if chunks and issue_type != 'good':
        print(f"\n❌ {issue_type.upper()} (showing first 2):")
        print("-" * 60)
        for chunk in chunks[:2]:
            print(f"  Vendor: {chunk['vendor']}")
            print(f"  URL: {chunk['url']}")
            print(f"  Size: {chunk['size']} chars")
            print(f"  Preview: {chunk['preview']}...")
            print()

# Show good examples
if issues['good']:
    print(f"\n✅ GOOD CHUNKS (showing first 3):")
    print("-" * 60)
    for chunk in issues['good'][:3]:
        print(f"  Vendor: {chunk['vendor']}")
        print(f"  URL: {chunk['url']}")
        print(f"  Size: {chunk['size']} chars")
        print(f"  Preview: {chunk['preview']}...")
        print()

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if len(issues['rss_feed']) > 0:
    print("⚠️  RSS feed content detected - scraper needs better filtering")

if len(issues['html_noise']) > 0:
    print("⚠️  HTML noise detected - content extraction needs improvement")

if len(issues['too_short']) > total * 0.2:
    print("⚠️  Too many short chunks - increase minimum chunk size")

if len(issues['low_content']) > total * 0.1:
    print("⚠️  Navigation/boilerplate content detected - better filtering needed")

good_pct = (len(issues['good']) / total * 100) if total > 0 else 0
print(f"\n📈 Overall Quality Score: {good_pct:.1f}% good chunks")

if good_pct < 50:
    print("   ❌ POOR - Scraper needs significant improvements")
elif good_pct < 75:
    print("   ⚠️  FAIR - Some improvements needed")
else:
    print("   ✅ GOOD - Quality acceptable")


