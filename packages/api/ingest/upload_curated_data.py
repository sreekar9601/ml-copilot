#!/usr/bin/env python3
"""Upload curated high-quality data to Qdrant."""

import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from production_data_curator import ProductionDataCurator

load_dotenv()

def main():
    """Upload curated data to Qdrant."""
    
    # Initialize clients
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'), 
        api_key=os.getenv('QDRANT_API_KEY')
    )
    
    # Load embedding model
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
    
    # Get curated documents
    curator = ProductionDataCurator()
    documents = curator.curate_high_quality_sources()
    
    print(f"📚 Uploading {len(documents)} curated documents to Qdrant...")
    
    # Clear existing collection
    try:
        client.delete_collection("ml-docs-copilot")
        print("✅ Cleared existing collection")
    except:
        print("ℹ️ Collection didn't exist")
    
    # Create new collection
    client.create_collection(
        collection_name="ml-docs-copilot",
        vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE)
    )
    print("✅ Created new collection")
    
    # Upload documents
    points = []
    for i, doc in enumerate(documents):
        # Generate embedding
        embedding = model.encode(doc.content).tolist()
        
        # Create point
        point = qmodels.PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": doc.content,
                "metadata": {
                    "source_url": doc.source_url,
                    "title": doc.title,
                    "vendor": doc.vendor,
                    "doc_type": doc.doc_type,
                    "topics": doc.topics,
                    "priority": doc.priority,
                    "quality_score": doc.quality_score,
                    "word_count": doc.metadata.get('word_count', 0),
                    "has_code_examples": doc.metadata.get('has_code_examples', False),
                    "technical_depth": doc.metadata.get('technical_depth', 'intermediate')
                }
            }
        )
        points.append(point)
    
    # Upload to Qdrant
    client.upsert(
        collection_name="ml-docs-copilot",
        points=points,
        wait=True
    )
    
    print(f"✅ Uploaded {len(points)} curated documents to Qdrant!")
    
    # Verify upload
    info = client.get_collection("ml-docs-copilot")
    print(f"✅ Collection now has {info.points_count} points")
    
    # Test search with comprehensive queries
    test_queries = [
        "How to deploy models with AWS SageMaker",
        "Kubernetes service networking best practices", 
        "MLflow model registry and tracking",
        "PyTorch DataLoader with MLflow tracking",
        "Docker container deployment strategies",
        "Complete ML pipeline integration"
    ]
    
    print("\n🔍 Testing comprehensive search results:")
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        test_embedding = model.encode(query).tolist()
        results = client.search("ml-docs-copilot", query_vector=test_embedding, limit=2)
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result.payload['metadata']['title']} (Score: {result.score:.3f})")
            print(f"   Vendor: {result.payload['metadata']['vendor']}")
            print(f"   Quality: {result.payload['metadata']['quality_score']:.2f}")
            print(f"   Has Code: {result.payload['metadata']['has_code_examples']}")
            print(f"   Content: {result.payload['text'][:150]}...")
            print()

if __name__ == "__main__":
    main()
