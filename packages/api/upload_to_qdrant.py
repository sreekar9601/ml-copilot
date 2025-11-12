#!/usr/bin/env python3
"""Directly upload the correct data to Qdrant."""

import os
import sys
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import numpy as np

# Add path for ingest module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ingest'))
from vertex_embedder import VertexAIEmbedder

load_dotenv()

# Initialize clients
client = QdrantClient(
    url=os.getenv('QDRANT_URL'), 
    api_key=os.getenv('QDRANT_API_KEY')
)

# Use Vertex AI embedder to match query embeddings
embedder = VertexAIEmbedder()

# Sample ML documentation content
ml_docs = [
    {
        "text": "AWS SageMaker is a fully managed machine learning service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly. SageMaker removes all the barriers that typically slow down developers who want to use machine learning.",
        "source_url": "https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
        "title": "What is Amazon SageMaker",
        "vendor": "AWS"
    },
    {
        "text": "Kubernetes Services provide a stable network endpoint for your pods. A Service is an abstraction which defines a logical set of Pods and a policy by which to access them. Services enable loose coupling between dependent Pods.",
        "source_url": "https://kubernetes.io/docs/concepts/services-networking/service/",
        "title": "Kubernetes Services",
        "vendor": "Kubernetes"
    },
    {
        "text": "Docker is a platform for developing, shipping, and running applications using containerization. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly.",
        "source_url": "https://docs.docker.com/get-started/overview/",
        "title": "What is Docker",
        "vendor": "Docker"
    },
    {
        "text": "MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It handles the entire ML pipeline from data preparation to model deployment and monitoring.",
        "source_url": "https://mlflow.org/docs/latest/tracking/",
        "title": "MLflow Tracking",
        "vendor": "MLflow"
    },
    {
        "text": "PyTorch DataLoader is a utility for loading data in batches during training. It provides efficient data loading with automatic batching, shuffling, and multiprocessing support for faster training.",
        "source_url": "https://pytorch.org/tutorials/beginner/data_loading_tutorial.html",
        "title": "PyTorch Data Loading",
        "vendor": "PyTorch"
    }
]

print("Uploading ML documentation to Qdrant...")

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
for i, doc in enumerate(ml_docs):
    # Generate embedding using Vertex AI
    embedding = embedder.encode_document(doc["text"]).tolist()
    
    # Create point
    point = qmodels.PointStruct(
        id=i,
        vector=embedding,
        payload={
            "text": doc["text"],
            "metadata": {
                "source_url": doc["source_url"],
                "title": doc["title"],
                "vendor": doc["vendor"]
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

print(f"✅ Uploaded {len(points)} documents to Qdrant!")

# Verify upload
info = client.get_collection("ml-docs-copilot")
print(f"✅ Collection now has {info.points_count} points")

# Test search
test_query = "How to deploy models with AWS SageMaker"
test_embedding = embedder.encode_query(test_query).tolist()
results = client.search("ml-docs-copilot", query_vector=test_embedding, limit=3)

print("\n🔍 Test search results:")
for i, result in enumerate(results):
    print(f"{i+1}. {result.payload['metadata']['title']} (Score: {result.score:.3f})")
    print(f"   {result.payload['text'][:100]}...")
    print()
