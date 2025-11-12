#!/usr/bin/env python3
"""Upload comprehensive ML documentation with detailed content."""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

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

# Comprehensive ML documentation content
comprehensive_docs = [
    # AWS SageMaker - Detailed deployment guide
    {
        "text": """
AWS SageMaker Model Deployment Guide

1. Prepare Your Model:
   - Save your trained model in a supported format (pickle, joblib, or native framework format)
   - Create a model.tar.gz file containing your model and any dependencies
   - Ensure your model follows the SageMaker inference specification

2. Create a SageMaker Model:
   ```python
   import boto3
   from sagemaker import get_execution_role
   
   role = get_execution_role()
   model = Model(
       model_data='s3://your-bucket/model.tar.gz',
       image_uri='your-inference-image-uri',
       role=role
   )
   ```

3. Deploy to Endpoint:
   ```python
   predictor = model.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large',
       endpoint_name='my-model-endpoint'
   )
   ```

4. Best Practices:
   - Use multi-model endpoints for cost optimization
   - Implement auto-scaling based on traffic
   - Set up monitoring and logging
   - Use A/B testing for model updates
   - Implement proper security with IAM roles
        """,
        "source_url": "https://docs.aws.amazon.com/sagemaker/latest/dg/model-deploy.html",
        "title": "AWS SageMaker Model Deployment Complete Guide",
        "vendor": "AWS",
        "topics": ["deployment", "aws", "sagemaker", "ml-ops"]
    },
    
    # Kubernetes Service Networking - Comprehensive guide
    {
        "text": """
Kubernetes Service Networking Best Practices

1. Service Types and Use Cases:
   - ClusterIP: Internal cluster communication (default)
   - NodePort: External access via node IP
   - LoadBalancer: Cloud provider load balancer integration
   - ExternalName: External service mapping

2. Service Discovery Best Practices:
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: ml-model-service
     labels:
       app: ml-model
   spec:
     selector:
       app: ml-model
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8080
     type: ClusterIP
   ```

3. Network Policies for Security:
   ```yaml
   apiVersion: networking.k8s.io/v1
   kind: NetworkPolicy
   metadata:
     name: ml-model-network-policy
   spec:
     podSelector:
       matchLabels:
         app: ml-model
     policyTypes:
     - Ingress
     - Egress
     ingress:
     - from:
       - namespaceSelector:
           matchLabels:
             name: allowed-namespace
   ```

4. Service Mesh Integration:
   - Use Istio for advanced traffic management
   - Implement circuit breakers and retries
   - Enable distributed tracing
   - Configure mTLS for service-to-service communication

5. Monitoring and Observability:
   - Use Prometheus for metrics collection
   - Implement health checks and readiness probes
   - Set up distributed tracing with Jaeger
   - Monitor service mesh metrics
        """,
        "source_url": "https://kubernetes.io/docs/concepts/services-networking/service/",
        "title": "Kubernetes Service Networking Best Practices",
        "vendor": "Kubernetes",
        "topics": ["networking", "kubernetes", "microservices", "security"]
    },
    
    # MLflow - Comprehensive tracking and registry
    {
        "text": """
MLflow Model Registry and Tracking Complete Guide

1. MLflow Tracking Setup:
   ```python
   import mlflow
   import mlflow.sklearn
   
   # Set tracking URI
   mlflow.set_tracking_uri("http://localhost:5000")
   
   # Start experiment
   with mlflow.start_run():
       # Log parameters
       mlflow.log_param("learning_rate", 0.01)
       mlflow.log_param("epochs", 100)
       
       # Log metrics
       mlflow.log_metric("accuracy", 0.95)
       mlflow.log_metric("loss", 0.05)
       
       # Log model
       mlflow.sklearn.log_model(model, "model")
   ```

2. Model Registry Workflow:
   ```python
   # Register model
   model_version = mlflow.register_model(
       model_uri="runs:/{run_id}/model",
       name="my_model"
   )
   
   # Transition to staging
   client = mlflow.tracking.MlflowClient()
   client.transition_model_version_stage(
       name="my_model",
       version=1,
       stage="Staging"
   )
   
   # Load model for inference
   model = mlflow.pyfunc.load_model(
       model_uri="models:/my_model/Production"
   )
   ```

3. Model Registry Best Practices:
   - Use semantic versioning for model versions
   - Implement approval workflows for production
   - Set up automated model validation
   - Monitor model performance in production
   - Implement model rollback strategies

4. Advanced Tracking Features:
   - Custom metrics and artifacts
   - Nested runs for hyperparameter tuning
   - Model comparison and selection
   - Integration with experiment tracking tools
        """,
        "source_url": "https://mlflow.org/docs/latest/model-registry/",
        "title": "MLflow Model Registry and Tracking Guide",
        "vendor": "MLflow",
        "topics": ["mlops", "tracking", "model-registry", "experiments"]
    },
    
    # PyTorch - Comprehensive guide
    {
        "text": """
PyTorch Complete Guide: From Basics to Advanced

1. PyTorch Fundamentals:
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   
   # Create tensors
   x = torch.tensor([1, 2, 3, 4])
   y = torch.randn(2, 3)
   
   # Basic operations
   z = x + y
   result = torch.matmul(x, y.T)
   ```

2. Data Loading with DataLoader:
   ```python
   from torch.utils.data import DataLoader, Dataset
   
   class CustomDataset(Dataset):
       def __init__(self, data, labels):
           self.data = data
           self.labels = labels
       
       def __len__(self):
           return len(self.data)
       
       def __getitem__(self, idx):
           return self.data[idx], self.labels[idx]
   
   # Create DataLoader
   dataset = CustomDataset(X_train, y_train)
   dataloader = DataLoader(
       dataset, 
       batch_size=32, 
       shuffle=True, 
       num_workers=4
   )
   
   # Training loop
   for batch_data, batch_labels in dataloader:
       # Training code here
       pass
   ```

3. Model Definition and Training:
   ```python
   class MLP(nn.Module):
       def __init__(self, input_size, hidden_size, output_size):
           super(MLP, self).__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, output_size)
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   
   # Training setup
   model = MLP(784, 128, 10)
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   
   # Training loop
   for epoch in range(num_epochs):
       for batch_data, batch_labels in dataloader:
           optimizer.zero_grad()
           outputs = model(batch_data)
           loss = criterion(outputs, batch_labels)
           loss.backward()
           optimizer.step()
   ```

4. Advanced Features:
   - Distributed training with torch.distributed
   - Model optimization with TorchScript
   - GPU acceleration and CUDA
   - Custom loss functions and metrics
   - Model checkpointing and saving
        """,
        "source_url": "https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html",
        "title": "PyTorch Complete Development Guide",
        "vendor": "PyTorch",
        "topics": ["deep-learning", "pytorch", "neural-networks", "training"]
    },
    
    # Docker - Comprehensive deployment strategies
    {
        "text": """
Docker Container Deployment Strategies for ML

1. Multi-Stage Docker Builds:
   ```dockerfile
   # Build stage
   FROM python:3.9-slim as builder
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Production stage
   FROM python:3.9-slim
   WORKDIR /app
   COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
   COPY . .
   EXPOSE 8080
   CMD ["python", "app.py"]
   ```

2. ML Model Serving with Docker:
   ```python
   # app.py - Model serving application
   from flask import Flask, request, jsonify
   import joblib
   
   app = Flask(__name__)
   model = joblib.load('model.pkl')
   
   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       prediction = model.predict([data['features']])
       return jsonify({'prediction': prediction[0]})
   
   if __name__ == '__main__':
       app.run(host='0.0.0.0', port=8080)
   ```

3. Docker Compose for ML Stack:
   ```yaml
   version: '3.8'
   services:
     ml-api:
       build: .
       ports:
         - "8080:8080"
       environment:
         - MODEL_PATH=/models/model.pkl
       volumes:
         - ./models:/models
     
     redis:
       image: redis:alpine
       ports:
         - "6379:6379"
     
     postgres:
       image: postgres:13
       environment:
         POSTGRES_DB: ml_tracking
         POSTGRES_USER: mlflow
         POSTGRES_PASSWORD: mlflow
   ```

4. Production Deployment Best Practices:
   - Use specific base image versions
   - Implement health checks
   - Set resource limits and requests
   - Use secrets management
   - Implement logging and monitoring
   - Use container orchestration (Kubernetes)
        """,
        "source_url": "https://docs.docker.com/get-started/overview/",
        "title": "Docker ML Deployment Strategies",
        "vendor": "Docker",
        "topics": ["containerization", "deployment", "ml-ops", "docker"]
    },
    
    # Multi-source integration example
    {
        "text": """
Complete ML Pipeline: PyTorch + MLflow + AWS SageMaker + Kubernetes

1. Training Pipeline with PyTorch and MLflow:
   ```python
   import mlflow
   import mlflow.pytorch
   import torch
   from torch.utils.data import DataLoader
   
   # Start MLflow run
   with mlflow.start_run():
       # Log hyperparameters
       mlflow.log_params({
           "learning_rate": 0.001,
           "batch_size": 32,
           "epochs": 100
       })
       
       # Training loop
       for epoch in range(epochs):
           for batch in dataloader:
               # Training code
               pass
       
       # Log model
       mlflow.pytorch.log_model(model, "model")
   ```

2. Model Registry and Versioning:
   ```python
   # Register model in MLflow
   model_version = mlflow.register_model(
       model_uri="runs:/{run_id}/model",
       name="pytorch_model"
   )
   
   # Transition to staging
   client.transition_model_version_stage(
       name="pytorch_model",
       version=1,
       stage="Staging"
   )
   ```

3. AWS SageMaker Deployment:
   ```python
   import sagemaker
   from sagemaker.pytorch import PyTorchModel
   
   # Create SageMaker model
   pytorch_model = PyTorchModel(
       model_data='s3://bucket/model.tar.gz',
       role=sagemaker.get_execution_role(),
       entry_point='inference.py',
       framework_version='1.9.0',
       py_version='py38'
   )
   
   # Deploy to endpoint
   predictor = pytorch_model.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large'
   )
   ```

4. Kubernetes Deployment:
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: ml-model-deployment
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: ml-model
     template:
       metadata:
         labels:
           app: ml-model
       spec:
         containers:
         - name: ml-model
           image: your-registry/ml-model:latest
           ports:
           - containerPort: 8080
           env:
           - name: MLFLOW_TRACKING_URI
             value: "http://mlflow-server:5000"
   ```

5. Monitoring and Observability:
   - Use Prometheus for metrics
   - Implement distributed tracing
   - Set up alerting for model drift
   - Monitor resource utilization
        """,
        "source_url": "https://mlflow.org/docs/latest/tracking/",
        "title": "Complete ML Pipeline Integration Guide",
        "vendor": "Multi-Vendor",
        "topics": ["integration", "mlops", "pipeline", "deployment"]
    }
]

print("Uploading comprehensive ML documentation to Qdrant...")

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
for i, doc in enumerate(comprehensive_docs):
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
                "vendor": doc["vendor"],
                "topics": doc.get("topics", [])
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

print(f"✅ Uploaded {len(points)} comprehensive documents to Qdrant!")

# Verify upload
info = client.get_collection("ml-docs-copilot")
print(f"✅ Collection now has {info.points_count} points")

# Test search
test_queries = [
    "How to deploy models with AWS SageMaker",
    "Kubernetes service networking best practices", 
    "MLflow model registry and tracking",
    "PyTorch DataLoader with MLflow tracking"
]

print("\n🔍 Testing comprehensive search results:")
for query in test_queries:
    print(f"\n--- Query: {query} ---")
    test_embedding = embedder.encode_query(query).tolist()
    results = client.search("ml-docs-copilot", query_vector=test_embedding, limit=2)
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result.payload['metadata']['title']} (Score: {result.score:.3f})")
        print(f"   {result.payload['text'][:200]}...")
        print()
