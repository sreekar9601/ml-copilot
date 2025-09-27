#!/usr/bin/env python3
"""
Create sample ML documentation data for testing the system.
This generates realistic documentation content for PyTorch, MLflow, Ray Serve, and KServe.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict

from .chunker import chunk_documents
from .upsert import upsert_document_chunks

logger = logging.getLogger(__name__)

# Sample documentation content
SAMPLE_DOCS = [
    {
        "url": "https://pytorch.org/docs/stable/data.html",
        "title": "PyTorch Data Loading and Processing",
        "content": """# PyTorch Data Loading and Processing

## DataLoader

The `torch.utils.data.DataLoader` combines a dataset and a sampler, and provides an iterable over the given dataset.

### Basic Usage

```python
from torch.utils.data import DataLoader, TensorDataset

# Create a simple dataset
data = torch.randn(1000, 10)
targets = torch.randint(0, 2, (1000,))
dataset = TensorDataset(data, targets)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over batches
for batch_data, batch_targets in dataloader:
    # Process batch
    pass
```

### Key Parameters

- `batch_size`: Number of samples per batch
- `shuffle`: Whether to shuffle the data
- `num_workers`: Number of subprocesses for data loading
- `pin_memory`: Whether to pin memory for faster GPU transfer

## Distributed Data Loading

For distributed training, use `DistributedSampler`:

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

## Custom Datasets

Create custom datasets by inheriting from `torch.utils.data.Dataset`:

```python
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
```

## Data Transforms

Use `torchvision.transforms` for image preprocessing:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```
"""
    },
    {
        "url": "https://pytorch.org/docs/stable/notes/ddp.html",
        "title": "PyTorch Distributed Data Parallel",
        "content": """# PyTorch Distributed Data Parallel (DDP)

## Overview

DistributedDataParallel (DDP) implements data parallelism at the module level which can run across multiple machines.

## Basic Setup

```python
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(100):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = ddp_model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    cleanup()
```

## Key Features

### Gradient Synchronization
DDP automatically synchronizes gradients across processes during backward pass.

### Process Group Management
DDP uses PyTorch's distributed communication package for gradient synchronization.

### Model Wrapping
DDP wraps the model and handles distributed communication transparently.

## Best Practices

1. **Use NCCL backend** for GPU training
2. **Set appropriate batch size** per GPU
3. **Use DistributedSampler** for data loading
4. **Handle process group initialization** properly

## Common Issues

### Deadlocks
- Ensure all processes call the same operations in the same order
- Use `torch.distributed.barrier()` for synchronization

### Memory Issues
- Use gradient accumulation for large models
- Consider model parallelism for very large models

## Performance Tips

- Use `find_unused_parameters=True` only when necessary
- Enable `broadcast_buffers=False` for models without batch norm
- Use `gradient_as_bucket_view=True` for memory efficiency
"""
    },
    {
        "url": "https://mlflow.org/docs/latest/tracking.html",
        "title": "MLflow Tracking",
        "content": """# MLflow Tracking

## Overview

MLflow Tracking is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code.

## Basic Usage

```python
import mlflow
import mlflow.sklearn

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("model.pkl")
```

## Tracking Server

Start the MLflow tracking server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts
```

## Experiment Management

```python
# Create experiment
experiment_id = mlflow.create_experiment("My Experiment")

# Set experiment
mlflow.set_experiment("My Experiment")

# Get experiment
experiment = mlflow.get_experiment(experiment_id)
```

## Advanced Features

### Nested Runs
```python
with mlflow.start_run():
    # Parent run
    mlflow.log_param("parent_param", "value")
    
    with mlflow.start_run(nested=True):
        # Child run
        mlflow.log_param("child_param", "value")
```

### Custom Tags
```python
mlflow.set_tag("team", "ml-team")
mlflow.set_tag("version", "1.0")
```

### Model Registry Integration
```python
# Log model with version
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="MyModel"
)
```

## UI Features

- **Experiment Comparison**: Compare runs across experiments
- **Parameter Tuning**: Visualize parameter vs metric relationships
- **Artifact Storage**: Browse and download logged artifacts
- **Model Versioning**: Track model versions and stages

## Best Practices

1. **Log early and often**: Log parameters, metrics, and artifacts
2. **Use meaningful names**: Choose descriptive experiment and run names
3. **Version your code**: Use Git integration for code versioning
4. **Organize experiments**: Use tags and experiment names for organization
"""
    },
    {
        "url": "https://mlflow.org/docs/latest/model-registry.html",
        "title": "MLflow Model Registry",
        "content": """# MLflow Model Registry

## Overview

The MLflow Model Registry is a centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of an MLflow Model.

## Model Lifecycle

Models in the registry have the following lifecycle stages:

1. **None**: Initial state
2. **Staging**: Testing and validation
3. **Production**: Live serving
4. **Archived**: Retired models

## Registering Models

```python
import mlflow
from mlflow.tracking import MlflowClient

# Log and register model
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="MyModel"
    )

# Or register existing model
client = MlflowClient()
client.create_registered_model("MyModel")
```

## Model Versioning

```python
# Create new version
client.create_model_version(
    name="MyModel",
    source="runs:/run_id/model",
    description="Version 1.0"
)

# Get model version
version = client.get_model_version("MyModel", 1)

# Update version
client.update_model_version(
    name="MyModel",
    version=1,
    description="Updated description"
)
```

## Stage Management

```python
# Transition to staging
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)
```

## Model Serving

```python
# Load model for serving
model = mlflow.sklearn.load_model(
    "models:/MyModel/Production"
)

# Make predictions
predictions = model.predict(data)
```

## Best Practices

### Model Naming
- Use consistent naming conventions
- Include team or project prefixes
- Use descriptive names

### Version Management
- Increment versions for each deployment
- Use semantic versioning when possible
- Document changes in version descriptions

### Stage Transitions
- Validate models in Staging before Production
- Archive old models when no longer needed
- Use automated testing for stage transitions

### Access Control
- Implement proper permissions
- Use service accounts for automated deployments
- Monitor model access and usage

## Integration with CI/CD

```python
# Automated model promotion
def promote_model(model_name, version, target_stage):
    client = MlflowClient()
    
    # Run validation tests
    if validate_model(model_name, version):
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage
        )
```

## Monitoring and Governance

- **Model Lineage**: Track model dependencies and data sources
- **Performance Monitoring**: Monitor model performance in production
- **Compliance**: Ensure models meet regulatory requirements
- **Audit Trail**: Track all model changes and deployments
"""
    },
    {
        "url": "https://docs.ray.io/en/latest/serve/getting-started.html",
        "title": "Ray Serve Getting Started",
        "content": """# Ray Serve Getting Started

## Overview

Ray Serve is a scalable model serving library for building online inference APIs. It's framework agnostic and can serve models built with any ML framework.

## Installation

```bash
pip install "ray[serve]"
```

## Basic Example

```python
import ray
from ray import serve
from fastapi import FastAPI

# Start Ray
ray.init()

# Start Serve
serve.start()

# Define a simple model
@serve.deployment
class MyModel:
    def __init__(self):
        self.model = load_model()
    
    def __call__(self, request):
        data = request["data"]
        return self.model.predict(data)

# Deploy the model
MyModel.deploy()

# Query the model
handle = serve.get_deployment("MyModel").get_handle()
result = ray.get(handle.remote({"data": input_data}))
```

## FastAPI Integration

```python
from fastapi import FastAPI
from ray import serve

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class FastAPIDeployment:
    def __init__(self):
        self.model = load_model()
    
    @app.get("/")
    def root(self):
        return {"message": "Hello World"}
    
    @app.post("/predict")
    def predict(self, data: dict):
        return self.model.predict(data["input"])

# Deploy
FastAPIDeployment.deploy()
```

## Configuration

```python
# Configure Serve
serve.start(
    http_options={
        "host": "0.0.0.0",
        "port": 8000
    }
)

# Configure deployment
@serve.deployment(
    num_replicas=3,
    ray_actor_options={"num_cpus": 2}
)
class MyModel:
    pass
```

## Scaling

### Automatic Scaling
```python
@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 2
    }
)
class MyModel:
    pass
```

### Manual Scaling
```python
# Scale up
MyModel.options(num_replicas=5).deploy()

# Scale down
MyModel.options(num_replicas=1).deploy()
```

## Model Composition

```python
@serve.deployment
class Preprocessor:
    def __call__(self, request):
        return preprocess(request["data"])

@serve.deployment
class Model:
    def __call__(self, request):
        return self.model.predict(request)

@serve.deployment
class Postprocessor:
    def __call__(self, request):
        return postprocess(request)

# Compose the pipeline
Preprocessor.deploy()
Model.deploy()
Postprocessor.deploy()
```

## Production Deployment

### Configuration File
```yaml
# serve_config.yaml
http_options:
  host: 0.0.0.0
  port: 8000

deployments:
  - name: MyModel
    num_replicas: 3
    ray_actor_options:
      num_cpus: 2
      num_gpus: 1
    autoscaling_config:
      min_replicas: 1
      max_replicas: 10
      target_ongoing_requests: 2
```

### Deploy with Config
```bash
serve deploy serve_config.yaml
```

## Monitoring

```python
# Get deployment status
serve.status()

# Get metrics
serve.get_deployment("MyModel").get_handle().get_metrics()
```

## Best Practices

1. **Resource Management**: Set appropriate CPU/GPU limits
2. **Health Checks**: Implement health check endpoints
3. **Error Handling**: Handle failures gracefully
4. **Monitoring**: Monitor performance and errors
5. **Security**: Implement authentication and authorization
"""
    },
    {
        "url": "https://kserve.github.io/website/latest/get_started/",
        "title": "KServe Getting Started",
        "content": """# KServe Getting Started

## Overview

KServe provides a Kubernetes-native serverless inferencing platform for machine learning models. It supports multiple ML frameworks and provides advanced features like autoscaling, canary deployments, and traffic management.

## Installation

### Prerequisites
- Kubernetes cluster (1.19+)
- kubectl configured
- Istio (for advanced features)

### Install KServe
```bash
# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml

# Install KServe Models Web App (optional)
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve-models-web-app.yaml
```

## Basic InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: gs://kfserving-examples/models/sklearn/1.0/model
```

### Deploy
```bash
kubectl apply -f sklearn-iris.yaml
```

### Test
```bash
# Get the service URL
SERVICE_URL=$(kubectl get inferenceservice sklearn-iris -o jsonpath='{.status.url}')

# Make a prediction
curl -v -H "Content-Type: application/json" \
  -d @./iris-input.json \
  $SERVICE_URL/v1/models/sklearn-iris:predict
```

## Model Frameworks

### Scikit-learn
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: gs://my-bucket/sklearn-model
      resources:
        requests:
          cpu: 100m
          memory: 1Gi
```

### PyTorch
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pytorch-cifar10
spec:
  predictor:
    pytorch:
      storageUri: gs://my-bucket/pytorch-model
      resources:
        requests:
          cpu: 100m
          memory: 1Gi
        limits:
          nvidia.com/gpu: 1
```

### TensorFlow
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: tensorflow-mnist
spec:
  predictor:
    tensorflow:
      storageUri: gs://my-bucket/tensorflow-model
      runtimeVersion: "2.8.0"
```

## Advanced Features

### Canary Deployment
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris-canary
spec:
  predictor:
    canaryTrafficPercent: 10
    sklearn:
      storageUri: gs://my-bucket/sklearn-model-v2
  canary:
    sklearn:
      storageUri: gs://my-bucket/sklearn-model-v1
```

### Autoscaling
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: gs://my-bucket/sklearn-model
      minReplicas: 1
      maxReplicas: 10
```

### Custom Predictor
```python
from kserve import Model, ModelServer, model_server
from typing import Dict, Any

class CustomModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()
    
    def load(self):
        # Load your model
        pass
    
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        # Make predictions
        return {"predictions": []}

if __name__ == "__main__":
    model = CustomModel("custom-model")
    ModelServer().start([model])
```

## Monitoring and Observability

### Metrics
- Request latency
- Request count
- Error rate
- Resource utilization

### Logging
```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sklearn-iris
spec:
  predictor:
    sklearn:
      storageUri: gs://my-bucket/sklearn-model
      logger:
        mode: all
        url: http://default-broker
```

## Best Practices

1. **Resource Management**: Set appropriate CPU/memory limits
2. **Health Checks**: Implement readiness and liveness probes
3. **Security**: Use proper RBAC and network policies
4. **Monitoring**: Set up comprehensive monitoring
5. **Testing**: Test models thoroughly before deployment
"""
    }
]

async def create_sample_data():
    """Create sample documentation data for testing."""
    logger.info("Creating sample ML documentation data...")
    
    # Create chunks from sample docs
    chunks = chunk_documents(SAMPLE_DOCS, chunk_size=500, overlap=50)
    
    # Store in databases
    data_dir = Path("./data")
    stats = upsert_document_chunks(chunks, data_dir)
    
    logger.info(f"Sample data created successfully!")
    logger.info(f"Stats: {stats}")
    
    return len(chunks)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(create_sample_data())

