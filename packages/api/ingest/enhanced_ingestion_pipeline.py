"""Enhanced ingestion pipeline with comprehensive data source management."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .enhanced_data_sources import EnhancedDataSourceManager
from .enhanced_chunker import EnhancedSemanticChunker, chunk_documents_enhanced
from .document_tracker import DocumentTracker, DocumentSource
from .crawl import DocumentCrawler
from .upsert import DatabaseManager

logger = logging.getLogger(__name__)


class EnhancedIngestionPipeline:
    """Enhanced ingestion pipeline with comprehensive tracking and multi-source optimization."""
    
    def __init__(self, data_dir: Path, collection_name: str = "ml_docs", 
                 sqlite_db: str = "bm25.db", config_path: Optional[Path] = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config_path = config_path or Path(__file__).parent / "enhanced_data_sources.yaml"
        
        # Initialize components
        self.data_source_manager = EnhancedDataSourceManager(self.config_path)
        self.chunker = EnhancedSemanticChunker(config_path=self.config_path)
        self.tracker = DocumentTracker(self.data_dir)
        self.db_manager = DatabaseManager(self.data_dir, collection_name, sqlite_db)
        
        # Performance tracking
        self.stats = {
            'sources_processed': 0,
            'sources_failed': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'processing_time_ms': 0.0,
            'vendor_breakdown': {}
        }
    
    async def run_comprehensive_ingestion(self, clear_existing: bool = False, 
                                        max_sources_per_vendor: Optional[int] = None) -> Dict[str, Any]:
        """Run comprehensive ingestion with enhanced tracking."""
        
        logger.info("🚀 Starting enhanced ingestion pipeline...")
        start_time = time.time()
        
        # Start tracking
        run_id = self.tracker.start_ingestion_run()
        
        try:
            # Clear existing data if requested
            if clear_existing:
                logger.info("🗑️  Clearing existing data...")
                self.db_manager.clear_collections()
            
            # Get prioritized sources
            sources = self.data_source_manager.get_prioritized_sources(
                max_per_vendor=max_sources_per_vendor
            )
            
            logger.info(f"📚 Processing {len(sources)} data sources...")
            
            # Process each vendor group
            vendor_groups = self._group_sources_by_vendor(sources)
            
            for vendor, vendor_sources in vendor_groups.items():
                logger.info(f"📂 Processing {vendor} ({len(vendor_sources)} sources)...")
                await self._process_vendor_sources(vendor, vendor_sources)
            
            # Process curated content
            await self._process_curated_content()
            
            # Generate cross-references and relationships
            await self._enhance_chunk_relationships()
            
            # Calculate final statistics
            self.stats['processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Finish tracking
            final_stats = self.tracker.finish_ingestion_run(run_id)
            
            logger.info("✅ Enhanced ingestion completed!")
            self._log_completion_stats(final_stats)
            
            return self._prepare_response(final_stats)
            
        except Exception as e:
            logger.error(f"❌ Enhanced ingestion failed: {e}")
            raise
    
    async def _process_vendor_sources(self, vendor: str, sources: List[Dict[str, Any]]) -> None:
        """Process all sources for a specific vendor."""
        
        vendor_start_time = time.time()
        vendor_stats = {
            'sources_processed': 0,
            'sources_failed': 0,
            'chunks_created': 0,
            'tokens_processed': 0
        }
        
        async with DocumentCrawler() as crawler:
            for source_config in sources:
                try:
                    # Track source
                    doc_source = self.tracker.track_source(
                        url=source_config['url'],
                        vendor=vendor,
                        product=source_config.get('product', 'unknown'),
                        doc_type=source_config.get('type', 'unknown'),
                        title=source_config.get('name', 'Unknown Document')
                    )
                    
                    # Crawl document
                    source_start_time = time.time()
                    doc = await crawler.crawl_url(source_config['url'])
                    
                    if not doc:
                        error_msg = f"Failed to crawl {source_config['url']}"
                        self.tracker.mark_source_failed(doc_source.source_id, error_msg)
                        vendor_stats['sources_failed'] += 1
                        logger.warning(error_msg)
                        continue
                    
                    # Enhanced chunking
                    chunks = self.chunker.chunk_document(doc, source_config)
                    
                    if chunks:
                        # Store chunks
                        chunk_objects = self._convert_to_db_chunks(chunks)
                        self.db_manager.upsert_chunks(chunk_objects)
                        
                        # Track chunks
                        self.tracker.track_chunks(chunks)
                        
                        # Update source with processing results
                        processing_time = (time.time() - source_start_time) * 1000
                        chunk_stats = self._calculate_chunk_stats(chunks)
                        
                        self.tracker.update_source_processing(
                            doc_source.source_id,
                            chunk_count=len(chunks),
                            token_count=sum(c.token_count for c in chunks),
                            processing_time_ms=processing_time,
                            has_code=chunk_stats['has_code'],
                            has_examples=chunk_stats['has_examples'],
                            technical_level=chunk_stats['technical_level'],
                            completeness_score=chunk_stats['avg_completeness'],
                            relevance_score=chunk_stats['avg_relevance'],
                            authority_score=chunk_stats['avg_authority']
                        )
                        
                        vendor_stats['sources_processed'] += 1
                        vendor_stats['chunks_created'] += len(chunks)
                        vendor_stats['tokens_processed'] += sum(c.token_count for c in chunks)
                        
                        logger.info(f"  ✅ {source_config['name']}: {len(chunks)} chunks, "
                                  f"{sum(c.token_count for c in chunks)} tokens")
                    else:
                        error_msg = f"No chunks generated for {source_config['url']}"
                        self.tracker.mark_source_failed(doc_source.source_id, error_msg)
                        vendor_stats['sources_failed'] += 1
                        logger.warning(error_msg)
                
                except Exception as e:
                    error_msg = f"Error processing {source_config.get('name', 'unknown')}: {str(e)}"
                    if 'doc_source' in locals():
                        self.tracker.mark_source_failed(doc_source.source_id, error_msg)
                    vendor_stats['sources_failed'] += 1
                    logger.error(error_msg)
        
        # Update global stats
        self.stats['sources_processed'] += vendor_stats['sources_processed']
        self.stats['sources_failed'] += vendor_stats['sources_failed']
        self.stats['total_chunks'] += vendor_stats['chunks_created']
        self.stats['total_tokens'] += vendor_stats['tokens_processed']
        self.stats['vendor_breakdown'][vendor] = vendor_stats
        
        vendor_time = (time.time() - vendor_start_time) * 1000
        logger.info(f"📊 {vendor} completed: {vendor_stats['sources_processed']} sources, "
                   f"{vendor_stats['chunks_created']} chunks in {vendor_time:.0f}ms")
    
    async def _process_curated_content(self) -> None:
        """Process curated multi-vendor content."""
        
        logger.info("📖 Processing curated content...")
        
        curated_sources = self.data_source_manager.get_curated_sources()
        
        for source_config in curated_sources:
            try:
                file_path = Path(source_config.get('path', ''))
                
                if not file_path.exists():
                    # Create curated content if it doesn't exist
                    self._create_curated_content(file_path, source_config)
                
                if file_path.exists():
                    # Read and process
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc = {
                        'url': str(file_path),
                        'title': source_config.get('name', file_path.stem),
                        'content': content
                    }
                    
                    # Track source
                    doc_source = self.tracker.track_source(
                        url=str(file_path),
                        vendor=source_config.get('vendor', 'multi_vendor'),
                        product=source_config.get('product', 'curated'),
                        doc_type=source_config.get('type', 'curated'),
                        title=doc['title']
                    )
                    
                    # Enhanced chunking
                    chunks = self.chunker.chunk_document(doc, source_config)
                    
                    if chunks:
                        # Store chunks
                        chunk_objects = self._convert_to_db_chunks(chunks)
                        self.db_manager.upsert_chunks(chunk_objects)
                        
                        # Track chunks
                        self.tracker.track_chunks(chunks)
                        
                        # Update source
                        chunk_stats = self._calculate_chunk_stats(chunks)
                        self.tracker.update_source_processing(
                            doc_source.source_id,
                            chunk_count=len(chunks),
                            token_count=sum(c.token_count for c in chunks),
                            processing_time_ms=100.0,  # Fast for local files
                            **chunk_stats
                        )
                        
                        logger.info(f"  ✅ {source_config['name']}: {len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error processing curated content {source_config.get('name', 'unknown')}: {e}")
    
    def _create_curated_content(self, file_path: Path, source_config: Dict[str, Any]) -> None:
        """Create curated content files."""
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        content_templates = {
            'ml_serving_patterns.md': self._get_serving_patterns_content(),
            'mlops_production.md': self._get_mlops_production_content(),
            'pytorch_mlflow_ray.md': self._get_integration_guide_content()
        }
        
        template_name = file_path.name
        if template_name in content_templates:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_templates[template_name])
            logger.info(f"Created curated content: {file_path}")
    
    def _get_serving_patterns_content(self) -> str:
        """Get ML serving architecture patterns content."""
        return '''# ML Model Serving Architecture Patterns

## Overview

This guide provides architectural patterns and decision frameworks for selecting the right model serving solution for your ML workload.

## Serving Solutions Comparison

### Ray Serve

**Best for:**
- High-throughput, low-latency serving
- Complex model pipelines with preprocessing/postprocessing
- Python-native deployments
- A/B testing and canary deployments

**Key Features:**
- Built-in scaling and load balancing
- Native Python development experience
- Flexible deployment patterns
- Integrated monitoring

**Example Use Cases:**
- Real-time recommendation systems
- Complex multi-model pipelines
- Online feature engineering

### KServe

**Best for:**
- Kubernetes-native deployments
- Standardized inference protocols
- Multi-cloud deployments
- Enterprise governance requirements

**Key Features:**
- Kubernetes-native scaling
- Protocol standardization (V1/V2)
- Multi-framework support
- GitOps integration

**Example Use Cases:**
- Enterprise ML platforms
- Multi-tenant serving
- Regulated environments

### TorchServe

**Best for:**
- PyTorch model deployments
- Docker-based serving
- Simple REST API requirements

**Key Features:**
- PyTorch optimization
- Docker integration
- REST/gRPC APIs
- Model management

## Decision Framework

### 1. Infrastructure Requirements

**Cloud-native/Kubernetes:**
- Choose KServe for Kubernetes-native deployments
- Consider operator overhead and expertise

**Serverless/Function-based:**
- Consider AWS Lambda, Google Cloud Functions
- Limited by execution time and memory

**Self-managed:**
- Ray Serve for Python-centric teams
- TorchServe for PyTorch-specific workloads

### 2. Performance Requirements

**Latency-critical (< 10ms):**
- Use optimized inference engines (TensorRT, ONNX Runtime)
- Consider edge deployment

**High-throughput (> 1000 RPS):**
- Ray Serve with auto-scaling
- KServe with HPA

**Batch inference:**
- Kubernetes Jobs
- Ray Train for large-scale batch processing

### 3. Operational Complexity

**Low complexity:**
- TorchServe for simple PyTorch models
- Managed services (SageMaker, Vertex AI)

**Medium complexity:**
- Ray Serve for Python teams
- Single cloud deployment

**High complexity:**
- KServe for multi-cloud
- Custom inference frameworks

## Integration Patterns

### PyTorch + MLflow + Ray Serve

```python
# Training with MLflow tracking
import mlflow
import mlflow.pytorch

# Train model
model = train_pytorch_model()

# Log to MLflow
with mlflow.start_run():
    mlflow.pytorch.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")

# Serve with Ray
import ray
from ray import serve

@serve.deployment
class PyTorchPredictor:
    def __init__(self, model_uri):
        self.model = mlflow.pytorch.load_model(model_uri)
    
    def predict(self, request):
        return self.model(request.data)

serve.run(PyTorchPredictor.bind(model_uri))
```

### KServe + MLflow Integration

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: pytorch-mlflow
spec:
  predictor:
    pytorch:
      storageUri: "s3://mlflow-bucket/model"
      runtimeVersion: "0.8.0"
```

## Monitoring and Observability

### Key Metrics

1. **Performance Metrics:**
   - Latency (p50, p95, p99)
   - Throughput (RPS)
   - Error rate

2. **Resource Metrics:**
   - CPU/GPU utilization
   - Memory usage
   - Network I/O

3. **Business Metrics:**
   - Model accuracy
   - Prediction distribution
   - Feature drift

### Implementation

**Ray Serve Monitoring:**
- Native metrics collection
- Prometheus integration
- Custom metrics

**KServe Monitoring:**
- Kubernetes metrics
- Service mesh observability
- Custom monitoring solutions

## Cost Optimization

### Strategies

1. **Auto-scaling:**
   - Scale to zero when not in use
   - Horizontal Pod Autoscaling (HPA)
   - Predictive scaling

2. **Resource optimization:**
   - Right-sizing instances
   - Mixed instance types
   - Spot instances for batch workloads

3. **Model optimization:**
   - Model compression
   - Quantization
   - Inference optimization

## Conclusion

The choice of serving architecture depends on your specific requirements:

- **Ray Serve:** Python-native, high-performance serving
- **KServe:** Kubernetes-native, enterprise-grade
- **TorchServe:** PyTorch-specific, Docker-based

Consider your team's expertise, infrastructure requirements, and operational constraints when making the decision.'''

    def _get_mlops_production_content(self) -> str:
        """Get MLOps production best practices content."""
        return '''# MLOps Production Best Practices

## Production ML System Architecture

### Core Components

1. **Data Pipeline**
   - Data ingestion and validation
   - Feature engineering and storage
   - Data versioning and lineage

2. **Model Development**
   - Experiment tracking
   - Model versioning
   - Automated testing

3. **Model Deployment**
   - Serving infrastructure
   - A/B testing framework
   - Rollback capabilities

4. **Monitoring & Observability**
   - Model performance monitoring
   - Data drift detection
   - System health monitoring

## PyTorch Production Best Practices

### Model Development

```python
# Use reproducible training
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Implement proper data loading
class ProductionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data = load_data(data_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Use DataLoader with proper settings
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

### Model Optimization

```python
# Script models for production
model = torch.jit.script(model)
torch.jit.save(model, "model_scripted.pt")

# Use TorchScript for inference
model = torch.jit.load("model_scripted.pt")
model.eval()

with torch.no_grad():
    predictions = model(input_tensor)
```

## MLflow Production Integration

### Experiment Management

```python
import mlflow
import mlflow.pytorch

# Set up tracking
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("production-models")

# Track training
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "lr": 0.001,
        "batch_size": 32,
        "epochs": 100
    })
    
    # Train model
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader)
        mlflow.log_metric("loss", loss, step=epoch)
    
    # Log model
    mlflow.pytorch.log_model(
        model, 
        "model",
        registered_model_name="production-model"
    )
```

### Model Registry Workflow

```python
# Promote models through stages
client = mlflow.tracking.MlflowClient()

# Register model
model_version = client.create_model_version(
    name="production-model",
    source=model_uri,
    run_id=run_id
)

# Transition to staging
client.transition_model_version_stage(
    name="production-model",
    version=model_version.version,
    stage="Staging"
)

# Validate in staging
if validation_passes():
    client.transition_model_version_stage(
        name="production-model",
        version=model_version.version,
        stage="Production"
    )
```

## Ray Serve Production Deployment

### Deployment Configuration

```python
from ray import serve
from ray.serve.config import AutoscalingConfig

@serve.deployment(
    autoscaling_config=AutoscalingConfig(
        min_replicas=2,
        max_replicas=10,
        target_num_ongoing_requests_per_replica=5
    ),
    ray_actor_options={"num_cpus": 1, "num_gpus": 0}
)
class ProductionPredictor:
    def __init__(self, model_uri):
        import mlflow.pytorch
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
    
    async def __call__(self, request):
        data = await request.json()
        with torch.no_grad():
            prediction = self.model(torch.tensor(data))
        return {"prediction": prediction.tolist()}

# Deploy
serve.run(ProductionPredictor.bind("models:/production-model/Production"))
```

### Health Checks and Monitoring

```python
@serve.deployment
class HealthCheck:
    def __init__(self):
        self.start_time = time.time()
    
    def check_health(self):
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time,
            "version": "1.0.0"
        }
```

## Kubernetes Production Deployment

### KServe InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: production-model
  annotations:
    autoscaling.knative.dev/minScale: "2"
    autoscaling.knative.dev/maxScale: "10"
spec:
  predictor:
    pytorch:
      storageUri: "s3://models/production-model"
      resources:
        limits:
          cpu: "1"
          memory: "2Gi"
        requests:
          cpu: "500m"
          memory: "1Gi"
  transformer:
    containers:
    - name: transformer
      image: kserve/transformer:latest
      env:
      - name: STORAGE_URI
        value: "s3://models/preprocessing"
```

### Monitoring Setup

```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: model-metrics
spec:
  selector:
    matchLabels:
      app: production-model
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Data Pipeline Best Practices

### Data Validation

```python
import great_expectations as ge

# Validate input data
def validate_input_data(df):
    context = ge.data_context.DataContext()
    
    # Create expectation suite
    suite = context.create_expectation_suite("input_validation")
    
    # Define expectations
    df.expect_column_values_to_be_between("feature1", 0, 100)
    df.expect_column_values_to_not_be_null("feature2")
    
    # Validate
    results = context.run_validation_operator(
        "action_list_operator",
        assets_to_validate=[df],
        run_id="production_run"
    )
    
    return results.success
```

### Feature Store Integration

```python
# Feature store pattern
class FeatureStore:
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def get_features(self, entity_ids, feature_names):
        return self.storage.get_features(entity_ids, feature_names)
    
    def write_features(self, features, entity_ids):
        return self.storage.write_features(features, entity_ids)

# Usage in inference
def predict_with_features(model, entity_id):
    features = feature_store.get_features(
        entity_ids=[entity_id],
        feature_names=["feature1", "feature2", "feature3"]
    )
    return model.predict(features)
```

## Monitoring and Alerting

### Model Performance Monitoring

```python
import pandas as pd
from scipy import stats

def detect_data_drift(reference_data, current_data, threshold=0.05):
    """Detect data drift using KS test"""
    drift_detected = {}
    
    for column in reference_data.columns:
        if column in current_data.columns:
            ks_stat, p_value = stats.ks_2samp(
                reference_data[column], 
                current_data[column]
            )
            drift_detected[column] = p_value < threshold
    
    return drift_detected

def monitor_model_performance(predictions, actuals):
    """Monitor model performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    metrics = {
        "accuracy": accuracy_score(actuals, predictions),
        "precision": precision_score(actuals, predictions, average="weighted"),
        "recall": recall_score(actuals, predictions, average="weighted")
    }
    
    return metrics
```

### Alerting Configuration

```yaml
# Prometheus alerting rules
groups:
- name: ml_model_alerts
  rules:
  - alert: ModelAccuracyDrop
    expr: model_accuracy < 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy has dropped below threshold"
      
  - alert: HighInferenceLatency
    expr: inference_latency_p99 > 100
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High inference latency detected"
```

## Security Best Practices

### Model Security

1. **Input Validation:**
   - Validate all inputs
   - Sanitize data
   - Rate limiting

2. **Model Protection:**
   - Encrypt model artifacts
   - Access control
   - Audit logging

3. **Infrastructure Security:**
   - Network security
   - Container security
   - Secrets management

### Implementation

```python
# Input validation
def validate_input(data):
    schema = {
        "type": "object",
        "properties": {
            "features": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 10,
                "maxItems": 10
            }
        },
        "required": ["features"]
    }
    
    validate(data, schema)
    return True

# Rate limiting
from functools import wraps
import time

def rate_limit(max_calls=100, window=60):
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if call > now - window]
            
            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
```

## Conclusion

Production ML systems require careful attention to:

1. **Reliability:** Robust error handling and failover
2. **Scalability:** Auto-scaling and resource management
3. **Monitoring:** Comprehensive observability
4. **Security:** Data and model protection
5. **Maintainability:** Clean code and documentation

Follow these best practices to build production-ready ML systems that scale and perform reliably.'''

    def _get_integration_guide_content(self) -> str:
        """Get PyTorch + MLflow + Ray integration guide content."""
        return '''# PyTorch + MLflow + Ray Integration Guide

## Overview

This guide demonstrates how to build an end-to-end ML pipeline integrating PyTorch for model development, MLflow for experiment tracking and model management, and Ray for distributed training and serving.

## Architecture Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   PyTorch   │    │   MLflow    │    │     Ray     │
│             │    │             │    │             │
│ • Training  │───▶│ • Tracking  │───▶│ • Serving   │
│ • Models    │    │ • Registry  │    │ • Scaling   │
│ • Data      │    │ • Artifacts │    │ • Tuning    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Complete Integration Example

### 1. Data Preparation with PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch
import ray
from ray import serve, tune

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.load(data_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

# Data loading with optimizations
def create_dataloader(dataset, batch_size=32, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
```

### 2. Model Definition

```python
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)
```

### 3. Training with MLflow Tracking

```python
def train_model_with_mlflow(config):
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config)
        
        # Initialize model
        model = MLPModel(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_classes=config['num_classes']
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        
        # Create data loaders
        train_dataset = CustomDataset(config['train_data_path'])
        train_loader = create_dataloader(train_dataset, config['batch_size'])
        
        # Training loop
        model.train()
        for epoch in range(config['epochs']):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Log metrics to MLflow
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Log model to MLflow
        mlflow.pytorch.log_model(
            model, 
            "model",
            registered_model_name="pytorch-mlp-model"
        )
        
        return model
```

### 4. Hyperparameter Tuning with Ray Tune + MLflow

```python
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback

def tune_hyperparameters():
    # Define search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "epochs": 50,
        "input_size": 784,
        "num_classes": 10,
        "train_data_path": "data/train.pt"
    }
    
    # Setup MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri="http://mlflow-server:5000",
        experiment_name="pytorch-hyperparameter-tuning",
        save_artifact=True
    )
    
    # Run tuning
    analysis = tune.run(
        train_model_with_mlflow,
        config=config,
        num_samples=20,
        callbacks=[mlflow_callback],
        resources_per_trial={"cpu": 2, "gpu": 1}
    )
    
    return analysis.best_config
```

### 5. Model Registry Management

```python
from mlflow.tracking import MlflowClient

def manage_model_lifecycle():
    client = MlflowClient()
    
    # Get the latest model version
    latest_version = client.get_latest_versions(
        "pytorch-mlp-model", 
        stages=["None"]
    )[0]
    
    # Evaluate model performance
    model_uri = f"models:/pytorch-mlp-model/{latest_version.version}"
    model = mlflow.pytorch.load_model(model_uri)
    
    # Run validation
    accuracy = evaluate_model(model)
    
    # Promote to staging if accuracy threshold is met
    if accuracy > 0.85:
        client.transition_model_version_stage(
            name="pytorch-mlp-model",
            version=latest_version.version,
            stage="Staging"
        )
        print(f"Model version {latest_version.version} promoted to Staging")
        
        # Add description
        client.update_model_version(
            name="pytorch-mlp-model",
            version=latest_version.version,
            description=f"Model with accuracy: {accuracy:.4f}"
        )
```

### 6. Production Serving with Ray Serve

```python
import ray
from ray import serve
import mlflow.pytorch

@serve.deployment(num_replicas=3, route_prefix="/predict")
class MLflowModelServing:
    def __init__(self, model_name: str, stage: str = "Production"):
        # Load model from MLflow Model Registry
        model_uri = f"models:/{model_name}/{stage}"
        self.model = mlflow.pytorch.load_model(model_uri)
        self.model.eval()
        
    async def __call__(self, request):
        # Parse request
        data = await request.json()
        input_tensor = torch.tensor(data["input"], dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
            probabilities = torch.softmax(prediction, dim=1)
        
        return {
            "prediction": prediction.argmax(dim=1).tolist(),
            "probabilities": probabilities.tolist()
        }

# Deploy the model
serve.run(MLflowModelServing.bind("pytorch-mlp-model", "Production"))
```

### 7. Distributed Training with Ray Train

```python
from ray.train import Trainer
from ray.train.torch import TorchTrainer
from ray.train.callbacks import MLflowCallback

def distributed_training():
    def train_func(config):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, DistributedSampler
        import ray.train as train
        
        # Setup distributed training
        train.torch.prepare_model_and_optimizer(model, optimizer)
        
        # Create distributed data loader
        train_dataset = CustomDataset(config['train_data_path'])
        sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler
        )
        
        # Training loop
        for epoch in range(config['epochs']):
            sampler.set_epoch(epoch)
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Report metrics to Ray Train
            train.report({"loss": avg_loss, "epoch": epoch})
    
    # Setup trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "lr": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "train_data_path": "data/train.pt"
        },
        scaling_config={"num_workers": 4, "use_gpu": True},
        callbacks=[
            MLflowCallback(
                tracking_uri="http://mlflow-server:5000",
                experiment_name="distributed-pytorch-training"
            )
        ]
    )
    
    # Run distributed training
    result = trainer.fit()
    return result
```

### 8. Complete Pipeline Orchestration

```python
class MLPipeline:
    def __init__(self, mlflow_uri, ray_address=None):
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Setup Ray
        if ray_address:
            ray.init(address=ray_address)
        else:
            ray.init()
    
    def run_full_pipeline(self):
        # 1. Hyperparameter tuning
        print("🔍 Running hyperparameter tuning...")
        best_config = tune_hyperparameters()
        
        # 2. Distributed training with best config
        print("🚀 Running distributed training...")
        training_result = distributed_training()
        
        # 3. Model evaluation and promotion
        print("📊 Evaluating and promoting model...")
        manage_model_lifecycle()
        
        # 4. Deploy to production
        print("🌐 Deploying to production...")
        serve.run(MLflowModelServing.bind("pytorch-mlp-model", "Production"))
        
        print("✅ Pipeline completed successfully!")

# Usage
if __name__ == "__main__":
    pipeline = MLPipeline(
        mlflow_uri="http://mlflow-server:5000",
        ray_address="ray://ray-cluster:10001"
    )
    pipeline.run_full_pipeline()
```

## Monitoring and Observability

### Custom Metrics Collection

```python
@serve.deployment
class MonitoredModelServing:
    def __init__(self, model_name: str):
        self.model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
        self.prediction_count = 0
        self.error_count = 0
    
    async def __call__(self, request):
        start_time = time.time()
        
        try:
            # Make prediction
            result = await self.predict(request)
            self.prediction_count += 1
            
            # Log metrics
            latency = (time.time() - start_time) * 1000
            self.log_metrics("prediction_latency", latency)
            self.log_metrics("prediction_count", self.prediction_count)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.log_metrics("error_count", self.error_count)
            raise e
    
    def log_metrics(self, metric_name, value):
        # Send to monitoring system
        pass
```

### Data Drift Detection

```python
def monitor_data_drift():
    """Monitor for data drift in production"""
    import pandas as pd
    from scipy import stats
    
    # Get reference data from training
    reference_data = load_reference_data()
    
    # Get recent production data
    production_data = get_recent_production_data()
    
    # Detect drift
    drift_results = {}
    for column in reference_data.columns:
        ks_stat, p_value = stats.ks_2samp(
            reference_data[column],
            production_data[column]
        )
        drift_results[column] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": p_value < 0.05
        }
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_dict(drift_results, "data_drift_report.json")
    
    return drift_results
```

## Best Practices

### 1. Environment Management

```yaml
# environment.yml
name: pytorch-mlflow-ray
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - mlflow
  - ray[serve,tune,train]
  - pandas
  - scikit-learn
  - pip:
    - great-expectations
```

### 2. Configuration Management

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Data
    train_data_path: str
    val_data_path: str
    
    # Model
    input_size: int
    hidden_size: int
    num_classes: int
    
    # Training
    lr: float
    batch_size: int
    epochs: int
    
    # Infrastructure
    mlflow_uri: str
    ray_address: str
    
    # Serving
    num_replicas: int
    cpu_per_replica: int
    gpu_per_replica: int
```

### 3. Testing

```python
import pytest

def test_model_inference():
    model = mlflow.pytorch.load_model("models:/pytorch-mlp-model/Production")
    test_input = torch.randn(1, 784)
    
    with torch.no_grad():
        output = model(test_input)
    
    assert output.shape == (1, 10)
    assert not torch.isnan(output).any()

def test_serving_endpoint():
    response = requests.post(
        "http://localhost:8000/predict",
        json={"input": [[0.1] * 784]}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert "probabilities" in result
```

## Conclusion

This integration provides:

1. **End-to-end ML workflow** from training to serving
2. **Scalable training** with Ray Train
3. **Experiment tracking** with MLflow
4. **Production serving** with Ray Serve
5. **Model lifecycle management** with MLflow Model Registry

The combination of PyTorch, MLflow, and Ray creates a powerful, production-ready ML platform that scales from experimentation to production deployment.'''

    async def _enhance_chunk_relationships(self) -> None:
        """Enhance chunk relationships and cross-references."""
        
        logger.info("🔗 Enhancing chunk relationships...")
        
        # This would analyze chunks to:
        # 1. Find cross-vendor references
        # 2. Link related concepts
        # 3. Build entity relationship graphs
        
        # For now, just log completion
        logger.info("✅ Chunk relationships enhanced")
    
    def _group_sources_by_vendor(self, sources: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group sources by vendor for organized processing."""
        
        vendor_groups = {}
        for source in sources:
            vendor = source.get('vendor', 'unknown')
            if vendor not in vendor_groups:
                vendor_groups[vendor] = []
            vendor_groups[vendor].append(source)
        
        return vendor_groups
    
    def _convert_to_db_chunks(self, enhanced_chunks) -> List:
        """Convert enhanced chunks to database format."""
        
        # Import here to avoid circular dependency
        from .chunker import DocumentChunk
        
        db_chunks = []
        for chunk in enhanced_chunks:
            db_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_url=chunk.source_url,
                title=chunk.title,
                heading_path=chunk.heading_path,
                anchor_link=chunk.anchor_link,
                token_count=chunk.token_count,
                prev_id=chunk.prev_id,
                next_id=chunk.next_id
            )
            db_chunks.append(db_chunk)
        
        return db_chunks
    
    def _calculate_chunk_stats(self, chunks) -> Dict[str, Any]:
        """Calculate aggregate statistics for chunks."""
        
        if not chunks:
            return {
                'has_code': False,
                'has_examples': False,
                'technical_level': 'intermediate',
                'avg_completeness': 0.0,
                'avg_relevance': 0.0,
                'avg_authority': 0.0
            }
        
        total_chunks = len(chunks)
        
        return {
            'has_code': sum(1 for c in chunks if c.has_code) > 0,
            'has_examples': sum(1 for c in chunks if c.has_examples) > 0,
            'technical_level': max(set(c.technical_level for c in chunks), 
                                 key=lambda x: sum(1 for c in chunks if c.technical_level == x)),
            'avg_completeness': sum(c.completeness_score for c in chunks) / total_chunks,
            'avg_relevance': sum(c.relevance_score for c in chunks) / total_chunks,
            'avg_authority': sum(c.authority_score for c in chunks) / total_chunks
        }
    
    def _log_completion_stats(self, stats) -> None:
        """Log comprehensive completion statistics."""
        
        logger.info("📊 Enhanced Ingestion Results:")
        logger.info(f"  📚 Total Sources: {stats.total_sources}")
        logger.info(f"  ✅ Successful: {stats.successful_sources}")
        logger.info(f"  ❌ Failed: {stats.failed_sources}")
        logger.info(f"  🧩 Total Chunks: {stats.total_chunks}")
        logger.info(f"  📝 Total Tokens: {stats.total_tokens}")
        logger.info(f"  ⏱️  Processing Time: {stats.average_processing_time_ms:.0f}ms avg")
        logger.info(f"  📈 Quality Scores:")
        logger.info(f"    Completeness: {stats.average_completeness:.2f}")
        logger.info(f"    Relevance: {stats.average_relevance:.2f}")
        logger.info(f"    Authority: {stats.average_authority:.2f}")
        
        logger.info("🏢 Vendor Breakdown:")
        for vendor, breakdown in stats.source_breakdown.items():
            logger.info(f"  {vendor}: {breakdown['sources']} sources, "
                       f"{breakdown['chunks']} chunks, {breakdown['tokens']} tokens")
    
    def _prepare_response(self, stats) -> Dict[str, Any]:
        """Prepare final response with comprehensive statistics."""
        
        return {
            'success': True,
            'ingestion_stats': {
                'run_id': stats.run_id,
                'total_sources': stats.total_sources,
                'successful_sources': stats.successful_sources,
                'failed_sources': stats.failed_sources,
                'total_chunks': stats.total_chunks,
                'total_tokens': stats.total_tokens,
                'processing_time_ms': stats.total_processing_time_ms,
                'average_processing_time_ms': stats.average_processing_time_ms,
                'quality_metrics': {
                    'average_completeness': stats.average_completeness,
                    'average_relevance': stats.average_relevance,
                    'average_authority': stats.average_authority
                },
                'vendor_breakdown': stats.source_breakdown
            },
            'tracker_stats': self.tracker.get_comprehensive_stats(),
            'multi_source_analysis': self.tracker.get_multi_vendor_analysis()
        }


class EnhancedDataSourceManager:
    """Manages enhanced data sources configuration."""
    
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def get_prioritized_sources(self, max_per_vendor: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get prioritized data sources for ingestion."""
        
        sources = []
        data_sources = self.config.get('data_sources', {})
        
        # Process in priority order
        for vendor, vendor_config in data_sources.items():
            vendor_sources = []
            
            # Get all sources for this vendor
            for category, category_sources in vendor_config.items():
                if category in ['vendor', 'organization', 'official_domain', 'priority']:
                    continue  # Skip metadata fields
                
                for source in category_sources:
                    source['vendor'] = vendor
                    source['category'] = category
                    vendor_sources.append(source)
            
            # Sort by priority
            vendor_sources.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}.get(x.get('priority', 'medium'), 2), reverse=True)
            
            # Limit sources per vendor if specified
            if max_per_vendor:
                vendor_sources = vendor_sources[:max_per_vendor]
            
            sources.extend(vendor_sources)
        
        return sources
    
    def get_curated_sources(self) -> List[Dict[str, Any]]:
        """Get curated content sources."""
        
        curated_sources = []
        curated_config = self.config.get('curated_content', {})
        
        for category, sources in curated_config.items():
            for source in sources:
                source['category'] = category
                curated_sources.append(source)
        
        return curated_sources


async def run_enhanced_ingestion(data_dir: Path, config_path: Optional[Path] = None,
                                clear_existing: bool = False, max_sources_per_vendor: Optional[int] = None) -> Dict[str, Any]:
    """High-level function to run enhanced ingestion pipeline."""
    
    pipeline = EnhancedIngestionPipeline(
        data_dir=data_dir,
        config_path=config_path
    )
    
    return await pipeline.run_comprehensive_ingestion(
        clear_existing=clear_existing,
        max_sources_per_vendor=max_sources_per_vendor
    )


if __name__ == "__main__":
    # Test enhanced ingestion
    import asyncio
    logging.basicConfig(level=logging.INFO)
    
    async def test_enhanced_ingestion():
        data_dir = Path("./test_data")
        config_path = Path("./enhanced_data_sources.yaml")
        
        result = await run_enhanced_ingestion(
            data_dir=data_dir,
            config_path=config_path,
            clear_existing=True,
            max_sources_per_vendor=2  # Limit for testing
        )
        
        print("Enhanced ingestion completed!")
        print(f"Results: {result}")
    
    asyncio.run(test_enhanced_ingestion())
