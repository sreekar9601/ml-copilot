"""Enhanced data source management module."""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


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


def get_default_sources() -> Dict[str, Any]:
    """Get default data sources when configuration file is missing."""
    return {
        'data_sources': {
            'pytorch': {
                'core_documentation': [
                    {
                        'name': 'PyTorch Data Loading',
                        'url': 'https://pytorch.org/docs/stable/data.html',
                        'type': 'api_reference',
                        'product': 'pytorch-core',
                        'version': 'stable',
                        'topics': ['data_loading', 'dataloader', 'dataset'],
                        'entities': ['DataLoader', 'Dataset', 'IterableDataset'],
                        'use_cases': ['training', 'inference', 'data_pipeline'],
                        'priority': 'high',
                        'estimated_chunks': 50
                    },
                    {
                        'name': 'PyTorch Neural Network Module',
                        'url': 'https://pytorch.org/docs/stable/nn.html',
                        'type': 'api_reference',
                        'product': 'pytorch-core',
                        'version': 'stable',
                        'topics': ['neural_networks', 'layers', 'loss_functions'],
                        'entities': ['Module', 'Sequential', 'Linear', 'Conv2d'],
                        'use_cases': ['model_building', 'training', 'inference'],
                        'priority': 'high',
                        'estimated_chunks': 80
                    }
                ],
                'tutorials': [
                    {
                        'name': 'PyTorch Quickstart Tutorial',
                        'url': 'https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html',
                        'type': 'tutorial',
                        'product': 'pytorch-tutorials',
                        'version': 'latest',
                        'topics': ['getting_started', 'basic_workflow', 'tensors'],
                        'entities': ['torch.Tensor', 'nn.Module', 'DataLoader'],
                        'use_cases': ['onboarding', 'learning', 'first_steps'],
                        'priority': 'high',
                        'estimated_chunks': 20
                    }
                ]
            },
            'mlflow': {
                'core_documentation': [
                    {
                        'name': 'MLflow Tracking',
                        'url': 'https://mlflow.org/docs/latest/tracking.html',
                        'type': 'feature_guide',
                        'product': 'mlflow-tracking',
                        'version': 'latest',
                        'topics': ['experiment_tracking', 'metrics', 'parameters'],
                        'entities': ['MLflowClient', 'start_run', 'log_metric'],
                        'use_cases': ['experiment_management', 'metrics_tracking'],
                        'priority': 'high',
                        'estimated_chunks': 45
                    },
                    {
                        'name': 'MLflow Model Registry',
                        'url': 'https://mlflow.org/docs/latest/model-registry.html',
                        'type': 'feature_guide',
                        'product': 'mlflow-registry',
                        'version': 'latest',
                        'topics': ['model_registry', 'versioning', 'governance'],
                        'entities': ['ModelRegistry', 'RegisteredModel', 'ModelVersion'],
                        'use_cases': ['model_management', 'model_governance'],
                        'priority': 'high',
                        'estimated_chunks': 35
                    }
                ]
            },
            'ray': {
                'serve': [
                    {
                        'name': 'Ray Serve Architecture',
                        'url': 'https://docs.ray.io/en/latest/serve/architecture.html',
                        'type': 'architecture_guide',
                        'product': 'ray-serve',
                        'version': 'latest',
                        'topics': ['architecture', 'deployment', 'scaling'],
                        'entities': ['Deployment', 'Application', 'ServeHandle'],
                        'use_cases': ['model_serving', 'microservices', 'scaling'],
                        'priority': 'high',
                        'estimated_chunks': 30
                    },
                    {
                        'name': 'Ray Serve Getting Started',
                        'url': 'https://docs.ray.io/en/latest/serve/getting-started.html',
                        'type': 'tutorial',
                        'product': 'ray-serve',
                        'version': 'latest',
                        'topics': ['getting_started', 'basic_deployment'],
                        'entities': ['serve.run', '@serve.deployment', 'FastAPI'],
                        'use_cases': ['onboarding', 'basic_serving', 'api_development'],
                        'priority': 'high',
                        'estimated_chunks': 25
                    }
                ]
            },
            'kserve': {
                'core': [
                    {
                        'name': 'KServe Getting Started',
                        'url': 'https://kserve.github.io/website/latest/get_started/',
                        'type': 'tutorial',
                        'product': 'kserve-core',
                        'version': 'latest',
                        'topics': ['getting_started', 'kubernetes', 'inference'],
                        'entities': ['InferenceService', 'Predictor', 'Transformer'],
                        'use_cases': ['kubernetes_deployment', 'model_serving'],
                        'priority': 'high',
                        'estimated_chunks': 30
                    },
                    {
                        'name': 'KServe InferenceService',
                        'url': 'https://kserve.github.io/website/latest/modelserving/inferenceservice/',
                        'type': 'api_reference',
                        'product': 'kserve-core',
                        'version': 'latest',
                        'topics': ['inference_service', 'predictor', 'transformer'],
                        'entities': ['InferenceService', 'Predictor', 'Transformer'],
                        'use_cases': ['model_deployment', 'inference'],
                        'priority': 'high',
                        'estimated_chunks': 40
                    }
                ]
            }
        },
        'curated_content': {
            'architectural_decisions': [
                {
                    'name': 'ML Serving Architecture Patterns',
                    'path': 'docs/curated/ml_serving_patterns.md',
                    'type': 'architecture_guide',
                    'vendor': 'multi_vendor',
                    'topics': ['architecture', 'patterns', 'decision_making'],
                    'entities': ['Ray Serve', 'KServe', 'TorchServe', 'SageMaker'],
                    'use_cases': ['architecture_decisions', 'technology_selection'],
                    'priority': 'high',
                    'estimated_chunks': 20
                }
            ],
            'best_practices': [
                {
                    'name': 'MLOps Production Best Practices',
                    'path': 'docs/curated/mlops_production.md',
                    'type': 'best_practices',
                    'vendor': 'multi_vendor',
                    'topics': ['mlops', 'production', 'monitoring'],
                    'entities': ['MLflow', 'PyTorch', 'Kubernetes', 'Ray'],
                    'use_cases': ['production_deployment', 'ops', 'best_practices'],
                    'priority': 'high',
                    'estimated_chunks': 25
                }
            ]
        }
    }
