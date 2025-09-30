"""Multi-source retrieval system for cross-vendor ML documentation queries."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

from .retrieval import HybridRetriever, RetrievalResult
from .clients import get_client, GENERATION_MODEL_NAME

logger = logging.getLogger(__name__)


@dataclass
class VendorContext:
    """Context about a specific vendor in a query."""
    vendor: str
    confidence: float
    keywords: List[str]
    entities: List[str]
    use_cases: List[str]


@dataclass
class MultiSourceQuery:
    """Analyzed query with multi-source requirements."""
    original_query: str
    primary_vendors: List[VendorContext]
    integration_intent: bool
    comparison_intent: bool
    cross_vendor_entities: List[str]
    required_use_cases: List[str]
    complexity_level: str  # simple, moderate, complex


class VendorDetector:
    """Detects vendors and technologies mentioned in queries."""
    
    def __init__(self):
        # Enhanced vendor detection patterns
        self.vendor_patterns = {
            'pytorch': {
                'keywords': ['pytorch', 'torch', 'torchvision', 'torchtext', 'torchaudio'],
                'entities': ['DataLoader', 'Dataset', 'Module', 'Tensor', 'nn.Linear', 'DistributedDataParallel'],
                'apis': ['torch.', 'torchvision.', 'nn.', 'F.'],
                'concepts': ['autograd', 'backward', 'gradient', 'dynamic graph']
            },
            'mlflow': {
                'keywords': ['mlflow', 'ml flow'],
                'entities': ['MLflowClient', 'start_run', 'log_metric', 'ModelRegistry', 'RegisteredModel'],
                'apis': ['mlflow.', 'mlflow.pytorch', 'mlflow.sklearn'],
                'concepts': ['experiment tracking', 'model registry', 'artifact', 'run']
            },
            'ray': {
                'keywords': ['ray', 'ray serve', 'ray tune', 'ray train'],
                'entities': ['serve.deployment', 'tune.run', 'train.Trainer', 'ServeHandle'],
                'apis': ['ray.', 'serve.', 'tune.', 'train.'],
                'concepts': ['distributed', 'scaling', 'hyperparameter tuning', 'model serving']
            },
            'kserve': {
                'keywords': ['kserve', 'k serve', 'kubeflow serving'],
                'entities': ['InferenceService', 'Predictor', 'Transformer', 'Explainer'],
                'apis': ['kserve.', 'v1beta1'],
                'concepts': ['kubernetes serving', 'inference service', 'serverless']
            },
            'aws': {
                'keywords': ['aws', 'amazon', 'sagemaker', 'ec2', 's3', 'lambda'],
                'entities': ['SageMaker', 'TrainingJob', 'Model', 'Endpoint', 'S3Bucket'],
                'apis': ['boto3', 'sagemaker.', 'aws.'],
                'concepts': ['cloud', 'managed service', 'auto scaling']
            },
            'kubernetes': {
                'keywords': ['kubernetes', 'k8s', 'kubectl', 'helm'],
                'entities': ['Deployment', 'Service', 'Pod', 'ConfigMap', 'Secret'],
                'apis': ['kubectl', 'k8s.io'],
                'concepts': ['container orchestration', 'cluster', 'namespace']
            },
            'tensorflow': {
                'keywords': ['tensorflow', 'tf', 'keras'],
                'entities': ['tf.keras', 'Model', 'Sequential', 'Layer'],
                'apis': ['tf.', 'tensorflow.', 'keras.'],
                'concepts': ['static graph', 'session', 'eager execution']
            }
        }
        
        # Integration patterns
        self.integration_keywords = [
            'integrate', 'integration', 'combine', 'together', 'with',
            'using both', 'along with', 'and', 'plus', 'workflow',
            'pipeline', 'end-to-end', 'complete', 'full stack'
        ]
        
        # Comparison patterns
        self.comparison_keywords = [
            'vs', 'versus', 'compare', 'comparison', 'difference',
            'better', 'best', 'alternative', 'instead of', 'choose',
            'pros and cons', 'trade-off', 'advantage', 'disadvantage'
        ]
    
    def analyze_query(self, query: str) -> MultiSourceQuery:
        """Analyze query for multi-source requirements."""
        query_lower = query.lower()
        
        # Detect vendors
        vendors = self._detect_vendors(query)
        
        # Detect integration intent
        integration_intent = any(keyword in query_lower for keyword in self.integration_keywords)
        
        # Detect comparison intent
        comparison_intent = any(keyword in query_lower for keyword in self.comparison_keywords)
        
        # Extract cross-vendor entities
        cross_vendor_entities = self._extract_cross_vendor_entities(query, vendors)
        
        # Extract use cases
        use_cases = self._extract_use_cases(query)
        
        # Assess complexity
        complexity = self._assess_complexity(vendors, integration_intent, comparison_intent)
        
        return MultiSourceQuery(
            original_query=query,
            primary_vendors=vendors,
            integration_intent=integration_intent,
            comparison_intent=comparison_intent,
            cross_vendor_entities=cross_vendor_entities,
            required_use_cases=use_cases,
            complexity_level=complexity
        )
    
    def _detect_vendors(self, query: str) -> List[VendorContext]:
        """Detect vendors mentioned in the query."""
        query_lower = query.lower()
        detected_vendors = []
        
        for vendor, patterns in self.vendor_patterns.items():
            confidence = 0.0
            matched_keywords = []
            matched_entities = []
            matched_concepts = []
            
            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    confidence += 0.3
                    matched_keywords.append(keyword)
            
            # Check entities
            for entity in patterns['entities']:
                if entity.lower() in query_lower:
                    confidence += 0.2
                    matched_entities.append(entity)
            
            # Check API patterns
            for api in patterns['apis']:
                if api in query_lower:
                    confidence += 0.15
                    matched_entities.append(api)
            
            # Check concepts
            for concept in patterns['concepts']:
                if concept in query_lower:
                    confidence += 0.1
                    matched_concepts.append(concept)
            
            if confidence > 0.1:  # Threshold for detection
                # Infer use cases based on context
                use_cases = self._infer_vendor_use_cases(query_lower, vendor)
                
                vendor_context = VendorContext(
                    vendor=vendor,
                    confidence=min(1.0, confidence),
                    keywords=matched_keywords,
                    entities=matched_entities,
                    use_cases=use_cases
                )
                detected_vendors.append(vendor_context)
        
        # Sort by confidence
        detected_vendors.sort(key=lambda x: x.confidence, reverse=True)
        return detected_vendors
    
    def _extract_cross_vendor_entities(self, query: str, vendors: List[VendorContext]) -> List[str]:
        """Extract entities that might span multiple vendors."""
        cross_vendor_entities = []
        
        if len(vendors) > 1:
            # Look for entities that could be used with multiple vendors
            multi_vendor_entities = [
                'model', 'training', 'inference', 'deployment', 'serving',
                'pipeline', 'workflow', 'data', 'dataset', 'metrics',
                'monitoring', 'logging', 'scaling', 'batch', 'real-time'
            ]
            
            query_lower = query.lower()
            for entity in multi_vendor_entities:
                if entity in query_lower:
                    cross_vendor_entities.append(entity)
        
        return cross_vendor_entities
    
    def _extract_use_cases(self, query: str) -> List[str]:
        """Extract use cases from the query."""
        use_cases = []
        query_lower = query.lower()
        
        use_case_patterns = {
            'training': ['train', 'training', 'fit', 'learn', 'optimize'],
            'inference': ['inference', 'predict', 'serve', 'deploy', 'production'],
            'data_preparation': ['data', 'preprocess', 'transform', 'load', 'batch'],
            'monitoring': ['monitor', 'track', 'log', 'observe', 'metrics'],
            'scaling': ['scale', 'distributed', 'parallel', 'cluster'],
            'experimentation': ['experiment', 'tune', 'hyperparameter', 'optimize'],
            'deployment': ['deploy', 'serve', 'production', 'endpoint', 'api']
        }
        
        for use_case, keywords in use_case_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                use_cases.append(use_case)
        
        return use_cases
    
    def _infer_vendor_use_cases(self, query_lower: str, vendor: str) -> List[str]:
        """Infer specific use cases for a vendor based on query context."""
        use_cases = []
        
        vendor_use_cases = {
            'pytorch': {
                'training': ['train', 'learning', 'backward', 'optimizer'],
                'data_loading': ['dataloader', 'dataset', 'batch', 'transform'],
                'distributed': ['ddp', 'distributed', 'parallel', 'multi_gpu'],
                'inference': ['eval', 'no_grad', 'inference', 'predict']
            },
            'mlflow': {
                'tracking': ['track', 'log', 'metric', 'parameter', 'artifact'],
                'registry': ['registry', 'model', 'version', 'stage'],
                'deployment': ['deploy', 'serve', 'endpoint', 'production']
            },
            'ray': {
                'serving': ['serve', 'deployment', 'endpoint', 'scale'],
                'tuning': ['tune', 'hyperparameter', 'optimize', 'search'],
                'training': ['train', 'distributed', 'parallel']
            },
            'kserve': {
                'serving': ['serve', 'inference', 'predict', 'endpoint'],
                'kubernetes': ['k8s', 'cluster', 'pod', 'deployment']
            }
        }
        
        if vendor in vendor_use_cases:
            for use_case, keywords in vendor_use_cases[vendor].items():
                if any(keyword in query_lower for keyword in keywords):
                    use_cases.append(use_case)
        
        return use_cases
    
    def _assess_complexity(self, vendors: List[VendorContext], 
                          integration_intent: bool, comparison_intent: bool) -> str:
        """Assess query complexity based on number of vendors and intent."""
        if len(vendors) == 0:
            return 'simple'
        elif len(vendors) == 1 and not integration_intent and not comparison_intent:
            return 'simple'
        elif len(vendors) == 2 or integration_intent or comparison_intent:
            return 'moderate'
        else:
            return 'complex'


class MultiSourceRetriever:
    """Enhanced retrieval system for multi-source queries."""
    
    def __init__(self):
        self.base_retriever = HybridRetriever()
        self.vendor_detector = VendorDetector()
        self.client = get_client()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve results optimized for multi-source queries."""
        
        # Analyze query for multi-source requirements
        analysis = self.vendor_detector.analyze_query(query)
        
        logger.info(f"Multi-source analysis: {len(analysis.primary_vendors)} vendors detected, "
                   f"integration_intent: {analysis.integration_intent}, "
                   f"comparison_intent: {analysis.comparison_intent}")
        
        if analysis.complexity_level == 'simple':
            # Use standard retrieval for simple queries
            return self.base_retriever.retrieve(query, top_k)
        
        # Multi-source retrieval strategy
        return self._multi_source_retrieve(analysis, top_k)
    
    def _multi_source_retrieve(self, analysis: MultiSourceQuery, top_k: int) -> List[RetrievalResult]:
        """Perform multi-source retrieval with vendor balancing."""
        
        all_results = []
        vendor_results = defaultdict(list)
        
        if analysis.integration_intent:
            # Integration queries need balanced representation
            results = self._integration_retrieval(analysis, top_k)
        elif analysis.comparison_intent:
            # Comparison queries need side-by-side vendor results
            results = self._comparison_retrieval(analysis, top_k)
        else:
            # General multi-vendor query
            results = self._balanced_vendor_retrieval(analysis, top_k)
        
        return results
    
    def _integration_retrieval(self, analysis: MultiSourceQuery, top_k: int) -> List[RetrievalResult]:
        """Retrieve results for integration queries."""
        
        # Generate integration-focused sub-queries
        sub_queries = self._generate_integration_queries(analysis)
        
        all_results = {}
        for sub_query in sub_queries:
            results = self.base_retriever.retrieve(sub_query, top_k // 2)
            
            for result in results:
                # Boost integration-relevant results
                if self._is_integration_relevant(result, analysis):
                    result.score *= 1.2
                
                # Avoid duplicates
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                elif result.score > all_results[result.chunk_id].score:
                    all_results[result.chunk_id] = result
        
        # Sort and balance by vendor
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return self._balance_vendor_representation(final_results, analysis, top_k)
    
    def _comparison_retrieval(self, analysis: MultiSourceQuery, top_k: int) -> List[RetrievalResult]:
        """Retrieve results for comparison queries."""
        
        vendor_results = {}
        target_vendors = [v.vendor for v in analysis.primary_vendors[:3]]  # Max 3 vendors for comparison
        
        # Get results for each vendor
        for vendor in target_vendors:
            vendor_query = f"{analysis.original_query} {vendor}"
            results = self.base_retriever.retrieve(vendor_query, top_k // len(target_vendors) + 2)
            
            # Filter to vendor-specific results
            vendor_specific = []
            for result in results:
                if self._is_vendor_specific(result, vendor):
                    vendor_specific.append(result)
            
            vendor_results[vendor] = vendor_specific[:top_k // len(target_vendors)]
        
        # Interleave results from different vendors
        final_results = []
        max_per_vendor = max(len(results) for results in vendor_results.values())
        
        for i in range(max_per_vendor):
            for vendor in target_vendors:
                if i < len(vendor_results[vendor]):
                    final_results.append(vendor_results[vendor][i])
        
        return final_results[:top_k]
    
    def _balanced_vendor_retrieval(self, analysis: MultiSourceQuery, top_k: int) -> List[RetrievalResult]:
        """Retrieve results with balanced vendor representation."""
        
        # Basic retrieval
        results = self.base_retriever.retrieve(analysis.original_query, top_k * 2)
        
        # Balance vendor representation
        return self._balance_vendor_representation(results, analysis, top_k)
    
    def _generate_integration_queries(self, analysis: MultiSourceQuery) -> List[str]:
        """Generate sub-queries focused on integration scenarios."""
        
        queries = [analysis.original_query]
        
        # Add vendor combination queries
        vendors = [v.vendor for v in analysis.primary_vendors[:3]]
        if len(vendors) >= 2:
            for i in range(len(vendors)):
                for j in range(i + 1, len(vendors)):
                    queries.append(f"{analysis.original_query} {vendors[i]} {vendors[j]} integration")
                    queries.append(f"how to use {vendors[i]} with {vendors[j]}")
        
        # Add use case specific queries
        for use_case in analysis.required_use_cases:
            queries.append(f"{use_case} {' '.join(vendors)} workflow")
        
        return queries[:5]  # Limit to prevent explosion
    
    def _is_integration_relevant(self, result: RetrievalResult, analysis: MultiSourceQuery) -> bool:
        """Check if result is relevant for integration scenarios."""
        
        content_lower = result.content.lower()
        vendor_names = [v.vendor for v in analysis.primary_vendors]
        
        # Check if multiple vendors are mentioned
        vendor_mentions = sum(1 for vendor in vendor_names if vendor in content_lower)
        if vendor_mentions >= 2:
            return True
        
        # Check for integration keywords
        integration_keywords = [
            'integration', 'combine', 'together', 'workflow', 'pipeline',
            'end-to-end', 'complete', 'full stack', 'connect', 'bridge'
        ]
        
        return any(keyword in content_lower for keyword in integration_keywords)
    
    def _is_vendor_specific(self, result: RetrievalResult, vendor: str) -> bool:
        """Check if result is specific to a particular vendor."""
        
        content_lower = result.content.lower()
        source_url = result.metadata.get('source_url', '').lower()
        
        # Check source URL
        vendor_domains = {
            'pytorch': 'pytorch.org',
            'mlflow': 'mlflow.org',
            'ray': 'ray.io',
            'kserve': 'kserve.github.io',
            'aws': 'aws.amazon.com',
            'kubernetes': 'kubernetes.io'
        }
        
        if vendor in vendor_domains and vendor_domains[vendor] in source_url:
            return True
        
        # Check content for vendor-specific terms
        vendor_patterns = self.vendor_detector.vendor_patterns.get(vendor, {})
        vendor_terms = (vendor_patterns.get('keywords', []) + 
                       vendor_patterns.get('entities', []) +
                       vendor_patterns.get('apis', []))
        
        vendor_mentions = sum(1 for term in vendor_terms if term.lower() in content_lower)
        
        return vendor_mentions >= 2
    
    def _balance_vendor_representation(self, results: List[RetrievalResult], 
                                     analysis: MultiSourceQuery, top_k: int) -> List[RetrievalResult]:
        """Balance vendor representation in results."""
        
        if len(analysis.primary_vendors) <= 1:
            return results[:top_k]
        
        # Group results by vendor
        vendor_groups = defaultdict(list)
        unclassified = []
        
        for result in results:
            vendor = self._classify_result_vendor(result)
            if vendor:
                vendor_groups[vendor].append(result)
            else:
                unclassified.append(result)
        
        # Calculate target distribution
        target_vendors = [v.vendor for v in analysis.primary_vendors[:3]]
        slots_per_vendor = top_k // len(target_vendors)
        remaining_slots = top_k % len(target_vendors)
        
        balanced_results = []
        
        # Add results from each vendor
        for i, vendor in enumerate(target_vendors):
            vendor_results = vendor_groups.get(vendor, [])
            target_count = slots_per_vendor + (1 if i < remaining_slots else 0)
            balanced_results.extend(vendor_results[:target_count])
        
        # Fill remaining slots with best unclassified or overflow results
        while len(balanced_results) < top_k:
            if unclassified:
                balanced_results.append(unclassified.pop(0))
            else:
                # Get overflow from vendor groups
                overflow = []
                for vendor in target_vendors:
                    vendor_results = vendor_groups.get(vendor, [])
                    start_idx = slots_per_vendor + (1 if vendor in target_vendors[:remaining_slots] else 0)
                    overflow.extend(vendor_results[start_idx:])
                
                overflow.sort(key=lambda x: x.score, reverse=True)
                if overflow:
                    balanced_results.append(overflow.pop(0))
                else:
                    break
        
        return balanced_results
    
    def _classify_result_vendor(self, result: RetrievalResult) -> Optional[str]:
        """Classify which vendor a result belongs to."""
        
        source_url = result.metadata.get('source_url', '').lower()
        content_lower = result.content.lower()
        
        # Check by source URL first
        vendor_domains = {
            'pytorch': 'pytorch.org',
            'mlflow': 'mlflow.org', 
            'ray': 'ray.io',
            'kserve': 'kserve.github.io',
            'aws': 'aws.amazon.com',
            'kubernetes': 'kubernetes.io'
        }
        
        for vendor, domain in vendor_domains.items():
            if domain in source_url:
                return vendor
        
        # Check by content
        for vendor, patterns in self.vendor_detector.vendor_patterns.items():
            vendor_score = 0
            
            # Count mentions of vendor-specific terms
            for keyword in patterns['keywords']:
                if keyword in content_lower:
                    vendor_score += 2
            
            for entity in patterns['entities']:
                if entity.lower() in content_lower:
                    vendor_score += 1
            
            if vendor_score >= 3:  # Threshold for classification
                return vendor
        
        return None
    
    def get_multi_source_stats(self) -> Dict[str, Any]:
        """Get statistics about multi-source query capabilities."""
        
        # This would analyze the current database for multi-source readiness
        stats = {
            'vendor_coverage': {},
            'integration_ready_chunks': 0,
            'cross_vendor_entities': [],
            'coverage_gaps': []
        }
        
        # Implementation would query the actual database
        # For now, return placeholder
        return stats


def retrieve_multi_source_documents(query: str, top_k: int = 10) -> List[RetrievalResult]:
    """High-level function for multi-source document retrieval."""
    retriever = MultiSourceRetriever()
    return retriever.retrieve(query, top_k)


# Global multi-source retriever instance
_multi_source_retriever = None


def get_multi_source_retriever() -> MultiSourceRetriever:
    """Get or create global multi-source retriever instance."""
    global _multi_source_retriever
    if _multi_source_retriever is None:
        _multi_source_retriever = MultiSourceRetriever()
    return _multi_source_retriever


if __name__ == "__main__":
    # Test multi-source retrieval
    logging.basicConfig(level=logging.INFO)
    
    # Test queries
    test_queries = [
        "How to use PyTorch DataLoader with MLflow tracking",
        "Compare Ray Serve vs KServe for model deployment",
        "End-to-end ML pipeline with PyTorch, MLflow, and Ray",
        "Kubernetes deployment for PyTorch models",
        "MLflow vs Ray Tune for hyperparameter optimization"
    ]
    
    detector = VendorDetector()
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = detector.analyze_query(query)
        
        print(f"Vendors: {[v.vendor for v in analysis.primary_vendors]}")
        print(f"Integration: {analysis.integration_intent}")
        print(f"Comparison: {analysis.comparison_intent}")
        print(f"Complexity: {analysis.complexity_level}")
        print(f"Use cases: {analysis.required_use_cases}")
        print("---")
