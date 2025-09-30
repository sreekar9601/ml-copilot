"""Enhanced semantic chunker with rich metadata extraction."""

import re
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnhancedDocumentChunk:
    """Enhanced document chunk with comprehensive metadata."""
    
    # Core content
    content: str
    chunk_id: str
    source_url: str
    title: str
    heading_path: str
    
    # Enhanced metadata
    vendor: str
    product: str
    version: str
    doc_type: str
    topics: List[str]
    entities: List[str]
    use_cases: List[str]
    priority: str
    
    # Quality and characteristics
    quality_score: float
    authority_score: float
    content_characteristics: Dict[str, Any]
    
    # Technical metadata
    token_count: int
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedSemanticChunker:
    """Enhanced semantic chunker with advanced metadata extraction."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Enhanced patterns for better classification
        self.vendor_patterns = {
            'pytorch': [r'pytorch', r'torch\.', r'torchvision', r'torchaudio'],
            'mlflow': [r'mlflow', r'mlflow\.', r'experiment', r'tracking'],
            'ray': [r'ray', r'ray\.', r'serve', r'@serve'],
            'kserve': [r'kserve', r'inference', r'predictor', r'transformer'],
            'aws': [r'aws', r'sagemaker', r'boto3', r's3'],
            'kubernetes': [r'k8s', r'kubernetes', r'kubectl', r'pod', r'deployment'],
            'docker': [r'docker', r'container', r'image', r'dockerfile']
        }
        
        self.doc_type_patterns = {
            'tutorial': [r'tutorial', r'getting.started', r'quickstart', r'guide'],
            'api_reference': [r'api', r'reference', r'docs', r'stable'],
            'architecture': [r'architecture', r'design', r'pattern', r'best.practice'],
            'troubleshooting': [r'troubleshoot', r'debug', r'issue', r'problem'],
            'example': [r'example', r'sample', r'demo', r'code']
        }
    
    def chunk_document(self, document: Dict[str, Any]) -> List[EnhancedDocumentChunk]:
        """Chunk a document with enhanced metadata extraction."""
        
        content = document.get('content', '')
        source_url = document.get('source_url', '')
        title = document.get('title', '')
        
        # Extract enhanced metadata
        vendor = self._extract_vendor(source_url, content)
        product = self._extract_product(source_url, content)
        version = self._extract_version(source_url, content)
        doc_type = self._classify_document_type(source_url, content)
        topics = self._extract_topics(content)
        entities = self._extract_entities(content, vendor)
        use_cases = self._extract_use_cases(content)
        priority = self._determine_priority(source_url, topics, doc_type)
        
        # Quality assessment
        quality_score = self._assess_quality(content, source_url, doc_type)
        authority_score = self._assess_authority(source_url, vendor)
        
        # Content characteristics
        content_characteristics = self._analyze_content_characteristics(content)
        
        # Split content into semantic chunks
        chunks = self._split_content_semantically(content, title)
        
        # Create enhanced chunks
        enhanced_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk_id = self._generate_chunk_id(source_url, i, chunk_content)
            
            chunk = EnhancedDocumentChunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_url=source_url,
                title=title,
                heading_path=self._extract_heading_path(chunk_content),
                vendor=vendor,
                product=product,
                version=version,
                doc_type=doc_type,
                topics=topics,
                entities=entities,
                use_cases=use_cases,
                priority=priority,
                quality_score=quality_score,
                authority_score=authority_score,
                content_characteristics=content_characteristics,
                token_count=len(chunk_content.split()),
                metadata={
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'source_type': 'web_scraped',
                    'extraction_method': 'enhanced_semantic'
                }
            )
            
            # Link chunks
            if i > 0:
                chunk.prev_id = enhanced_chunks[-1].chunk_id
                enhanced_chunks[-1].next_id = chunk.chunk_id
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _extract_vendor(self, source_url: str, content: str) -> str:
        """Extract vendor from URL and content."""
        url_lower = source_url.lower()
        content_lower = content.lower()
        
        for vendor, patterns in self.vendor_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower) or re.search(pattern, content_lower):
                    return vendor
        
        return 'unknown'
    
    def _extract_product(self, source_url: str, content: str) -> str:
        """Extract product from URL and content."""
        url_lower = source_url.lower()
        
        if 'pytorch' in url_lower:
            if 'torchvision' in url_lower:
                return 'pytorch-torchvision'
            elif 'torchaudio' in url_lower:
                return 'pytorch-torchaudio'
            else:
                return 'pytorch-core'
        elif 'mlflow' in url_lower:
            return 'mlflow-core'
        elif 'ray' in url_lower:
            if 'serve' in url_lower:
                return 'ray-serve'
            else:
                return 'ray-core'
        elif 'kserve' in url_lower:
            return 'kserve-core'
        elif 'aws' in url_lower or 'sagemaker' in url_lower:
            return 'aws-sagemaker'
        elif 'kubernetes' in url_lower or 'k8s' in url_lower:
            return 'kubernetes-core'
        elif 'docker' in url_lower:
            return 'docker-core'
        
        return 'unknown'
    
    def _extract_version(self, source_url: str, content: str) -> str:
        """Extract version information."""
        # Look for version patterns in URL
        version_match = re.search(r'/(\d+\.\d+)/', source_url)
        if version_match:
            return version_match.group(1)
        
        # Look for version in content
        version_match = re.search(r'version[:\s]+([0-9.]+)', content, re.IGNORECASE)
        if version_match:
            return version_match.group(1)
        
        return 'latest'
    
    def _classify_document_type(self, source_url: str, content: str) -> str:
        """Classify document type."""
        url_lower = source_url.lower()
        content_lower = content.lower()
        
        for doc_type, patterns in self.doc_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower) or re.search(pattern, content_lower):
                    return doc_type
        
        return 'documentation'
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from content."""
        topics = []
        content_lower = content.lower()
        
        topic_patterns = {
            'data_loading': [r'dataloader', r'dataset', r'data.loading', r'batch'],
            'model_training': [r'training', r'train', r'loss', r'optimizer', r'epoch'],
            'model_serving': [r'serving', r'inference', r'deploy', r'endpoint'],
            'experiment_tracking': [r'experiment', r'tracking', r'metrics', r'logging'],
            'model_registry': [r'registry', r'version', r'staging', r'production'],
            'scaling': [r'scale', r'distributed', r'parallel', r'cluster'],
            'monitoring': [r'monitor', r'observability', r'logging', r'metrics'],
            'security': [r'security', r'auth', r'permission', r'access']
        }
        
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    topics.append(topic)
                    break
        
        return list(set(topics))
    
    def _extract_entities(self, content: str, vendor: str) -> List[str]:
        """Extract technical entities from content."""
        entities = []
        
        entity_patterns = {
            'pytorch': [r'DataLoader', r'Dataset', r'Module', r'Sequential', r'Linear', r'Conv2d', r'ReLU', r'CrossEntropyLoss', r'Adam', r'SGD'],
            'mlflow': [r'MLflowClient', r'start_run', r'log_metric', r'log_param', r'log_model', r'ModelRegistry', r'RegisteredModel'],
            'ray': [r'@serve\.deployment', r'serve\.run', r'Deployment', r'Application', r'ServeHandle', r'Ray'],
            'kserve': [r'InferenceService', r'Predictor', r'Transformer', r'Explainer', r'KServe'],
            'aws': [r'SageMaker', r'Endpoint', r'Model', r'TrainingJob', r'InferenceJob'],
            'kubernetes': [r'Pod', r'Deployment', r'Service', r'ConfigMap', r'Secret', r'Ingress']
        }
        
        if vendor in entity_patterns:
            for entity in entity_patterns[vendor]:
                if re.search(entity, content):
                    entities.append(entity)
        
        return list(set(entities))
    
    def _extract_use_cases(self, content: str) -> List[str]:
        """Extract use cases from content."""
        use_cases = []
        content_lower = content.lower()
        
        use_case_patterns = {
            'training': [r'training', r'train', r'learn', r'fit'],
            'inference': [r'inference', r'predict', r'score', r'deploy'],
            'experimentation': [r'experiment', r'test', r'try', r'explore'],
            'production': [r'production', r'prod', r'live', r'serve'],
            'monitoring': [r'monitor', r'watch', r'observe', r'track'],
            'scaling': [r'scale', r'distribute', r'parallel', r'cluster']
        }
        
        for use_case, patterns in use_case_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    use_cases.append(use_case)
                    break
        
        return list(set(use_cases))
    
    def _determine_priority(self, source_url: str, topics: List[str], doc_type: str) -> str:
        """Determine priority based on source and content."""
        url_lower = source_url.lower()
        
        # High priority indicators
        if any(indicator in url_lower for indicator in ['getting.started', 'quickstart', 'tutorial']):
            return 'high'
        
        if doc_type in ['tutorial', 'getting_started']:
            return 'high'
        
        if any(topic in ['data_loading', 'model_training', 'model_serving'] for topic in topics):
            return 'high'
        
        # Medium priority indicators
        if doc_type in ['api_reference', 'architecture']:
            return 'medium'
        
        # Low priority for everything else
        return 'low'
    
    def _assess_quality(self, content: str, source_url: str, doc_type: str) -> float:
        """Assess content quality (0.0 to 1.0)."""
        score = 0.5  # Base score
        
        # Length bonus
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.1
        
        # Code presence bonus
        if '```' in content or 'def ' in content or 'class ' in content:
            score += 0.1
        
        # Official source bonus
        if any(domain in source_url for domain in ['pytorch.org', 'mlflow.org', 'docs.ray.io', 'kserve.github.io']):
            score += 0.2
        
        # Tutorial/guide bonus
        if doc_type in ['tutorial', 'getting_started']:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_authority(self, source_url: str, vendor: str) -> float:
        """Assess source authority (0.0 to 1.0)."""
        authority_domains = {
            'pytorch.org': 1.0,
            'mlflow.org': 1.0,
            'docs.ray.io': 1.0,
            'kserve.github.io': 0.9,
            'docs.aws.amazon.com': 1.0,
            'kubernetes.io': 1.0,
            'docs.docker.com': 1.0
        }
        
        for domain, score in authority_domains.items():
            if domain in source_url:
                return score
        
        return 0.5  # Default authority score
    
    def _analyze_content_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze content characteristics."""
        characteristics = {
            'has_code': '```' in content or 'def ' in content,
            'has_examples': 'example' in content.lower() or 'sample' in content.lower(),
            'has_diagrams': '```mermaid' in content or 'graph' in content.lower(),
            'has_api_references': 'api' in content.lower() or 'reference' in content.lower(),
            'has_tutorials': 'step' in content.lower() or 'tutorial' in content.lower(),
            'content_length': len(content),
            'paragraph_count': len(content.split('\n\n')),
            'sentence_count': len(re.findall(r'[.!?]+', content))
        }
        
        return characteristics
    
    def _split_content_semantically(self, content: str, title: str) -> List[str]:
        """Split content into semantic chunks."""
        # Split by headings first
        sections = re.split(r'\n(#{1,6}\s)', content)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            # If adding this section would exceed max size, start new chunk
            if len(current_chunk) + len(section) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += section
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no good split points, split by paragraphs
        if len(chunks) == 1 and len(chunks[0]) > self.max_chunk_size:
            paragraphs = chunks[0].split('\n\n')
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += paragraph + '\n\n'
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_heading_path(self, content: str) -> str:
        """Extract heading path from chunk content."""
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        if headings:
            return ' > '.join([h[1].strip() for h in headings[:3]])  # First 3 headings
        return ""
    
    def _generate_chunk_id(self, source_url: str, chunk_index: int, content: str) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{hashlib.md5(source_url.encode()).hexdigest()[:8]}_{chunk_index}_{content_hash}"


def chunk_documents_enhanced(documents: List[Dict[str, Any]]) -> List[EnhancedDocumentChunk]:
    """Chunk multiple documents with enhanced metadata."""
    chunker = EnhancedSemanticChunker()
    all_chunks = []
    
    for doc in documents:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error chunking document {doc.get('source_url', 'unknown')}: {e}")
            continue
    
    return all_chunks