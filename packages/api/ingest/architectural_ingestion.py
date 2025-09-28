"""Architectural knowledge ingestion pipeline with fallback strategies."""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass

from .semantic_chunker import SemanticChunker, ArchitecturalChunk
from .advanced_retrieval import AdvancedRetriever
from .upsert import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class ArchitecturalDocument:
    """Represents an architectural document with rich metadata."""
    content: str
    title: str
    source_url: str
    doc_type: str
    topics: List[str]
    priority: str
    metadata: Dict[str, Any]

class ArchitecturalIngestionPipeline:
    """Ingestion pipeline focused on architectural knowledge."""
    
    def __init__(self, data_dir: Path, collection_name: str = "architectural_knowledge"):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.chunker = SemanticChunker()
        self.db_manager = DatabaseManager(data_dir, collection_name)
        
        # Load data sources configuration
        self.data_sources = self._load_data_sources()
        
        # Create curated content if it doesn't exist
        self._create_curated_content()
    
    def _load_data_sources(self) -> Dict[str, Any]:
        """Load the architectural data sources configuration."""
        
        sources_file = Path(__file__).parent / "architectural_data_sources.yaml"
        
        if not sources_file.exists():
            logger.warning("Data sources file not found, using default configuration")
            return self._get_default_sources()
        
        with open(sources_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_sources(self) -> Dict[str, Any]:
        """Get default data sources when configuration file is missing."""
        
        return {
            'tier1_core_technical': {
                'pytorch': [
                    {
                        'name': 'PyTorch Distributed Training',
                        'source': 'https://pytorch.org/docs/stable/distributed.html',
                        'type': 'api_reference',
                        'topics': ['distributed_training', 'ddp', 'performance'],
                        'priority': 'high'
                    }
                ]
            },
            'alternative_sources': {
                'curated_content': [
                    {
                        'name': 'MLOps Best Practices',
                        'path': './docs/curated/mlops_best_practices.md',
                        'type': 'curated',
                        'topics': ['mlops', 'best_practices', 'architecture'],
                        'priority': 'high'
                    }
                ]
            }
        }
    
    def _create_curated_content(self):
        """Create curated architectural content as fallback."""
        
        curated_dir = self.data_dir / "curated"
        curated_dir.mkdir(parents=True, exist_ok=True)
        
        # Create MLOps best practices document
        mlops_best_practices = curated_dir / "mlops_best_practices.md"
        if not mlops_best_practices.exists():
            self._write_mlops_best_practices(mlops_best_practices)
        
        # Create architecture decision records
        adr_file = curated_dir / "architecture_decisions.md"
        if not adr_file.exists():
            self._write_architecture_decisions(adr_file)
    
    def _write_mlops_best_practices(self, file_path: Path):
        """Write MLOps best practices curated content."""
        
        content = """# MLOps Best Practices for Production ML Systems

## Architecture Patterns

### Model Serving Architecture

**When to use Ray Serve:**
- Need for high-throughput, low-latency serving
- Complex model pipelines with preprocessing/postprocessing
- A/B testing and canary deployments
- Integration with existing Ray ecosystem

**When to use KServe:**
- Kubernetes-native deployment requirements
- Need for standardized inference protocols (Seldon, TensorFlow Serving)
- Multi-cloud or hybrid cloud deployments
- Compliance and governance requirements

### Data Pipeline Architecture

**PyTorch DataLoader Best Practices:**
- Use `num_workers > 0` for I/O bound workloads
- Pin memory with `pin_memory=True` for GPU training
- Use `persistent_workers=True` to avoid worker restart overhead
- Implement proper error handling and retry logic

**MLflow Experiment Tracking:**
- Log all hyperparameters, metrics, and artifacts
- Use nested runs for complex experiments
- Implement proper run naming conventions
- Set up automated model validation gates

### Monitoring and Observability

**Key Metrics to Track:**
- Model performance metrics (accuracy, precision, recall)
- Data drift detection (statistical tests, distribution changes)
- Infrastructure metrics (CPU, memory, GPU utilization)
- Business metrics (conversion rates, user engagement)

**Alerting Strategy:**
- Set up alerts for model performance degradation
- Monitor data quality and pipeline health
- Track deployment success rates
- Monitor resource utilization and costs

## Decision Framework

### Technology Selection Criteria

1. **Scalability Requirements**
   - Expected request volume
   - Latency requirements
   - Geographic distribution

2. **Operational Complexity**
   - Team expertise level
   - Maintenance overhead
   - Integration requirements

3. **Cost Considerations**
   - Infrastructure costs
   - Development time
   - Operational overhead

### Common Anti-Patterns to Avoid

1. **Over-engineering early-stage systems**
2. **Ignoring data quality and monitoring**
3. **Tight coupling between components**
4. **Lack of proper testing and validation**
5. **Insufficient documentation and runbooks**
"""
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _write_architecture_decisions(self, file_path: Path):
        """Write architecture decision records."""
        
        content = """# Architecture Decision Records (ADRs)

## ADR-001: Model Serving Platform Selection

**Status:** Accepted
**Date:** 2024-01-15

### Context
We need to choose between Ray Serve and KServe for model serving in our ML platform.

### Decision
Use Ray Serve for high-throughput serving and KServe for Kubernetes-native deployments.

### Rationale
- Ray Serve provides better performance for high-throughput scenarios
- KServe offers better Kubernetes integration and standardization
- Both can coexist in the same architecture for different use cases

### Consequences
- Need to maintain expertise in both platforms
- More complex deployment pipeline
- Better flexibility for different serving requirements

## ADR-002: Experiment Tracking Tool

**Status:** Accepted
**Date:** 2024-01-20

### Context
Select an experiment tracking solution for our ML workflows.

### Decision
Use MLflow as the primary experiment tracking tool.

### Rationale
- Comprehensive feature set (tracking, registry, deployment)
- Strong Python ecosystem integration
- Good balance of simplicity and functionality
- Active community and development

### Consequences
- Standardized experiment tracking across teams
- Need for MLflow expertise
- Potential vendor lock-in considerations

## ADR-003: Distributed Training Strategy

**Status:** Accepted
**Date:** 2024-01-25

### Context
Choose between PyTorch DDP and Ray Train for distributed training.

### Decision
Use PyTorch DDP for model training and Ray Train for hyperparameter optimization.

### Rationale
- PyTorch DDP provides better performance for single-model training
- Ray Train excels at hyperparameter optimization and multi-model training
- Clear separation of concerns between training and optimization

### Consequences
- Need to maintain both PyTorch and Ray expertise
- More complex training pipeline
- Better performance for both use cases
"""
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    async def ingest_architectural_knowledge(self, clear_existing: bool = True) -> Dict[str, Any]:
        """Main ingestion pipeline for architectural knowledge."""
        
        logger.info("🚀 Starting architectural knowledge ingestion...")
        
        results = {
            'documents_processed': 0,
            'chunks_created': 0,
            'errors': []
        }
        
        try:
            # Initialize database
            if clear_existing:
                logger.info("🗑️  Clearing existing data...")
                self.db_manager.initialize_chromadb()
                self.db_manager.initialize_sqlite()
                self.db_manager.clear_collections()
            
            # Process each tier of data sources
            for tier_name, tier_sources in self.data_sources.items():
                if tier_name.startswith('tier'):
                    logger.info(f"📚 Processing {tier_name}...")
                    
                    for category, sources in tier_sources.items():
                        logger.info(f"  📂 Processing {category}...")
                        
                        for source in sources:
                            try:
                                doc = await self._process_source(source)
                                if doc:
                                    chunks = self._chunk_document(doc)
                                    await self._store_chunks(chunks)
                                    
                                    results['documents_processed'] += 1
                                    results['chunks_created'] += len(chunks)
                                    
                                    logger.info(f"    ✅ Processed: {doc.title}")
                                
                            except Exception as e:
                                error_msg = f"Failed to process {source.get('name', 'unknown')}: {e}"
                                logger.error(error_msg)
                                results['errors'].append(error_msg)
            
            # Process alternative sources (curated content)
            logger.info("📖 Processing curated content...")
            curated_sources = self.data_sources.get('alternative_sources', {}).get('curated_content', [])
            
            for source in curated_sources:
                try:
                    doc = await self._process_curated_source(source)
                    if doc:
                        chunks = self._chunk_document(doc)
                        await self._store_chunks(chunks)
                        
                        results['documents_processed'] += 1
                        results['chunks_created'] += len(chunks)
                        
                        logger.info(f"  ✅ Processed curated: {doc.title}")
                
                except Exception as e:
                    error_msg = f"Failed to process curated content {source.get('name', 'unknown')}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            logger.info("✅ Architectural knowledge ingestion completed!")
            logger.info(f"📊 Results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ingestion failed: {e}")
            results['errors'].append(str(e))
            return results
    
    async def _process_source(self, source: Dict[str, Any]) -> Optional[ArchitecturalDocument]:
        """Process a single data source."""
        
        source_url = source.get('source', '')
        source_type = source.get('type', '')
        
        # Try web scraping first
        if source_url.startswith('http'):
            try:
                content = await self._scrape_web_content(source_url)
                if content:
                    return ArchitecturalDocument(
                        content=content,
                        title=source.get('name', ''),
                        source_url=source_url,
                        doc_type=source_type,
                        topics=source.get('topics', []),
                        priority=source.get('priority', 'medium'),
                        metadata={
                            'source_category': 'web_scraped',
                            'scraping_success': True
                        }
                    )
            except Exception as e:
                logger.warning(f"Web scraping failed for {source_url}: {e}")
        
        # Fallback to curated content or local files
        return await self._process_fallback_source(source)
    
    async def _scrape_web_content(self, url: str) -> Optional[str]:
        """Scrape web content with proper headers and error handling."""
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text if len(text) > 100 else None
            
        except Exception as e:
            logger.warning(f"Failed to scrape {url}: {e}")
            return None
    
    async def _process_fallback_source(self, source: Dict[str, Any]) -> Optional[ArchitecturalDocument]:
        """Process fallback source (local files, curated content)."""
        
        # This would implement fallback strategies
        # For now, return None to indicate no content available
        return None
    
    async def _process_curated_source(self, source: Dict[str, Any]) -> Optional[ArchitecturalDocument]:
        """Process curated content source."""
        
        file_path = Path(source.get('path', ''))
        
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return ArchitecturalDocument(
                    content=content,
                    title=source.get('name', file_path.stem),
                    source_url=str(file_path),
                    doc_type=source.get('type', 'curated'),
                    topics=source.get('topics', []),
                    priority=source.get('priority', 'high'),
                    metadata={
                        'source_category': 'curated',
                        'file_path': str(file_path)
                    }
                )
            except Exception as e:
                logger.error(f"Failed to read curated file {file_path}: {e}")
        
        return None
    
    def _chunk_document(self, doc: ArchitecturalDocument) -> List[ArchitecturalChunk]:
        """Chunk an architectural document using semantic chunking."""
        
        document_dict = {
            'content': doc.content,
            'url': doc.source_url,
            'title': doc.title,
            'id': f"{doc.title}_{hash(doc.source_url)}"
        }
        
        return self.chunker.chunk_document(document_dict)
    
    async def _store_chunks(self, chunks: List[ArchitecturalChunk]):
        """Store chunks in the database."""
        
        if not chunks:
            return
        
        # Convert to the format expected by the database manager
        from .chunker import DocumentChunk
        
        chunk_objects = []
        for chunk in chunks:
            chunk_obj = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_url=chunk.source_url,
                title=chunk.title,
                heading_path=chunk.heading_path,
                anchor_link="",
                token_count=chunk.token_count,
                prev_id=chunk.prev_id,
                next_id=chunk.next_id
            )
            chunk_objects.append(chunk_obj)
        
        # Store in database
        self.db_manager.upsert_chunks(chunk_objects)

async def main():
    """Main function to run the architectural ingestion pipeline."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize pipeline
    data_dir = Path('./data')
    pipeline = ArchitecturalIngestionPipeline(data_dir)
    
    # Run ingestion
    results = await pipeline.ingest_architectural_knowledge(clear_existing=True)
    
    print("🎉 Architectural knowledge ingestion completed!")
    print(f"📊 Results: {results}")

if __name__ == '__main__':
    asyncio.run(main())
