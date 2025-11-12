#!/usr/bin/env python3
"""
Production-grade ingestion pipeline for ML Documentation Copilot.
Focuses on data quality, comprehensive coverage, and reliable processing.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

from production_scraper import ProductionScraper, ScrapedDocument
from production_chunker import ProductionChunker, ProductionChunk
from upsert import DatabaseManager
from vertex_embedder import VertexAIEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionIngestionPipeline:
    """Production-grade ingestion pipeline with quality controls."""
    
    def __init__(self, data_dir: Path, collection_name: str = "ml_docs_production"):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.scraper = None
        self.chunker = ProductionChunker(max_chunk_size=800, overlap=100)
        # Use Vertex AI embedder to match query embeddings
        self.embedder = VertexAIEmbedder()
        self.db_manager = DatabaseManager(data_dir, collection_name)
        
        # Quality metrics
        self.quality_metrics = {
            'total_documents': 0,
            'high_quality_documents': 0,
            'total_chunks': 0,
            'high_quality_chunks': 0,
            'vendors_covered': set(),
            'topics_covered': set(),
            'chunk_types': {},
            'quality_distribution': {}
        }
    
    async def run_full_ingestion(self, sources_config: Path, clear_existing: bool = True):
        """Run the complete production ingestion pipeline."""
        logger.info("🚀 Starting Production ML Documentation Ingestion")
        
        try:
            # Step 1: Scrape high-quality documentation
            logger.info("📥 Step 1: Scraping documentation sources...")
            documents = await self._scrape_documentation(sources_config)
            
            # Step 2: Chunk documents with quality controls
            logger.info("✂️ Step 2: Chunking documents with quality controls...")
            all_chunks = await self._chunk_documents(documents)
            
            # Step 3: Quality filtering and validation
            logger.info("🔍 Step 3: Quality filtering and validation...")
            filtered_chunks = self._filter_high_quality_chunks(all_chunks)
            
            # Step 4: Generate embeddings
            logger.info("🧠 Step 4: Generating embeddings...")
            await self._generate_embeddings(filtered_chunks)
            
            # Step 5: Store in databases
            logger.info("💾 Step 5: Storing in databases...")
            await self._store_chunks(filtered_chunks, clear_existing)
            
            # Step 6: Generate quality report
            logger.info("📊 Step 6: Generating quality report...")
            self._generate_quality_report()
            
            logger.info("✅ Production ingestion completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Production ingestion failed: {e}")
            raise
    
    async def _scrape_documentation(self, sources_config: Path) -> List[ScrapedDocument]:
        """Scrape documentation with quality controls."""
        async with ProductionScraper(sources_config, self.data_dir / "scraped_data") as scraper:
            documents = await scraper.scrape_all_sources()
            
            # Update metrics
            self.quality_metrics['total_documents'] = len(documents)
            self.quality_metrics['high_quality_documents'] = len([d for d in documents if d.quality_score > 0.7])
            
            # Track vendors and topics
            for doc in documents:
                self.quality_metrics['vendors_covered'].add(doc.source_vendor)
                self.quality_metrics['topics_covered'].update(doc.topics)
            
            logger.info(f"📥 Scraped {len(documents)} documents from {len(self.quality_metrics['vendors_covered'])} vendors")
            return documents
    
    async def _chunk_documents(self, documents: List[ScrapedDocument]) -> List[ProductionChunk]:
        """Chunk documents with production-grade quality controls."""
        all_chunks = []
        
        for doc in documents:
            try:
                # Convert ScrapedDocument to dict for chunker
                doc_dict = {
                    'url': doc.url,
                    'title': doc.title,
                    'content': doc.content,
                    'source_vendor': doc.source_vendor,
                    'doc_type': doc.doc_type,
                    'topics': doc.topics,
                    'priority': doc.priority,
                    'quality_score': doc.quality_score,
                    'heading_structure': doc.heading_structure,
                    'metadata': doc.metadata
                }
                
                chunks = self.chunker.chunk_document(doc_dict)
                all_chunks.extend(chunks)
                
                # Update metrics
                self.quality_metrics['total_chunks'] += len(chunks)
                self.quality_metrics['high_quality_chunks'] += len([c for c in chunks if c.quality_score > 0.7])
                
                # Track chunk types
                for chunk in chunks:
                    chunk_type = chunk.chunk_type
                    self.quality_metrics['chunk_types'][chunk_type] = self.quality_metrics['chunk_types'].get(chunk_type, 0) + 1
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.title}: {e}")
                continue
        
        logger.info(f"✂️ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _filter_high_quality_chunks(self, chunks: List[ProductionChunk]) -> List[ProductionChunk]:
        """Filter chunks by quality standards."""
        filtered_chunks = []
        
        for chunk in chunks:
            # Quality filters
            if (chunk.quality_score < 0.4 or
                chunk.word_count < 20 or
                chunk.word_count > 2000 or
                not chunk.content.strip()):
                continue
            
            # Content quality checks
            if self._is_high_quality_chunk(chunk):
                filtered_chunks.append(chunk)
        
        logger.info(f"🔍 Filtered to {len(filtered_chunks)} high-quality chunks from {len(chunks)} total")
        return filtered_chunks
    
    def _is_high_quality_chunk(self, chunk: ProductionChunk) -> bool:
        """Check if a chunk meets high quality standards."""
        # Must have substantial content
        if chunk.word_count < 30:
            return False
        
        # Must have good quality score
        if chunk.quality_score < 0.5:
            return False
        
        # Prefer chunks with code examples or practical content
        if chunk.has_code or chunk.has_examples:
            return True
        
        # Prefer chunks with good structure
        if chunk.heading_path and len(chunk.content.split('\n')) > 3:
            return True
        
        # Accept high-quality conceptual content
        return chunk.quality_score > 0.7
    
    async def _generate_embeddings(self, chunks: List[ProductionChunk]):
        """Generate embeddings for chunks."""
        logger.info(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        
        # This would be handled by the DatabaseManager during upsert
        # We're just logging the process here
        pass
    
    async def _store_chunks(self, chunks: List[ProductionChunk], clear_existing: bool):
        """Store chunks in databases with quality tracking."""
        logger.info(f"💾 Storing {len(chunks)} chunks in databases...")
        
        try:
            # Convert ProductionChunks to format expected by DatabaseManager
            chunk_objects = []
            for chunk in chunks:
                chunk_obj = type('Chunk', (), {
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'source_url': chunk.source_url,
                    'title': chunk.title,
                    'token_count': chunk.word_count,
                    'prev_id': chunk.prev_chunk_id,
                    'next_id': chunk.next_chunk_id
                })()
                chunk_objects.append(chunk_obj)
            
            # Store in databases
            self.db_manager.upsert_chunks(chunk_objects)
            
            logger.info("✅ Successfully stored chunks in databases")
            
        except Exception as e:
            logger.error(f"❌ Error storing chunks: {e}")
            raise
    
    def _generate_quality_report(self):
        """Generate comprehensive quality report."""
        report = {
            'ingestion_summary': {
                'total_documents': self.quality_metrics['total_documents'],
                'high_quality_documents': self.quality_metrics['high_quality_documents'],
                'total_chunks': self.quality_metrics['total_chunks'],
                'high_quality_chunks': self.quality_metrics['high_quality_chunks'],
                'quality_ratio': self.quality_metrics['high_quality_chunks'] / max(self.quality_metrics['total_chunks'], 1)
            },
            'coverage': {
                'vendors_covered': list(self.quality_metrics['vendors_covered']),
                'topics_covered': list(self.quality_metrics['topics_covered']),
                'chunk_types': self.quality_metrics['chunk_types']
            },
            'quality_metrics': {
                'average_quality_score': 0.0,  # Would calculate from actual chunks
                'code_example_ratio': 0.0,   # Would calculate from actual chunks
                'tutorial_ratio': 0.0        # Would calculate from actual chunks
            }
        }
        
        # Save report
        report_file = self.data_dir / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Quality report saved to {report_file}")
        logger.info(f"📈 Quality Summary:")
        logger.info(f"   - Documents: {report['ingestion_summary']['total_documents']}")
        logger.info(f"   - High Quality: {report['ingestion_summary']['high_quality_documents']}")
        logger.info(f"   - Chunks: {report['ingestion_summary']['total_chunks']}")
        logger.info(f"   - High Quality Chunks: {report['ingestion_summary']['high_quality_chunks']}")
        logger.info(f"   - Vendors: {len(report['coverage']['vendors_covered'])}")
        logger.info(f"   - Topics: {len(report['coverage']['topics_covered'])}")

async def main():
    """Run production ingestion pipeline."""
    # Configuration
    data_dir = Path("./data")
    sources_config = Path(__file__).parent / "production_sources.yaml"
    collection_name = "ml_docs_production"
    
    # Initialize pipeline
    pipeline = ProductionIngestionPipeline(data_dir, collection_name)
    
    # Run ingestion
    await pipeline.run_full_ingestion(sources_config, clear_existing=True)

if __name__ == "__main__":
    asyncio.run(main())
