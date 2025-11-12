#!/usr/bin/env python3
"""
Complete End-to-End ML Documentation Ingestion
Includes: Web Scraping → Chunking → Embedding → Qdrant Storage
With strict 4GB size limit
"""

import os
import sys
import asyncio
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

# Setup environment
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from smart_web_scraper import SmartWebScraper, ScrapedPage
from github_doc_scraper import GitHubDocScraper, GitHubDoc
from vertex_embedder import CachedVertexEmbedder
from chunker import DocumentChunk
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Track ingestion statistics."""
    vendor: str
    pages_scraped: int
    chunks_created: int
    size_mb: float
    status: str
    error: Optional[str] = None


class CompleteMLIngestion:
    """
    Complete end-to-end ML documentation ingestion system.
    
    Pipeline:
    1. Web Scraping (with size limits)
    2. Content Chunking
    3. Vertex AI Embedding
    4. Qdrant Storage
    5. Progress Tracking
    """
    
    # Conservative limits to stay under 4GB
    MAX_SOURCES_TO_INGEST = 40  # Ingest 40 of the 60+ sources
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    EMBEDDING_BATCH_SIZE = 10
    
    def __init__(self, sources_file: str, qdrant_url: str, qdrant_api_key: str):
        self.sources_file = sources_file
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        self.scraper = None
        self.github_scraper = GitHubDocScraper()
        self.embedder = CachedVertexEmbedder()
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Load sources
        with open(sources_file, 'r') as f:
            self.sources_config = yaml.safe_load(f)
        
        # Stats
        self.stats: List[IngestionStats] = []
        self.total_chunks = 0
        
        logger.info(f"Initialized Complete ML Ingestion - Run ID: {self.run_id}")
    
    def _create_chunks_from_page(
        self, 
        page: ScrapedPage, 
        vendor: str, 
        tier: str
    ) -> List[DocumentChunk]:
        """Convert scraped page to chunks."""
        chunks = []
        
        # Combine content with code blocks
        full_content = page.content
        if page.code_blocks:
            full_content += "\n\nCode Examples:\n" + "\n\n".join(page.code_blocks[:5])
        
        # Split into chunks
        words = full_content.split()
        for i in range(0, len(words), self.CHUNK_SIZE - self.CHUNK_OVERLAP):
            chunk_words = words[i:i + self.CHUNK_SIZE]
            chunk_content = ' '.join(chunk_words)
            
            if len(chunk_content) < 100:  # Skip very small chunks
                continue
            
            chunk_id = f"{vendor.lower().replace(' ', '_')}_{hash(page.url) % 100000}_{i}"
            
            # Create heading path
            heading_path = " > ".join(page.headings[:3]) if page.headings else vendor
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                source_url=page.url,
                title=page.title,
                heading_path=heading_path,
                anchor_link=page.url,
                token_count=len(chunk_words),
                prev_id=None,
                next_id=None
            )
            chunks.append(chunk)
        
        # Link chunks
        for j in range(len(chunks) - 1):
            chunks[j].next_id = chunks[j + 1].chunk_id
        for j in range(1, len(chunks)):
            chunks[j].prev_id = chunks[j - 1].chunk_id
        
        return chunks
    
    def _convert_github_doc_to_page(self, doc: GitHubDoc) -> ScrapedPage:
        """Convert GitHubDoc to ScrapedPage format for compatibility."""
        return ScrapedPage(
            url=doc.url,
            title=doc.title,
            content=doc.content,
            headings=[doc.title],
            code_blocks=[],
            links=[],
            size_bytes=len(doc.content.encode('utf-8'))
        )
    
    def ingest_github_source(
        self,
        repo_url: str,
        vendor: str,
        tier: str,
        base_url: str,
        sparse_paths: Optional[List[str]] = None
    ) -> IngestionStats:
        """Ingest documentation from a GitHub repository."""
        logger.info(f"[{tier}] Starting GitHub ingestion: {vendor} from {repo_url}")
        
        try:
            # Clone and parse GitHub docs
            clone_path = self.github_scraper.clone_repo(repo_url, sparse_paths)
            
            # Determine which parsing method to use
            if "tutorials" in repo_url.lower():
                github_docs = self.github_scraper.scrape_pytorch_tutorials()
            else:
                github_docs = self.github_scraper.scrape_pytorch_docs()
            
            if not github_docs:
                logger.warning(f"[{tier}] No documents found in {repo_url}")
                return IngestionStats(
                    vendor=vendor,
                    pages_scraped=0,
                    chunks_created=0,
                    size_mb=0,
                    status='failed',
                    error='No documents found'
                )
            
            logger.info(f"[{tier}] {vendor}: Found {len(github_docs)} documents")
            
            # Convert GitHub docs to ScrapedPage format
            pages = [self._convert_github_doc_to_page(doc) for doc in github_docs]
            
            # Step 2: Create chunks
            all_chunks = []
            for page in pages:
                chunks = self._create_chunks_from_page(page, vendor, tier)
                all_chunks.extend(chunks)
            
            logger.info(f"[{tier}] {vendor}: Created {len(all_chunks)} chunks")
            
            # Step 3: Generate embeddings and upload to Qdrant in batches
            for i in range(0, len(all_chunks), self.EMBEDDING_BATCH_SIZE):
                batch = all_chunks[i:i + self.EMBEDDING_BATCH_SIZE]
                
                # Generate embeddings
                texts = [chunk.content for chunk in batch]
                embeddings = self.embedder.encode_batch(texts)
                
                # Create Qdrant points
                points = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    point = qmodels.PointStruct(
                        id=hash(chunk.chunk_id) % (2**63),
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk.content,
                            "source": chunk.source_url,
                            "title": chunk.title,
                            "vendor": vendor,
                            "heading_path": chunk.heading_path,
                            "tier": tier
                        }
                    )
                    points.append(point)
                
                # Upload batch to Qdrant
                self.qdrant_client.upsert(
                    collection_name="ml_docs",
                    points=points
                )
                
                logger.info(f"[{tier}] {vendor}: Uploaded batch {i//self.EMBEDDING_BATCH_SIZE + 1} ({len(batch)} chunks)")
            
            # Calculate size
            total_size = sum(len(page.content.encode('utf-8')) for page in pages)
            size_mb = total_size / (1024 * 1024)
            
            self.total_chunks += len(all_chunks)
            
            logger.info(f"[{tier}] {vendor}: SUCCESS - {len(pages)} pages, {len(all_chunks)} chunks, {size_mb:.2f}MB")
            
            return IngestionStats(
                vendor=vendor,
                pages_scraped=len(pages),
                chunks_created=len(all_chunks),
                size_mb=size_mb,
                status='success'
            )
            
        except Exception as e:
            logger.error(f"[{tier}] {vendor}: FAILED - {str(e)}", exc_info=True)
            return IngestionStats(
                vendor=vendor,
                pages_scraped=0,
                chunks_created=0,
                size_mb=0,
                status='failed',
                error=str(e)
            )
    
    async def ingest_source(
        self,
        url: str,
        vendor: str,
        tier: str,
        max_depth: int
    ) -> IngestionStats:
        """Ingest a single documentation source."""
        logger.info(f"[{tier}] Starting ingestion: {vendor}")
        
        try:
            # Step 1: Scrape documentation
            pages = await self.scraper.scrape_documentation(url, vendor, max_depth)
            
            if not pages:
                logger.warning(f"[{tier}] No pages scraped for {vendor}")
                return IngestionStats(
                    vendor=vendor,
                    pages_scraped=0,
                    chunks_created=0,
                    size_mb=0,
                    status='failed',
                    error='No pages scraped'
                )
            
            logger.info(f"[{tier}] {vendor}: Scraped {len(pages)} pages")
            
            # Step 2: Create chunks
            all_chunks = []
            for page in pages:
                chunks = self._create_chunks_from_page(page, vendor, tier)
                all_chunks.extend(chunks)
            
            logger.info(f"[{tier}] {vendor}: Created {len(all_chunks)} chunks")
            
            # Step 3: Generate embeddings and upload to Qdrant in batches
            for i in range(0, len(all_chunks), self.EMBEDDING_BATCH_SIZE):
                batch = all_chunks[i:i + self.EMBEDDING_BATCH_SIZE]
                
                # Generate embeddings
                texts = [chunk.content for chunk in batch]
                embeddings = self.embedder.encode_batch(texts)
                
                # Create Qdrant points
                points = []
                for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    point = qmodels.PointStruct(
                        id=hash(chunk.chunk_id) % (2**63),
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk.content,
                            "metadata": {
                                "chunk_id": chunk.chunk_id,
                                "source_url": chunk.source_url,
                                "title": chunk.title,
                                "heading_path": chunk.heading_path,
                                "anchor_link": chunk.anchor_link,
                                "vendor": vendor,
                                "tier": tier,
                                "token_count": chunk.token_count,
                                "prev_id": chunk.prev_id or "",
                                "next_id": chunk.next_id or ""
                            }
                        }
                    )
                    points.append(point)
                
                # Upload batch to Qdrant
                self.qdrant_client.upsert(
                    collection_name="ml_docs",
                    points=points,
                    wait=True
                )
                
                logger.info(
                    f"[{tier}] {vendor}: Uploaded batch {i // self.EMBEDDING_BATCH_SIZE + 1} "
                    f"({len(points)} chunks)"
                )
            
            # Calculate size
            source_size_mb = self.scraper.source_sizes.get(vendor, 0) / 1024 / 1024
            
            # Update stats
            self.total_chunks += len(all_chunks)
            
            logger.info(
                f"[{tier}] {vendor}: SUCCESS - {len(pages)} pages, "
                f"{len(all_chunks)} chunks, {source_size_mb:.2f}MB"
            )
            
            return IngestionStats(
                vendor=vendor,
                pages_scraped=len(pages),
                chunks_created=len(all_chunks),
                size_mb=source_size_mb,
                status='success'
            )
            
        except Exception as e:
            logger.error(f"[{tier}] {vendor}: FAILED - {e}", exc_info=True)
            return IngestionStats(
                vendor=vendor,
                pages_scraped=0,
                chunks_created=0,
                size_mb=0,
                status='failed',
                error=str(e)
            )
    
    async def run_complete_ingestion(
        self,
        tier_filter: Optional[str] = None,
        max_sources: Optional[int] = None
    ):
        """Run complete end-to-end ingestion."""
        logger.info(f"\n{'='*60}")
        logger.info(f"COMPLETE END-TO-END ML DOCUMENTATION INGESTION")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Max Size: {SmartWebScraper.MAX_TOTAL_SIZE_GB}GB")
        logger.info(f"{'='*60}\n")
        
        # Initialize scraper
        async with SmartWebScraper() as scraper:
            self.scraper = scraper
            
            # Setup Qdrant collection
            self._setup_qdrant_collection()
            
            # Collect sources to ingest
            sources_to_ingest = []
            source_count = 0
            max_to_ingest = max_sources or self.MAX_SOURCES_TO_INGEST
            
            for tier_name, tier_data in self.sources_config.items():
                if tier_filter and tier_name != tier_filter:
                    continue
                
                # Handle list-based tier structure (like pytorch_github_sources.yaml)
                if isinstance(tier_data, list):
                    for source_item in tier_data:
                        if source_count >= max_to_ingest:
                            break
                        
                        vendor = source_item.get('name', source_item.get('vendor', 'Unknown'))
                        source_type = source_item.get('source_type', 'web')
                        
                        if source_type == 'github':
                            sources_to_ingest.append({
                                'type': 'github',
                                'repo_url': source_item.get('repo_url'),
                                'base_url': source_item.get('base_url'),
                                'sparse_paths': source_item.get('sparse_paths'),
                                'vendor': vendor,
                                'tier': tier_name,
                                'priority': 'high'
                            })
                        else:
                            url = source_item.get('url')
                            if url:
                                sources_to_ingest.append({
                                    'type': 'web',
                                    'url': url,
                                    'vendor': vendor,
                                    'tier': tier_name,
                                    'max_depth': source_item.get('max_depth', 1),
                                    'priority': 'medium'
                                })
                        source_count += 1
                
                # Handle dict-based tier structure (like comprehensive_ml_sources.yaml)
                elif isinstance(tier_data, dict):
                    for category_name, category_data in tier_data.items():
                        vendor = category_data.get('name', category_name)
                        priority = category_data.get('priority', 'medium')
                        sources = category_data.get('sources', [])
                        
                        for source in sources:
                            if source_count >= max_to_ingest:
                                break
                            
                            url = source.get('url')
                            max_depth = source.get('max_depth', 1)  # Default to 1 for size control
                            
                            if url:
                                sources_to_ingest.append({
                                    'type': 'web',
                                    'url': url,
                                    'vendor': vendor,
                                    'tier': tier_name,
                                    'max_depth': max_depth,
                                    'priority': priority
                                })
                                source_count += 1
                        
                        if source_count >= max_to_ingest:
                            break
                    
                    if source_count >= max_to_ingest:
                        break
            
            logger.info(f"Selected {len(sources_to_ingest)} sources to ingest\n")
            
            # Ingest sources sequentially (for better control)
            for idx, source_info in enumerate(sources_to_ingest, 1):
                logger.info(f"\n--- Source {idx}/{len(sources_to_ingest)} ---")
                
                # Handle GitHub sources
                if source_info.get('type') == 'github':
                    stat = self.ingest_github_source(
                        repo_url=source_info['repo_url'],
                        vendor=source_info['vendor'],
                        tier=source_info['tier'],
                        base_url=source_info['base_url'],
                        sparse_paths=source_info.get('sparse_paths')
                    )
                # Handle web scraping sources
                else:
                    stat = await self.ingest_source(
                        url=source_info['url'],
                        vendor=source_info['vendor'],
                        tier=source_info['tier'],
                        max_depth=source_info['max_depth']
                    )
                
                self.stats.append(stat)
                
                # Check size limits
                size_report = scraper.get_size_report()
                if size_report['utilization_percent'] > 90:
                    logger.warning(
                        f"Approaching size limit ({size_report['utilization_percent']:.1f}%), "
                        f"stopping ingestion"
                    )
                    break
            
            # Print final summary
            self._print_summary(scraper)
    
    def _setup_qdrant_collection(self):
        """Setup Qdrant collection."""
        try:
            # Try to get existing collection
            self.qdrant_client.get_collection("ml_docs")
            logger.info("Using existing Qdrant collection 'ml_docs'")
        except:
            # Create new collection
            self.qdrant_client.create_collection(
                collection_name="ml_docs",
                vectors_config=qmodels.VectorParams(
                    size=768,  # Vertex AI embedding dimension
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info("Created new Qdrant collection 'ml_docs'")
    
    def _print_summary(self, scraper: SmartWebScraper):
        """Print final ingestion summary."""
        print("\n" + "="*60)
        print("COMPLETE INGESTION SUMMARY")
        print("="*60)
        
        print(f"\nRun ID: {self.run_id}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall stats
        successful = sum(1 for s in self.stats if s.status == 'success')
        failed = sum(1 for s in self.stats if s.status == 'failed')
        total_pages = sum(s.pages_scraped for s in self.stats)
        
        print(f"\nSources:")
        print(f"  SUCCESS: {successful}")
        print(f"  FAILED: {failed}")
        print(f"  TOTAL: {len(self.stats)}")
        
        print(f"\nContent:")
        print(f"  Pages Scraped: {total_pages}")
        print(f"  Chunks Created: {self.total_chunks}")
        
        # Size report
        size_report = scraper.get_size_report()
        print(f"\nSize Usage:")
        print(f"  Total: {size_report['total_size_gb']:.2f}GB / {size_report['max_size_gb']}GB")
        print(f"  Utilization: {size_report['utilization_percent']:.1f}%")
        print(f"  Pages: {size_report['total_pages']}")
        
        # Top sources by size
        print(f"\nTop Sources by Size:")
        sources_sorted = sorted(
            size_report['sources'].items(),
            key=lambda x: x[1]['size_mb'],
            reverse=True
        )[:10]
        
        for vendor, info in sources_sorted:
            print(f"  {vendor}: {info['size_mb']:.2f}MB")
        
        # Qdrant stats
        try:
            collection_info = self.qdrant_client.get_collection("ml_docs")
            print(f"\nQdrant Collection:")
            print(f"  Total Points: {collection_info.points_count}")
            print(f"  Status: {collection_info.status}")
        except Exception as e:
            print(f"\nQdrant Collection: Error - {e}")
        
        # Save detailed report
        report_file = f"complete_ingestion_report_{self.run_id}.json"
        report = {
            'run_id': self.run_id,
            'completed_at': datetime.now().isoformat(),
            'sources': [
                {
                    'vendor': s.vendor,
                    'pages': s.pages_scraped,
                    'chunks': s.chunks_created,
                    'size_mb': s.size_mb,
                    'status': s.status,
                    'error': s.error
                }
                for s in self.stats
            ],
            'totals': {
                'sources': len(self.stats),
                'successful': successful,
                'failed': failed,
                'total_pages': total_pages,
                'total_chunks': self.total_chunks,
                'total_size_gb': size_report['total_size_gb'],
                'utilization_percent': size_report['utilization_percent']
            },
            'size_report': size_report
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed Report: {report_file}")
        print(f"Logs: complete_ingestion.log")
        
        print("\n" + "="*60)
        print("INGESTION COMPLETE!")
        print("="*60 + "\n")


async def main():
    """Main entry point."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description='Complete End-to-End ML Documentation Ingestion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest Tier 1 only (Core ML - ~8 sources):
  python complete_end_to_end_ingestion.py --tier tier_1_core_ml
  
  # Ingest first 20 sources:
  python complete_end_to_end_ingestion.py --max-sources 20
  
  # Ingest all (up to 40 sources, ~3.5GB limit):
  python complete_end_to_end_ingestion.py
        """
    )
    
    parser.add_argument('--sources', 
                       default='comprehensive_ml_sources.yaml',
                       help='Path to sources YAML file')
    parser.add_argument('--tier', 
                       default=None,
                       help='Specific tier to ingest')
    parser.add_argument('--max-sources',
                       type=int,
                       default=None,
                       help='Maximum number of sources to ingest')
    
    args = parser.parse_args()
    
    # Get Qdrant credentials
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return
    
    # Run ingestion
    ingestion = CompleteMLIngestion(
        sources_file=args.sources,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    
    await ingestion.run_complete_ingestion(
        tier_filter=args.tier,
        max_sources=args.max_sources
    )


if __name__ == "__main__":
    asyncio.run(main())

