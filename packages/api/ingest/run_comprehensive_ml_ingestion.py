#!/usr/bin/env python3
"""
Complete ML Documentation Ingestion System
Scrapes, chunks, embeds, and stores comprehensive ML documentation
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

# Import local modules
try:
    from comprehensive_ingestion_tracker import IngestionTracker, IngestionStatus
    from vertex_embedder import CachedVertexEmbedder
except ImportError:
    from ingest.comprehensive_ingestion_tracker import IngestionTracker, IngestionStatus
    from ingest.vertex_embedder import CachedVertexEmbedder

# Import from ingest package
try:
    from chunker import DocumentChunk
except ImportError:
    # Try as package import
    import importlib.util
    spec = importlib.util.spec_from_file_location("chunker", os.path.join(os.path.dirname(__file__), "chunker.py"))
    chunker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunker_module)
    DocumentChunk = chunker_module.DocumentChunk

# Import DatabaseManager
try:
    from upsert import DatabaseManager
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location("upsert", os.path.join(os.path.dirname(__file__), "upsert.py"))
    upsert_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upsert_module)
    DatabaseManager = upsert_module.DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveMLDocumentationIngestion:
    """Complete ingestion system for ML documentation."""
    
    def __init__(self, sources_file: str, data_dir: Path):
        self.sources_file = sources_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.tracker = IngestionTracker()
        self.embedder = CachedVertexEmbedder()
        self.db_manager = DatabaseManager(data_dir, collection_name="ml_docs")
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load sources configuration
        with open(sources_file, 'r') as f:
            self.sources_config = yaml.safe_load(f)
        
        logger.info(f"Initialized ingestion system - Run ID: {self.run_id}")
    
    def _create_mock_chunks(self, url: str, vendor: str, tier: str) -> List[DocumentChunk]:
        """
        Create mock chunks for a documentation source.
        TODO: Replace with actual scraping when ready.
        """
        # For now, create sample chunks to demonstrate the system
        chunks = []
        
        base_content = f"""
        Documentation from {vendor}
        Source: {url}
        Tier: {tier}
        
        This is sample content that demonstrates the ingestion system.
        In production, this would be actual scraped documentation content.
        
        Key topics covered:
        - Installation and setup
        - Core concepts
        - API reference
        - Best practices
        - Code examples
        """
        
        for i in range(5):  # Create 5 sample chunks per source
            chunk = DocumentChunk(
                chunk_id=f"{vendor.lower().replace(' ', '_')}_{tier}_{i}_{hash(url) % 10000}",
                content=f"{base_content}\n\nSection {i+1} specific content...",
                source_url=url,
                title=f"{vendor} Documentation - Section {i+1}",
                heading_path=f"{vendor} > Documentation > Section {i+1}",
                anchor_link=f"{url}#section-{i+1}",
                token_count=len(base_content.split()),
                prev_id=None if i == 0 else f"{vendor.lower().replace(' ', '_')}_{tier}_{i-1}_{hash(url) % 10000}",
                next_id=None if i == 4 else f"{vendor.lower().replace(' ', '_')}_{tier}_{i+1}_{hash(url) % 10000}"
            )
            chunks.append(chunk)
        
        # Link chunks
        for i in range(len(chunks) - 1):
            chunks[i].next_id = chunks[i + 1].chunk_id
        for i in range(1, len(chunks)):
            chunks[i].prev_id = chunks[i - 1].chunk_id
        
        return chunks
    
    async def ingest_source(self, url: str, vendor: str, tier: str, 
                           source_type: str, max_depth: int, priority: str) -> IngestionStatus:
        """Ingest a single documentation source."""
        status = IngestionStatus(
            url=url,
            vendor=vendor,
            tier=tier,
            status='in_progress',
            started_at=datetime.now().isoformat()
        )
        self.tracker.update_source_status(status)
        
        try:
            logger.info(f"🔄 [{tier}] Ingesting {vendor}: {url}")
            
            # Step 1: Scrape documentation (using mock data for now)
            # TODO: Integrate with actual scraping system
            await asyncio.sleep(0.1)  # Simulate network delay
            chunks = self._create_mock_chunks(url, vendor, tier)
            
            logger.info(f"  📄 Created {len(chunks)} chunks from {vendor}")
            
            # Step 2: Store chunks in database (with Vertex AI embeddings)
            self.db_manager.upsert_chunks(chunks, batch_size=32)
            
            logger.info(f"  💾 Stored {len(chunks)} chunks in vector DB")
            
            # Update status
            status.status = 'success'
            status.chunks_created = len(chunks)
            status.completed_at = datetime.now().isoformat()
            
            logger.info(f"✅ [{tier}] Successfully ingested {vendor}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ [{tier}] Failed to ingest {vendor} ({url}): {e}", exc_info=True)
            status.status = 'failed'
            status.error_message = str(e)
            status.completed_at = datetime.now().isoformat()
            status.retry_count += 1
        
        self.tracker.update_source_status(status)
        return status
    
    async def ingest_tier(self, tier_name: str, tier_data: Dict[str, Any]) -> None:
        """Ingest all sources in a tier."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 Starting Tier: {tier_name}")
        logger.info(f"{'='*60}\n")
        
        tasks = []
        for category_name, category_data in tier_data.items():
            vendor = category_data.get('name', category_name)
            priority = category_data.get('priority', 'medium')
            sources = category_data.get('sources', [])
            
            for source in sources:
                url = source.get('url')
                source_type = source.get('type', 'guide')
                max_depth = source.get('max_depth', 2)
                
                if url:
                    task = self.ingest_source(
                        url, vendor, tier_name, source_type, max_depth, priority
                    )
                    tasks.append(task)
        
        # Run all sources in this tier concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = sum(1 for r in results if isinstance(r, IngestionStatus) and r.status == 'success')
        failures = sum(1 for r in results if isinstance(r, IngestionStatus) and r.status == 'failed')
        total_chunks = sum(r.chunks_created for r in results if isinstance(r, IngestionStatus))
        
        logger.info(f"\n✅ Tier {tier_name} Complete:")
        logger.info(f"   Success: {successes}/{len(tasks)}")
        logger.info(f"   Failed: {failures}/{len(tasks)}")
        logger.info(f"   Total Chunks: {total_chunks}\n")
    
    async def run_full_ingestion(self, tier_filter: Optional[str] = None) -> None:
        """Run the complete ingestion process."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 COMPREHENSIVE ML DOCUMENTATION INGESTION")
        logger.info(f"   Run ID: {self.run_id}")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")
        
        # Initialize tracking
        total_sources = sum(
            len(category_data.get('sources', []))
            for tier_data in self.sources_config.values()
            for category_data in tier_data.values()
            if not tier_filter or tier_filter in self.sources_config
        )
        self.tracker.start_run(self.run_id, total_sources)
        
        # Initialize all sources as pending
        for tier_name, tier_data in self.sources_config.items():
            if tier_filter and tier_name != tier_filter:
                continue
            
            for category_name, category_data in tier_data.items():
                vendor = category_data.get('name', category_name)
                sources = category_data.get('sources', [])
                
                for source in sources:
                    url = source.get('url')
                    if url:
                        status = IngestionStatus(
                            url=url,
                            vendor=vendor,
                            tier=tier_name,
                            status='pending'
                        )
                        self.tracker.update_source_status(status)
        
        # Ingest tier by tier (sequential for better control)
        for tier_name, tier_data in self.sources_config.items():
            if tier_filter and tier_name != tier_filter:
                continue
            
            await self.ingest_tier(tier_name, tier_data)
        
        # Complete the run
        self.tracker.complete_run(self.run_id)
        
        # Print final summary
        self.print_final_summary()
    
    def print_final_summary(self) -> None:
        """Print final ingestion summary."""
        summary = self.tracker.get_status_summary()
        
        print("\n" + "="*60)
        print("📊 FINAL INGESTION SUMMARY")
        print("="*60)
        
        print(f"\n🆔 Run ID: {self.run_id}")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n📈 Overall Status:")
        total_sources = 0
        total_chunks = 0
        for status, data in summary['status_counts'].items():
            count = data['count']
            chunks = data['chunks'] or 0
            total_sources += count
            total_chunks += chunks
            print(f"  ✓ {status.upper()}: {count} sources ({chunks} chunks)")
        
        print(f"\n📦 Total: {total_sources} sources, {total_chunks} chunks")
        
        print("\n🎯 By Tier:")
        for tier, stats in sorted(summary['tier_stats'].items()):
            print(f"  {tier}:")
            for status, count in stats.items():
                emoji = "✅" if status == "success" else "❌" if status == "failed" else "⏳"
                print(f"    {emoji} {status}: {count}")
        
        # Get database stats
        db_stats = self.db_manager.get_collection_stats()
        print(f"\n💾 Vector Database Stats:")
        for key, value in db_stats.items():
            print(f"  {key}: {value}")
        
        # Export detailed report
        report_file = f"ingestion_report_{self.run_id}.json"
        self.tracker.export_report(report_file)
        print(f"\n📄 Detailed Report: {report_file}")
        print(f"📋 Tracking DB: {self.tracker.db_path}")
        print(f"📝 Logs: comprehensive_ingestion.log")
        
        print("\n" + "="*60)
        print("✨ Ingestion Complete!")
        print("="*60 + "\n")
    
    def close(self):
        """Clean up resources."""
        self.tracker.close()
        self.db_manager.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive ML Documentation Ingestion System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all documentation:
  python run_comprehensive_ml_ingestion.py
  
  # Ingest only Tier 1 (Core ML):
  python run_comprehensive_ml_ingestion.py --tier tier_1_core_ml
  
  # Check current status:
  python run_comprehensive_ml_ingestion.py --status
        """
    )
    
    parser.add_argument('--sources', 
                       default='comprehensive_ml_sources.yaml',
                       help='Path to sources YAML file')
    parser.add_argument('--data-dir', 
                       default='../../data',
                       help='Data directory for storage')
    parser.add_argument('--tier', 
                       default=None,
                       help='Specific tier to ingest (e.g., tier_1_core_ml)')
    parser.add_argument('--status', 
                       action='store_true',
                       help='Show current ingestion status and exit')
    
    args = parser.parse_args()
    
    if args.status:
        # Just show status
        tracker = IngestionTracker()
        summary = tracker.get_status_summary()
        
        print("\n📊 Current Ingestion Status")
        print("="*60)
        print(f"\nOverall:")
        for status, data in summary['status_counts'].items():
            print(f"  {status}: {data['count']} sources, {data['chunks']} chunks")
        
        print(f"\nBy Tier:")
        for tier, stats in summary['tier_stats'].items():
            print(f"  {tier}: {stats}")
        
        tracker.export_report("current_status.json")
        print(f"\n📄 Full report: current_status.json\n")
        tracker.close()
        return
    
    # Run ingestion
    ingestion = ComprehensiveMLDocumentationIngestion(args.sources, args.data_dir)
    
    try:
        await ingestion.run_full_ingestion(tier_filter=args.tier)
    finally:
        ingestion.close()


if __name__ == "__main__":
    asyncio.run(main())

