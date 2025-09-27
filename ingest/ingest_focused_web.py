#!/usr/bin/env python3
"""
Focused web scraping and ingestion for ML documentation.
Optimized for Qdrant Cloud free tier (1GB = ~330k vectors).
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ingest.focused_scraper import scrape_focused_ml_docs
from ingest.chunker import SemanticChunker
from ingest.upsert import upsert_document_chunks
from api.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion pipeline for focused web-scraped ML docs."""
    
    logger.info("🚀 Starting focused ML documentation ingestion...")
    logger.info(f"Target: Qdrant Cloud ({settings.qdrant_url})")
    logger.info(f"Collection: {settings.qdrant_collection_name}")
    
    try:
        # Step 1: Scrape focused ML documentation
        logger.info("📡 Step 1: Scraping high-value ML documentation...")
        scraped_docs = await scrape_focused_ml_docs()
        
        if not scraped_docs:
            logger.error("❌ No documents scraped. Exiting.")
            return
        
        logger.info(f"✅ Scraped {len(scraped_docs)} documents")
        
        # Calculate total word count and estimate chunks
        total_words = sum(doc['word_count'] for doc in scraped_docs)
        estimated_chunks = total_words // (settings.chunk_size // 2)  # Rough estimate
        
        logger.info(f"📊 Total words: {total_words:,}")
        logger.info(f"📦 Estimated chunks: {estimated_chunks:,}")
        
        if estimated_chunks > 300000:  # Close to Qdrant limit
            logger.warning(f"⚠️  Estimated chunks ({estimated_chunks:,}) approaching Qdrant limit (330k)")
        
        # Step 2: Chunk documents
        logger.info("🔪 Step 2: Chunking documents...")
        chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        
        all_chunks = []
        for doc in scraped_docs:
            try:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)
                logger.debug(f"Chunked '{doc['title']}': {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error chunking document '{doc['title']}': {e}")
        
        logger.info(f"✅ Created {len(all_chunks)} chunks")
        
        if len(all_chunks) > 300000:
            logger.warning(f"⚠️  Actual chunks ({len(all_chunks):,}) approaching Qdrant limit!")
            logger.info("Consider reducing chunk count or upgrading Qdrant plan")
        
        # Step 3: Upsert to databases
        logger.info("💾 Step 3: Upserting to Qdrant Cloud + SQLite...")
        
        stats = upsert_document_chunks(
            chunks=all_chunks,
            data_dir=settings.data_dir,
            collection_name=settings.qdrant_collection_name,
            sqlite_db=settings.sqlite_db
        )
        
        logger.info("✅ Ingestion completed successfully!")
        logger.info("📈 Final Statistics:")
        for key, value in stats.items():
            logger.info(f"  - {key}: {value:,}" if isinstance(value, int) else f"  - {key}: {value}")
        
        # Estimate storage usage
        if len(all_chunks) > 0:
            estimated_storage_mb = (len(all_chunks) * 3072) / (1024 * 1024)  # 3KB per vector
            logger.info(f"🗂️  Estimated Qdrant storage: {estimated_storage_mb:.1f} MB")
            
            free_tier_usage = (estimated_storage_mb / 1024) * 100  # % of 1GB
            logger.info(f"📊 Free tier usage: {free_tier_usage:.1f}%")
            
            if free_tier_usage > 90:
                logger.warning("⚠️  Approaching Qdrant Cloud free tier limit!")
    
    except KeyboardInterrupt:
        logger.info("🛑 Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
