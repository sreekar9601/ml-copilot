"""Comprehensive ML documentation ingestion to Qdrant Cloud.

This script scrapes extensive ML documentation and ingests it into Qdrant Cloud
for maximum knowledge coverage in the ML documentation copilot.
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .comprehensive_scraper import scrape_comprehensive_ml_docs
from .chunker import SemanticChunker
from .upsert import upsert_document_chunks

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


async def ingest_comprehensive_ml_docs():
    """Main ingestion pipeline for comprehensive ML documentation."""
    
    logger.info("🚀 Starting comprehensive ML documentation ingestion...")
    
    # Check Qdrant configuration
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "ml-docs-copilot")
    
    if not qdrant_url or not qdrant_api_key:
        logger.error("❌ Qdrant credentials not found in environment variables")
        logger.error("Please set QDRANT_URL, QDRANT_API_KEY, and QDRANT_COLLECTION")
        return
    
    logger.info(f"Target: Qdrant Cloud ({qdrant_url})")
    logger.info(f"Collection: {collection_name}")
    
    # Step 1: Scrape comprehensive documentation
    logger.info("📡 Step 1: Scraping comprehensive ML documentation...")
    docs = await scrape_comprehensive_ml_docs()
    
    if not docs:
        logger.error("❌ No documents scraped")
        return
    
    logger.info(f"✅ Scraped {len(docs)} documents")
    
    # Calculate statistics
    total_words = sum(doc['word_count'] for doc in docs)
    estimated_chunks = total_words // 200  # Rough estimate: 200 words per chunk
    
    logger.info(f"📊 Total words: {total_words:,}")
    logger.info(f"📦 Estimated chunks: {estimated_chunks:,}")
    
    # Step 2: Chunk documents
    logger.info("🔪 Step 2: Chunking documents...")
    chunker = SemanticChunker()
    
    all_chunks = []
    for doc in docs:
        try:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error chunking document '{doc['title']}': {e}")
            continue
    
    logger.info(f"✅ Created {len(all_chunks)} chunks")
    
    # Step 3: Upsert to Qdrant Cloud + SQLite
    logger.info("💾 Step 3: Upserting to Qdrant Cloud + SQLite...")
    
    try:
        await upsert_document_chunks(
            chunks=all_chunks,
            data_dir=Path("./data"),
            collection_name=collection_name,
            sqlite_db="bm25.db"
        )
        
        logger.info("✅ Ingestion completed successfully!")
        
        # Final statistics
        logger.info("📈 Final Statistics:")
        logger.info(f"  - Documents scraped: {len(docs)}")
        logger.info(f"  - Chunks created: {len(all_chunks)}")
        logger.info(f"  - Total words: {total_words:,}")
        
        # Estimate Qdrant storage usage
        estimated_storage_mb = (len(all_chunks) * 768 * 4) / (1024 * 1024)  # 768-dim vectors * 4 bytes
        free_tier_usage = (estimated_storage_mb / 1024) * 100  # 1GB free tier
        
        logger.info(f"🗂️  Estimated Qdrant storage: {estimated_storage_mb:.1f} MB")
        logger.info(f"📊 Free tier usage: {free_tier_usage:.1f}%")
        
        if free_tier_usage > 80:
            logger.warning("⚠️  Approaching Qdrant free tier limit!")
        
    except Exception as e:
        logger.error(f"❌ Error during ingestion: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(ingest_comprehensive_ml_docs())
