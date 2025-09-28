#!/usr/bin/env python3
"""Fix the ingestion by using working documentation sources."""

import asyncio
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fix_ingestion():
    """Fix the ingestion with working sources."""
    
    # Clear the existing database completely
    logger.info("🗑️  Clearing existing database...")
    data_dir = Path('./data')
    if data_dir.exists():
        import shutil
        shutil.rmtree(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Use the working seeds file
    seeds_file = Path('./ingest/seeds_working.yaml')
    if not seeds_file.exists():
        logger.error(f"❌ Seeds file not found: {seeds_file}")
        return
    
    logger.info(f"📄 Using seeds file: {seeds_file}")
    
    # Run ingestion with working sources
    logger.info("🚀 Starting ingestion with working sources...")
    
    # Import and run the ingestion
    from ingest.main import run_ingestion_pipeline
    
    try:
        result = await run_ingestion_pipeline(
            seeds_path=seeds_file,
            data_dir=data_dir,
            chunk_size=500,
            chunk_overlap=50,
            collection_name="ml_docs",
            sqlite_db="bm25.db",
            clear_existing=True
        )
        
        logger.info("✅ Ingestion completed!")
        logger.info(f"📊 Results: {result}")
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(fix_ingestion())
