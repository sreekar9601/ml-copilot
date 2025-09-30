#!/usr/bin/env python3
"""Simplified enhanced ingestion using existing infrastructure."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.main import run_ingestion_pipeline
from ingest.enhanced_data_sources import get_default_sources
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_enhanced_ingestion_simple():
    """Run enhanced ingestion with simplified approach using existing pipeline."""
    
    logger.info("Starting enhanced ingestion with existing infrastructure...")
    
    # Create enhanced seeds file from our data sources
    enhanced_sources = get_default_sources()
    
    # Convert to simple seeds format for existing pipeline
    urls = []
    for vendor, vendor_config in enhanced_sources.get('data_sources', {}).items():
        for category, sources in vendor_config.items():
            if isinstance(sources, list):
                for source in sources:
                    if 'url' in source:
                        urls.append(source['url'])
                        logger.info(f"Added {vendor}/{category}: {source['name']}")
    
    # Create seeds file
    seeds_data = {'urls': urls}
    seeds_path = Path(__file__).parent / "enhanced_seeds.yaml"
    
    with open(seeds_path, 'w') as f:
        yaml.dump(seeds_data, f, default_flow_style=False)
    
    logger.info(f"Created enhanced seeds file: {seeds_path}")
    logger.info(f"Total URLs: {len(urls)}")
    
    # Run with existing pipeline
    try:
        result = await run_ingestion_pipeline(
            seeds_path=seeds_path,
            data_dir=Path("./data"),
            chunk_size=500,
            chunk_overlap=50,
            collection_name="ml_docs_enhanced",
            sqlite_db="bm25_enhanced.db",
            clear_existing=True
        )
        
        logger.info("Enhanced ingestion completed!")
        logger.info(f"Results: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced ingestion failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_enhanced_ingestion_simple())
