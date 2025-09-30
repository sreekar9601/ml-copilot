#!/usr/bin/env python3
"""Command-line interface for enhanced ingestion pipeline."""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from ingest
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.enhanced_ingestion_pipeline import run_enhanced_ingestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main entry point for enhanced ingestion."""
    parser = argparse.ArgumentParser(description="Enhanced ML Documentation Ingestion Pipeline")
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data"),
        help="Directory for storing databases"
    )
    
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).parent / "enhanced_data_sources.yaml",
        help="Path to enhanced data sources configuration"
    )
    
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing data before ingestion"
    )
    
    parser.add_argument(
        "--max-sources-per-vendor",
        type=int,
        default=None,
        help="Maximum number of sources to process per vendor"
    )
    
    parser.add_argument(
        "--vendor-filter",
        nargs="+",
        default=None,
        help="Process only specific vendors (e.g., pytorch mlflow)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.config_path.exists():
        logger.warning(f"Config file not found: {args.config_path}")
        logger.info("Using default configuration...")
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("🚀 Starting enhanced ingestion pipeline...")
        logger.info(f"📁 Data directory: {args.data_dir}")
        logger.info(f"📄 Config file: {args.config_path}")
        logger.info(f"🗑️  Clear existing: {args.clear_existing}")
        logger.info(f"📊 Max sources per vendor: {args.max_sources_per_vendor}")
        
        # Run enhanced ingestion
        result = await run_enhanced_ingestion(
            data_dir=args.data_dir,
            config_path=args.config_path,
            clear_existing=args.clear_existing,
            max_sources_per_vendor=args.max_sources_per_vendor
        )
        
        logger.info("✅ Enhanced ingestion completed successfully!")
        logger.info(f"📊 Results: {result}")
        
    except Exception as e:
        logger.error(f"❌ Enhanced ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
