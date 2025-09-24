"""Ingestion script for local documentation files."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from .local_reader import load_local_docs
from .chunker import chunk_documents
from .upsert import upsert_document_chunks

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('local_ingestion.log')
    ]
)

logger = logging.getLogger(__name__)


async def run_local_ingestion(
    docs_dir: Path,
    data_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    collection_name: str = "ml_docs",
    sqlite_db: str = "bm25.db",
    clear_existing: bool = False
) -> Dict[str, Any]:
    """Run the local documentation ingestion pipeline."""
    
    logger.info("Starting local ML documentation ingestion pipeline")
    logger.info(f"Docs directory: {docs_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
    
    pipeline_stats = {
        "documents_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": []
    }
    
    try:
        # Step 1: Load local documentation files
        logger.info("Step 1: Loading local documentation files...")
        documents = load_local_docs(docs_dir)
        
        if not documents:
            logger.error("No documents were found in the docs directory")
            return pipeline_stats
        
        pipeline_stats["documents_processed"] = len(documents)
        logger.info(f"Successfully loaded {len(documents)} documents")
        
        # Log document sources for verification
        source_counts = {}
        for doc in documents:
            source = doc.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Document sources:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"  - {source}: {count} files")
        
        # Step 2: Chunk documents
        logger.info("Step 2: Chunking documents...")
        chunks = chunk_documents(
            documents, 
            chunk_size=chunk_size, 
            overlap=chunk_overlap
        )
        
        if not chunks:
            logger.error("No chunks were created from documents")
            return pipeline_stats
        
        pipeline_stats["chunks_created"] = len(chunks)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Log chunk distribution by source
        chunk_counts = {}
        for chunk in chunks:
            source = chunk.source_url
            chunk_counts[source] = chunk_counts.get(source, 0) + 1
        
        logger.info("Chunk distribution by source:")
        for source, count in sorted(chunk_counts.items()):
            logger.info(f"  - {source}: {count} chunks")
        
        # Step 3: Clear existing data if requested
        if clear_existing:
            logger.info("Step 3: Clearing existing data...")
            from .upsert import DatabaseManager
            db_manager = DatabaseManager(data_dir, collection_name, sqlite_db)
            try:
                db_manager.initialize_chromadb()
                db_manager.initialize_sqlite()
                db_manager.clear_collections()
                logger.info("Existing data cleared")
            finally:
                db_manager.close()
        
        # Step 4: Store chunks in databases
        logger.info("Step 4: Storing chunks in databases...")
        storage_stats = upsert_document_chunks(
            chunks, 
            data_dir, 
            collection_name, 
            sqlite_db
        )
        
        pipeline_stats["chunks_stored"] = storage_stats.get("chromadb_count", 0)
        pipeline_stats["storage_stats"] = storage_stats
        
        logger.info("Local ingestion pipeline completed successfully!")
        logger.info(f"Final stats: {pipeline_stats}")
        
        return pipeline_stats
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        pipeline_stats["errors"].append(error_msg)
        raise


async def main():
    """Main entry point for the local ingestion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Local ML Documentation Ingestion Pipeline")
    parser.add_argument(
        "--docs-dir", 
        type=Path, 
        default=Path("./docs"),
        help="Directory containing documentation files"
    )
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("./data"),
        help="Directory for storing databases"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=500,
        help="Target chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50,
        help="Overlap between chunks in tokens"
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="ml_docs",
        help="ChromaDB collection name"
    )
    parser.add_argument(
        "--sqlite-db", 
        type=str, 
        default="bm25.db",
        help="SQLite database filename"
    )
    parser.add_argument(
        "--clear", 
        action="store_true",
        help="Clear existing data before ingestion"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.docs_dir.exists():
        logger.error(f"Docs directory not found: {args.docs_dir}")
        sys.exit(1)
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the pipeline
        stats = await run_local_ingestion(
            docs_dir=args.docs_dir,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            collection_name=args.collection,
            sqlite_db=args.sqlite_db,
            clear_existing=args.clear
        )
        
        print("\n" + "="*50)
        print("LOCAL INGESTION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Chunks stored: {stats['chunks_stored']}")
        
        if stats.get('storage_stats'):
            print(f"ChromaDB count: {stats['storage_stats'].get('chromadb_count', 'N/A')}")
            print(f"SQLite FTS count: {stats['storage_stats'].get('sqlite_fts_count', 'N/A')}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Local ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
