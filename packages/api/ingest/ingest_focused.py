"""Focused ingestion script for key documentation files only."""

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Focus on key documentation files
FOCUSED_PATTERNS = [
    # PyTorch key files
    "**/data.rst",
    "**/nn.rst", 
    "**/distributed.rst",
    "**/ddp.rst",
    "**/tutorials/**/*data*.rst",
    "**/tutorials/**/*distributed*.rst",
    
    # MLflow key files
    "**/tracking*.mdx",
    "**/model-registry*.mdx",
    "**/models*.mdx",
    "**/getting-started*.mdx",
    
    # Ray Serve key files
    "**/serve/**/*getting-started*.rst",
    "**/serve/**/*key-concepts*.rst",
    "**/serve/**/*production*.rst",
    "**/serve/**/*deployment*.rst",
]


def find_focused_docs(docs_dir: Path) -> List[Path]:
    """Find only the most important documentation files."""
    focused_files = []
    
    for pattern in FOCUSED_PATTERNS:
        files = list(docs_dir.glob(pattern))
        focused_files.extend(files)
    
    # Remove duplicates and sort
    focused_files = list(set(focused_files))
    focused_files.sort()
    
    logger.info(f"Found {len(focused_files)} focused documentation files")
    return focused_files


async def run_focused_ingestion(
    docs_dir: Path,
    data_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    collection_name: str = "ml_docs",
    sqlite_db: str = "bm25.db",
    clear_existing: bool = False
) -> Dict[str, Any]:
    """Run focused documentation ingestion pipeline."""
    
    logger.info("Starting focused ML documentation ingestion pipeline")
    logger.info(f"Docs directory: {docs_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    pipeline_stats = {
        "documents_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": []
    }
    
    try:
        # Step 1: Find focused documentation files
        logger.info("Step 1: Finding key documentation files...")
        focused_files = find_focused_docs(docs_dir)
        
        if not focused_files:
            logger.error("No focused documentation files found")
            return pipeline_stats
        
        # Step 2: Process only focused files
        logger.info("Step 2: Processing focused documentation files...")
        from .local_reader import LocalDocReader
        
        reader = LocalDocReader(docs_dir)
        documents = []
        
        for file_path in focused_files:
            try:
                doc = reader.process_file(file_path)
                if doc:
                    documents.append(doc)
                    logger.info(f"Processed: {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        if not documents:
            logger.error("No documents were successfully processed")
            return pipeline_stats
        
        pipeline_stats["documents_processed"] = len(documents)
        logger.info(f"Successfully processed {len(documents)} documents")
        
        # Step 3: Chunk documents
        logger.info("Step 3: Chunking documents...")
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
        
        # Step 4: Clear existing data if requested
        if clear_existing:
            logger.info("Step 4: Clearing existing data...")
            from .upsert import DatabaseManager
            db_manager = DatabaseManager(data_dir, collection_name, sqlite_db)
            try:
                db_manager.initialize_chromadb()
                db_manager.initialize_sqlite()
                db_manager.clear_collections()
                logger.info("Existing data cleared")
            finally:
                db_manager.close()
        
        # Step 5: Store chunks in databases
        logger.info("Step 5: Storing chunks in databases...")
        storage_stats = upsert_document_chunks(
            chunks, 
            data_dir, 
            collection_name, 
            sqlite_db
        )
        
        pipeline_stats["chunks_stored"] = storage_stats.get("chromadb_count", 0)
        pipeline_stats["storage_stats"] = storage_stats
        
        logger.info("Focused ingestion pipeline completed successfully!")
        logger.info(f"Final stats: {pipeline_stats}")
        
        return pipeline_stats
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        pipeline_stats["errors"].append(error_msg)
        raise


async def main():
    """Main entry point for the focused ingestion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Focused ML Documentation Ingestion Pipeline")
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
        stats = await run_focused_ingestion(
            docs_dir=args.docs_dir,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            collection_name=args.collection,
            sqlite_db=args.sqlite_db,
            clear_existing=args.clear
        )
        
        print("\n" + "="*50)
        print("FOCUSED INGESTION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Chunks stored: {stats['chunks_stored']}")
        
        if stats.get('storage_stats'):
            print(f"ChromaDB count: {stats['storage_stats'].get('chromadb_count', 'N/A')}")
            print(f"SQLite FTS count: {stats['storage_stats'].get('sqlite_fts_count', 'N/A')}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Focused ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
