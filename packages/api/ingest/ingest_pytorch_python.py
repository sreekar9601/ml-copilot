"""Focused ingestion script for PyTorch Python API documentation."""

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

# Focus on PyTorch Python API documentation
PYTORCH_PYTHON_PATTERNS = [
    # Look for Python API documentation
    "**/pytorch-docs/**/*.rst",
    "**/pytorch-docs/**/*.md",
    # Exclude C++ specific files
    "!**/cpp/**",
    "!**/libtorch*",
    "!**/cuda*",
    "!**/cpu*",
    # Focus on core Python modules
    "**/nn*",
    "**/data*", 
    "**/utils*",
    "**/optim*",
    "**/autograd*",
    "**/torch*",
]


def find_pytorch_python_docs(docs_dir: Path) -> List[Path]:
    """Find PyTorch Python API documentation files."""
    all_files = []
    
    # Get all PyTorch files first
    pytorch_dir = docs_dir / "pytorch-docs"
    if not pytorch_dir.exists():
        logger.error(f"PyTorch docs directory not found: {pytorch_dir}")
        return []
    
    for ext in ['.rst', '.md']:
        pattern = f"**/*{ext}"
        files = list(pytorch_dir.glob(pattern))
        all_files.extend(files)
    
    # Filter for Python API documentation
    filtered_files = []
    for file_path in all_files:
        # Skip C++ specific files
        if any(excluded in str(file_path) for excluded in [
            '/cpp/', '/libtorch', '/cuda', '/cpu', '/cudnn'
        ]):
            continue
            
        # Skip very large files (>1MB)
        if file_path.stat().st_size > 1024 * 1024:
            logger.warning(f"Skipping large file: {file_path}")
            continue
            
        # Skip files that are likely not documentation
        if file_path.name in ['Makefile', 'requirements.txt', 'setup.py', 'conf.py']:
            continue
            
        filtered_files.append(file_path)
    
    # Remove duplicates and sort
    filtered_files = list(set(filtered_files))
    filtered_files.sort()
    
    logger.info(f"Found {len(filtered_files)} PyTorch Python API documentation files")
    
    # Log some examples
    logger.info("Sample files to be processed:")
    for i, file_path in enumerate(filtered_files[:10]):
        logger.info(f"  {i+1}. {file_path.relative_to(docs_dir)}")
    
    if len(filtered_files) > 10:
        logger.info(f"  ... and {len(filtered_files) - 10} more files")
    
    return filtered_files


async def run_pytorch_python_ingestion(
    docs_dir: Path,
    data_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    collection_name: str = "ml_docs",
    sqlite_db: str = "bm25.db",
    clear_existing: bool = False
) -> Dict[str, Any]:
    """Run PyTorch Python API documentation ingestion pipeline."""
    
    logger.info("Starting PyTorch Python API documentation ingestion pipeline")
    logger.info(f"Docs directory: {docs_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    pipeline_stats = {
        "documents_processed": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
        "errors": []
    }
    
    try:
        # Step 1: Find PyTorch Python API documentation files
        logger.info("Step 1: Finding PyTorch Python API documentation files...")
        pytorch_files = find_pytorch_python_docs(docs_dir)
        
        if not pytorch_files:
            logger.error("No PyTorch Python API documentation files found")
            return pipeline_stats
        
        # Step 2: Process files
        logger.info("Step 2: Processing PyTorch Python API documentation files...")
        from .local_reader import LocalDocReader
        
        reader = LocalDocReader(docs_dir)
        documents = []
        
        for file_path in pytorch_files:
            try:
                doc = reader.process_file(file_path)
                if doc and doc['content'].strip():  # Only add non-empty docs
                    documents.append(doc)
                    logger.debug(f"Processed: {file_path.name}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                pipeline_stats["errors"].append(f"{file_path}: {e}")
        
        if not documents:
            logger.error("No documents were successfully processed")
            return pipeline_stats
        
        pipeline_stats["documents_processed"] = len(documents)
        logger.info(f"Successfully processed {len(documents)} documents")
        
        # Log document distribution by source
        source_counts = {}
        for doc in documents:
            source = doc.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Document distribution by source:")
        for source, count in sorted(source_counts.items()):
            logger.info(f"  - {source}: {count} files")
        
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
        
        # Step 4: Store chunks in databases (don't clear existing data)
        logger.info("Step 4: Storing chunks in databases...")
        storage_stats = upsert_document_chunks(
            chunks, 
            data_dir, 
            collection_name, 
            sqlite_db
        )
        
        pipeline_stats["chunks_stored"] = storage_stats.get("chromadb_count", 0)
        pipeline_stats["storage_stats"] = storage_stats
        
        logger.info("PyTorch Python API ingestion pipeline completed successfully!")
        logger.info(f"Final stats: {pipeline_stats}")
        
        return pipeline_stats
        
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        logger.error(error_msg)
        pipeline_stats["errors"].append(error_msg)
        raise


async def main():
    """Main entry point for the PyTorch Python API ingestion script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Python API Documentation Ingestion Pipeline")
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.docs_dir.exists():
        logger.error(f"Docs directory not found: {args.docs_dir}")
        sys.exit(1)
    
    # Create data directory
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the pipeline
        stats = await run_pytorch_python_ingestion(
            docs_dir=args.docs_dir,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            collection_name=args.collection,
            sqlite_db=args.sqlite_db,
            clear_existing=False  # Don't clear existing MLflow data
        )
        
        print("\n" + "="*50)
        print("PYTORCH PYTHON API INGESTION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Chunks stored: {stats['chunks_stored']}")
        
        if stats.get('storage_stats'):
            print(f"ChromaDB count: {stats['storage_stats'].get('chromadb_count', 'N/A')}")
            print(f"SQLite FTS count: {stats['storage_stats'].get('sqlite_fts_count', 'N/A')}")
        
        if stats.get('errors'):
            print(f"Errors encountered: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"PyTorch Python API ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
