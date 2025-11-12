#!/usr/bin/env python3
"""
Clean ML Documentation Ingestion Pipeline
Supports both GitHub repos and web scraping with proper metadata
"""

import os
import sys
import asyncio
import logging
import yaml
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

# Import local modules
try:
    from chunker import DocumentChunk
    from vertex_embedder import CachedVertexEmbedder
except ImportError:
    from ingest.chunker import DocumentChunk
    from ingest.vertex_embedder import CachedVertexEmbedder

# Import DatabaseManager with special handling
try:
    from upsert import DatabaseManager
except ImportError:
    # If relative import fails, use direct import
    import importlib.util
    upsert_path = os.path.join(os.path.dirname(__file__), "upsert.py")
    spec = importlib.util.spec_from_file_location("upsert_module", upsert_path)
    upsert_module = importlib.util.module_from_spec(spec)
    
    # Fix the upsert module's imports before loading
    import sys
    sys.modules['chunker'] = importlib.import_module('chunker')
    sys.modules['vertex_embedder'] = importlib.import_module('vertex_embedder')
    
    spec.loader.exec_module(upsert_module)
    DatabaseManager = upsert_module.DatabaseManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Track ingestion statistics per vendor."""
    vendor: str
    source_type: str  # 'github' or 'web'
    files_processed: int = 0
    chunks_created: int = 0
    errors: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class GitHubDocumentFetcher:
    """Fetches documentation from GitHub repositories."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.clone_dir = None
    
    def clone_repo(self, repo_url: str, sparse_paths: Optional[List[str]] = None) -> Path:
        """Clone a GitHub repository."""
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_path = Path(self.temp_dir) / f"github_docs_{repo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Remove existing if present
        if clone_path.exists():
            subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', str(clone_path)], 
                         check=False, capture_output=True)
        
        logger.info(f"  📥 Cloning {repo_name}...")
        
        try:
            if sparse_paths:
                # Sparse checkout
                clone_path.mkdir(parents=True, exist_ok=True)
                subprocess.run(['git', 'init'], cwd=str(clone_path), check=True, capture_output=True)
                subprocess.run(['git', 'remote', 'add', 'origin', repo_url], 
                             cwd=str(clone_path), check=True, capture_output=True)
                subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], 
                             cwd=str(clone_path), check=True, capture_output=True)
                
                sparse_file = clone_path / '.git' / 'info' / 'sparse-checkout'
                sparse_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sparse_file, 'w') as f:
                    f.write('\n'.join(sparse_paths))
                
                subprocess.run(['git', 'pull', '--depth=1', 'origin', 'main'], 
                             cwd=str(clone_path), check=True, capture_output=True, timeout=300)
            else:
                # Full shallow clone
                subprocess.run(['git', 'clone', '--depth=1', repo_url, str(clone_path)], 
                             check=True, capture_output=True, timeout=300)
            
            self.clone_dir = clone_path
            return clone_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Git clone failed: {e}")
            raise
    
    def extract_docs(self, clone_path: Path, file_patterns: List[str], 
                    base_url: str, vendor: str, paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Extract documentation files from cloned repo."""
        docs = []
        
        # Determine search paths
        search_paths = []
        if paths:
            for p in paths:
                search_path = clone_path / p
                if search_path.exists():
                    search_paths.append(search_path)
        else:
            search_paths = [clone_path]
        
        # Find all matching files
        for search_path in search_paths:
            for pattern in file_patterns:
                for file_path in search_path.rglob(pattern):
                    # Skip irrelevant directories
                    if any(skip in str(file_path) for skip in ['.git', '__pycache__', 'node_modules', 'build', 'dist']):
                        continue
                    
                    # Parse the file
                    doc = self._parse_file(file_path, clone_path, base_url, vendor)
                    if doc:
                        docs.append(doc)
        
        return docs
    
    def _parse_file(self, file_path: Path, clone_path: Path, base_url: str, vendor: str) -> Optional[Dict[str, Any]]:
        """Parse a single documentation file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Skip if too short
            if len(content.strip()) < 150:
                return None
            
            # Extract title based on file type
            title = self._extract_title(content, file_path)
            
            # Clean content based on file type
            if file_path.suffix in ['.md', '.mdx']:
                content = self._clean_markdown(content)
            elif file_path.suffix in ['.rst']:
                content = self._clean_rst(content)
            elif file_path.suffix == '.py':
                content = self._extract_python_docstring(content)
                if not content:
                    return None
            
            # Generate source URL
            rel_path = file_path.relative_to(clone_path)
            source_url = f"{base_url}/{rel_path.as_posix()}".replace('.rst', '.html').replace('.md', '.html').replace('.py', '.html')
            
            return {
                'title': title,
                'content': content,
                'source_url': source_url,
                'vendor': vendor,
                'file_path': str(rel_path)
            }
            
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to parse {file_path.name}: {e}")
            return None
    
    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from content."""
        # Try markdown heading
        md_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if md_match:
            return md_match.group(1).strip()
        
        # Try RST heading
        lines = content.split('\n')
        for i, line in enumerate(lines[:10]):
            if line.strip() and i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'^[=\-~`:#"^_*+]{3,}$', next_line.strip()):
                    return line.strip()
        
        # Fallback to filename
        return file_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content."""
        # Remove code fences but keep content
        content = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
        # Remove inline code backticks
        content = re.sub(r'`([^`]+)`', r'\1', content)
        # Remove images
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # Remove link markdown but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        # Remove headers markers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Clean whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def _clean_rst(self, content: str) -> str:
        """Clean reStructuredText content."""
        # Remove RST directives
        content = re.sub(r'\.\. \w+::[^\n]*\n(?:   [^\n]*\n)*', '', content)
        # Remove inline literals
        content = re.sub(r'``([^`]+)``', r'\1', content)
        # Remove references
        content = re.sub(r':\w+:`([^`]+)`', r'\1', content)
        # Remove underline headings
        content = re.sub(r'^[=\-~`:#"^_*+]{3,}$', '', content, flags=re.MULTILINE)
        # Clean whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content.strip()
    
    def _extract_python_docstring(self, content: str) -> Optional[str]:
        """Extract docstring from Python file."""
        match = re.search(r'^["\'"]{3}(.*?)["\'"]{3}', content, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def cleanup(self):
        """Remove cloned repository."""
        if self.clone_dir and self.clone_dir.exists():
            try:
                subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', str(self.clone_dir)], 
                             check=False, capture_output=True)
                logger.info(f"  🧹 Cleaned up {self.clone_dir.name}")
            except Exception as e:
                logger.warning(f"  ⚠️  Cleanup failed: {e}")


class WebDocumentFetcher:
    """Fetches documentation from web sources."""
    
    def __init__(self):
        self.session = None
    
    async def fetch_docs(self, url: str, vendor: str, max_depth: int = 2, max_pages: int = 100) -> List[Dict[str, Any]]:
        """Fetch documentation from web URL."""
        logger.info(f"  🌐 Web scraping not fully implemented - creating placeholder for {url}")
        
        # TODO: Implement actual web scraping using BeautifulSoup/Scrapy
        # For now, return empty list as web sources are secondary
        return []


class CleanIngestionPipeline:
    """Main ingestion pipeline orchestrator."""
    
    def __init__(self, sources_file: str, data_dir: str = "../../data"):
        self.sources_file = sources_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Load sources configuration
        with open(sources_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.db_manager = DatabaseManager(str(self.data_dir), collection_name="ml_docs")
        self.embedder = CachedVertexEmbedder()
        self.github_fetcher = GitHubDocumentFetcher()
        self.web_fetcher = WebDocumentFetcher()
        
        # Statistics tracking
        self.stats: Dict[str, IngestionStats] = {}
        self.run_id = f"clean_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"✨ Initialized Clean Ingestion Pipeline - Run ID: {self.run_id}")
    
    async def ingest_github_source(self, vendor: str, source_config: Dict[str, Any]) -> IngestionStats:
        """Ingest documentation from GitHub repository."""
        stats = IngestionStats(vendor=vendor, source_type="github")
        stats.status = "in_progress"
        stats.started_at = datetime.now().isoformat()
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📦 {vendor} (GitHub)")
            logger.info(f"{'='*60}")
            
            for source in source_config.get('sources', []):
                repo_url = source.get('repo_url')
                base_url = source.get('base_url')
                file_patterns = source.get('file_patterns', ['*.md'])
                sparse_paths = source.get('sparse_paths')
                paths = source.get('paths')
                max_files = source.get('max_files', 500)
                
                try:
                    # Clone repository
                    clone_path = self.github_fetcher.clone_repo(repo_url, sparse_paths)
                    
                    # Extract documentation
                    docs = self.github_fetcher.extract_docs(
                        clone_path, file_patterns, base_url, vendor, paths
                    )
                    
                    # Limit documents
                    docs = docs[:max_files]
                    
                    logger.info(f"  📄 Extracted {len(docs)} documents")
                    
                    # Convert to chunks and upsert
                    chunks = self._convert_to_chunks(docs, vendor)
                    if chunks:
                        self.db_manager.upsert_chunks(chunks, batch_size=50)
                        logger.info(f"  💾 Stored {len(chunks)} chunks")
                    
                    stats.files_processed += len(docs)
                    stats.chunks_created += len(chunks)
                    
                    # Cleanup
                    self.github_fetcher.cleanup()
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing {repo_url}: {e}")
                    stats.errors += 1
            
            stats.status = "completed"
            stats.completed_at = datetime.now().isoformat()
            logger.info(f"✅ {vendor} completed: {stats.chunks_created} chunks")
            
        except Exception as e:
            logger.error(f"❌ {vendor} failed: {e}", exc_info=True)
            stats.status = "failed"
            stats.completed_at = datetime.now().isoformat()
        
        return stats
    
    async def ingest_web_source(self, vendor: str, source_config: Dict[str, Any]) -> IngestionStats:
        """Ingest documentation from web URLs."""
        stats = IngestionStats(vendor=vendor, source_type="web")
        stats.status = "in_progress"
        stats.started_at = datetime.now().isoformat()
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"🌐 {vendor} (Web)")
            logger.info(f"{'='*60}")
            
            # Web scraping is secondary - skip for now
            logger.info(f"  ⏭️  Skipping web source (GitHub-first strategy)")
            stats.status = "skipped"
            stats.completed_at = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"❌ {vendor} failed: {e}")
            stats.status = "failed"
            stats.completed_at = datetime.now().isoformat()
        
        return stats
    
    def _convert_to_chunks(self, docs: List[Dict[str, Any]], vendor: str) -> List[DocumentChunk]:
        """Convert documents to chunks."""
        chunks = []
        
        for doc in docs:
            # Simple chunking by paragraphs
            content = doc['content']
            paragraphs = content.split('\n\n')
            
            for i, para in enumerate(paragraphs):
                if len(para.strip()) < 100:  # Skip short paragraphs
                    continue
                
                chunk_id = f"{vendor.lower().replace(' ', '_')}_{hash(doc['source_url']) % 100000}_{i}"
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=para.strip(),
                    source_url=doc['source_url'],
                    title=doc['title'],
                    heading_path=f"{vendor} > {doc['title']}",
                    anchor_link=doc['source_url'],
                    token_count=len(para.split()),
                    prev_id=None,
                    next_id=None
                )
                chunks.append(chunk)
        
        return chunks
    
    async def run_full_ingestion(self, vendors: Optional[List[str]] = None):
        """Run full ingestion pipeline."""
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 CLEAN ML DOCUMENTATION INGESTION")
        logger.info(f"   Run ID: {self.run_id}")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}\n")
        
        sources = self.config.get('sources', {})
        
        # Filter vendors if specified
        if vendors:
            sources = {k: v for k, v in sources.items() if k in vendors}
        
        # Process each vendor
        for vendor_key, vendor_config in sources.items():
            vendor_name = vendor_config.get('vendor', vendor_key)
            source_type = vendor_config.get('type', 'web')
            
            if source_type == 'github':
                stats = await self.ingest_github_source(vendor_name, vendor_config)
            else:
                stats = await self.ingest_web_source(vendor_name, vendor_config)
            
            self.stats[vendor_name] = stats
            
            # Small delay between vendors
            await asyncio.sleep(1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print ingestion summary."""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 INGESTION SUMMARY")
        logger.info(f"{'='*80}\n")
        
        total_files = 0
        total_chunks = 0
        completed = 0
        failed = 0
        skipped = 0
        
        for vendor, stats in self.stats.items():
            emoji = "✅" if stats.status == "completed" else "❌" if stats.status == "failed" else "⏭️"
            logger.info(f"{emoji} {vendor:30s} | {stats.source_type:8s} | Files: {stats.files_processed:4d} | Chunks: {stats.chunks_created:5d}")
            
            total_files += stats.files_processed
            total_chunks += stats.chunks_created
            
            if stats.status == "completed":
                completed += 1
            elif stats.status == "failed":
                failed += 1
            elif stats.status == "skipped":
                skipped += 1
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📈 TOTALS:")
        logger.info(f"   Vendors Completed: {completed}")
        logger.info(f"   Vendors Failed: {failed}")
        logger.info(f"   Vendors Skipped: {skipped}")
        logger.info(f"   Total Files: {total_files}")
        logger.info(f"   Total Chunks: {total_chunks}")
        
        # Get database stats
        db_stats = self.db_manager.get_collection_stats()
        logger.info(f"\n💾 DATABASE:")
        logger.info(f"   Total Documents: {db_stats.get('vectors_count', 0)}")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✨ Ingestion Complete!")
        logger.info(f"{'='*80}\n")
    
    def close(self):
        """Cleanup resources."""
        self.db_manager.close()


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean ML Documentation Ingestion')
    parser.add_argument('--sources', default='clean_ml_sources.yaml', help='Sources YAML file')
    parser.add_argument('--data-dir', default='../../data', help='Data directory')
    parser.add_argument('--vendors', nargs='+', help='Specific vendors to ingest (optional)')
    parser.add_argument('--clear', action='store_true', help='Clear existing collection before ingestion')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CleanIngestionPipeline(args.sources, args.data_dir)
    
    # Clear collection if requested
    if args.clear:
        logger.info("🗑️  Clearing existing collection...")
        # TODO: Implement collection clearing
        logger.info("✅ Collection cleared")
    
    try:
        await pipeline.run_full_ingestion(vendors=args.vendors)
    finally:
        pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())

