"""
GitHub Documentation Scraper
Clones and parses documentation from GitHub repositories (Markdown, RST, etc.)
Bypasses web scraping 403 issues by using the open-source repos directly.
"""

import os
import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GitHubDoc:
    """Represents a parsed documentation file from GitHub."""
    title: str
    content: str
    file_path: str
    url: str
    source: str

class GitHubDocScraper:
    """Scrapes documentation from GitHub repositories."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.clone_dir = None
        
    def clone_repo(self, repo_url: str, sparse_paths: Optional[List[str]] = None) -> Path:
        """
        Clone a GitHub repository (sparse checkout if paths specified).
        
        Args:
            repo_url: GitHub repository URL
            sparse_paths: Optional list of paths to checkout (e.g., ['docs/', 'tutorials/'])
        
        Returns:
            Path to cloned repository
        """
        # Extract repo name from URL
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        clone_path = Path(self.temp_dir) / f"github_docs_{repo_name}"
        
        # Remove existing clone if present
        if clone_path.exists():
            logger.info(f"Removing existing clone at {clone_path}")
            subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', str(clone_path)], 
                         check=False, capture_output=True)
        
        logger.info(f"Cloning {repo_url} to {clone_path}")
        
        try:
            if sparse_paths:
                # Sparse checkout for large repos
                clone_path.mkdir(parents=True, exist_ok=True)
                
                subprocess.run(['git', 'init'], cwd=str(clone_path), check=True, capture_output=True)
                subprocess.run(['git', 'remote', 'add', 'origin', repo_url], 
                             cwd=str(clone_path), check=True, capture_output=True)
                subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], 
                             cwd=str(clone_path), check=True, capture_output=True)
                
                # Write sparse-checkout paths
                sparse_file = clone_path / '.git' / 'info' / 'sparse-checkout'
                sparse_file.parent.mkdir(parents=True, exist_ok=True)
                with open(sparse_file, 'w') as f:
                    f.write('\n'.join(sparse_paths))
                
                # Pull only specified paths
                subprocess.run(['git', 'pull', '--depth=1', 'origin', 'main'], 
                             cwd=str(clone_path), check=True, capture_output=True)
            else:
                # Full shallow clone
                subprocess.run(['git', 'clone', '--depth=1', repo_url, str(clone_path)], 
                             check=True, capture_output=True)
            
            self.clone_dir = clone_path
            logger.info(f"Successfully cloned {repo_name}")
            return clone_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Git clone failed: {e}")
            raise
    
    def parse_markdown(self, file_path: Path, base_url: str, source: str) -> Optional[GitHubDoc]:
        """Parse a Markdown file into a GitHubDoc."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract title from first heading or filename
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem.replace('_', ' ').title()
            
            # Clean content
            content = self._clean_markdown(content)
            
            # Skip if content is too short
            if len(content.strip()) < 100:
                return None
            
            # Generate web URL
            rel_path = file_path.relative_to(self.clone_dir)
            doc_url = f"{base_url}/{rel_path.as_posix().replace('.md', '.html')}"
            
            return GitHubDoc(
                title=title,
                content=content,
                file_path=str(rel_path),
                url=doc_url,
                source=source
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def parse_rst(self, file_path: Path, base_url: str, source: str) -> Optional[GitHubDoc]:
        """Parse a reStructuredText file into a GitHubDoc."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract title from RST heading
            lines = content.split('\n')
            title = file_path.stem.replace('_', ' ').title()
            for i, line in enumerate(lines[:10]):
                if line.strip() and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if re.match(r'^[=\-~`:#"^_*+]{3,}$', next_line.strip()):
                        title = line.strip()
                        break
            
            # Clean content
            content = self._clean_rst(content)
            
            # Skip if content is too short
            if len(content.strip()) < 100:
                return None
            
            # Generate web URL
            rel_path = file_path.relative_to(self.clone_dir)
            doc_url = f"{base_url}/{rel_path.as_posix().replace('.rst', '.html')}"
            
            return GitHubDoc(
                title=title,
                content=content,
                file_path=str(rel_path),
                url=doc_url,
                source=source
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return None
    
    def _clean_markdown(self, content: str) -> str:
        """Clean markdown content."""
        # Remove code fences but keep content
        content = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
        
        # Remove inline code backticks
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Remove image references
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        
        # Remove link markdown but keep text
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        
        # Remove markdown headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Clean up whitespace
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
        
        # Clean up whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def scrape_pytorch_tutorials(self) -> List[GitHubDoc]:
        """Scrape PyTorch tutorials from GitHub."""
        repo_url = "https://github.com/pytorch/tutorials.git"
        base_url = "https://pytorch.org/tutorials"
        
        clone_path = self.clone_repo(repo_url)
        docs = []
        
        # Find all tutorial files
        for pattern in ['*.py', '*.rst', '*.md']:
            for file_path in clone_path.rglob(pattern):
                # Skip non-tutorial files
                if any(skip in str(file_path) for skip in ['.git', 'build', '__pycache__', 'test']):
                    continue
                
                if file_path.suffix == '.py':
                    # Parse Python tutorial files (they have docstrings)
                    doc = self._parse_python_tutorial(file_path, base_url)
                elif file_path.suffix == '.rst':
                    doc = self.parse_rst(file_path, base_url, "PyTorch Tutorials")
                else:
                    doc = self.parse_markdown(file_path, base_url, "PyTorch Tutorials")
                
                if doc:
                    docs.append(doc)
        
        logger.info(f"Scraped {len(docs)} PyTorch tutorial documents")
        return docs
    
    def scrape_pytorch_docs(self) -> List[GitHubDoc]:
        """Scrape PyTorch core documentation from GitHub."""
        repo_url = "https://github.com/pytorch/pytorch.git"
        base_url = "https://pytorch.org/docs/stable"
        
        # Use sparse checkout for just docs folder
        clone_path = self.clone_repo(repo_url, sparse_paths=['docs/source/'])
        docs = []
        
        docs_dir = clone_path / 'docs' / 'source'
        if not docs_dir.exists():
            logger.warning(f"Docs directory not found at {docs_dir}")
            return docs
        
        # Find all doc files
        for pattern in ['*.rst', '*.md']:
            for file_path in docs_dir.rglob(pattern):
                if any(skip in str(file_path) for skip in ['.git', 'build', '__pycache__']):
                    continue
                
                if file_path.suffix == '.rst':
                    doc = self.parse_rst(file_path, base_url, "PyTorch Docs")
                else:
                    doc = self.parse_markdown(file_path, base_url, "PyTorch Docs")
                
                if doc:
                    docs.append(doc)
        
        logger.info(f"Scraped {len(docs)} PyTorch documentation files")
        return docs
    
    def _parse_python_tutorial(self, file_path: Path, base_url: str) -> Optional[GitHubDoc]:
        """Parse Python tutorial files (which contain docstrings)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Extract module docstring
            docstring_match = re.search(r'^["\'"]{3}(.*?)["\'"]{3}', content, re.DOTALL | re.MULTILINE)
            if not docstring_match:
                return None
            
            docstring = docstring_match.group(1).strip()
            
            # Skip if too short
            if len(docstring) < 100:
                return None
            
            # Extract title from first line
            lines = docstring.split('\n')
            title = lines[0].strip() if lines else file_path.stem.replace('_', ' ').title()
            
            # Clean title
            title = re.sub(r'^#+\s*', '', title)
            title = re.sub(r'[*_]+', '', title)
            
            # Generate URL
            rel_path = file_path.relative_to(self.clone_dir)
            doc_url = f"{base_url}/{rel_path.as_posix().replace('.py', '.html')}"
            
            return GitHubDoc(
                title=title,
                content=docstring,
                file_path=str(rel_path),
                url=doc_url,
                source="PyTorch Tutorials"
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Python tutorial {file_path}: {e}")
            return None
    
    def cleanup(self):
        """Remove cloned repositories."""
        if self.clone_dir and self.clone_dir.exists():
            try:
                subprocess.run(['cmd', '/c', 'rmdir', '/s', '/q', str(self.clone_dir)], 
                             check=False, capture_output=True)
                logger.info(f"Cleaned up {self.clone_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {self.clone_dir}: {e}")


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    scraper = GitHubDocScraper()
    
    try:
        print("Testing PyTorch Tutorials scraping...")
        docs = scraper.scrape_pytorch_tutorials()
        print(f"✅ Found {len(docs)} tutorial documents")
        
        if docs:
            print(f"\nSample document:")
            print(f"  Title: {docs[0].title}")
            print(f"  Path: {docs[0].file_path}")
            print(f"  URL: {docs[0].url}")
            print(f"  Content length: {len(docs[0].content)} chars")
            print(f"  Preview: {docs[0].content[:200]}...")
        
    finally:
        scraper.cleanup()


