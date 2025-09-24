"""Local documentation file reader for processing downloaded docs."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
import markdown
from docutils.core import publish_doctree
from docutils.writers.html4css1 import Writer
from docutils.core import publish_parts

logger = logging.getLogger(__name__)


class LocalDocReader:
    """Reads and processes local documentation files."""
    
    def __init__(self, docs_dir: Path):
        self.docs_dir = Path(docs_dir)
        self.supported_extensions = {'.rst', '.md', '.mdx'}
    
    def find_doc_files(self) -> List[Path]:
        """Find all documentation files in the docs directory."""
        doc_files = []
        
        for ext in self.supported_extensions:
            pattern = f"**/*{ext}"
            files = list(self.docs_dir.glob(pattern))
            doc_files.extend(files)
        
        logger.info(f"Found {len(doc_files)} documentation files")
        return doc_files
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read content from a file, handling different formats."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def convert_rst_to_markdown(self, rst_content: str) -> str:
        """Convert reStructuredText to markdown."""
        try:
            # Use docutils to convert RST to HTML, then extract text
            parts = publish_parts(
                source=rst_content,
                writer_name='html',
                settings_overrides={'output_encoding': 'unicode'}
            )
            
            html_content = parts['html_body']
            
            # Convert HTML to markdown (simple conversion)
            markdown_content = self._html_to_markdown(html_content)
            return markdown_content
            
        except Exception as e:
            logger.warning(f"Error converting RST to markdown: {e}")
            # Fallback: return original content with basic cleanup
            return self._clean_rst_content(rst_content)
    
    def convert_mdx_to_markdown(self, mdx_content: str) -> str:
        """Convert MDX to markdown by removing JSX components."""
        try:
            # Remove JSX components and imports
            content = re.sub(r'import.*?from.*?;', '', mdx_content, flags=re.MULTILINE)
            content = re.sub(r'<[A-Z][^>]*>.*?</[A-Z][^>]*>', '', content, flags=re.DOTALL)
            content = re.sub(r'<[A-Z][^>]*/>', '', content)
            
            # Remove remaining JSX syntax
            content = re.sub(r'\{[^}]*\}', '', content)
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"Error converting MDX to markdown: {e}")
            return mdx_content
    
    def _html_to_markdown(self, html_content: str) -> str:
        """Simple HTML to markdown conversion."""
        # Basic HTML tag removal and conversion
        content = html_content
        
        # Convert headers
        content = re.sub(r'<h1[^>]*>(.*?)</h1>', r'# \1', content, flags=re.DOTALL)
        content = re.sub(r'<h2[^>]*>(.*?)</h2>', r'## \1', content, flags=re.DOTALL)
        content = re.sub(r'<h3[^>]*>(.*?)</h3>', r'### \1', content, flags=re.DOTALL)
        content = re.sub(r'<h4[^>]*>(.*?)</h4>', r'#### \1', content, flags=re.DOTALL)
        
        # Convert code blocks
        content = re.sub(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', r'```\n\1\n```', content, flags=re.DOTALL)
        content = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', content)
        
        # Convert lists
        content = re.sub(r'<ul[^>]*>', '', content)
        content = re.sub(r'</ul>', '', content)
        content = re.sub(r'<ol[^>]*>', '', content)
        content = re.sub(r'</ol>', '', content)
        content = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1', content, flags=re.DOTALL)
        
        # Convert paragraphs
        content = re.sub(r'<p[^>]*>(.*?)</p>', r'\1\n\n', content, flags=re.DOTALL)
        
        # Remove remaining HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _clean_rst_content(self, rst_content: str) -> str:
        """Basic RST content cleaning."""
        content = rst_content
        
        # Remove RST directives
        content = re.sub(r'\.\. [^:]+::.*?\n', '', content)
        content = re.sub(r'\.\. [^:]+:.*?\n', '', content)
        
        # Convert headers
        content = re.sub(r'^([A-Z][A-Z\s]+)$', r'# \1', content, flags=re.MULTILINE)
        content = re.sub(r'^([A-Z][a-z\s]+)$', r'## \1', content, flags=re.MULTILINE)
        
        # Convert code blocks
        content = re.sub(r'::\s*\n(.*?)(?=\n\S|\Z)', r'```\n\1\n```', content, flags=re.DOTALL)
        
        return content
    
    def extract_title_from_content(self, content: str, file_path: Path) -> str:
        """Extract title from content or derive from file path."""
        # Try to find title in content
        lines = content.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
            elif line and not line.startswith('```') and len(line) < 100:
                return line
        
        # Fallback: derive from file path
        return file_path.stem.replace('_', ' ').replace('-', ' ').title()
    
    def process_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Process a single documentation file."""
        content = self.read_file_content(file_path)
        if not content:
            return None
        
        # Convert based on file extension
        if file_path.suffix == '.rst':
            markdown_content = self.convert_rst_to_markdown(content)
        elif file_path.suffix == '.mdx':
            markdown_content = self.convert_mdx_to_markdown(content)
        else:  # .md
            markdown_content = content
        
        # Extract title
        title = self.extract_title_from_content(markdown_content, file_path)
        
        # Create URL-like identifier
        relative_path = file_path.relative_to(self.docs_dir)
        url = f"file://{relative_path.as_posix()}"
        
        return {
            'url': url,
            'title': title,
            'content': markdown_content,
            'source': str(relative_path)
        }
    
    def process_all_docs(self) -> List[Dict[str, str]]:
        """Process all documentation files."""
        doc_files = self.find_doc_files()
        processed_docs = []
        
        for file_path in doc_files:
            try:
                doc = self.process_file(file_path)
                if doc:
                    processed_docs.append(doc)
                    logger.debug(f"Processed: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Successfully processed {len(processed_docs)} documentation files")
        return processed_docs


def load_local_docs(docs_dir: Path) -> List[Dict[str, str]]:
    """Load and process all local documentation files."""
    reader = LocalDocReader(docs_dir)
    return reader.process_all_docs()


if __name__ == "__main__":
    # Test the local reader
    logging.basicConfig(level=logging.INFO)
    
    docs_path = Path("./docs")
    if docs_path.exists():
        docs = load_local_docs(docs_path)
        print(f"Loaded {len(docs)} documents")
        
        for doc in docs[:3]:  # Show first 3
            print(f"Title: {doc['title']}")
            print(f"Source: {doc['source']}")
            print(f"Content length: {len(doc['content'])}")
            print("---")
    else:
        print("Docs directory not found")
