"""Focused web scraper for high-value ML documentation.

Targets specific documentation sections for maximum value within Qdrant Cloud free tier limits.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import yaml

logger = logging.getLogger(__name__)


class FocusedMLScraper:
    """High-value ML documentation scraper optimized for Qdrant Cloud free tier."""
    
    def __init__(self, max_pages_per_source: int = 50, delay_seconds: float = 1.0):
        self.max_pages_per_source = max_pages_per_source
        self.delay_seconds = delay_seconds
        self.scraped_urls: Set[str] = set()
        self.session = None
        
        # Headers to mimic real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    def is_documentation_page(self, url: str, soup: BeautifulSoup) -> bool:
        """Check if a page contains meaningful documentation content."""
        # Check for common documentation indicators
        text_content = soup.get_text()
        
        # Must have substantial content
        if len(text_content) < 500:
            return False
        
        # Should contain typical documentation keywords
        doc_keywords = ['api', 'tutorial', 'guide', 'example', 'reference', 'documentation']
        content_lower = text_content.lower()
        
        if not any(keyword in content_lower for keyword in doc_keywords):
            return False
        
        # Should have code examples or structured content
        has_code = soup.find('code') or soup.find('pre') or '```' in text_content
        has_headers = soup.find(['h1', 'h2', 'h3', 'h4'])
        
        return has_code or has_headers
    
    def clean_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from BeautifulSoup object."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
        
        # Remove navigation elements by class/id
        for element in soup.find_all(['div', 'section'], class_=re.compile(r'nav|sidebar|menu|footer|header|breadcrumb')):
            element.decompose()
        
        # Extract main content area if it exists
        main_content = soup.find(['main', 'article']) or soup.find('div', class_=re.compile(r'content|main|documentation'))
        
        if main_content:
            content_text = main_content.get_text(separator='\n', strip=True)
        else:
            content_text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace and format
        lines = [line.strip() for line in content_text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Remove excessive blank lines
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        
        return clean_text
    
    def extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        # Try different title sources
        title_selectors = [
            'h1',
            'title', 
            '[role="heading"]',
            '.page-title',
            '.doc-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        # Fallback to URL-based title
        path = urlparse(url).path
        return path.split('/')[-1].replace('-', ' ').replace('_', ' ').title() or 'Documentation'
    
    async def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """Scrape a single page and return structured content."""
        if url in self.scraped_urls:
            return None
        
        try:
            logger.info(f"Scraping: {url}")
            response = await self.session.get(url, follow_redirects=True)
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for {url}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if this is a meaningful documentation page
            if not self.is_documentation_page(url, soup):
                logger.debug(f"Skipping non-documentation page: {url}")
                return None
            
            # Extract content
            title = self.extract_title(soup, url)
            content = self.clean_content(soup)
            
            if len(content) < 200:  # Skip pages with minimal content
                logger.debug(f"Skipping minimal content page: {url}")
                return None
            
            self.scraped_urls.add(url)
            
            # Add delay to be respectful
            await asyncio.sleep(self.delay_seconds)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'scraped_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def find_related_links(self, soup: BeautifulSoup, base_url: str, allowed_patterns: List[str]) -> List[str]:
        """Find related documentation links on the current page."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if URL matches allowed patterns
            if any(pattern in full_url for pattern in allowed_patterns):
                # Avoid duplicates and external links
                if full_url not in self.scraped_urls and urlparse(full_url).netloc == urlparse(base_url).netloc:
                    links.append(full_url)
        
        return list(set(links))  # Remove duplicates
    
    async def scrape_source(self, source_config: Dict) -> List[Dict[str, str]]:
        """Scrape a specific documentation source."""
        base_url = source_config['base_url']
        start_urls = source_config['start_urls']
        allowed_patterns = source_config.get('allowed_patterns', [])
        max_pages = min(source_config.get('max_pages', self.max_pages_per_source), self.max_pages_per_source)
        
        logger.info(f"Starting scrape of {source_config['name']} (max {max_pages} pages)")
        
        scraped_docs = []
        urls_to_scrape = list(start_urls)
        
        while urls_to_scrape and len(scraped_docs) < max_pages:
            url = urls_to_scrape.pop(0)
            
            doc = await self.scrape_page(url)
            if doc:
                scraped_docs.append(doc)
                logger.info(f"Scraped {len(scraped_docs)}/{max_pages}: {doc['title']}")
                
                # Find more related links if we haven't reached the limit
                if len(scraped_docs) < max_pages:
                    try:
                        response = await self.session.get(url)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        related_links = self.find_related_links(soup, base_url, allowed_patterns)
                        
                        # Add new links to queue (prioritize by not already being queued)
                        for link in related_links:
                            if link not in urls_to_scrape and link not in self.scraped_urls:
                                urls_to_scrape.append(link)
                                
                    except Exception as e:
                        logger.error(f"Error finding related links for {url}: {e}")
        
        logger.info(f"Completed {source_config['name']}: {len(scraped_docs)} pages scraped")
        return scraped_docs


async def scrape_focused_ml_docs() -> List[Dict[str, str]]:
    """Scrape focused ML documentation optimized for Qdrant Cloud free tier."""
    
    # Configuration for high-value ML documentation sources
    sources = [
        {
            'name': 'PyTorch Core',
            'base_url': 'https://pytorch.org',
            'start_urls': [
                'https://pytorch.org/docs/stable/torch.html',
                'https://pytorch.org/docs/stable/nn.html', 
                'https://pytorch.org/docs/stable/data.html',
                'https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html',
                'https://pytorch.org/tutorials/beginner/basics/data_tutorial.html'
            ],
            'allowed_patterns': ['/docs/stable/', '/tutorials/'],
            'max_pages': 50
        },
        {
            'name': 'Scikit-learn',
            'base_url': 'https://scikit-learn.org',
            'start_urls': [
                'https://scikit-learn.org/stable/user_guide.html',
                'https://scikit-learn.org/stable/modules/classes.html',
                'https://scikit-learn.org/stable/auto_examples/index.html'
            ],
            'allowed_patterns': ['/stable/user_guide', '/stable/modules/', '/stable/auto_examples/'],
            'max_pages': 30
        },
        {
            'name': 'MLflow',
            'base_url': 'https://mlflow.org',
            'start_urls': [
                'https://mlflow.org/docs/latest/tracking.html',
                'https://mlflow.org/docs/latest/model-registry.html',
                'https://mlflow.org/docs/latest/models.html',
                'https://mlflow.org/docs/latest/deployment/index.html'
            ],
            'allowed_patterns': ['/docs/latest/'],
            'max_pages': 30
        },
        {
            'name': 'Ray Serve',
            'base_url': 'https://docs.ray.io',
            'start_urls': [
                'https://docs.ray.io/en/latest/serve/index.html',
                'https://docs.ray.io/en/latest/serve/getting_started.html',
                'https://docs.ray.io/en/latest/serve/deployment-guide/index.html'
            ],
            'allowed_patterns': ['/en/latest/serve/'],
            'max_pages': 20
        },
        {
            'name': 'TensorFlow Guide',
            'base_url': 'https://www.tensorflow.org',
            'start_urls': [
                'https://www.tensorflow.org/guide/keras',
                'https://www.tensorflow.org/guide/data',
                'https://www.tensorflow.org/guide/checkpoint'
            ],
            'allowed_patterns': ['/guide/', '/tutorials/'],
            'max_pages': 20
        }
    ]
    
    all_docs = []
    
    async with FocusedMLScraper(max_pages_per_source=50, delay_seconds=1.0) as scraper:
        for source in sources:
            try:
                source_docs = await scraper.scrape_source(source)
                all_docs.extend(source_docs)
                logger.info(f"Total docs so far: {len(all_docs)}")
                
                # Respect rate limits between sources
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Error scraping {source['name']}: {e}")
    
    logger.info(f"Total scraped documents: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run the focused scraping
    docs = asyncio.run(scrape_focused_ml_docs())
    print(f"Scraped {len(docs)} high-value ML documentation pages")
    
    # Show sample results
    for i, doc in enumerate(docs[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Title: {doc['title']}")
        print(f"URL: {doc['url']}")
        print(f"Word count: {doc['word_count']}")
        print(f"Content preview: {doc['content'][:200]}...")
