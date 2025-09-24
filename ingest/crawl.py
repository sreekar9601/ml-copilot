"""Web crawling module for fetching and cleaning documentation content."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from readability import Document

logger = logging.getLogger(__name__)


class DocumentCrawler:
    """Crawls and processes web documentation into clean Markdown."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.aclose()
    
    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL with retries and redirect handling."""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching URL (attempt {attempt + 1}): {url}")
                response = await self.session.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning(f"Access forbidden for {url} - may need different user agent or headers")
                elif e.response.status_code == 404:
                    logger.warning(f"Page not found: {url}")
                else:
                    logger.warning(f"HTTP {e.response.status_code} error fetching {url}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except httpx.HTTPError as e:
                logger.warning(f"HTTP error fetching {url}: {e}")
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None
        return None
    
    def clean_html(self, html: str, base_url: str) -> Optional[str]:
        """Extract main content from HTML and convert to clean Markdown."""
        try:
            # Use readability to extract main content
            doc = Document(html)
            clean_html = doc.summary()
            
            # Parse with BeautifulSoup for further processing
            soup = BeautifulSoup(clean_html, 'html.parser')
            
            # Add IDs to headings for anchor links
            self._add_heading_ids(soup)
            
            # Convert relative links to absolute
            self._resolve_links(soup, base_url)
            
            # Convert to Markdown
            markdown = md(
                str(soup),
                heading_style="ATX",
                bullets="-",
                strip=['script', 'style']
            )
            
            # Clean up extra whitespace
            lines = [line.strip() for line in markdown.split('\n')]
            cleaned_lines = []
            prev_empty = False
            
            for line in lines:
                if not line:
                    if not prev_empty:
                        cleaned_lines.append('')
                    prev_empty = True
                else:
                    cleaned_lines.append(line)
                    prev_empty = False
            
            return '\n'.join(cleaned_lines).strip()
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            return None
    
    def _add_heading_ids(self, soup: BeautifulSoup) -> None:
        """Add ID attributes to heading tags for deep linking."""
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if not tag.get('id'):
                # Create ID from heading text
                heading_text = tag.get_text().strip()
                heading_id = self._slugify(heading_text)
                tag['id'] = heading_id
    
    def _resolve_links(self, soup: BeautifulSoup, base_url: str) -> None:
        """Convert relative links to absolute URLs."""
        for tag in soup.find_all(['a', 'img']):
            attr = 'href' if tag.name == 'a' else 'src'
            if tag.get(attr):
                absolute_url = urljoin(base_url, tag[attr])
                tag[attr] = absolute_url
    
    @staticmethod
    def _slugify(text: str) -> str:
        """Convert text to a URL-safe slug."""
        import re
        # Convert to lowercase and replace non-alphanumeric with hyphens
        slug = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
        slug = re.sub(r'[\s-]+', '-', slug).strip('-')
        return slug
    
    async def crawl_url(self, url: str) -> Optional[Dict[str, str]]:
        """Crawl a single URL and return processed content."""
        html = await self.fetch_url(url)
        if not html:
            return None
        
        markdown = self.clean_html(html, url)
        if not markdown:
            return None
        
        # Extract title from URL or content
        title = self._extract_title(html, url)
        
        return {
            'url': url,
            'title': title,
            'content': markdown,
            'domain': urlparse(url).netloc
        }
    
    def _extract_title(self, html: str, url: str) -> str:
        """Extract page title from HTML or derive from URL."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
        except Exception:
            pass
        
        # Fallback: use URL path
        path = urlparse(url).path
        return path.split('/')[-1] or urlparse(url).netloc
    
    async def crawl_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """Crawl multiple URLs concurrently."""
        tasks = [self.crawl_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        crawled_docs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error crawling {urls[i]}: {result}")
            elif result:
                crawled_docs.append(result)
        
        logger.info(f"Successfully crawled {len(crawled_docs)}/{len(urls)} URLs")
        return crawled_docs


async def load_and_crawl_seeds(seeds_path: Path) -> List[Dict[str, str]]:
    """Load URLs from seeds.yaml and crawl them."""
    import yaml
    
    try:
        with open(seeds_path, 'r', encoding='utf-8') as f:
            seeds = yaml.safe_load(f)
        
        urls = seeds.get('urls', [])
        logger.info(f"Loaded {len(urls)} URLs from seeds file")
        
        async with DocumentCrawler() as crawler:
            return await crawler.crawl_urls(urls)
            
    except Exception as e:
        logger.error(f"Error loading/crawling seeds: {e}")
        return []


if __name__ == "__main__":
    # Test the crawler
    import sys
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        seeds_path = Path(__file__).parent / "seeds.yaml"
        docs = await load_and_crawl_seeds(seeds_path)
        print(f"Crawled {len(docs)} documents")
        for doc in docs[:3]:  # Show first 3
            print(f"Title: {doc['title']}")
            print(f"URL: {doc['url']}")
            print(f"Content length: {len(doc['content'])}")
            print("---")
    
    asyncio.run(test())
