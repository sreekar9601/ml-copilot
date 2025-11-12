#!/usr/bin/env python3
"""
Smart Web Scraper with Size Controls
Scrapes documentation with strict size limits to stay under 4GB total
"""

import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ScrapedPage:
    """Represents a scraped documentation page."""
    url: str
    title: str
    content: str
    headings: List[str]
    code_blocks: List[str]
    links: List[str]
    size_bytes: int


class SmartWebScraper:
    """
    Web scraper with intelligent size management.
    
    Features:
    - Per-source size limits
    - Total size tracking (4GB cap)
    - Respectful crawling (rate limiting)
    - Content quality filtering
    """
    
    # Size limits
    MAX_TOTAL_SIZE_GB = 3.5  # Stay under 4GB with buffer
    MAX_SOURCE_SIZE_MB = 100  # Max 100MB per documentation source
    MAX_PAGE_SIZE_MB = 5     # Max 5MB per page
    
    # Crawl settings
    MAX_PAGES_PER_SOURCE = 200  # Limit pages per source
    REQUEST_DELAY = 1.0  # Seconds between requests (increased for reliability)
    TIMEOUT = 30  # Request timeout
    MAX_RETRIES = 3  # Retry failed requests
    BACKOFF_FACTOR = 2  # Exponential backoff multiplier
    
    def __init__(self):
        self.total_size_bytes = 0
        self.source_sizes = {}
        self.visited_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Use headers that mimic a real browser to avoid being blocked
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.TIMEOUT),
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _is_valid_doc_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid documentation."""
        parsed = urlparse(url)
        
        # Must be same domain
        if base_domain not in parsed.netloc:
            return False
        
        # Skip common non-doc URLs
        skip_patterns = [
            '/search', '/login', '/signup', '/download',
            '.pdf', '.zip', '.tar.gz', '.jpg', '.png', '.gif', '.svg',
            '/api/v1/', '/api/v2/',  # API endpoints
            '#', 'javascript:', 'mailto:',
            'rss.xml', '/feed', '/atom',  # RSS/Atom feeds
            '/changelog', '/releases',  # Often too verbose
            '.txt', '.json', '.xml'  # Raw data files
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from page."""
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract code blocks from page."""
        code_blocks = []
        
        # Find code blocks
        for code in soup.find_all(['code', 'pre']):
            code_text = code.get_text(strip=True)
            if len(code_text) > 10:  # Skip very short snippets
                code_blocks.append(code_text)
        
        return code_blocks[:50]  # Limit to 50 code blocks per page
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract heading hierarchy from page."""
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            text = heading.get_text(strip=True)
            if text:
                headings.append(text)
        return headings[:20]  # Limit to 20 headings
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract valid documentation links."""
        links = []
        base_domain = urlparse(base_url).netloc
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            if self._is_valid_doc_url(full_url, base_domain):
                links.append(full_url)
        
        return list(set(links))[:100]  # Limit and deduplicate
    
    async def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page with retry logic."""
        if url in self.visited_urls:
            return None
        
        # Retry logic with exponential backoff
        for attempt in range(self.MAX_RETRIES):
            try:
                async with self.session.get(url) as response:
                    # Handle rate limiting
                    if response.status == 429:
                        wait_time = self.BACKOFF_FACTOR ** attempt
                        logger.warning(f"Rate limited on {url}, waiting {wait_time}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        await asyncio.sleep(wait_time)
                        continue
                    
                    if response.status != 200:
                        if attempt == self.MAX_RETRIES - 1:
                            logger.warning(f"Failed to fetch {url}: {response.status} (after {self.MAX_RETRIES} attempts)")
                        return None
                    
                    html = await response.text()
                    size_bytes = len(html.encode('utf-8'))
                
                # Check size limits
                if size_bytes > self.MAX_PAGE_SIZE_MB * 1024 * 1024:
                    logger.warning(f"Page too large, skipping: {url} ({size_bytes / 1024 / 1024:.2f}MB)")
                    return None
                
                # Parse HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract content
                title = soup.find('title')
                title = title.get_text(strip=True) if title else url
                
                content = self._extract_text_content(soup)
                headings = self._extract_headings(soup)
                code_blocks = self._extract_code_blocks(soup)
                links = self._extract_links(soup, url)
                
                # Quality check - must have substantial content
                if len(content) < 500:  # Skip pages with < 500 chars
                    logger.debug(f"Insufficient content, skipping: {url}")
                    return None
                
                self.visited_urls.add(url)
                
                return ScrapedPage(
                    url=url,
                    title=title,
                    content=content,
                    headings=headings,
                    code_blocks=code_blocks,
                    links=links,
                    size_bytes=size_bytes
                )
                
            except asyncio.TimeoutError:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Timeout scraping {url} (after {self.MAX_RETRIES} attempts)")
                else:
                    wait_time = self.BACKOFF_FACTOR ** attempt
                    logger.warning(f"Timeout on {url}, retrying in {wait_time}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    await asyncio.sleep(wait_time)
                    continue
            except Exception as e:
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Error scraping {url}: {e} (after {self.MAX_RETRIES} attempts)")
                else:
                    wait_time = self.BACKOFF_FACTOR ** attempt
                    await asyncio.sleep(wait_time)
                    continue
        
        return None
    
    async def scrape_documentation(
        self, 
        start_url: str, 
        vendor: str,
        max_depth: int = 2
    ) -> List[ScrapedPage]:
        """
        Scrape documentation from a starting URL.
        
        Args:
            start_url: Starting URL to scrape
            vendor: Vendor name for tracking
            max_depth: Maximum depth to crawl
        
        Returns:
            List of scraped pages
        """
        logger.info(f"Starting scrape: {vendor} from {start_url}")
        
        # Initialize tracking
        if vendor not in self.source_sizes:
            self.source_sizes[vendor] = 0
        
        pages = []
        to_visit = [(start_url, 0)]  # (url, depth)
        visited = set()
        
        while to_visit and len(pages) < self.MAX_PAGES_PER_SOURCE:
            # Check size limits
            if self.total_size_bytes > self.MAX_TOTAL_SIZE_GB * 1024 * 1024 * 1024:
                logger.warning(f"Total size limit reached ({self.MAX_TOTAL_SIZE_GB}GB), stopping")
                break
            
            if self.source_sizes[vendor] > self.MAX_SOURCE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Source size limit reached for {vendor} ({self.MAX_SOURCE_SIZE_MB}MB)")
                break
            
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            
            # Scrape page
            page = await self.scrape_page(url)
            
            if page:
                pages.append(page)
                self.total_size_bytes += page.size_bytes
                self.source_sizes[vendor] += page.size_bytes
                
                logger.info(
                    f"  [{len(pages)}/{self.MAX_PAGES_PER_SOURCE}] {vendor}: {url} "
                    f"({page.size_bytes / 1024:.1f}KB, "
                    f"total: {self.total_size_bytes / 1024 / 1024:.1f}MB)"
                )
                
                # Add new links to visit (if not at max depth)
                if depth < max_depth:
                    for link in page.links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
            
            # Rate limiting
            await asyncio.sleep(self.REQUEST_DELAY)
        
        logger.info(
            f"Completed {vendor}: {len(pages)} pages, "
            f"{self.source_sizes[vendor] / 1024 / 1024:.2f}MB"
        )
        
        return pages
    
    def get_size_report(self) -> Dict:
        """Get size usage report."""
        return {
            'total_size_mb': self.total_size_bytes / 1024 / 1024,
            'total_size_gb': self.total_size_bytes / 1024 / 1024 / 1024,
            'max_size_gb': self.MAX_TOTAL_SIZE_GB,
            'utilization_percent': (self.total_size_bytes / (self.MAX_TOTAL_SIZE_GB * 1024 * 1024 * 1024)) * 100,
            'sources': {
                vendor: {
                    'size_mb': size / 1024 / 1024,
                    'size_bytes': size
                }
                for vendor, size in self.source_sizes.items()
            },
            'total_pages': len(self.visited_urls)
        }


async def test_scraper():
    """Test the scraper with a small example."""
    async with SmartWebScraper() as scraper:
        # Test with a small documentation site
        pages = await scraper.scrape_documentation(
            start_url="https://pytorch.org/docs/stable/torch.html",
            vendor="PyTorch Test",
            max_depth=1  # Only scrape 1 level deep for testing
        )
        
        print(f"\nScraped {len(pages)} pages")
        print("\nSize Report:")
        report = scraper.get_size_report()
        print(json.dumps(report, indent=2))
        
        if pages:
            print(f"\nSample page:")
            print(f"Title: {pages[0].title}")
            print(f"Content length: {len(pages[0].content)} chars")
            print(f"Headings: {pages[0].headings[:3]}")
            print(f"Code blocks: {len(pages[0].code_blocks)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_scraper())

