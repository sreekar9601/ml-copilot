#!/usr/bin/env python3
"""
Production-grade web scraper for ML documentation.
Focuses on data quality, reliability, and comprehensive coverage.
"""

import asyncio
import aiohttp
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time
import re
from bs4 import BeautifulSoup
import markdownify
from readability import Document

logger = logging.getLogger(__name__)

@dataclass
class ScrapedDocument:
    """Represents a scraped document with quality metadata."""
    url: str
    title: str
    content: str
    source_vendor: str
    doc_type: str
    topics: List[str]
    priority: str
    quality_score: float
    word_count: int
    has_code_examples: bool
    heading_structure: List[str]
    metadata: Dict[str, any]

class ProductionScraper:
    """Production-grade scraper with quality controls and comprehensive coverage."""
    
    def __init__(self, sources_config: Path, output_dir: Path):
        self.sources_config = sources_config
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Load configuration
        with open(sources_config, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.quality_standards = self.config['quality_standards']
        self.scraping_config = self.config['scraping_config']
        
        # Session configuration
        self.session = None
        self.rate_limit = self.scraping_config['rate_limit']
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            'User-Agent': self.scraping_config['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def scrape_all_sources(self) -> List[ScrapedDocument]:
        """Scrape all configured data sources with quality controls."""
        all_documents = []
        
        for vendor, vendor_config in self.config['data_sources'].items():
            logger.info(f"Scraping {vendor} documentation...")
            
            for source in vendor_config['sources']:
                try:
                    documents = await self._scrape_source(vendor, source)
                    all_documents.extend(documents)
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit)
                    
                except Exception as e:
                    logger.error(f"Error scraping {source['url']}: {e}")
                    continue
        
        # Quality filtering
        filtered_documents = self._filter_by_quality(all_documents)
        
        logger.info(f"Scraped {len(all_documents)} documents, {len(filtered_documents)} passed quality filters")
        return filtered_documents
    
    async def _scrape_source(self, vendor: str, source: Dict) -> List[ScrapedDocument]:
        """Scrape a single source with comprehensive content extraction."""
        documents = []
        
        try:
            # Fetch main page
            main_doc = await self._fetch_and_parse(source['url'], vendor, source)
            if main_doc:
                documents.append(main_doc)
            
            # Find related pages
            related_urls = await self._find_related_pages(source['url'], vendor)
            
            for url in related_urls[:5]:  # Limit to 5 related pages
                try:
                    doc = await self._fetch_and_parse(url, vendor, source)
                    if doc:
                        documents.append(doc)
                    await asyncio.sleep(self.rate_limit)
                except Exception as e:
                    logger.warning(f"Error scraping related page {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping source {source['url']}: {e}")
        
        return documents
    
    async def _fetch_and_parse(self, url: str, vendor: str, source: Dict) -> Optional[ScrapedDocument]:
        """Fetch and parse a single URL with quality assessment."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                html = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract title
                title = self._extract_title(soup, url)
                
                # Extract main content
                content = self._extract_content(soup, url)
                
                if not content or len(content) < self.quality_standards['min_content_length']:
                    logger.debug(f"Content too short for {url}")
                    return None
                
                # Quality assessment
                quality_score = self._assess_content_quality(content, soup)
                if quality_score < 0.3:  # Minimum quality threshold
                    logger.debug(f"Quality score too low ({quality_score:.2f}) for {url}")
                    return None
                
                # Extract metadata
                word_count = len(content.split())
                has_code_examples = self._has_code_examples(content)
                heading_structure = self._extract_headings(soup)
                
                return ScrapedDocument(
                    url=url,
                    title=title,
                    content=content,
                    source_vendor=vendor,
                    doc_type=source['type'],
                    topics=source['topics'],
                    priority=source['priority'],
                    quality_score=quality_score,
                    word_count=word_count,
                    has_code_examples=has_code_examples,
                    heading_structure=heading_structure,
                    metadata={
                        'scraped_at': time.time(),
                        'response_status': response.status,
                        'content_type': response.headers.get('content-type', ''),
                        'base_url': source.get('base_url', ''),
                        'vendor': vendor
                    }
                )
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract document title with fallbacks."""
        # Try multiple title sources
        title_selectors = [
            'h1',
            'title',
            '[data-testid="title"]',
            '.page-title',
            '.document-title'
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                return element.get_text().strip()
        
        # Fallback to URL-based title
        return url.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
    
    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract main content with readability improvements."""
        # Remove unwanted elements
        for element in soup(['nav', 'footer', 'header', 'aside', 'script', 'style', 'noscript']):
            element.decompose()
        
        # Try to find main content area
        content_selectors = [
            'main',
            'article',
            '.content',
            '.documentation',
            '.docs-content',
            '#content',
            '.page-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if not main_content:
            return ""
        
        # Convert to markdown
        content = markdownify.markdownify(str(main_content), heading_style="ATX")
        
        # Clean up content
        content = self._clean_content(content)
        
        return content
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        # Remove empty lines at start/end
        content = content.strip()
        
        return content
    
    def _assess_content_quality(self, content: str, soup: BeautifulSoup) -> float:
        """Assess content quality based on multiple factors."""
        score = 0.0
        
        # Length factor (0-0.3)
        word_count = len(content.split())
        if word_count > 500:
            score += 0.3
        elif word_count > 200:
            score += 0.2
        elif word_count > 100:
            score += 0.1
        
        # Code examples factor (0-0.3)
        if self._has_code_examples(content):
            score += 0.3
        
        # Structure factor (0-0.2)
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if len(headings) > 2:
            score += 0.2
        elif len(headings) > 0:
            score += 0.1
        
        # Technical content factor (0-0.2)
        technical_terms = ['function', 'class', 'method', 'api', 'config', 'parameter', 'example', 'tutorial']
        technical_count = sum(1 for term in technical_terms if term.lower() in content.lower())
        if technical_count > 3:
            score += 0.2
        elif technical_count > 1:
            score += 0.1
        
        return min(score, 1.0)
    
    def _has_code_examples(self, content: str) -> bool:
        """Check if content has code examples."""
        code_indicators = ['```', '`', 'def ', 'class ', 'import ', 'from ', 'function', 'const ', 'var ']
        return any(indicator in content for indicator in code_indicators)
    
    def _extract_headings(self, soup: BeautifulSoup) -> List[str]:
        """Extract document heading structure."""
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        return [h.get_text().strip() for h in headings[:10]]  # Limit to first 10 headings
    
    async def _find_related_pages(self, base_url: str, vendor: str) -> List[str]:
        """Find related pages to scrape for comprehensive coverage."""
        try:
            async with self.session.get(base_url) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find links to related documentation
                related_links = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    # Filter for documentation links
                    if self._is_documentation_link(full_url, vendor):
                        related_links.append(full_url)
                
                return list(set(related_links))[:10]  # Limit to 10 related pages
                
        except Exception as e:
            logger.error(f"Error finding related pages for {base_url}: {e}")
            return []
    
    def _is_documentation_link(self, url: str, vendor: str) -> bool:
        """Check if a link is likely to be documentation content."""
        # Include patterns
        include_patterns = [
            '/tutorial', '/guide', '/docs/', '/documentation/',
            '/api/', '/reference/', '/examples/', '/how-to'
        ]
        
        # Exclude patterns
        exclude_patterns = [
            '/download', '/changelog', '/version', '/api/raw',
            '/search', '/login', '/register', '/contact'
        ]
        
        url_lower = url.lower()
        
        # Check exclude patterns first
        if any(pattern in url_lower for pattern in exclude_patterns):
            return False
        
        # Check include patterns
        return any(pattern in url_lower for pattern in include_patterns)
    
    def _filter_by_quality(self, documents: List[ScrapedDocument]) -> List[ScrapedDocument]:
        """Filter documents by quality standards."""
        filtered = []
        
        for doc in documents:
            # Apply quality filters
            if (doc.word_count < self.quality_standards['min_content_length'] or
                doc.word_count > self.quality_standards['max_content_length'] or
                doc.quality_score < 0.3):
                continue
            
            # Check for required elements
            if not doc.title or not doc.content:
                continue
            
            filtered.append(doc)
        
        return filtered

async def main():
    """Main function to run production scraping."""
    logging.basicConfig(level=logging.INFO)
    
    sources_config = Path(__file__).parent / "production_sources.yaml"
    output_dir = Path(__file__).parent / "scraped_data"
    
    async with ProductionScraper(sources_config, output_dir) as scraper:
        documents = await scraper.scrape_all_sources()
        
        # Save results
        output_file = output_dir / "scraped_documents.json"
        import json
        with open(output_file, 'w') as f:
            json.dump([doc.__dict__ for doc in documents], f, indent=2)
        
        logger.info(f"Scraped {len(documents)} high-quality documents")
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
