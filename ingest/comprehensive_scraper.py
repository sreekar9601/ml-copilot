"""Comprehensive ML documentation scraper for maximum coverage.

Targets extensive ML documentation sources for a comprehensive knowledge base.
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


class ComprehensiveMLScraper:
    """Comprehensive ML documentation scraper for maximum knowledge coverage."""
    
    def __init__(self, max_pages_per_source: int = 300, delay_seconds: float = 0.3):
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
        text_content = soup.get_text().lower()
        
        # Skip pages with too little content
        if len(text_content) < 500:
            return False
            
        # Look for documentation keywords
        doc_keywords = [
            'api', 'reference', 'guide', 'tutorial', 'example', 'function',
            'class', 'method', 'parameter', 'return', 'usage', 'installation',
            'configuration', 'training', 'model', 'dataset', 'pipeline'
        ]
        
        keyword_count = sum(1 for keyword in doc_keywords if keyword in text_content)
        return keyword_count >= 3
    
    def extract_clean_text(self, soup: BeautifulSoup, url: str) -> str:
        """Extract clean text content from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_title(self, soup: BeautifulSoup, url: str) -> str:
        """Extract page title."""
        title = soup.find('title')
        if title:
            return title.get_text().strip()
        
        # Fallback to h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text().strip()
        
        # Fallback to URL
        return urlparse(url).path.split('/')[-1] or 'Untitled'
    
    def extract_links(self, soup: BeautifulSoup, base_url: str, allowed_patterns: List[str]) -> List[str]:
        """Extract relevant documentation links."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Check if URL matches allowed patterns
            if any(pattern in full_url for pattern in allowed_patterns):
                # Avoid fragments and query parameters for now
                clean_url = full_url.split('#')[0].split('?')[0]
                if clean_url not in self.scraped_urls:
                    links.append(clean_url)
        
        return links
    
    async def scrape_page(self, url: str) -> Optional[Dict[str, str]]:
        """Scrape a single page and return document data."""
        if url in self.scraped_urls:
            return None
            
        try:
            response = await self.session.get(url, follow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            if not self.is_documentation_page(url, soup):
                return None
            
            title = self.extract_title(soup, url)
            content = self.extract_clean_text(soup, url)
            
            if len(content) < 200:  # Skip very short pages
                return None
            
            self.scraped_urls.add(url)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.warning(f"Error scraping {url}: {e}")
            return None
    
    async def scrape_source(self, source: Dict[str, any]) -> List[Dict[str, str]]:
        """Scrape all pages from a documentation source."""
        logger.info(f"Starting scrape of {source['name']} (max {source['max_pages']} pages)")
        
        docs = []
        urls_to_visit = source['start_urls'].copy()
        visited_count = 0
        
        while urls_to_visit and visited_count < source['max_pages']:
            current_batch = urls_to_visit[:10]  # Process in batches
            urls_to_visit = urls_to_visit[10:]
            
            # Scrape current batch
            tasks = [self.scrape_page(url) for url in current_batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result:
                    docs.append(result)
                    visited_count += 1
                    
                    # Extract new links from this page
                    try:
                        response = await self.session.get(result['url'])
                        soup = BeautifulSoup(response.content, 'html.parser')
                        new_links = self.extract_links(soup, source['base_url'], source['allowed_patterns'])
                        
                        # Add new links to visit (avoid duplicates)
                        for link in new_links:
                            if link not in self.scraped_urls and link not in urls_to_visit:
                                urls_to_visit.append(link)
                    except:
                        pass
            
            # Rate limiting
            await asyncio.sleep(self.delay_seconds)
            
            if visited_count % 10 == 0:
                logger.info(f"Scraped {visited_count}/{source['max_pages']} pages from {source['name']}")
        
        logger.info(f"Completed {source['name']}: {len(docs)} pages scraped")
        return docs


async def scrape_comprehensive_ml_docs():
    """Scrape comprehensive ML documentation from multiple sources."""
    
    sources = [
        {
            'name': 'PyTorch Core',
            'base_url': 'https://pytorch.org',
            'start_urls': [
                'https://pytorch.org/docs/stable/torch.html',
                'https://pytorch.org/docs/stable/nn.html',
                'https://pytorch.org/docs/stable/data.html',
                'https://pytorch.org/docs/stable/optim.html',
                'https://pytorch.org/docs/stable/distributed.html',
                'https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html',
                'https://pytorch.org/tutorials/beginner/basics/data_tutorial.html',
                'https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html',
                'https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html',
            ],
            'allowed_patterns': ['/docs/stable/', '/tutorials/'],
            'max_pages': 200
        },
        {
            'name': 'Scikit-learn',
            'base_url': 'https://scikit-learn.org',
            'start_urls': [
                'https://scikit-learn.org/stable/user_guide.html',
                'https://scikit-learn.org/stable/modules/classes.html',
                'https://scikit-learn.org/stable/auto_examples/index.html',
                'https://scikit-learn.org/stable/modules/preprocessing.html',
                'https://scikit-learn.org/stable/modules/model_selection.html',
                'https://scikit-learn.org/stable/modules/ensemble.html',
            ],
            'allowed_patterns': ['/stable/user_guide', '/stable/modules/', '/stable/auto_examples/'],
            'max_pages': 150
        },
        {
            'name': 'MLflow',
            'base_url': 'https://mlflow.org',
            'start_urls': [
                'https://mlflow.org/docs/latest/tracking.html',
                'https://mlflow.org/docs/latest/model-registry.html',
                'https://mlflow.org/docs/latest/models.html',
                'https://mlflow.org/docs/latest/deployment/index.html',
                'https://mlflow.org/docs/latest/python_api/index.html',
                'https://mlflow.org/docs/latest/concepts.html',
            ],
            'allowed_patterns': ['/docs/latest/'],
            'max_pages': 100
        },
        {
            'name': 'TensorFlow Guide',
            'base_url': 'https://www.tensorflow.org',
            'start_urls': [
                'https://www.tensorflow.org/guide/keras',
                'https://www.tensorflow.org/guide/data',
                'https://www.tensorflow.org/guide/checkpoint',
                'https://www.tensorflow.org/guide/variable',
                'https://www.tensorflow.org/guide/keras/functional',
                'https://www.tensorflow.org/guide/keras/sequential_model',
                'https://www.tensorflow.org/guide/keras/train_and_evaluate',
                'https://www.tensorflow.org/guide/keras/save_and_serialize',
            ],
            'allowed_patterns': ['/guide/', '/tutorials/'],
            'max_pages': 150
        },
        {
            'name': 'Hugging Face Transformers',
            'base_url': 'https://huggingface.co',
            'start_urls': [
                'https://huggingface.co/docs/transformers/index',
                'https://huggingface.co/docs/transformers/task_summary',
                'https://huggingface.co/docs/transformers/model_doc/bert',
                'https://huggingface.co/docs/transformers/model_doc/gpt2',
                'https://huggingface.co/docs/transformers/training',
                'https://huggingface.co/docs/transformers/preprocessing',
            ],
            'allowed_patterns': ['/docs/transformers/'],
            'max_pages': 100
        },
        {
            'name': 'XGBoost',
            'base_url': 'https://xgboost.readthedocs.io',
            'start_urls': [
                'https://xgboost.readthedocs.io/en/stable/python/python_intro.html',
                'https://xgboost.readthedocs.io/en/stable/tutorials/index.html',
                'https://xgboost.readthedocs.io/en/stable/parameter.html',
                'https://xgboost.readthedocs.io/en/stable/python/python_api.html',
            ],
            'allowed_patterns': ['/en/stable/'],
            'max_pages': 80
        },
        {
            'name': 'LightGBM',
            'base_url': 'https://lightgbm.readthedocs.io',
            'start_urls': [
                'https://lightgbm.readthedocs.io/en/latest/Python-Intro.html',
                'https://lightgbm.readthedocs.io/en/latest/Parameters.html',
                'https://lightgbm.readthedocs.io/en/latest/Python-API.html',
                'https://lightgbm.readthedocs.io/en/latest/Quick-Start.html',
            ],
            'allowed_patterns': ['/en/latest/'],
            'max_pages': 60
        },
        {
            'name': 'Ray Serve',
            'base_url': 'https://docs.ray.io',
            'start_urls': [
                'https://docs.ray.io/en/latest/serve/index.html',
                'https://docs.ray.io/en/latest/serve/getting_started.html',
                'https://docs.ray.io/en/latest/serve/deployment-guide/index.html',
                'https://docs.ray.io/en/latest/serve/advanced-config.html',
                'https://docs.ray.io/en/latest/serve/api/index.html',
            ],
            'allowed_patterns': ['/en/latest/serve/'],
            'max_pages': 80
        },
        {
            'name': 'Kubernetes ML',
            'base_url': 'https://kubernetes.io',
            'start_urls': [
                'https://kubernetes.io/docs/concepts/workloads/controllers/job/',
                'https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/',
                'https://kubernetes.io/docs/concepts/services-networking/service/',
                'https://kubernetes.io/docs/concepts/storage/persistent-volumes/',
            ],
            'allowed_patterns': ['/docs/concepts/'],
            'max_pages': 50
        },
        {
            'name': 'Docker ML',
            'base_url': 'https://docs.docker.com',
            'start_urls': [
                'https://docs.docker.com/get-started/overview/',
                'https://docs.docker.com/build/',
                'https://docs.docker.com/compose/',
                'https://docs.docker.com/engine/reference/builder/',
            ],
            'allowed_patterns': ['/docs/'],
            'max_pages': 40
        }
    ]

    all_docs = []
    
    async with ComprehensiveMLScraper(max_pages_per_source=300, delay_seconds=0.3) as scraper:
        for source in sources:
            try:
                docs = await scraper.scrape_source(source)
                all_docs.extend(docs)
                logger.info(f"Total docs so far: {len(all_docs)}")
            except Exception as e:
                logger.error(f"Error scraping {source['name']}: {e}")
                continue
    
    logger.info(f"Total scraped documents: {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(scrape_comprehensive_ml_docs())
