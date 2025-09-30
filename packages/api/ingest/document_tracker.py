"""Document tracking and analytics for enhanced ingestion."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DocumentSource:
    """Represents a document source with metadata."""
    url: str
    title: str
    vendor: str
    product: str
    doc_type: str
    topics: List[str]
    priority: str
    quality_score: float
    authority_score: float
    chunk_count: int
    ingested_at: str
    status: str = "success"


class DocumentTracker:
    """Tracks and analyzes document ingestion."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.tracking_file = data_dir / "document_tracking.json"
        self.sources: List[DocumentSource] = []
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load existing tracking data."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    data = json.load(f)
                    self.sources = [DocumentSource(**source) for source in data.get('sources', [])]
            except Exception as e:
                logger.error(f"Error loading tracking data: {e}")
                self.sources = []
        else:
            self.sources = []
    
    def _save_tracking_data(self):
        """Save tracking data to file."""
        try:
            data = {
                'sources': [asdict(source) for source in self.sources],
                'last_updated': datetime.now().isoformat(),
                'total_sources': len(self.sources)
            }
            
            with open(self.tracking_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tracking data: {e}")
    
    def log_document_ingestion(self, 
                              url: str, 
                              title: str, 
                              vendor: str, 
                              product: str, 
                              doc_type: str, 
                              topics: List[str], 
                              priority: str, 
                              quality_score: float, 
                              authority_score: float, 
                              chunk_count: int, 
                              status: str = "success"):
        """Log a document ingestion event."""
        
        source = DocumentSource(
            url=url,
            title=title,
            vendor=vendor,
            product=product,
            doc_type=doc_type,
            topics=topics,
            priority=priority,
            quality_score=quality_score,
            authority_score=authority_score,
            chunk_count=chunk_count,
            ingested_at=datetime.now().isoformat(),
            status=status
        )
        
        # Update existing source or add new one
        existing_index = next((i for i, s in enumerate(self.sources) if s.url == url), None)
        if existing_index is not None:
            self.sources[existing_index] = source
        else:
            self.sources.append(source)
        
        self._save_tracking_data()
        logger.info(f"Logged ingestion: {vendor}/{product} - {title}")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get comprehensive ingestion statistics."""
        if not self.sources:
            return {
                'total_sources': 0,
                'total_chunks': 0,
                'vendors': {},
                'products': {},
                'doc_types': {},
                'topics': {},
                'quality_distribution': {},
                'authority_distribution': {},
                'priority_distribution': {},
                'status_distribution': {}
            }
        
        # Basic counts
        total_sources = len(self.sources)
        total_chunks = sum(source.chunk_count for source in self.sources)
        
        # Vendor distribution
        vendors = {}
        for source in self.sources:
            vendors[source.vendor] = vendors.get(source.vendor, 0) + 1
        
        # Product distribution
        products = {}
        for source in self.sources:
            products[source.product] = products.get(source.product, 0) + 1
        
        # Document type distribution
        doc_types = {}
        for source in self.sources:
            doc_types[source.doc_type] = doc_types.get(source.doc_type, 0) + 1
        
        # Topic distribution
        topics = {}
        for source in self.sources:
            for topic in source.topics:
                topics[topic] = topics.get(topic, 0) + 1
        
        # Quality distribution
        quality_scores = [source.quality_score for source in self.sources]
        quality_distribution = {
            'high': len([s for s in quality_scores if s >= 0.8]),
            'medium': len([s for s in quality_scores if 0.5 <= s < 0.8]),
            'low': len([s for s in quality_scores if s < 0.5])
        }
        
        # Authority distribution
        authority_scores = [source.authority_score for source in self.sources]
        authority_distribution = {
            'high': len([s for s in authority_scores if s >= 0.8]),
            'medium': len([s for s in authority_scores if 0.5 <= s < 0.8]),
            'low': len([s for s in authority_scores if s < 0.5])
        }
        
        # Priority distribution
        priority_distribution = {}
        for source in self.sources:
            priority_distribution[source.priority] = priority_distribution.get(source.priority, 0) + 1
        
        # Status distribution
        status_distribution = {}
        for source in self.sources:
            status_distribution[source.status] = status_distribution.get(source.status, 0) + 1
        
        return {
            'total_sources': total_sources,
            'total_chunks': total_chunks,
            'vendors': vendors,
            'products': products,
            'doc_types': doc_types,
            'topics': topics,
            'quality_distribution': quality_distribution,
            'authority_distribution': authority_distribution,
            'priority_distribution': priority_distribution,
            'status_distribution': status_distribution,
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'average_authority_score': sum(authority_scores) / len(authority_scores) if authority_scores else 0
        }