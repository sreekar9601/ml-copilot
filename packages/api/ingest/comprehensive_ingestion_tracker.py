#!/usr/bin/env python3
"""
Comprehensive ML Documentation Ingestion System with Tracking
Ingests documentation from comprehensive_ml_sources.yaml with progress tracking
"""

import os
import sys
import json
import yaml
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IngestionStatus:
    """Track ingestion status for each source."""
    url: str
    vendor: str
    tier: str
    status: str  # 'pending', 'in_progress', 'success', 'failed'
    chunks_created: int = 0
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    retry_count: int = 0


class IngestionTracker:
    """Tracks ingestion progress in SQLite database."""
    
    def __init__(self, db_path: str = "ingestion_tracking.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize tracking database."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                vendor TEXT NOT NULL,
                tier TEXT NOT NULL,
                status TEXT NOT NULL,
                chunks_created INTEGER DEFAULT 0,
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                retry_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_sources INTEGER,
                successful INTEGER DEFAULT 0,
                failed INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                status TEXT NOT NULL
            )
        """)
        
        self.conn.commit()
        logger.info(f"Initialized tracking database at {self.db_path}")
    
    def start_run(self, run_id: str, total_sources: int) -> None:
        """Start a new ingestion run."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ingestion_runs (run_id, started_at, total_sources, status)
            VALUES (?, ?, ?, 'in_progress')
        """, (run_id, datetime.now().isoformat(), total_sources))
        self.conn.commit()
        logger.info(f"Started ingestion run: {run_id} with {total_sources} sources")
    
    def update_source_status(self, status: IngestionStatus) -> None:
        """Update status for a specific source."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ingestion_status 
            (url, vendor, tier, status, chunks_created, error_message, 
             started_at, completed_at, retry_count, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            status.url, status.vendor, status.tier, status.status,
            status.chunks_created, status.error_message,
            status.started_at, status.completed_at, status.retry_count,
            datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def complete_run(self, run_id: str) -> None:
        """Mark a run as completed and update statistics."""
        cursor = self.conn.cursor()
        
        # Get statistics
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
                SUM(chunks_created) as total_chunks
            FROM ingestion_status
        """)
        successful, failed, total_chunks = cursor.fetchone()
        
        # Update run
        cursor.execute("""
            UPDATE ingestion_runs 
            SET completed_at = ?, successful = ?, failed = ?, 
                total_chunks = ?, status = 'completed'
            WHERE run_id = ?
        """, (datetime.now().isoformat(), successful or 0, failed or 0, 
              total_chunks or 0, run_id))
        
        self.conn.commit()
        logger.info(f"Completed run {run_id}: {successful} successful, {failed} failed, {total_chunks} chunks")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get overall ingestion status summary."""
        cursor = self.conn.cursor()
        
        # Overall stats
        cursor.execute("""
            SELECT status, COUNT(*) as count, SUM(chunks_created) as chunks
            FROM ingestion_status
            GROUP BY status
        """)
        status_counts = {row[0]: {"count": row[1], "chunks": row[2]} 
                        for row in cursor.fetchall()}
        
        # By tier
        cursor.execute("""
            SELECT tier, status, COUNT(*) as count
            FROM ingestion_status
            GROUP BY tier, status
        """)
        tier_stats = {}
        for tier, status, count in cursor.fetchall():
            if tier not in tier_stats:
                tier_stats[tier] = {}
            tier_stats[tier][status] = count
        
        # Recent runs
        cursor.execute("""
            SELECT * FROM ingestion_runs
            ORDER BY started_at DESC
            LIMIT 5
        """)
        recent_runs = [dict(zip([col[0] for col in cursor.description], row))
                      for row in cursor.fetchall()]
        
        return {
            "status_counts": status_counts,
            "tier_stats": tier_stats,
            "recent_runs": recent_runs
        }
    
    def get_pending_sources(self) -> List[Dict[str, Any]]:
        """Get sources that haven't been ingested yet."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT url, vendor, tier FROM ingestion_status
            WHERE status IN ('pending', 'failed') AND retry_count < 3
            ORDER BY tier, vendor
        """)
        return [{"url": row[0], "vendor": row[1], "tier": row[2]} 
                for row in cursor.fetchall()]
    
    def export_report(self, output_file: str = "ingestion_report.json") -> None:
        """Export detailed ingestion report."""
        summary = self.get_status_summary()
        
        # Add detailed source list
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM ingestion_status
            ORDER BY tier, vendor, status
        """)
        sources = [dict(zip([col[0] for col in cursor.description], row))
                  for row in cursor.fetchall()]
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "sources": sources
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported ingestion report to {output_file}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class ComprehensiveMLIngestion:
    """Main ingestion system for comprehensive ML documentation."""
    
    def __init__(self, sources_file: str, data_dir: Path):
        self.sources_file = sources_file
        self.data_dir = Path(data_dir)
        self.tracker = IngestionTracker()
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load sources
        with open(sources_file, 'r') as f:
            self.sources_config = yaml.safe_load(f)
    
    def initialize_tracking(self) -> None:
        """Initialize tracking for all sources."""
        total_sources = 0
        
        for tier_name, tier_data in self.sources_config.items():
            for category_name, category_data in tier_data.items():
                vendor = category_data.get('name', category_name)
                sources = category_data.get('sources', [])
                
                for source in sources:
                    url = source.get('url')
                    if url:
                        status = IngestionStatus(
                            url=url,
                            vendor=vendor,
                            tier=tier_name,
                            status='pending'
                        )
                        self.tracker.update_source_status(status)
                        total_sources += 1
        
        self.tracker.start_run(self.run_id, total_sources)
        logger.info(f"Initialized tracking for {total_sources} sources")
    
    async def ingest_source(self, url: str, vendor: str, tier: str, 
                           source_type: str, max_depth: int) -> IngestionStatus:
        """Ingest a single documentation source."""
        status = IngestionStatus(
            url=url,
            vendor=vendor,
            tier=tier,
            status='in_progress',
            started_at=datetime.now().isoformat()
        )
        self.tracker.update_source_status(status)
        
        try:
            logger.info(f"🔄 Ingesting {vendor}: {url}")
            
            # TODO: Implement actual scraping and ingestion
            # This is a placeholder - you'll need to integrate with your scraping system
            # For now, we'll simulate the process
            
            await asyncio.sleep(0.1)  # Simulate work
            
            # Simulate success
            status.status = 'success'
            status.chunks_created = 10  # Placeholder
            status.completed_at = datetime.now().isoformat()
            
            logger.info(f"✅ Successfully ingested {vendor}: {status.chunks_created} chunks")
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest {vendor} ({url}): {e}")
            status.status = 'failed'
            status.error_message = str(e)
            status.completed_at = datetime.now().isoformat()
            status.retry_count += 1
        
        self.tracker.update_source_status(status)
        return status
    
    async def run_ingestion(self, tier_filter: Optional[str] = None) -> None:
        """Run the complete ingestion process."""
        logger.info(f"🚀 Starting comprehensive ML documentation ingestion: {self.run_id}")
        
        self.initialize_tracking()
        
        tasks = []
        for tier_name, tier_data in self.sources_config.items():
            # Skip if tier filter is specified and doesn't match
            if tier_filter and tier_name != tier_filter:
                continue
            
            for category_name, category_data in tier_data.items():
                vendor = category_data.get('name', category_name)
                sources = category_data.get('sources', [])
                
                for source in sources:
                    url = source.get('url')
                    source_type = source.get('type', 'guide')
                    max_depth = source.get('max_depth', 2)
                    
                    if url:
                        task = self.ingest_source(url, vendor, tier_name, 
                                                 source_type, max_depth)
                        tasks.append(task)
        
        # Run all ingestions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Complete the run
        self.tracker.complete_run(self.run_id)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print ingestion summary."""
        summary = self.tracker.get_status_summary()
        
        print("\n" + "="*60)
        print("📊 INGESTION SUMMARY")
        print("="*60)
        
        print("\n📈 Overall Status:")
        for status, data in summary['status_counts'].items():
            print(f"  {status.upper()}: {data['count']} sources, {data['chunks']} chunks")
        
        print("\n🎯 By Tier:")
        for tier, stats in summary['tier_stats'].items():
            print(f"  {tier}:")
            for status, count in stats.items():
                print(f"    {status}: {count}")
        
        # Export detailed report
        self.tracker.export_report(f"ingestion_report_{self.run_id}.json")
        print(f"\n📄 Detailed report: ingestion_report_{self.run_id}.json")
        print(f"📋 Tracking database: {self.tracker.db_path}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive ML Documentation Ingestion')
    parser.add_argument('--sources', default='comprehensive_ml_sources.yaml',
                       help='Path to sources YAML file')
    parser.add_argument('--data-dir', default='./data',
                       help='Data directory for storage')
    parser.add_argument('--tier', default=None,
                       help='Specific tier to ingest (e.g., tier_1_core_ml)')
    parser.add_argument('--status', action='store_true',
                       help='Show current ingestion status and exit')
    
    args = parser.parse_args()
    
    if args.status:
        # Just show status
        tracker = IngestionTracker()
        summary = tracker.get_status_summary()
        print(json.dumps(summary, indent=2))
        tracker.export_report("current_status.json")
        tracker.close()
        return
    
    # Run ingestion
    ingestion = ComprehensiveMLIngestion(args.sources, args.data_dir)
    await ingestion.run_ingestion(tier_filter=args.tier)
    ingestion.tracker.close()


if __name__ == "__main__":
    asyncio.run(main())


