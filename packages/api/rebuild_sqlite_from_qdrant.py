"""Rebuild SQLite FTS5 index from Qdrant data."""
import os
import sqlite3
from pathlib import Path
from qdrant_client import QdrantClient
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def rebuild_sqlite_from_qdrant():
    """Rebuild SQLite FTS5 database from Qdrant collection."""
    
    # Configuration
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    qdrant_collection = os.getenv('QDRANT_COLLECTION', 'ml_docs')
    sqlite_path = Path('./data/bm25.db')
    
    print(f"Connecting to Qdrant: {qdrant_url}")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    # Get collection info
    collection_info = qdrant_client.get_collection(collection_name=qdrant_collection)
    total_points = collection_info.points_count
    print(f"Total points in Qdrant: {total_points}")
    
    # Delete old SQLite database
    if sqlite_path.exists():
        print(f"Deleting old SQLite database: {sqlite_path}")
        sqlite_path.unlink()
    
    # Create new SQLite database
    print("Creating new SQLite FTS5 table...")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()
    
    # Create FTS5 table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
        USING fts5(
            chunk_id UNINDEXED,
            content,
            tokenize = 'porter unicode61'
        )
    """)
    conn.commit()
    
    # Scroll through all points in Qdrant and add to SQLite
    print("Fetching data from Qdrant and populating SQLite...")
    offset = None
    batch_size = 100
    total_added = 0
    
    with tqdm(total=total_points) as pbar:
        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not results:
                break
            
            # Prepare batch insert
            batch_data = []
            for point in results:
                chunk_id = str(point.id)
                content = point.payload.get('text', '')
                if content:  # Only add if content exists
                    batch_data.append((chunk_id, content))
            
            # Insert batch
            if batch_data:
                cursor.executemany(
                    "INSERT INTO documents_fts (chunk_id, content) VALUES (?, ?)",
                    batch_data
                )
                conn.commit()
                total_added += len(batch_data)
            
            pbar.update(len(results))
            
            if next_offset is None:
                break
            offset = next_offset
    
    conn.close()
    
    print(f"\n✅ SQLite FTS5 rebuild complete!")
    print(f"   Total points in Qdrant: {total_points}")
    print(f"   Total added to SQLite: {total_added}")
    print(f"   Database: {sqlite_path}")

if __name__ == "__main__":
    rebuild_sqlite_from_qdrant()

