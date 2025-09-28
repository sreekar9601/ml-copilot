#!/usr/bin/env python3
"""Check the database contents to diagnose retrieval issues."""

import sqlite3
from pathlib import Path

def check_database():
    """Check database contents."""
    data_dir = Path('./data')
    sqlite_path = data_dir / 'bm25.db'
    
    if not sqlite_path.exists():
        print(f'❌ SQLite database not found at: {sqlite_path}')
        return
    
    print(f'✅ Found database at: {sqlite_path}')
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    try:
        # Check table structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f'📊 Tables: {[t[0] for t in tables]}')
        
        # Check documents_fts table (this seems to be the main table)
        cursor.execute('SELECT COUNT(*) FROM documents_fts')
        count = cursor.fetchone()[0]
        print(f'📦 Total documents in FTS table: {count}')
        
        if count == 0:
            print('❌ No documents found in database!')
            return
        
        # Check sample of source URLs from FTS table
        cursor.execute('SELECT DISTINCT source_url FROM documents_fts LIMIT 10')
        urls = cursor.fetchall()
        print('🔗 Sample source URLs:')
        for url in urls:
            print(f'  - {url[0]}')
        
        # Check for PyTorch content specifically
        cursor.execute("SELECT COUNT(*) FROM documents_fts WHERE source_url LIKE '%pytorch%'")
        pytorch_count = cursor.fetchone()[0]
        print(f'🔥 PyTorch documents: {pytorch_count}')
        
        # Check for TensorFlow content (this might be the problem)
        cursor.execute("SELECT COUNT(*) FROM documents_fts WHERE source_url LIKE '%tensorflow%' OR content LIKE '%tensorflow%'")
        tf_count = cursor.fetchone()[0]
        print(f'⚠️  TensorFlow documents: {tf_count}')
        
        # Check sample content
        cursor.execute('SELECT content, source_url FROM documents_fts LIMIT 3')
        samples = cursor.fetchall()
        print('📄 Sample content:')
        for content, url in samples:
            print(f'  URL: {url}')
            print(f'  Content: {content[:100]}...')
            print()
            
    except Exception as e:
        print(f'❌ Error checking database: {e}')
    finally:
        conn.close()

if __name__ == '__main__':
    check_database()
