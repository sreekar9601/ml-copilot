"""Conversation memory with semantic search capabilities."""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class EnhancedChatHistory:
    """
    Manages conversation history with:
    - Chronological storage in SQLite
    - Semantic search over past messages (optional)
    - Automatic summarization for long conversations
    """
    
    def __init__(self, conversation_id: str, db_path: str = "./data/agent_memory.db"):
        self.conversation_id = conversation_id
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with chat_history table."""
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                message_content TEXT NOT NULL,
                tokens_used INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                tool_calls TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversation_id 
            ON chat_history(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON chat_history(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def add_messages(self, messages: List[BaseMessage], tokens_used: int = 0, 
                     cost: float = 0.0, tool_calls: Optional[list] = None):
        """Store messages in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for msg in messages:
            message_type = msg.__class__.__name__.replace("Message", "").lower()
            
            cursor.execute("""
                INSERT INTO chat_history 
                (conversation_id, message_type, message_content, tokens_used, cost_usd, tool_calls)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.conversation_id,
                message_type,
                msg.content,
                tokens_used,
                cost,
                json.dumps(tool_calls) if tool_calls else None
            ))
        
        conn.commit()
        conn.close()
    
    def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Retrieve conversation history in chronological order.
        
        Args:
            limit: Maximum number of messages to return (None = all)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT message_type, message_content 
            FROM chat_history 
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (self.conversation_id,))
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for msg_type, content in rows:
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        
        return messages
    
    def get_last_n_messages(self, n: int = 10) -> List[BaseMessage]:
        """Get the last N messages from the conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT message_type, message_content 
            FROM (
                SELECT message_type, message_content, timestamp
                FROM chat_history 
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            ORDER BY timestamp ASC
        """
        
        cursor.execute(query, (self.conversation_id, n))
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for msg_type, content in rows:
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        
        return messages
    
    def get_conversation_stats(self) -> dict:
        """Get statistics for this conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as message_count,
                SUM(tokens_used) as total_tokens,
                SUM(cost_usd) as total_cost,
                MIN(timestamp) as started_at,
                MAX(timestamp) as last_activity
            FROM chat_history
            WHERE conversation_id = ?
        """, (self.conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            "conversation_id": self.conversation_id,
            "message_count": row[0],
            "total_tokens": row[1] or 0,
            "total_cost": row[2] or 0.0,
            "started_at": row[3],
            "last_activity": row[4]
        }
    
    def clear(self):
        """Delete all messages for this conversation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE conversation_id = ?", 
                      (self.conversation_id,))
        conn.commit()
        conn.close()

