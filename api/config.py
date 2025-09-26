"""Configuration management using Pydantic settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    
    # Qdrant (managed vector DB)
    qdrant_url: str | None = Field(default=None, env="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="ml-docs-copilot", env="QDRANT_COLLECTION")
    
    # Data storage
    data_dir: Path = Field(default=Path("./data"), env="DATA_DIR")
    chroma_collection: str = Field(default="ml_docs", env="CHROMA_COLLECTION")
    sqlite_db: str = Field(default="bm25.db", env="SQLITE_DB")
    
    # Model settings
    embedding_model: str = Field(default="gemini-embedding-001", env="EMBEDDING_MODEL")
    
    # Chunking settings
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Search settings
    top_k_vector: int = Field(default=10, env="TOP_K_VECTOR")
    top_k_keyword: int = Field(default=10, env="TOP_K_KEYWORD")
    rrf_k: int = Field(default=60, env="RRF_K")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def chroma_path(self) -> Path:
        """Path to ChromaDB persistent directory."""
        return self.data_dir / "chroma"
    
    @property
    def sqlite_path(self) -> Path:
        """Path to SQLite database file."""
        return self.data_dir / self.sqlite_db


# Global settings instance
settings = Settings()
