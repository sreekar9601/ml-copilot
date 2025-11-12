"""Agent system configuration using Pydantic settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AgentConfig(BaseSettings):
    """Configuration for the agentic system."""
    
    # LLM Settings
    orchestrator_model: str = "gemini-2.5-flash"
    specialist_model: str = "gemini-2.5-flash"
    max_tokens: int = 8192  # Increased for comprehensive answers
    temperature: float = 0.2
    
    # Agent Behavior
    max_iterations: int = 5  # Reduced to prevent infinite loops
    max_execution_time: int = 120  # seconds
    enable_self_reflection: bool = False  # Disabled to simplify
    
    # Cost Control
    max_cost_per_session: float = 0.50  # dollars
    max_llm_calls_per_session: int = 20
    cost_per_1k_input_tokens: float = 0.000075  # Gemini Flash pricing
    cost_per_1k_output_tokens: float = 0.0003
    
    # Memory
    memory_backend: str = "sqlite"  # or "redis"
    max_history_messages: int = 20
    memory_db_path: str = "./data/agent_memory.db"
    redis_url: Optional[str] = None
    
    # Tools
    enable_code_execution: bool = True
    enable_web_search: bool = True
    sandbox_timeout: int = 30
    
    # Observability
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "ml-copilot-v3"
    langsmith_tracing: bool = False
    
    # Google Cloud / Vertex AI (inherited from existing config)
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None
    google_location: str = "us-central1"
    
    # Qdrant (inherited from existing config)
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "ml_docs"
    
    # External API keys
    e2b_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env
    )


# Global config instance
config = AgentConfig()


def get_config() -> AgentConfig:
    """Get the global config instance."""
    return config

