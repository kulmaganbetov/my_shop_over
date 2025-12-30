"""Application configuration using Pydantic settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Find project root (where .env should be located)
# Go up from app/core/config.py -> app/core -> app -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Debug: print which .env file we're looking for
print(f"[CONFIG] Looking for .env at: {ENV_FILE}")
print(f"[CONFIG] .env exists: {ENV_FILE.exists()}")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/overshop"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/overshop"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # FTP Configuration
    ftp_host: str = "over-shop.kz"
    ftp_port: int = 21
    ftp_user: str = "zoomos1"
    ftp_password: str = "FJsV6cFv"
    ftp_csv_filename: str = "Dealer.csv"

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"

    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Application
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    # Auto-sync settings
    auto_sync_on_startup: bool = True  # Check and sync on FastAPI startup
    embedding_batch_size: int = 500  # Products per embedding batch (OpenAI supports up to 2048)
    embedding_interval_seconds: int = 30  # How often to check for missing embeddings
    product_sync_interval_seconds: int = 3600  # How often to sync from FTP (1 hour)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    s = Settings()
    # Debug output to verify configuration
    print(f"[CONFIG] Loaded settings:")
    print(f"[CONFIG]   REDIS_URL: {s.redis_url}")
    print(f"[CONFIG]   DATABASE_URL: {s.database_url[:50]}...")
    print(f"[CONFIG]   FTP_HOST: {s.ftp_host}")
    return s


settings = get_settings()
