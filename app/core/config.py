"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
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
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-haiku-20240307"

    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Application
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
