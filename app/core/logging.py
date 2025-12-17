"""Logging configuration for the application."""

import logging
import sys
from typing import Optional

from app.core.config import settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Configure application-wide logging."""

    log_level = getattr(logging, level or settings.log_level.upper(), logging.INFO)

    # Default format with timestamp, level, module, and message
    log_format = format_string or (
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set specific loggers
    loggers_config = {
        "app": log_level,
        "app.tasks": log_level,
        "app.services": log_level,
        "app.llm": log_level,
        "app.db": log_level,
        "app.api": log_level,
        # Reduce noise from libraries
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "openai": logging.WARNING,
        "urllib3": logging.WARNING,
        "asyncio": logging.WARNING,
        "sqlalchemy.engine": logging.WARNING if not settings.debug else logging.INFO,
    }

    for logger_name, logger_level in loggers_config.items():
        logging.getLogger(logger_name).setLevel(logger_level)

    logging.info("=" * 60)
    logging.info("Logging configured successfully")
    logging.info(f"Log level: {logging.getLevelName(log_level)}")
    logging.info(f"Environment: {settings.app_env}")
    logging.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(name)
