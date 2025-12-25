"""Logging configuration for the application."""

import logging
import sys
from pathlib import Path
from typing import Optional
from collections import deque

from app.core.config import settings

# In-memory log buffer for admin panel (shared)
_admin_log_buffer: deque = deque(maxlen=500)


class AdminLogHandler(logging.Handler):
    """Custom handler that stores logs in memory for admin panel."""

    def emit(self, record):
        try:
            from datetime import datetime
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": self.format(record),
            }
            _admin_log_buffer.append(entry)
        except Exception:
            pass  # Don't break logging if buffer fails


def get_admin_logs(limit: int = 100) -> list:
    """Get logs from the admin buffer."""
    return list(_admin_log_buffer)[-limit:]


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

    # Short format for file logs (easier to parse in admin)
    file_format = "%(asctime)s | %(levelname)s | %(message)s"
    admin_format = "%(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler for admin log viewing
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(file_format, datefmt="%H:%M:%S"))
    file_handler.setLevel(log_level)
    handlers.append(file_handler)

    # Add admin buffer handler
    admin_handler = AdminLogHandler()
    admin_handler.setFormatter(logging.Formatter(admin_format))
    admin_handler.setLevel(log_level)
    handlers.append(admin_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # Override any existing config
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
