"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from collections import deque
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import text

from app.api.routes import router as api_router
from app.api.admin import router as admin_router
from app.core.config import settings
from app.db.base import async_engine

# Static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"

# Ensure uploads directory exists
UPLOADS_DIR.mkdir(exist_ok=True)

# Global in-memory log buffer for admin dashboard
admin_log_buffer: deque = deque(maxlen=1000)


class AdminLogHandler(logging.Handler):
    """Custom handler that stores logs in memory for admin panel."""

    def emit(self, record):
        try:
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": self.format(record),
                "timestamp": datetime.now().isoformat(),
            }
            admin_log_buffer.append(entry)
        except Exception:
            pass


def get_admin_logs(limit: int = 200) -> list:
    """Get logs from the admin buffer."""
    return list(admin_log_buffer)[-limit:]


# Configure logging with admin buffer
log_format = "%(asctime)s | %(levelname)-7s | %(message)s"
handlers = [
    logging.StreamHandler(),  # Console output
    AdminLogHandler(),  # Admin buffer
]

# Add file handler
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "app.log", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter(log_format, datefmt="%H:%M:%S"))
handlers.append(file_handler)

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=log_format,
    datefmt="%H:%M:%S",
    handlers=handlers,
    force=True,
)

# Silence noisy loggers
for noisy_logger in ["httpx", "httpcore", "sqlalchemy", "urllib3", "asyncio"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


async def check_and_trigger_startup_tasks():
    """Check database state and trigger necessary tasks on startup."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    logger.info("=" * 60)
    logger.info("STARTUP: Checking data pipeline status...")
    logger.info("=" * 60)

    async_session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        try:
            # Check products count
            result = await session.execute(text("SELECT COUNT(*) FROM products"))
            products_count = result.scalar()
            logger.info(f"  Products in database: {products_count}")

            # Check embeddings count
            result = await session.execute(text("SELECT COUNT(*) FROM product_embeddings"))
            embeddings_count = result.scalar()
            logger.info(f"  Products with embeddings: {embeddings_count}")

            # Calculate missing embeddings
            missing_embeddings = products_count - embeddings_count
            logger.info(f"  Products without embeddings: {missing_embeddings}")

            # Import Celery tasks
            from app.tasks.ingestion import sync_products_from_ftp
            from app.tasks.embeddings import update_missing_embeddings

            # Trigger product sync if no products
            if products_count == 0:
                logger.info("")
                logger.info("  → No products found. Triggering FTP sync...")
                task = sync_products_from_ftp.delay()
                logger.info(f"  → Product sync task queued: {task.id}")
            else:
                logger.info("  ✓ Products already synced")

            # Trigger embeddings if many are missing
            if missing_embeddings > 0:
                logger.info("")
                logger.info(f"  → {missing_embeddings} products missing embeddings.")
                logger.info(f"  → Triggering embedding generation (batch_size={settings.embedding_batch_size})...")
                task = update_missing_embeddings.delay(batch_size=settings.embedding_batch_size)
                logger.info(f"  → Embedding task queued: {task.id}")

                # Note: This will only process first batch.
                # Celery Beat will continue processing in background.
            else:
                logger.info("  ✓ All products have embeddings")

            logger.info("")
            logger.info("=" * 60)
            logger.info("STARTUP: Data pipeline check complete")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"  ✗ Startup check failed: {e}")
            logger.info("  Note: Make sure PostgreSQL is running and tables exist.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting OverShop AI Assistant")

    # Run startup tasks in background (don't block server startup)
    if settings.auto_sync_on_startup:
        asyncio.create_task(check_and_trigger_startup_tasks())
    else:
        logger.info("Auto-sync disabled (AUTO_SYNC_ON_STARTUP=false)")

    yield
    logger.info("Shutting down OverShop AI Assistant")


app = FastAPI(
    title="OverShop AI Assistant",
    description="""
    AI-powered assistant for over-shop.kz e-commerce store.

    Features:
    - PC Build Assistant: Help users build compatible PCs
    - Product Search: Semantic search using embeddings
    - Store FAQ: Answer questions about delivery, warranty, payment, etc.

    Architecture:
    - LLM is used ONLY for intent detection and response generation
    - All business logic is deterministic
    - PC compatibility rules are rule-based
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1", tags=["chat"])
app.include_router(admin_router, prefix="/api/v1", tags=["admin"])

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount uploads directory for serving uploaded files
if UPLOADS_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


@app.get("/")
async def root():
    """Serve the chat interface."""
    chat_file = STATIC_DIR / "index.html"
    if chat_file.exists():
        return FileResponse(chat_file)
    return {
        "service": "OverShop AI Assistant",
        "version": "0.1.0",
        "chat": "/chat",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/chat")
async def chat_page():
    """Serve the chat interface."""
    chat_file = STATIC_DIR / "index.html"
    if chat_file.exists():
        return FileResponse(chat_file)
    return {"error": "Chat interface not found"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
