"""FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api.routes import router as api_router
from app.api.admin import router as admin_router
from app.core.config import settings
from app.db.base import async_engine

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "OverShop AI Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
