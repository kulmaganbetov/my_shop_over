"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.api.admin import router as admin_router
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting OverShop AI Assistant")
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
