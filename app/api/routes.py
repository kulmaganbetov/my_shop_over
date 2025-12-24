"""API routes for the AI assistant."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_async_session
from app.db.repositories import ChatRepository, ProductRepository
from app.llm.service import LLMService
from app.schemas.chat import ChatRequest, ChatResponse, ProductSearchResponse
from app.schemas.common import ProductSchema
from app.services.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_orchestrator(
    session: AsyncSession = Depends(get_async_session),
) -> Orchestrator:
    """Dependency for getting the orchestrator."""
    return Orchestrator(session)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """
    Main chat endpoint for the AI assistant.

    Uses LLM as orchestrator to decide which backend tool to call.
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        response = await orchestrator.process_message(request.message, session_id)
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/products/search", response_model=ProductSearchResponse)
async def search_products(
    query: str = Query(..., min_length=1, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    manufacturer: Optional[str] = Query(None, description="Filter by manufacturer"),
    in_stock_only: bool = Query(True, description="Only show products in stock"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Search products endpoint.

    Supports semantic search using embeddings and structured filters.
    """
    from app.services.product_search import ProductSearchService
    from app.schemas.llm import ProductSearchParams

    llm_service = LLMService()
    search_service = ProductSearchService(session, llm_service)

    params = ProductSearchParams(
        query=query,
        category=category,
        min_price=min_price,
        max_price=max_price,
        manufacturer=manufacturer,
        in_stock_only=in_stock_only,
        limit=limit,
    )

    return await search_service.search(params)


@router.get("/products/{sku}", response_model=ProductSchema)
async def get_product(
    sku: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get product by SKU."""
    repo = ProductRepository(session)
    product = await repo.get_by_sku(sku)

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    return ProductSchema.model_validate(product)


@router.get("/products/categories/list")
async def list_categories(
    session: AsyncSession = Depends(get_async_session),
):
    """List all product categories."""
    repo = ProductRepository(session)
    categories = await repo.get_categories()
    return {"categories": categories}


@router.get("/products/manufacturers/list")
async def list_manufacturers(
    category: Optional[str] = Query(None, description="Filter by category"),
    session: AsyncSession = Depends(get_async_session),
):
    """List all manufacturers."""
    repo = ProductRepository(session)
    manufacturers = await repo.get_manufacturers(category)
    return {"manufacturers": manufacturers}


@router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get chat history for a session (for client to restore chat on refresh)."""
    chat_repo = ChatRepository(session)
    chat_session = await chat_repo.get_session_by_id(session_id)

    if not chat_session:
        return {"messages": [], "status": "bot"}

    messages = await chat_repo.get_session_messages(session_id, limit=50)

    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ],
        "status": chat_session.status or "bot",
        "escalation_reason": chat_session.escalation_reason,
    }


@router.get("/chat/status/{session_id}")
async def get_chat_status(
    session_id: str,
    last_message_id: Optional[int] = Query(None, description="Last message ID client has"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    Get session status and new messages (for polling).
    Returns only messages after last_message_id if provided.
    """
    chat_repo = ChatRepository(session)
    chat_session = await chat_repo.get_session_by_id(session_id)

    if not chat_session:
        return {"status": "bot", "new_messages": []}

    # Get messages after last_message_id
    all_messages = await chat_repo.get_session_messages(session_id, limit=50)

    new_messages = []
    if last_message_id:
        for msg in all_messages:
            if msg.id > last_message_id:
                new_messages.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if msg.created_at else None,
                })
    else:
        # Return last message ID for tracking
        if all_messages:
            new_messages = [{"id": all_messages[-1].id}]

    return {
        "status": chat_session.status or "bot",
        "new_messages": new_messages,
        "last_message_id": all_messages[-1].id if all_messages else 0,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "overshop-assistant"}


@router.get("/health/db")
async def db_health_check(
    session: AsyncSession = Depends(get_async_session),
):
    """Database health check."""
    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")
