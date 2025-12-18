"""Admin API routes for management operations."""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import get_async_session
from app.db.repositories import FAQRepository
from app.llm.service import LLMService
from app.services.faq import FAQService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/debug/config")
async def debug_config():
    """Debug endpoint to check loaded configuration."""
    from app.core.config import ENV_FILE

    return {
        "env_file_path": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
        "redis_url": settings.redis_url,
        "database_url": settings.database_url[:50] + "...",
        "ftp_host": settings.ftp_host,
        "ftp_user": settings.ftp_user,
        "llm_provider": settings.llm_provider,
        "log_level": settings.log_level,
    }


@router.get("/debug/celery")
async def debug_celery():
    """Debug endpoint to check Celery configuration and connectivity."""
    from app.tasks.celery_app import celery_app
    import redis

    debug_info = {
        "broker_url": settings.redis_url,
        "celery_broker": celery_app.conf.broker_url,
        "registered_tasks": [],
        "redis_connected": False,
        "redis_keys": [],
        "celery_queue_length": 0,
    }

    # Check registered tasks
    try:
        debug_info["registered_tasks"] = list(celery_app.tasks.keys())
    except Exception as e:
        debug_info["registered_tasks_error"] = str(e)

    # Check Redis connectivity
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        debug_info["redis_connected"] = True
        debug_info["redis_keys"] = r.keys("*")[:20]  # First 20 keys
        debug_info["celery_queue_length"] = r.llen("celery")
    except Exception as e:
        debug_info["redis_error"] = str(e)

    return debug_info


@router.post("/sync/products")
async def trigger_product_sync(background_tasks: BackgroundTasks):
    """Trigger product synchronization from FTP."""
    from app.tasks.ingestion import sync_products_from_ftp
    from app.tasks.celery_app import celery_app

    logger.info("=" * 50)
    logger.info("PRODUCT SYNC REQUEST RECEIVED")
    logger.info("=" * 50)
    logger.info(f"  Celery broker URL: {celery_app.conf.broker_url}")
    logger.info(f"  Task name: {sync_products_from_ftp.name}")

    try:
        logger.info("  Sending task to Celery...")
        task = sync_products_from_ftp.delay()
        logger.info(f"  ✓ Task sent successfully!")
        logger.info(f"  Task ID: {task.id}")
        logger.info(f"  Task backend: {task.backend}")

        return {
            "status": "queued",
            "task_id": task.id,
            "task_name": sync_products_from_ftp.name,
            "broker_url": celery_app.conf.broker_url,
            "message": "Product sync task has been queued",
        }
    except Exception as e:
        logger.error(f"  ✗ Failed to queue product sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/embeddings")
async def trigger_embedding_update(
    full_regenerate: bool = Query(False, description="Regenerate all embeddings"),
    batch_size: int = Query(100, ge=1, le=500, description="Batch size"),
):
    """Trigger embedding update for products."""
    from app.tasks.embeddings import update_missing_embeddings, regenerate_all_embeddings

    try:
        if full_regenerate:
            task = regenerate_all_embeddings.delay(batch_size)
        else:
            task = update_missing_embeddings.delay(batch_size)

        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Embedding update task has been queued",
            "full_regenerate": full_regenerate,
        }
    except Exception as e:
        logger.error(f"Failed to queue embedding update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faq/seed")
async def seed_default_faq(
    session: AsyncSession = Depends(get_async_session),
):
    """Seed default FAQ content."""
    try:
        llm_service = LLMService()
        faq_service = FAQService(session, llm_service)
        await faq_service.seed_default_faq()
        return {"status": "success", "message": "Default FAQ content seeded"}
    except Exception as e:
        logger.error(f"Failed to seed FAQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/faq")
async def create_faq(
    title: str,
    content: str,
    category: Optional[str] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """Create or update FAQ document."""
    try:
        llm_service = LLMService()
        faq_repo = FAQRepository(session)

        # Get embedding
        embedding_text = f"{title}: {content}"
        embedding = await llm_service.get_embedding(embedding_text)

        faq = await faq_repo.upsert_faq(
            title=title,
            content=content,
            category=category,
            embedding=embedding,
        )

        return {
            "status": "success",
            "id": faq.id,
            "title": faq.title,
        }
    except Exception as e:
        logger.error(f"Failed to create FAQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a Celery task."""
    from app.tasks.celery_app import celery_app

    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None,
    }


# ============= Chat Session Admin Endpoints =============

@router.get("/sessions")
async def get_all_sessions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    manager_only: bool = Query(False, description="Show only sessions with manager requests"),
    session: AsyncSession = Depends(get_async_session),
):
    """Get all chat sessions for admin panel."""
    from app.services.chat_logger import ChatLogger

    chat_logger = ChatLogger(session)
    sessions = await chat_logger.get_all_sessions(
        limit=limit,
        offset=offset,
        manager_only=manager_only,
    )
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session_details(
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get full conversation for a specific session."""
    from app.services.chat_logger import ChatLogger

    chat_logger = ChatLogger(session)
    messages = await chat_logger.get_session_messages(session_id)
    summary = await chat_logger.get_session_summary(session_id)

    return {
        "session": summary,
        "messages": messages,
    }


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Get summary of a chat session."""
    from app.services.chat_logger import ChatLogger

    chat_logger = ChatLogger(session)
    return await chat_logger.get_session_summary(session_id)


@router.get("/manager-requests")
async def get_manager_requests(
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_async_session),
):
    """Get all sessions where manager was requested."""
    from app.services.chat_logger import ChatLogger

    chat_logger = ChatLogger(session)
    sessions = await chat_logger.get_all_sessions(
        limit=limit,
        manager_only=True,
    )
    return {"manager_requests": sessions, "count": len(sessions)}


@router.get("/stats")
async def get_admin_stats(
    session: AsyncSession = Depends(get_async_session),
):
    """Get statistics for admin dashboard."""
    from sqlalchemy import select, func
    from app.db.models import ChatSession, ChatMessage, Product

    # Count total sessions
    sessions_count = await session.execute(
        select(func.count(ChatSession.id))
    )
    total_sessions = sessions_count.scalar_one()

    # Count total messages
    messages_count = await session.execute(
        select(func.count(ChatMessage.id))
    )
    total_messages = messages_count.scalar_one()

    # Count products
    products_count = await session.execute(
        select(func.count(Product.id))
    )
    total_products = products_count.scalar_one()

    # Count manager requests
    manager_count = await session.execute(
        select(func.count(ChatMessage.id))
        .where(ChatMessage.intent == "call_manager")
    )
    total_manager_requests = manager_count.scalar_one()

    # Get intent distribution
    intent_dist = await session.execute(
        select(ChatMessage.intent, func.count(ChatMessage.id))
        .where(ChatMessage.intent.isnot(None))
        .group_by(ChatMessage.intent)
        .order_by(func.count(ChatMessage.id).desc())
        .limit(10)
    )

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_products": total_products,
        "total_manager_requests": total_manager_requests,
        "intent_distribution": [
            {"intent": row[0], "count": row[1]}
            for row in intent_dist.all()
        ],
    }
