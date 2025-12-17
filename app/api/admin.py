"""Admin API routes for management operations."""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_async_session
from app.db.repositories import FAQRepository
from app.llm.service import LLMService
from app.services.faq import FAQService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/sync/products")
async def trigger_product_sync(background_tasks: BackgroundTasks):
    """Trigger product synchronization from FTP."""
    from app.tasks.ingestion import sync_products_from_ftp

    try:
        task = sync_products_from_ftp.delay()
        return {
            "status": "queued",
            "task_id": task.id,
            "message": "Product sync task has been queued",
        }
    except Exception as e:
        logger.error(f"Failed to queue product sync: {e}")
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
