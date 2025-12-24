"""Admin API routes for management operations."""

import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import get_async_session
from app.db.models import AdminUser, ChatSession, ChatMessage, Product, ManagerChat
from app.db.repositories import FAQRepository
from app.llm.service import LLMService
from app.services.faq import FAQService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Simple session store (in production, use Redis)
admin_sessions: dict[str, dict] = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash


async def get_current_admin(
    admin_token: Optional[str] = Cookie(None),
    session: AsyncSession = Depends(get_async_session),
) -> Optional[AdminUser]:
    """Get current admin user from session token."""
    if not admin_token or admin_token not in admin_sessions:
        return None

    admin_data = admin_sessions[admin_token]
    if datetime.utcnow() > admin_data["expires"]:
        del admin_sessions[admin_token]
        return None

    result = await session.execute(
        select(AdminUser).where(AdminUser.id == admin_data["user_id"])
    )
    return result.scalar_one_or_none()


async def require_admin(
    admin: Optional[AdminUser] = Depends(get_current_admin),
) -> AdminUser:
    """Require authenticated admin user."""
    if not admin:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return admin


async def require_super_admin(
    admin: AdminUser = Depends(require_admin),
) -> AdminUser:
    """Require admin with 'admin' role."""
    if admin.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return admin


# ============= Auth Endpoints =============

class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    role: str = "manager"


@router.post("/auth/login")
async def login(
    request: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_async_session),
):
    """Login to admin panel."""
    result = await session.execute(
        select(AdminUser).where(
            AdminUser.username == request.username,
            AdminUser.is_active == True
        )
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session token
    token = secrets.token_urlsafe(32)
    admin_sessions[token] = {
        "user_id": user.id,
        "expires": datetime.utcnow() + timedelta(hours=24),
    }

    # Update last login
    user.last_login = datetime.utcnow()
    await session.commit()

    # Set cookie
    response.set_cookie(
        key="admin_token",
        value=token,
        httponly=True,
        max_age=86400,  # 24 hours
        samesite="lax",
    )

    return {
        "success": True,
        "user": {
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        }
    }


@router.post("/auth/logout")
async def logout(
    response: Response,
    admin_token: Optional[str] = Cookie(None),
):
    """Logout from admin panel."""
    if admin_token and admin_token in admin_sessions:
        del admin_sessions[admin_token]

    response.delete_cookie("admin_token")
    return {"success": True}


@router.get("/auth/me")
async def get_me(
    admin: AdminUser = Depends(require_admin),
):
    """Get current admin user info."""
    return {
        "id": admin.id,
        "username": admin.username,
        "full_name": admin.full_name,
        "role": admin.role,
    }


@router.post("/auth/init")
async def init_admin(
    session: AsyncSession = Depends(get_async_session),
):
    """Initialize default admin user if none exists."""
    result = await session.execute(
        select(func.count(AdminUser.id))
    )
    count = result.scalar_one()

    if count > 0:
        return {"message": "Admin users already exist", "created": False}

    # Create default admin
    admin = AdminUser(
        username="admin",
        password_hash=hash_password("admin123"),
        full_name="Administrator",
        role="admin",
        is_active=True,
    )
    session.add(admin)
    await session.commit()

    return {
        "message": "Default admin created",
        "created": True,
        "username": "admin",
        "password": "admin123",
    }


# ============= User Management =============

@router.get("/users")
async def list_users(
    admin: AdminUser = Depends(require_super_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """List all admin users."""
    result = await session.execute(
        select(AdminUser).order_by(AdminUser.created_at.desc())
    )
    users = result.scalars().all()

    return {
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "full_name": u.full_name,
                "role": u.role,
                "is_active": u.is_active,
                "last_login": u.last_login.isoformat() if u.last_login else None,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ]
    }


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    admin: AdminUser = Depends(require_super_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Create a new admin/manager user."""
    # Check if username exists
    result = await session.execute(
        select(AdminUser).where(AdminUser.username == request.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")

    if request.role not in ["admin", "manager"]:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'manager'")

    user = AdminUser(
        username=request.username,
        password_hash=hash_password(request.password),
        full_name=request.full_name,
        role=request.role,
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
    }


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: AdminUser = Depends(require_super_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Delete an admin/manager user."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    result = await session.execute(
        select(AdminUser).where(AdminUser.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await session.delete(user)
    await session.commit()
    return {"success": True}


@router.put("/users/{user_id}/toggle")
async def toggle_user(
    user_id: int,
    admin: AdminUser = Depends(require_super_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Toggle user active status."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot disable yourself")

    result = await session.execute(
        select(AdminUser).where(AdminUser.id == user_id)
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = not user.is_active
    await session.commit()
    return {"success": True, "is_active": user.is_active}


# ============= Manager Chat =============

class ManagerMessageRequest(BaseModel):
    message: str


@router.post("/sessions/{session_id}/reply")
async def send_manager_reply(
    session_id: str,
    request: ManagerMessageRequest,
    admin: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """Send a manager reply to a customer session."""
    from app.db.repositories import ChatRepository

    # Add manager message to manager_chats
    manager_msg = ManagerChat(
        session_id=session_id,
        manager_id=admin.id,
        message=request.message,
    )
    db.add(manager_msg)

    # Also add to chat_messages for the customer to see
    chat_msg = ChatMessage(
        session_id=session_id,
        role="manager",
        content=f"👤 **Менеджер {admin.full_name or admin.username}**: {request.message}",
        intent="manager_reply",
    )
    db.add(chat_msg)

    # Resolve escalation - change status back to "bot"
    chat_repo = ChatRepository(db)
    await chat_repo.resolve_escalation(session_id, "bot")

    await db.commit()

    logger.info(f"[MANAGER] {admin.username} replied to session {session_id}")

    return {"success": True, "message_id": manager_msg.id}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    admin: AdminUser = Depends(require_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Get all messages for a session including manager replies."""
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()

    return {
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]
    }


# ============= Debug Endpoints =============

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

    try:
        debug_info["registered_tasks"] = list(celery_app.tasks.keys())
    except Exception as e:
        debug_info["registered_tasks_error"] = str(e)

    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        debug_info["redis_connected"] = True
        debug_info["redis_keys"] = r.keys("*")[:20]
        debug_info["celery_queue_length"] = r.llen("celery")
    except Exception as e:
        debug_info["redis_error"] = str(e)

    return debug_info


# ============= Sync Endpoints =============

@router.post("/sync/products")
async def trigger_product_sync(
    background_tasks: BackgroundTasks,
    admin: AdminUser = Depends(require_admin),
):
    """Trigger product synchronization from FTP."""
    from app.tasks.ingestion import sync_products_from_ftp
    from app.tasks.celery_app import celery_app

    logger.info("Product sync requested by admin")

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
    full_regenerate: bool = Query(False),
    batch_size: int = Query(100, ge=1, le=500),
    admin: AdminUser = Depends(require_admin),
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
        }
    except Exception as e:
        logger.error(f"Failed to queue embedding update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= FAQ Management =============

@router.post("/faq/seed")
async def seed_default_faq(
    admin: AdminUser = Depends(require_admin),
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
    admin: AdminUser = Depends(require_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Create or update FAQ document."""
    try:
        llm_service = LLMService()
        faq_repo = FAQRepository(session)

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
    manager_only: bool = Query(False),
    admin: AdminUser = Depends(require_admin),
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
    admin: AdminUser = Depends(require_admin),
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
    admin: AdminUser = Depends(require_admin),
    session: AsyncSession = Depends(get_async_session),
):
    """Get summary of a chat session."""
    from app.services.chat_logger import ChatLogger

    chat_logger = ChatLogger(session)
    return await chat_logger.get_session_summary(session_id)


@router.get("/manager-requests")
async def get_manager_requests(
    limit: int = Query(50, ge=1, le=200),
    admin: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """Get all sessions waiting for manager response."""
    from app.db.repositories import ChatRepository

    chat_repo = ChatRepository(db)
    waiting_sessions = await chat_repo.get_sessions_waiting_manager()

    sessions_data = []
    for s in waiting_sessions[:limit]:
        # Get all messages for count and first user message
        messages = await chat_repo.get_session_messages(s.session_id, limit=100)
        user_messages = [m for m in messages if m.role == "user"]
        first_user_message = user_messages[0].content[:80] if user_messages else ""
        last_message_time = messages[-1].created_at if messages else s.updated_at

        sessions_data.append({
            "session_id": s.session_id,
            "status": s.status,
            "escalation_reason": s.escalation_reason,
            "message_count": len(messages),
            "first_user_message": first_user_message,
            "last_message_time": last_message_time.isoformat() if last_message_time else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        })

    return {
        "waiting_manager": sessions_data,
        "count": len(sessions_data),
        "urgent": len([s for s in sessions_data]),  # All are urgent
    }


@router.get("/manager-queue")
async def get_manager_queue_count(
    admin: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """Get count of sessions waiting for manager - for notifications."""
    result = await db.execute(
        select(func.count(ChatSession.id))
        .where(ChatSession.status == "waiting_manager")
    )
    count = result.scalar_one()

    return {
        "waiting_count": count,
        "has_urgent": count > 0,
    }


@router.get("/stats")
async def get_admin_stats(
    admin: AdminUser = Depends(require_admin),
    db: AsyncSession = Depends(get_async_session),
):
    """Get statistics for admin dashboard."""
    # Count total sessions
    sessions_count = await db.execute(
        select(func.count(ChatSession.id))
    )
    total_sessions = sessions_count.scalar_one()

    # Count sessions waiting for manager (URGENT!)
    waiting_count = await db.execute(
        select(func.count(ChatSession.id))
        .where(ChatSession.status == "waiting_manager")
    )
    waiting_manager = waiting_count.scalar_one()

    # Count total messages
    messages_count = await db.execute(
        select(func.count(ChatMessage.id))
    )
    total_messages = messages_count.scalar_one()

    # Count products
    products_count = await db.execute(
        select(func.count(Product.id))
    )
    total_products = products_count.scalar_one()

    # Count manager requests (historical)
    manager_count = await db.execute(
        select(func.count(ChatMessage.id))
        .where(ChatMessage.intent == "call_manager")
    )
    total_manager_requests = manager_count.scalar_one()

    # Get intent distribution
    intent_dist = await db.execute(
        select(ChatMessage.intent, func.count(ChatMessage.id))
        .where(ChatMessage.intent.isnot(None))
        .group_by(ChatMessage.intent)
        .order_by(func.count(ChatMessage.id).desc())
        .limit(10)
    )

    return {
        "total_sessions": total_sessions,
        "waiting_manager": waiting_manager,  # Urgent - show prominently!
        "total_messages": total_messages,
        "total_products": total_products,
        "total_manager_requests": total_manager_requests,
        "intent_distribution": [
            {"intent": row[0], "count": row[1]}
            for row in intent_dist.all()
        ],
    }
