"""Chat logging service for tracking all interactions."""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession

logger = logging.getLogger(__name__)


class ChatLogger:
    """Service for logging chat interactions for admin review."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log_user_message(
        self,
        session_id: str,
        message: str,
        detected_intent: Optional[str] = None,
    ) -> None:
        """Log user message with detected intent."""
        logger.info(
            f"[USER] session={session_id} intent={detected_intent} message={message[:100]}"
        )

    async def log_assistant_response(
        self,
        session_id: str,
        response: str,
        intent: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Log assistant response."""
        status = "SUCCESS" if success else "ERROR"
        logger.info(
            f"[ASSISTANT] session={session_id} intent={intent} status={status} "
            f"response={response[:100]}"
        )
        if error:
            logger.error(f"[ASSISTANT_ERROR] session={session_id} error={error}")

    async def log_system_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
        context: Optional[dict] = None,
    ) -> None:
        """Log system errors (timeouts, API failures, etc.)."""
        logger.error(
            f"[SYSTEM_ERROR] session={session_id} type={error_type} "
            f"error={error_message} context={context}"
        )

    async def log_manager_request(
        self,
        session_id: str,
        reason: str,
        user_confirmed: bool = False,
    ) -> None:
        """Log manager handoff request."""
        status = "CONFIRMED" if user_confirmed else "REQUESTED"
        logger.warning(
            f"[MANAGER_REQUEST] session={session_id} status={status} reason={reason}"
        )

    async def get_session_summary(self, session_id: str) -> dict:
        """Get summary of a chat session for admin view."""
        from sqlalchemy import select, func

        # Get session info
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            return {}

        # Get message count and first/last messages
        messages_result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        messages = list(messages_result.scalars().all())

        first_message = messages[0].content[:50] if messages else ""
        last_message = messages[-1].content[:50] if messages else ""

        # Check if manager was requested
        manager_requested = any(
            m.intent == "call_manager" for m in messages if m.intent
        )

        return {
            "session_id": session_id,
            "created_at": chat_session.created_at.isoformat() if chat_session.created_at else None,
            "message_count": len(messages),
            "first_message": first_message,
            "last_message": last_message,
            "manager_requested": manager_requested,
        }

    async def get_all_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        manager_only: bool = False,
    ) -> list[dict]:
        """Get all chat sessions for admin panel."""
        from sqlalchemy import select, func, distinct

        # Get unique session IDs with their first message time
        subquery = (
            select(
                ChatMessage.session_id,
                func.min(ChatMessage.created_at).label("first_msg_time"),
                func.max(ChatMessage.created_at).label("last_msg_time"),
                func.count(ChatMessage.id).label("msg_count"),
            )
            .group_by(ChatMessage.session_id)
            .order_by(func.max(ChatMessage.created_at).desc())
            .limit(limit)
            .offset(offset)
            .subquery()
        )

        result = await self.session.execute(
            select(
                subquery.c.session_id,
                subquery.c.first_msg_time,
                subquery.c.last_msg_time,
                subquery.c.msg_count,
            )
        )

        sessions = []
        for row in result.all():
            # Get ChatSession to get status
            chat_session_result = await self.session.execute(
                select(ChatSession).where(ChatSession.session_id == row.session_id)
            )
            chat_session = chat_session_result.scalar_one_or_none()
            status = chat_session.status if chat_session else "bot"
            escalation_reason = chat_session.escalation_reason if chat_session else None

            # Get first and last user message
            msgs_result = await self.session.execute(
                select(ChatMessage)
                .where(
                    ChatMessage.session_id == row.session_id,
                    ChatMessage.role == "user"
                )
                .order_by(ChatMessage.created_at.asc())
            )
            user_msgs = list(msgs_result.scalars().all())

            # Check for manager request - use first() instead of scalar_one_or_none to handle multiple rows
            manager_result = await self.session.execute(
                select(ChatMessage.id)
                .where(
                    ChatMessage.session_id == row.session_id,
                    ChatMessage.intent == "call_manager"
                )
                .limit(1)
            )
            has_manager_request = manager_result.scalar() is not None

            if manager_only and not has_manager_request:
                continue

            sessions.append({
                "session_id": row.session_id,
                "status": status,
                "escalation_reason": escalation_reason,
                "first_message_time": row.first_msg_time.isoformat() if row.first_msg_time else None,
                "last_message_time": row.last_msg_time.isoformat() if row.last_msg_time else None,
                "message_count": row.msg_count,
                "first_user_message": user_msgs[0].content[:80] + "..." if user_msgs else "",
                "last_user_message": user_msgs[-1].content[:80] + "..." if user_msgs else "",
                "manager_requested": has_manager_request,
            })

        return sessions

    async def get_session_messages(self, session_id: str) -> list[dict]:
        """Get all messages for a specific session."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )

        messages = []
        for msg in result.scalars().all():
            messages.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "intent": msg.intent,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
                "extra_data": msg.extra_data,
            })

        return messages
