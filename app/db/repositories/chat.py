"""Chat session and message repository."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatMessage, ChatSession
from app.db.repositories.base import BaseRepository


class ChatRepository(BaseRepository[ChatSession]):
    """Repository for chat session and message operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, ChatSession)

    async def get_or_create_session(self, session_id: str) -> ChatSession:
        """Get existing session or create a new one."""
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            chat_session = ChatSession(session_id=session_id, context={})
            self.session.add(chat_session)
            await self.session.commit()
            await self.session.refresh(chat_session)

        return chat_session

    async def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        """Get session by session_id."""
        result = await self.session.execute(
            select(ChatSession).where(ChatSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def update_session_context(self, session_id: str, context: dict) -> ChatSession:
        """Update session context by MERGING with existing context."""
        chat_session = await self.get_or_create_session(session_id)
        # Create a NEW dict to ensure SQLAlchemy detects the change
        existing_context = dict(chat_session.context or {})
        existing_context.update(context)
        chat_session.context = existing_context
        # Mark as modified to ensure SQLAlchemy saves the change
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(chat_session, "context")
        await self.session.commit()
        await self.session.refresh(chat_session)
        return chat_session

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        intent: Optional[str] = None,
        extra_data: Optional[dict] = None,
    ) -> ChatMessage:
        """Add a message to the session."""
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent,
            extra_data=extra_data or {},
        )
        self.session.add(message)
        await self.session.commit()
        await self.session.refresh(message)
        return message

    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[ChatMessage]:
        """Get recent messages from a session."""
        result = await self.session.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        return list(reversed(messages))

    async def escalate_to_manager(
        self,
        session_id: str,
        reason: str,
    ) -> ChatSession:
        """Escalate session to manager."""
        chat_session = await self.get_or_create_session(session_id)
        chat_session.status = "waiting_manager"
        chat_session.escalation_reason = reason
        await self.session.commit()
        await self.session.refresh(chat_session)
        return chat_session

    async def resolve_escalation(
        self,
        session_id: str,
        new_status: str = "bot",
    ) -> ChatSession:
        """Resolve escalation - manager responded."""
        chat_session = await self.get_session_by_id(session_id)
        if chat_session:
            chat_session.status = new_status
            await self.session.commit()
            await self.session.refresh(chat_session)
        return chat_session

    async def get_sessions_waiting_manager(self) -> list[ChatSession]:
        """Get all sessions needing manager attention (waiting or active)."""
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.status.in_(["waiting_manager", "manager_active"]))
            .order_by(
                # waiting_manager first (more urgent)
                ChatSession.status.asc(),
                ChatSession.updated_at.desc()
            )
        )
        return list(result.scalars().all())

    async def get_all_active_sessions(self, limit: int = 50) -> list[ChatSession]:
        """Get all active sessions sorted by status priority."""
        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.status != "closed")
            .order_by(
                # waiting_manager first, then by updated_at
                ChatSession.status.desc(),
                ChatSession.updated_at.desc()
            )
            .limit(limit)
        )
        return list(result.scalars().all())
