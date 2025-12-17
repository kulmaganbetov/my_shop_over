"""FAQ repository with vector search for RAG."""

from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import FAQDocument
from app.db.repositories.base import BaseRepository


class FAQRepository(BaseRepository[FAQDocument]):
    """Repository for FAQ document operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, FAQDocument)

    async def search_by_content(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> list[FAQDocument]:
        """Search FAQ documents by content using ILIKE."""
        conditions = [
            FAQDocument.content.ilike(f"%{query}%")
            | FAQDocument.title.ilike(f"%{query}%")
        ]

        if category:
            conditions.append(FAQDocument.category == category)

        result = await self.session.execute(
            select(FAQDocument)
            .where(and_(*conditions))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 5,
        category: Optional[str] = None,
        similarity_threshold: float = 0.5,
    ) -> list[tuple[FAQDocument, float]]:
        """Search FAQ documents using vector similarity."""
        conditions = []

        if category:
            conditions.append(FAQDocument.category == category)

        # Vector similarity search
        distance = FAQDocument.embedding.cosine_distance(embedding)

        query = (
            select(FAQDocument, (1 - distance).label("similarity"))
            .where(FAQDocument.embedding.isnot(None))
        )

        if conditions:
            query = query.where(and_(*conditions))

        query = query.having((1 - distance) >= similarity_threshold)
        query = query.order_by(distance).limit(limit)

        result = await self.session.execute(query)
        return [(row.FAQDocument, row.similarity) for row in result.all()]

    async def get_by_category(self, category: str) -> list[FAQDocument]:
        """Get all FAQ documents in a category."""
        result = await self.session.execute(
            select(FAQDocument).where(FAQDocument.category == category)
        )
        return list(result.scalars().all())

    async def get_categories(self) -> list[str]:
        """Get all unique FAQ categories."""
        result = await self.session.execute(
            select(FAQDocument.category)
            .where(FAQDocument.category.isnot(None))
            .distinct()
        )
        return [row[0] for row in result.all()]

    async def upsert_faq(self, title: str, content: str, **kwargs) -> FAQDocument:
        """Insert or update FAQ document by title."""
        result = await self.session.execute(
            select(FAQDocument).where(FAQDocument.title == title)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.content = content
            for key, value in kwargs.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing
        else:
            faq = FAQDocument(title=title, content=content, **kwargs)
            self.session.add(faq)
            await self.session.commit()
            await self.session.refresh(faq)
            return faq
