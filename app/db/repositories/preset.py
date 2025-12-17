"""PC preset repository."""

from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PCPreset
from app.db.repositories.base import BaseRepository


class PresetRepository(BaseRepository[PCPreset]):
    """Repository for PC preset operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, PCPreset)

    async def get_by_purpose(self, purpose: str) -> list[PCPreset]:
        """Get presets by purpose."""
        result = await self.session.execute(
            select(PCPreset).where(
                and_(
                    PCPreset.purpose == purpose,
                    PCPreset.is_active == True,
                )
            )
        )
        return list(result.scalars().all())

    async def get_by_budget(
        self,
        budget: int,
        purpose: Optional[str] = None,
    ) -> list[PCPreset]:
        """Get presets within budget range."""
        conditions = [
            PCPreset.budget_min <= budget,
            PCPreset.budget_max >= budget,
            PCPreset.is_active == True,
        ]

        if purpose:
            conditions.append(PCPreset.purpose == purpose)

        result = await self.session.execute(
            select(PCPreset)
            .where(and_(*conditions))
            .order_by(PCPreset.budget_min)
        )
        return list(result.scalars().all())

    async def get_all_active(self) -> list[PCPreset]:
        """Get all active presets."""
        result = await self.session.execute(
            select(PCPreset).where(PCPreset.is_active == True)
        )
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> Optional[PCPreset]:
        """Get preset by name."""
        result = await self.session.execute(
            select(PCPreset).where(PCPreset.name == name)
        )
        return result.scalar_one_or_none()
