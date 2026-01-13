"""Build preset repository for Smart Presets system."""

from typing import Optional, List

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import BuildPreset
from app.db.repositories.base import BaseRepository


class PresetRepository(BaseRepository[BuildPreset]):
    """Repository for build preset operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, BuildPreset)

    async def get_by_purpose(self, purpose: str) -> List[BuildPreset]:
        """Get presets by purpose."""
        result = await self.session.execute(
            select(BuildPreset).where(
                and_(
                    BuildPreset.purpose == purpose,
                    BuildPreset.is_active == True,
                )
            ).order_by(BuildPreset.priority.desc())
        )
        return list(result.scalars().all())

    async def get_by_budget(
        self,
        budget: int,
        purpose: Optional[str] = None,
        category_tag: Optional[str] = None,
    ) -> List[BuildPreset]:
        """Get presets within ±10% budget range."""
        min_budget = int(budget * 0.9)
        max_budget = int(budget * 1.1)

        conditions = [
            BuildPreset.target_budget >= min_budget,
            BuildPreset.target_budget <= max_budget,
            BuildPreset.is_active == True,
        ]

        if purpose:
            conditions.append(BuildPreset.purpose == purpose)

        if category_tag:
            conditions.append(BuildPreset.category_tag.ilike(f"%{category_tag}%"))

        result = await self.session.execute(
            select(BuildPreset)
            .where(and_(*conditions))
            .order_by(BuildPreset.priority.desc(), BuildPreset.target_budget)
        )
        return list(result.scalars().all())

    async def get_all_active(self) -> List[BuildPreset]:
        """Get all active presets."""
        result = await self.session.execute(
            select(BuildPreset)
            .where(BuildPreset.is_active == True)
            .order_by(BuildPreset.target_budget, BuildPreset.category_tag)
        )
        return list(result.scalars().all())

    async def get_by_name(self, name: str) -> Optional[BuildPreset]:
        """Get preset by name."""
        result = await self.session.execute(
            select(BuildPreset).where(BuildPreset.name == name)
        )
        return result.scalar_one_or_none()

    async def get_by_category(self, category_tag: str) -> List[BuildPreset]:
        """Get presets by category tag (Intel, AMD, Workstation, etc.)."""
        result = await self.session.execute(
            select(BuildPreset).where(
                and_(
                    BuildPreset.category_tag.ilike(f"%{category_tag}%"),
                    BuildPreset.is_active == True,
                )
            ).order_by(BuildPreset.target_budget)
        )
        return list(result.scalars().all())
