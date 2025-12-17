"""Product repository with vector search capabilities."""

from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Product, ProductEmbedding
from app.db.repositories.base import BaseRepository


class ProductRepository(BaseRepository[Product]):
    """Repository for product operations with vector search."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Product)

    async def get_by_sku(self, sku: str) -> Optional[Product]:
        """Get product by SKU."""
        result = await self.session.execute(
            select(Product).where(Product.sku == sku)
        )
        return result.scalar_one_or_none()

    async def search_by_name(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> list[Product]:
        """Search products by name using ILIKE."""
        conditions = [Product.name.ilike(f"%{query}%")]

        if category:
            conditions.append(Product.category == category)
        if in_stock_only:
            conditions.append(Product.stock > 0)

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(Product.stock.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        manufacturer: Optional[str] = None,
        in_stock_only: bool = True,
        component_type: Optional[str] = None,
    ) -> list[tuple[Product, float]]:
        """Search products using vector similarity."""
        # Build filter conditions
        conditions = [Product.is_active == True]

        if category:
            conditions.append(Product.category == category)
        if min_price is not None:
            conditions.append(
                or_(Product.price >= min_price, Product.discount_price >= min_price)
            )
        if max_price is not None:
            conditions.append(
                or_(Product.price <= max_price, Product.discount_price <= max_price)
            )
        if manufacturer:
            conditions.append(Product.manufacturer.ilike(f"%{manufacturer}%"))
        if in_stock_only:
            conditions.append(Product.stock > 0)
        if component_type:
            conditions.append(Product.component_type == component_type)

        # Vector similarity search using cosine distance
        distance = ProductEmbedding.embedding.cosine_distance(embedding)

        result = await self.session.execute(
            select(Product, (1 - distance).label("similarity"))
            .join(ProductEmbedding, Product.id == ProductEmbedding.product_id)
            .where(and_(*conditions))
            .order_by(distance)
            .limit(limit)
        )

        return [(row.Product, row.similarity) for row in result.all()]

    async def get_by_component_type(
        self,
        component_type: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock_only: bool = True,
        limit: int = 50,
    ) -> list[Product]:
        """Get products by component type for PC builds."""
        conditions = [
            Product.component_type == component_type,
            Product.is_active == True,
        ]

        if min_price is not None:
            conditions.append(
                or_(Product.price >= min_price, Product.discount_price >= min_price)
            )
        if max_price is not None:
            conditions.append(
                or_(Product.price <= max_price, Product.discount_price <= max_price)
            )
        if in_stock_only:
            conditions.append(Product.stock > 0)

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).asc()
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_categories(self) -> list[str]:
        """Get all unique categories."""
        result = await self.session.execute(
            select(Product.category)
            .where(Product.category.isnot(None))
            .distinct()
        )
        return [row[0] for row in result.all()]

    async def get_manufacturers(self, category: Optional[str] = None) -> list[str]:
        """Get all unique manufacturers, optionally filtered by category."""
        query = select(Product.manufacturer).where(
            Product.manufacturer.isnot(None)
        )
        if category:
            query = query.where(Product.category == category)

        result = await self.session.execute(query.distinct())
        return [row[0] for row in result.all()]

    async def upsert_product(self, product_data: dict) -> Product:
        """Insert or update a product by SKU."""
        existing = await self.get_by_sku(product_data["sku"])

        if existing:
            for key, value in product_data.items():
                if hasattr(existing, key):
                    setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing
        else:
            product = Product(**product_data)
            self.session.add(product)
            await self.session.commit()
            await self.session.refresh(product)
            return product

    async def save_embedding(
        self, product_id: int, embedding: list[float], embedding_text: str
    ) -> ProductEmbedding:
        """Save or update product embedding."""
        # Check if embedding exists
        result = await self.session.execute(
            select(ProductEmbedding).where(ProductEmbedding.product_id == product_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.embedding = embedding
            existing.embedding_text = embedding_text
            await self.session.commit()
            return existing
        else:
            product_embedding = ProductEmbedding(
                product_id=product_id,
                embedding=embedding,
                embedding_text=embedding_text,
            )
            self.session.add(product_embedding)
            await self.session.commit()
            return product_embedding
