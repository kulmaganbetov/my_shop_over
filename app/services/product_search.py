"""Product search service with vector search support."""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import ProductRepository
from app.llm.service import LLMService
from app.schemas.chat import ProductSearchResponse
from app.schemas.common import ProductSchema
from app.schemas.llm import ProductSearchParams

logger = logging.getLogger(__name__)


class ProductSearchService:
    """Service for searching products using text and vector search."""

    def __init__(self, session: AsyncSession, llm_service: LLMService):
        self.session = session
        self.llm_service = llm_service
        self.product_repo = ProductRepository(session)

    async def search(self, params: ProductSearchParams) -> ProductSearchResponse:
        """Search for products using combined text and vector search."""
        products = []
        filters_applied = {}

        # Track applied filters
        if params.category:
            filters_applied["category"] = params.category
        if params.min_price:
            filters_applied["min_price"] = params.min_price
        if params.max_price:
            filters_applied["max_price"] = params.max_price
        if params.manufacturer:
            filters_applied["manufacturer"] = params.manufacturer
        if params.in_stock_only:
            filters_applied["in_stock_only"] = params.in_stock_only

        try:
            # Try vector search first
            embedding = await self.llm_service.get_embedding(params.query)
            results = await self.product_repo.vector_search(
                embedding=embedding,
                limit=params.limit,
                category=params.category,
                min_price=params.min_price,
                max_price=params.max_price,
                manufacturer=params.manufacturer,
                in_stock_only=params.in_stock_only,
            )

            products = [
                ProductSchema.model_validate(product)
                for product, score in results
            ]

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")

            # Fallback to text search
            text_results = await self.product_repo.search_by_name(
                query=params.query,
                limit=params.limit,
                category=params.category,
                in_stock_only=params.in_stock_only,
            )

            products = [
                ProductSchema.model_validate(product)
                for product in text_results
            ]

        return ProductSearchResponse(
            products=products,
            total_count=len(products),
            query=params.query,
            filters_applied=filters_applied,
        )

    async def search_by_component_type(
        self,
        component_type: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 10,
    ) -> list[ProductSchema]:
        """Search products by PC component type."""
        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=True,
            limit=limit,
        )
        return [ProductSchema.model_validate(p) for p in products]

    async def get_categories(self) -> list[str]:
        """Get all product categories."""
        return await self.product_repo.get_categories()

    async def get_manufacturers(self, category: Optional[str] = None) -> list[str]:
        """Get all manufacturers, optionally filtered by category."""
        return await self.product_repo.get_manufacturers(category)
