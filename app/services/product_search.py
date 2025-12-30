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

# Accessory keywords to filter out when searching for main components
ACCESSORY_KEYWORDS = [
    "держатель", "подставка", "вентилятор для", "кабель", "переходник",
    "кронштейн", "крепление", "стойка", "органайзер", "бокс для",
    "термопаста", "салфетки", "чистящ", "защитн", "наклейк",
]

# Categories where we should filter out accessories
MAIN_COMPONENT_CATEGORIES = [
    "Видеокарты", "Процессоры", "Материнские платы", "Оперативная память",
    "SSD накопители", "Жесткие диски", "Блоки питания", "Корпуса",
]


class ProductSearchService:
    """Service for searching products using text and vector search."""

    def __init__(self, session: AsyncSession, llm_service: LLMService):
        self.session = session
        self.llm_service = llm_service
        self.product_repo = ProductRepository(session)

    def _is_accessory(self, product_name: str) -> bool:
        """Check if product is an accessory based on name."""
        name_lower = product_name.lower()
        return any(kw in name_lower for kw in ACCESSORY_KEYWORDS)

    def _filter_accessories(self, products: list, category: str | None) -> list:
        """Filter out accessories when searching for main components."""
        if not category or category not in MAIN_COMPONENT_CATEGORIES:
            return products
        return [p for p in products if not self._is_accessory(p.name)]

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
            # Request more results so we have enough after filtering
            search_limit = params.limit * 3 if params.category in MAIN_COMPONENT_CATEGORIES else params.limit
            results = await self.product_repo.vector_search(
                embedding=embedding,
                limit=search_limit,
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

            # Filter out accessories if searching for main components
            products = self._filter_accessories(products, params.category)
            products = products[:params.limit]

            # If vector search returned nothing, try text search as fallback
            if not products:
                logger.info(f"Vector search empty, trying text search for: {params.query}")
                products = await self._text_search_fallback(params)

        except Exception as e:
            logger.warning(f"Vector search failed, falling back to text search: {e}")
            products = await self._text_search_fallback(params)

        return ProductSearchResponse(
            products=products,
            total_count=len(products),
            query=params.query,
            filters_applied=filters_applied,
        )

    async def _text_search_fallback(self, params: ProductSearchParams) -> list[ProductSchema]:
        """Fallback text search when vector search fails or returns empty."""
        # Try with category first
        text_results = await self.product_repo.search_by_name(
            query=params.query,
            limit=params.limit * 2,
            category=params.category,
            in_stock_only=params.in_stock_only,
        )

        # If nothing found with category, try without
        if not text_results and params.category:
            logger.info(f"No results with category '{params.category}', searching without")
            text_results = await self.product_repo.search_by_name(
                query=params.query,
                limit=params.limit * 2,
                category=None,
                in_stock_only=params.in_stock_only,
            )

        products = [
            ProductSchema.model_validate(product)
            for product in text_results
        ]

        # Filter out accessories
        products = self._filter_accessories(products, params.category)
        return products[:params.limit]

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
