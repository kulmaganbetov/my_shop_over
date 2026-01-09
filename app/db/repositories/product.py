"""Product repository with vector search and smart component queries."""

import logging
from typing import Optional, List, Tuple, Dict, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, func, or_, select, text, cast, String
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import JSONB

from app.db.models import Product, ProductEmbedding
from app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ProductRepository(BaseRepository[Product]):
    """Repository for product operations with vector search and smart queries."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Product)

    # =========================================================================
    # BASIC QUERIES
    # =========================================================================

    async def get_by_sku(self, sku: str) -> Optional[Product]:
        """Get product by SKU."""
        result = await self.session.execute(
            select(Product).where(Product.sku == sku)
        )
        return result.scalar_one_or_none()

    async def get_by_kaspi_code(self, code: str) -> Optional[Product]:
        """Get product by Kaspi code."""
        result = await self.session.execute(
            select(Product).where(Product.kaspi_code == code)
        )
        return result.scalar_one_or_none()

    async def search_by_name(
        self,
        query: str,
        limit: int = 10,
        category: Optional[str] = None,
        in_stock_only: bool = True,
    ) -> List[Product]:
        """Search products by name using ILIKE."""
        conditions = [Product.name.ilike(f"%{query}%")]

        if category:
            conditions.append(Product.category.ilike(f"%{category}%"))
        if in_stock_only:
            conditions.append(Product.stock > 0)

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(Product.stock.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # =========================================================================
    # SMART COMPONENT QUERIES (JSONB-based)
    # All methods MUST filter by stock > 0
    # =========================================================================

    async def get_products_by_strict_category(
        self,
        category: str,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Get products by STRICT category match - NO vector search, NO hallucinations.

        CRITICAL: This method is the primary search method.
        Vector search should only be used as fallback for fuzzy queries.

        If no products match, returns EMPTY list (never hallucinate products).

        Args:
            category: Exact category string to match
            min_price: Minimum price filter
            max_price: Maximum price filter
            limit: Maximum results
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            Product.category == category,  # EXACT match
        ]

        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Strict category '{category}': found {len(products)} products")
        return products

    async def find_cpus(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        socket: Optional[str] = None,
        brand: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find CPUs with optional filters.

        CRITICAL: Always filters by stock > 0
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Процессор%"),
                Product.component_type == "cpu",
            ),
        ]

        # Price filter using effective price
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # JSONB filters for specifications
        if socket:
            conditions.append(
                Product.specifications["socket"].astext == socket
            )
        if brand:
            conditions.append(
                Product.specifications["brand"].astext.ilike(f"%{brand}%")
            )
        if tier:
            conditions.append(
                Product.specifications["tier"].astext == tier
            )

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def find_compatible_motherboards(
        self,
        socket: str,
        ram_type: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        tier: Optional[str] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find motherboards compatible with given CPU socket.

        CRITICAL: Always filters by stock > 0

        Args:
            socket: CPU socket (e.g., "AM5", "LGA1700")
            ram_type: Optional RAM type filter (e.g., "DDR5")
            min_price: Minimum price
            max_price: Maximum price
            tier: Optional tier filter (budget, mid, high, enthusiast)
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Материнск%"),
                Product.component_type == "motherboard",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Socket filter - CRITICAL for compatibility
        # Use JSONB if available, otherwise name pattern matching
        socket_condition = or_(
            Product.specifications["socket"].astext == socket,
            Product.name.ilike(f"%{socket}%"),
        )

        # Also match by chipset patterns
        socket_chipset_map = {
            "AM5": ["X870", "X670", "B650", "A620"],
            "AM4": ["X570", "B550", "A520", "X470", "B450", "A320"],
            "LGA1700": ["Z790", "B760", "H770", "H610", "Z690", "B660"],
            "LGA1851": ["Z890", "B860", "H810"],
            "LGA1200": ["Z590", "B560", "H570", "H510", "Z490"],
        }

        chipsets = socket_chipset_map.get(socket, [])
        if chipsets:
            chipset_conditions = [Product.name.ilike(f"%{cs}%") for cs in chipsets]
            socket_condition = or_(socket_condition, *chipset_conditions)

        conditions.append(socket_condition)

        # RAM type filter
        if ram_type:
            ram_condition = or_(
                Product.specifications["ram_type"].astext == ram_type,
                Product.name.ilike(f"%{ram_type}%"),
                Product.name.ilike(f"%D5%") if ram_type == "DDR5" else Product.name.ilike(f"%D4%"),
            )
            conditions.append(ram_condition)

        # Tier filter - avoid Z chipsets for budget builds
        if tier == "budget":
            # Exclude enthusiast chipsets for budget builds
            exclude_chipsets = ["Z790", "Z690", "X670E", "X870E"]
            for exc in exclude_chipsets:
                conditions.append(~Product.name.ilike(f"%{exc}%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} motherboards for socket={socket}")
        return products

    async def find_compatible_ram(
        self,
        ram_type: str,
        min_size_gb: int = 8,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find RAM compatible with given type.

        CRITICAL: Always filters by stock > 0

        Args:
            ram_type: RAM type (e.g., "DDR5", "DDR4")
            min_size_gb: Minimum size in GB (default 8)
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Оперативная память%"),
                Product.category.ilike("%RAM%"),
                Product.component_type == "ram",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # RAM type filter
        ram_condition = or_(
            Product.specifications["ram_type"].astext == ram_type,
            Product.name.ilike(f"%{ram_type}%"),
        )
        conditions.append(ram_condition)

        # Exclude laptop RAM
        conditions.append(~Product.name.ilike("%SO-DIMM%"))
        conditions.append(~Product.name.ilike("%SODIMM%"))
        conditions.append(~Product.name.ilike("%ноутбук%"))

        # Exclude server RAM
        conditions.append(~Product.name.ilike("%ECC%"))
        conditions.append(~Product.name.ilike("%RDIMM%"))
        conditions.append(~Product.name.ilike("%сервер%"))
        conditions.append(~Product.name.ilike("%registered%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} RAM modules for type={ram_type}")
        return products

    async def find_gpus(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        brand: Optional[str] = None,
        min_vram: Optional[int] = None,
        tier: Optional[str] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find GPUs with optional filters.

        CRITICAL: Always filters by stock > 0
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Видеокарт%"),
                Product.component_type == "gpu",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Exclude workstation cards
        conditions.append(~Product.name.ilike("%Quadro%"))
        conditions.append(~Product.name.ilike("%Tesla%"))
        conditions.append(~Product.name.ilike("%A100%"))

        # NOTE: RTX 50-series (5050, 5060, 5070, 5080, 5090) exists - DO NOT block
        # Only block truly non-existent GPUs
        # RTX 60-series doesn't exist
        conditions.append(~Product.name.ilike("%RTX 6050%"))
        conditions.append(~Product.name.ilike("%RTX 6060%"))
        # AMD RX 8000 series doesn't exist
        conditions.append(~Product.name.ilike("%RX 8000%"))
        conditions.append(~Product.name.ilike("%RX 8700%"))
        conditions.append(~Product.name.ilike("%RX 8800%"))
        conditions.append(~Product.name.ilike("%RX 8900%"))

        # Brand filter
        if brand:
            if brand.upper() == "NVIDIA":
                conditions.append(
                    or_(
                        Product.name.ilike("%RTX%"),
                        Product.name.ilike("%GTX%"),
                        Product.name.ilike("%GeForce%"),
                    )
                )
            elif brand.upper() == "AMD":
                conditions.append(
                    or_(
                        Product.name.ilike("%RX %"),
                        Product.name.ilike("%Radeon%"),
                    )
                )

        # VRAM filter
        if min_vram:
            conditions.append(
                Product.specifications["vram_gb"].as_integer() >= min_vram
            )

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} GPUs")
        return products

    async def find_psus(
        self,
        min_wattage: int = 500,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        efficiency: Optional[str] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find PSUs with sufficient wattage.

        CRITICAL: Always filters by stock > 0

        Args:
            min_wattage: Minimum required wattage
            efficiency: Optional efficiency rating (e.g., "Gold", "Platinum")
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Блок%питан%"),
                Product.category.ilike("%PSU%"),
                Product.component_type == "psu",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Wattage filter - use name pattern matching since JSONB might not have it
        # Match patterns like "650W", "750 W", "850W"
        wattage_patterns = []
        for w in range(min_wattage, 1600, 50):
            wattage_patterns.append(Product.name.ilike(f"%{w}W%"))
            wattage_patterns.append(Product.name.ilike(f"%{w} W%"))

        conditions.append(or_(*wattage_patterns))

        # Efficiency filter
        if efficiency:
            conditions.append(Product.name.ilike(f"%{efficiency}%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} PSUs with min {min_wattage}W")
        return products

    async def find_storage(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        storage_type: Optional[str] = None,  # "SSD", "NVMe", "HDD"
        min_capacity_gb: Optional[int] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find storage devices.

        CRITICAL: Always filters by stock > 0
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
        ]

        # Category filter
        if storage_type == "NVMe" or storage_type == "SSD":
            conditions.append(
                or_(
                    Product.category.ilike("%SSD%"),
                    Product.category.ilike("%Твердотельн%"),
                    Product.category.ilike("%накопител%"),
                    Product.component_type == "storage",
                )
            )
        elif storage_type == "HDD":
            conditions.append(
                or_(
                    Product.category.ilike("%HDD%"),
                    Product.category.ilike("%Жестк%"),
                )
            )
        else:
            conditions.append(
                or_(
                    Product.category.ilike("%SSD%"),
                    Product.category.ilike("%накопител%"),
                    Product.component_type == "storage",
                )
            )

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Exclude external drives
        conditions.append(~Product.name.ilike("%внешн%"))
        conditions.append(~Product.name.ilike("%external%"))
        conditions.append(~Product.name.ilike("%portable%"))
        conditions.append(~Product.name.ilike("%Внешний SSD%"))
        conditions.append(~Product.name.ilike("%Внешний HDD%"))

        # Exclude server storage (CRITICAL: blocks HPE MSA, rack systems, etc.)
        conditions.append(~Product.name.ilike("%сервер%"))
        conditions.append(~Product.name.ilike("%enterprise%"))
        conditions.append(~Product.name.ilike("%Система хранения%"))
        conditions.append(~Product.name.ilike("%Storage System%"))
        conditions.append(~Product.name.ilike("%HPE%"))
        conditions.append(~Product.name.ilike("%MSA %"))
        conditions.append(~Product.name.ilike("%MSA2%"))
        conditions.append(~Product.name.ilike("%Rack%"))
        conditions.append(~Product.name.ilike("%SAN %"))
        conditions.append(~Product.name.ilike("%NAS %"))
        conditions.append(~Product.name.ilike("%iSCSI%"))
        conditions.append(~Product.name.ilike("%RAID%"))
        conditions.append(~Product.name.ilike("%JBOD%"))
        conditions.append(~Product.category.ilike("%Система хранения%"))
        conditions.append(~Product.category.ilike("%серверн%"))

        # CRITICAL: Exclude USB flash drives (they are NOT storage for PC builds)
        conditions.append(~Product.name.ilike("%USB%"))
        conditions.append(~Product.name.ilike("%Flash%"))
        conditions.append(~Product.name.ilike("%флешк%"))
        conditions.append(~Product.name.ilike("%флэшк%"))
        conditions.append(~Product.name.ilike("%накопитель USB%"))
        conditions.append(~Product.category.ilike("%Flash%"))
        conditions.append(~Product.category.ilike("%USB%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} storage devices")
        return products

    async def find_cases(
        self,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        form_factor: Optional[str] = None,  # ATX, mATX, ITX
        limit: int = 20,
    ) -> List[Product]:
        """Find PC cases.

        CRITICAL: Always filters by stock > 0
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Корпус%"),
                Product.component_type == "case",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Form factor filter
        if form_factor:
            conditions.append(Product.name.ilike(f"%{form_factor}%"))

        # CRITICAL: Exclude fans/ventilators (they are NOT cases)
        conditions.append(~Product.name.ilike("%вентилятор%"))
        conditions.append(~Product.name.ilike("%Fan %"))
        conditions.append(~Product.name.ilike("% Fan%"))
        conditions.append(~Product.name.ilike("%кулер%"))
        conditions.append(~Product.name.ilike("%охлаждени%"))
        conditions.append(~Product.category.ilike("%вентилятор%"))
        conditions.append(~Product.category.ilike("%кулер%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} cases")
        return products

    async def find_coolers(
        self,
        socket: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        limit: int = 20,
    ) -> List[Product]:
        """Find CPU coolers.

        CRITICAL: Always filters by stock > 0
        """
        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(
                Product.category.ilike("%Кулер%"),
                Product.category.ilike("%охлажд%"),
                Product.component_type == "cooler",
            ),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        # Socket compatibility - coolers often support multiple sockets
        if socket:
            socket_kw = []
            if socket in ["AM4", "AM5"]:
                socket_kw = ["AM4", "AM5", "AMD"]
            elif socket == "LGA1700":
                socket_kw = ["LGA1700", "1700", "Intel"]
            elif socket == "LGA1851":
                socket_kw = ["LGA1851", "1851", "1700", "Intel"]  # 1700 coolers often work

            if socket_kw:
                socket_conditions = [Product.name.ilike(f"%{kw}%") for kw in socket_kw]
                conditions.append(or_(*socket_conditions))

        # CRITICAL: Exclude cooler accessories (NOT actual coolers!)
        # We need Active Coolers (Radiator + Fan), not mounting kits
        conditions.append(~Product.name.ilike("%Bracket%"))
        conditions.append(~Product.name.ilike("%крепление%"))
        conditions.append(~Product.name.ilike("%крепления%"))
        conditions.append(~Product.name.ilike("%Mount%"))
        conditions.append(~Product.name.ilike("%Thermal Paste%"))
        conditions.append(~Product.name.ilike("%термопаст%"))
        conditions.append(~Product.name.ilike("%Screw%"))
        conditions.append(~Product.name.ilike("%винт%"))
        conditions.append(~Product.name.ilike("%комплект%"))
        conditions.append(~Product.name.ilike("%Kit%"))
        conditions.append(~Product.name.ilike("%Adapter%"))
        conditions.append(~Product.name.ilike("%переходник%"))

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} coolers (excluding accessories)")
        return products

    # =========================================================================
    # PERIPHERALS QUERIES
    # =========================================================================

    async def find_peripherals(
        self,
        peripheral_type: str,  # monitor, mouse, keyboard, headset
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        limit: int = 10,
    ) -> List[Product]:
        """Find peripherals by type.

        CRITICAL: Always filters by stock > 0
        """
        type_categories = {
            "monitor": ["%Монитор%", "%дисплей%"],
            "mouse": ["%Мышь%", "%мышка%"],
            "keyboard": ["%Клавиатур%"],
            "headset": ["%Наушник%", "%гарнитур%"],
            "mousepad": ["%Коврик%"],
            "webcam": ["%Веб-камер%", "%камера%"],
        }

        categories = type_categories.get(peripheral_type.lower(), [f"%{peripheral_type}%"])

        conditions = [
            Product.stock > 0,
            Product.is_active == True,
            or_(*[Product.category.ilike(cat) for cat in categories]),
        ]

        # Price filters
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )

        products = list(result.scalars().all())
        logger.info(f"[REPO] Found {len(products)} {peripheral_type}")
        return products

    # =========================================================================
    # VECTOR SEARCH (kept for fuzzy search)
    # =========================================================================

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
    ) -> List[Tuple[Product, float]]:
        """Search products using vector similarity."""
        conditions = [Product.is_active == True]

        if category:
            conditions.append(Product.category.ilike(f"%{category}%"))
        if min_price is not None and min_price > 0:
            conditions.append(
                or_(Product.price >= min_price, Product.discount_price >= min_price)
            )
        if max_price is not None and max_price > 0:
            conditions.append(
                or_(Product.price <= max_price, Product.discount_price <= max_price)
            )
        if manufacturer:
            conditions.append(Product.manufacturer.ilike(f"%{manufacturer}%"))
        if in_stock_only:
            conditions.append(Product.stock > 0)
        if component_type:
            conditions.append(Product.component_type == component_type)

        distance = ProductEmbedding.embedding.cosine_distance(embedding)

        result = await self.session.execute(
            select(Product, (1 - distance).label("similarity"))
            .join(ProductEmbedding, Product.id == ProductEmbedding.product_id)
            .where(and_(*conditions))
            .order_by(distance)
            .limit(limit)
        )

        return [(row.Product, row.similarity) for row in result.all()]

    # =========================================================================
    # EXISTING METHODS
    # =========================================================================

    async def get_by_component_type(
        self,
        component_type: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        in_stock_only: bool = True,
        limit: int = 50,
        search_keywords: List[str] = None,
    ) -> List[Product]:
        """Get products by component type or category for PC builds."""
        type_conditions = [Product.component_type == component_type]

        if search_keywords:
            for kw in search_keywords:
                type_conditions.append(Product.category == kw)
                type_conditions.append(Product.category.ilike(f"%{kw}%"))

        conditions = [
            or_(*type_conditions),
            Product.is_active == True,
        ]

        if min_price is not None and min_price > 0:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )
        if max_price is not None and max_price > 0:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )
        if in_stock_only:
            conditions.append(Product.stock > 0)

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(
                func.coalesce(Product.discount_price, Product.price).desc()
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_categories(self) -> List[str]:
        """Get all unique categories."""
        result = await self.session.execute(
            select(Product.category)
            .where(Product.category.isnot(None))
            .distinct()
        )
        return [row[0] for row in result.all()]

    async def get_manufacturers(self, category: Optional[str] = None) -> List[str]:
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

    async def update_product_specs(self, product_id: int, specs: Dict[str, Any]) -> bool:
        """Update product specifications JSONB field.

        Used to enrich products with parsed specs.
        """
        try:
            result = await self.session.execute(
                select(Product).where(Product.id == product_id)
            )
            product = result.scalar_one_or_none()
            if product:
                # Merge with existing specs
                current_specs = product.specifications or {}
                current_specs.update(specs)
                product.specifications = current_specs
                await self.session.commit()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update specs for product {product_id}: {e}")
            return False
