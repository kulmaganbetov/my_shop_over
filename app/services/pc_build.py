"""PC Build service with deterministic compatibility rules."""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import PresetRepository, ProductRepository
from app.llm.service import LLMService
from app.schemas.chat import PCBuildResult
from app.schemas.common import ProductSchema
from app.schemas.llm import PCBuildParams, PCPurpose
from app.services.compatibility import (
    BUDGET_ALLOCATION,
    CompatibilityEngine,
    CompatibilityStatus,
)

logger = logging.getLogger(__name__)


# Default budgets by purpose (in tenge)
DEFAULT_BUDGETS = {
    "gaming": 500000,      # Игровой ПК
    "office": 200000,      # Офисный ПК
    "work": 700000,        # Рабочая станция
    "budget": 350000,      # Бюджетный ПК
    "default": 450000,     # По умолчанию
}

# Component types mapping - exact category names from DB
# Format: (exact_category_name, [fallback_keywords])
COMPONENT_TYPES = {
    "cpu": ("Процессоры", ["Процессоры"]),
    "gpu": ("Видеокарты", ["Видеокарты"]),
    "motherboard": ("Материнские платы", ["Материнские платы"]),
    "ram": ("Оперативная память", ["Оперативная память"]),
    "storage": ("Твердотельные диски (SSD)", ["Твердотельные диски", "SSD"]),
    "psu": ("Блоки питания", ["Блоки питания"]),
    "case": ("Корпуса", ["Корпуса"]),
    "cooler": ("Кулеры для процессоров", ["Кулеры для процессоров", "Кулеры"]),
}


class PCBuildService:
    """Service for recommending PC builds.

    Uses DETERMINISTIC rules for compatibility.
    LLM is only used for generating explanations.
    """

    def __init__(self, session: AsyncSession, llm_service: LLMService):
        self.session = session
        self.llm_service = llm_service
        self.product_repo = ProductRepository(session)
        self.preset_repo = PresetRepository(session)
        self.compatibility_engine = CompatibilityEngine()

    async def recommend_build(self, params: PCBuildParams) -> PCBuildResult:
        """Recommend a PC build based on parameters."""
        budget = params.budget or 0
        if params.budget_min and params.budget_max:
            budget = (params.budget_min + params.budget_max) // 2
        elif params.budget_min:
            budget = params.budget_min
        elif params.budget_max:
            budget = params.budget_max

        purpose = params.purpose.value if isinstance(params.purpose, PCPurpose) else params.purpose

        # Use default budget if not specified
        if budget <= 0:
            budget = DEFAULT_BUDGETS.get(purpose, DEFAULT_BUDGETS["default"])
            logger.info(f"Using default budget for {purpose}: {budget} тенге")

        # Check for matching presets first
        presets = await self.preset_repo.get_by_budget(budget, purpose)
        if presets:
            preset_result = await self._build_from_preset(presets[0], budget)
            # If preset has at least 3 valid components, use it
            valid_components = sum(1 for v in preset_result.build.values() if v is not None)
            if valid_components >= 3:
                return preset_result
            logger.info(f"Preset has only {valid_components} valid components, building from scratch")

        # Build from scratch
        return await self._build_from_scratch(budget, purpose, params)

    async def _build_from_preset(
        self,
        preset,
        budget: int,
    ) -> PCBuildResult:
        """Build from a predefined preset."""
        build = {}
        total_price = 0
        warnings = []

        component_skus = preset.components

        for component_type, sku in component_skus.items():
            product = await self.product_repo.get_by_sku(sku)
            if product and product.stock > 0:
                build[component_type] = ProductSchema.model_validate(product)
                price = product.discount_price or product.price or 0
                total_price += price
            else:
                build[component_type] = None
                warnings.append(f"{component_type}: рекомендуемый товар недоступен")

        # Check compatibility
        compatibility_notes = []
        build_specs = self._extract_build_specs(build)
        compatibility_results = self.compatibility_engine.check_all_compatibility(build_specs)
        overall_status = self.compatibility_engine.get_overall_status(compatibility_results)

        for result in compatibility_results:
            if result.status != CompatibilityStatus.OK:
                compatibility_notes.append(result.message)

        return PCBuildResult(
            build=build,
            total_price=total_price,
            compatibility=overall_status.value,
            compatibility_notes=compatibility_notes,
            warnings=warnings,
        )

    async def _build_from_scratch(
        self,
        budget: int,
        purpose: str,
        params: PCBuildParams,
    ) -> PCBuildResult:
        """Build a PC from scratch based on budget and purpose."""
        # Allocate budget
        budget_allocation = self.compatibility_engine.allocate_budget(budget, purpose)

        build = {}
        total_price = 0
        warnings = []
        compatibility_notes = []

        # Select components in order of importance
        component_order = ["cpu", "gpu", "motherboard", "ram", "storage", "psu", "case"]

        for component in component_order:
            component_budget = budget_allocation.get(component, 0)
            component_config = COMPONENT_TYPES.get(component)

            if not component_config:
                continue

            component_type, search_keywords = component_config

            # Get products for this component type within budget range
            # Use min_price to find products that match budget, not just cheapest
            min_budget = int(component_budget * 0.4)  # 40% of allocated budget
            max_budget = int(component_budget * 1.0)  # 100% of allocated budget (strict)

            products = await self.product_repo.get_by_component_type(
                component_type=component_type,
                min_price=min_budget,
                max_price=max_budget,
                in_stock_only=True,
                limit=20,
                search_keywords=search_keywords,
            )
            logger.info(f"Found {len(products)} products for {component} (budget: {min_budget}-{max_budget})")

            # If no products in range, try without min_price constraint
            if not products:
                products = await self.product_repo.get_by_component_type(
                    component_type=component_type,
                    max_price=max_budget,
                    in_stock_only=True,
                    limit=20,
                    search_keywords=search_keywords,
                )
                logger.info(f"Fallback: found {len(products)} products for {component}")

            if products:
                # Select best product within budget
                selected = self._select_best_component(
                    products,
                    component_budget,
                    purpose,
                    build,
                )
                if selected:
                    build[component] = ProductSchema.model_validate(selected)
                    price = selected.discount_price or selected.price or 0
                    total_price += price
                else:
                    build[component] = None
                    warnings.append(f"{component}: не найден подходящий товар в бюджете")
            else:
                build[component] = None
                warnings.append(f"{component}: товары не найдены")

        # Validate compatibility
        build_specs = self._extract_build_specs(build)
        compatibility_results = self.compatibility_engine.check_all_compatibility(build_specs)
        overall_status = self.compatibility_engine.get_overall_status(compatibility_results)

        for result in compatibility_results:
            if result.status != CompatibilityStatus.OK:
                compatibility_notes.append(result.message)

        return PCBuildResult(
            build=build,
            total_price=total_price,
            compatibility=overall_status.value,
            compatibility_notes=compatibility_notes,
            warnings=warnings,
        )

    def _select_best_component(
        self,
        products: list,
        budget: int,
        purpose: str,
        current_build: dict,
    ) -> Optional[any]:
        """Select the best component within budget.

        This uses DETERMINISTIC rules based on:
        1. Price within budget
        2. Stock availability
        3. Compatibility with current build
        4. Exclude server components
        """
        # Filter out server components first
        filtered_products = []
        for product in products:
            category_lower = (product.category or "").lower()
            name_lower = (product.name or "").lower()

            # Skip server components
            if "сервер" in category_lower or "для сервера" in category_lower:
                continue
            if "ecc" in name_lower or "rdimm" in name_lower or "lrdimm" in name_lower:
                continue

            filtered_products.append(product)

        # Use filtered list, fallback to original if all filtered out
        products = filtered_products if filtered_products else products

        # Filter products within budget (strict - no overflow)
        candidates = []
        for product in products:
            price = product.discount_price or product.price or 0
            if price <= budget and product.stock > 0:
                candidates.append((product, price))

        if not candidates:
            # If nothing in budget, return cheapest available
            if products:
                return min(products, key=lambda p: p.discount_price or p.price or float('inf'))
            return None

        # Sort by price descending (get best within budget)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Return the most expensive within budget (best value)
        return candidates[0][0]

    def _extract_build_specs(self, build: dict) -> dict:
        """Extract specifications for compatibility checking."""
        specs = {}

        # Extract specs from products
        for component_type, product in build.items():
            if product is None:
                continue

            # Handle both ProductSchema objects and dicts
            if isinstance(product, dict):
                product_specs = product.get("specifications") or {}
                product_name = product.get("name", "")
            else:
                product_specs = product.specifications or {}
                product_name = product.name if hasattr(product, 'name') else ""

            if component_type == "cpu":
                specs["cpu_socket"] = product_specs.get("socket")
                specs["cpu_tdp"] = product_specs.get("tdp", 65)

            elif component_type == "motherboard":
                specs["motherboard_socket"] = product_specs.get("socket")
                specs["motherboard_ram_type"] = product_specs.get("ram_type")

            elif component_type == "ram":
                specs["ram_type"] = product_specs.get("type")

            elif component_type == "gpu":
                specs["gpu_model"] = product_name
                specs["gpu_length"] = product_specs.get("length", 300)
                specs["gpu_tdp"] = product_specs.get("tdp", 200)

            elif component_type == "psu":
                specs["psu_wattage"] = product_specs.get("wattage", 500)

            elif component_type == "case":
                specs["case_max_gpu_length"] = product_specs.get("max_gpu_length", 350)
                specs["case_max_cooler_height"] = product_specs.get("max_cooler_height", 160)

            elif component_type == "cooler":
                specs["cooler_height"] = product_specs.get("height", 150)

        # Calculate total TDP
        total_tdp = specs.get("cpu_tdp", 0) + specs.get("gpu_tdp", 0) + 100  # +100 for other components
        specs["total_tdp"] = total_tdp

        return specs

    async def get_component_alternatives(
        self,
        component_type: str,
        current_build: dict,
        budget: Optional[int] = None,
        preference: Optional[str] = None,
        limit: int = 5,
    ) -> list[ProductSchema]:
        """Get alternative components for replacement.

        Returns products without strict compatibility filtering since
        many products lack complete specifications for compatibility checks.
        Uses reasonable budget based on the current component's price.
        """
        component_config = COMPONENT_TYPES.get(component_type)
        if not component_config:
            return []

        db_component_type, search_keywords = component_config

        # Check if preference is about price (cheaper/better)
        preference_lower = (preference or "").lower()
        wants_cheaper = any(word in preference_lower for word in ["дешев", "cheap", "бюджет", "недорог"])
        wants_better = any(word in preference_lower for word in ["дорож", "лучш", "better", "мощн", "топов"])

        # Calculate budget based on current component price and preference
        current_component = current_build.get(component_type)
        current_price = 0
        if current_component:
            if isinstance(current_component, dict):
                current_price = current_component.get("discount_price") or current_component.get("price") or 0
            else:
                current_price = getattr(current_component, "discount_price", None) or getattr(current_component, "price", 0)

        if current_price > 0:
            if wants_cheaper:
                # Show products BELOW current price (20% to 90% of current)
                min_budget = int(current_price * 0.2)
                max_budget = int(current_price * 0.95)  # Must be cheaper
                logger.info(f"Cheaper alternatives: {min_budget}-{max_budget} (current: {current_price})")
            elif wants_better:
                # Show products ABOVE current price (110% to 300%)
                min_budget = int(current_price * 1.05)
                max_budget = int(current_price * 3.0)
                logger.info(f"Better alternatives: {min_budget}-{max_budget} (current: {current_price})")
            else:
                # Allow alternatives from 50% to 200% of current component price
                min_budget = int(current_price * 0.5)
                max_budget = int(current_price * 2.0)
                logger.info(f"Component alternatives budget: {min_budget}-{max_budget} (current: {current_price})")
        else:
            min_budget = None
            max_budget = budget

        # Get products from DB
        products = await self.product_repo.get_by_component_type(
            component_type=db_component_type,
            min_price=min_budget,
            max_price=max_budget,
            in_stock_only=True,
            limit=50,  # Get more to filter
            search_keywords=search_keywords,
        )

        if not products:
            # Fallback without min price
            products = await self.product_repo.get_by_component_type(
                component_type=db_component_type,
                max_price=max_budget,
                in_stock_only=True,
                limit=50,
                search_keywords=search_keywords,
            )

        if not products:
            return []

        # Exclude server components (they're too expensive and not for consumers)
        filtered_products = []
        for product in products:
            category_lower = (product.category or "").lower()
            name_lower = (product.name or "").lower()

            # Skip server components
            if "сервер" in category_lower or "для сервера" in category_lower:
                continue
            if "xeon" in name_lower or "epyc" in name_lower:
                continue
            if "ecc" in name_lower or "rdimm" in name_lower:
                continue

            filtered_products.append(product)

        products = filtered_products if filtered_products else products

        # If user has a brand preference (e.g., "Intel", "AMD"), filter by it
        # Skip price preferences as they're already handled above
        if preference and not wants_cheaper and not wants_better:
            preference_filtered = []
            for product in products:
                name_lower = product.name.lower() if product.name else ""
                manufacturer_lower = product.manufacturer.lower() if product.manufacturer else ""

                # Check if preference matches name or manufacturer
                if preference_lower in name_lower or preference_lower in manufacturer_lower:
                    preference_filtered.append(product)

            # If we found matching products, use them
            if preference_filtered:
                products = preference_filtered
                logger.info(f"Found {len(products)} products matching preference '{preference}'")
            else:
                logger.info(f"No products matching preference '{preference}', showing all alternatives")

        # Sort by price (ascending for reasonable alternatives)
        products.sort(key=lambda p: p.discount_price or p.price or 0)

        # Return products
        return [ProductSchema.model_validate(p) for p in products[:limit]]
