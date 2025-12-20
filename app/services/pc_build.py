"""PC Build service with STRICT compatibility rules.

GOLDEN RULE: Compatibility = deterministic logic, NOT AI.

Selection pipeline (strict order):
1. CPU (by budget and purpose)
2. Motherboard (MUST match CPU socket + RAM type)
3. RAM (MUST match motherboard RAM type)
4. GPU (by budget)
5. PSU (MUST support total TDP)
6. Case (MUST fit GPU)
7. Storage (almost no restrictions)
"""

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
from app.services.specs_extractor import (
    SpecsExtractor,
    Socket,
    RAMType,
    CPUSpecs,
    MotherboardSpecs,
    RAMSpecs,
)

logger = logging.getLogger(__name__)


# Default budgets by purpose (in tenge)
DEFAULT_BUDGETS = {
    "gaming": 500000,
    "office": 200000,
    "work": 700000,
    "budget": 350000,
    "default": 450000,
}

# Component types mapping
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
        self.specs_extractor = SpecsExtractor()

    async def recommend_build(self, params: PCBuildParams) -> PCBuildResult:
        """Recommend a PC build based on parameters.

        Uses STRICT compatibility pipeline:
        1. CPU → 2. MB (matching socket) → 3. RAM (matching type) → 4. GPU → 5. PSU → 6. Case → 7. Storage
        """
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

        # Build with strict compatibility
        return await self._build_with_compatibility(budget, purpose, params)

    async def _build_with_compatibility(
        self,
        budget: int,
        purpose: str,
        params: PCBuildParams,
    ) -> PCBuildResult:
        """Build PC with STRICT compatibility checks.

        Pipeline: CPU → MB → RAM → GPU → PSU → Case → Storage
        """
        budget_allocation = self.compatibility_engine.allocate_budget(budget, purpose)

        build = {}
        total_price = 0
        warnings = []
        compatibility_notes = []

        # ============================================
        # Step 1: SELECT CPU
        # ============================================
        cpu_budget = budget_allocation.get("cpu", 0)
        cpu, cpu_specs, cpu_price = await self._select_cpu(cpu_budget, purpose)

        if not cpu:
            warnings.append("cpu: не найден подходящий процессор")
            cpu_specs = CPUSpecs()  # Empty specs
        else:
            build["cpu"] = ProductSchema.model_validate(cpu)
            total_price += cpu_price
            logger.info(f"Selected CPU: {cpu.name}, socket: {cpu_specs.socket}")

        # ============================================
        # Step 2: SELECT MOTHERBOARD (must match CPU socket)
        # ============================================
        mb_budget = budget_allocation.get("motherboard", 0)
        mb, mb_specs, mb_price = await self._select_motherboard(mb_budget, cpu_specs)

        if not mb:
            warnings.append("motherboard: не найдена совместимая материнская плата")
            mb_specs = MotherboardSpecs()
        else:
            build["motherboard"] = ProductSchema.model_validate(mb)
            total_price += mb_price
            logger.info(f"Selected MB: {mb.name}, socket: {mb_specs.socket}, RAM: {mb_specs.ram_type}")

        # ============================================
        # Step 3: SELECT RAM (must match MB RAM type)
        # ============================================
        ram_budget = budget_allocation.get("ram", 0)
        ram, ram_specs, ram_price = await self._select_ram(ram_budget, mb_specs, cpu_specs)

        if not ram:
            warnings.append("ram: не найдена совместимая оперативная память")
        else:
            build["ram"] = ProductSchema.model_validate(ram)
            total_price += ram_price
            logger.info(f"Selected RAM: {ram.name}, type: {ram_specs.ram_type}")

        # ============================================
        # Step 4: SELECT GPU
        # ============================================
        gpu_budget = budget_allocation.get("gpu", 0)
        gpu, gpu_specs, gpu_price = await self._select_gpu(gpu_budget, purpose)

        if not gpu:
            warnings.append("gpu: не найдена видеокарта")
        else:
            build["gpu"] = ProductSchema.model_validate(gpu)
            total_price += gpu_price
            logger.info(f"Selected GPU: {gpu.name}")

        # ============================================
        # Step 5: SELECT PSU (must support total TDP)
        # ============================================
        psu_budget = budget_allocation.get("psu", 0)
        total_tdp = (cpu_specs.tdp if cpu_specs else 65) + (gpu_specs.tdp if gpu_specs else 200) + 100
        psu, psu_specs, psu_price = await self._select_psu(psu_budget, total_tdp)

        if not psu:
            warnings.append("psu: не найден подходящий блок питания")
        else:
            build["psu"] = ProductSchema.model_validate(psu)
            total_price += psu_price
            logger.info(f"Selected PSU: {psu.name}, wattage: {psu_specs.wattage}W")

        # ============================================
        # Step 6: SELECT CASE
        # ============================================
        case_budget = budget_allocation.get("case", 0)
        case_product, case_price = await self._select_case(case_budget, gpu_specs)

        if not case_product:
            warnings.append("case: не найден корпус")
        else:
            build["case"] = ProductSchema.model_validate(case_product)
            total_price += case_price

        # ============================================
        # Step 7: SELECT STORAGE
        # ============================================
        storage_budget = budget_allocation.get("storage", 0)
        storage, storage_price = await self._select_storage(storage_budget)

        if not storage:
            warnings.append("storage: не найден накопитель")
        else:
            build["storage"] = ProductSchema.model_validate(storage)
            total_price += storage_price

        # ============================================
        # Validate final build compatibility
        # ============================================
        if cpu_specs.socket and mb_specs.socket and cpu_specs.socket != mb_specs.socket:
            compatibility_notes.append(f"ОШИБКА: CPU ({cpu_specs.socket}) не совместим с материнской платой ({mb_specs.socket})")

        if mb_specs.ram_type and ram_specs and ram_specs.ram_type and mb_specs.ram_type != ram_specs.ram_type:
            compatibility_notes.append(f"ОШИБКА: RAM ({ram_specs.ram_type}) не совместима с материнской платой ({mb_specs.ram_type})")

        overall_status = "ok" if not compatibility_notes else "error"

        return PCBuildResult(
            build=build,
            total_price=total_price,
            compatibility=overall_status,
            compatibility_notes=compatibility_notes,
            warnings=warnings,
        )

    async def _select_cpu(
        self,
        budget: int,
        purpose: str,
    ) -> tuple[Optional[any], CPUSpecs, int]:
        """Select CPU within budget."""
        component_type, keywords = COMPONENT_TYPES["cpu"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            min_price=int(budget * 0.5),
            max_price=budget,
            in_stock_only=True,
            limit=30,
            search_keywords=keywords,
        )

        if not products:
            products = await self.product_repo.get_by_component_type(
                component_type=component_type,
                max_price=budget,
                in_stock_only=True,
                limit=30,
                search_keywords=keywords,
            )

        # Filter and score products
        best_cpu = None
        best_specs = CPUSpecs()
        best_score = -1

        for product in products:
            # Filter out server CPUs
            name_lower = (product.name or "").lower()
            if "xeon" in name_lower or "epyc" in name_lower:
                continue

            specs = self.specs_extractor.extract_cpu_specs(
                product.name or "",
                product.specifications or {},
            )

            # Skip if we can't determine socket
            if not specs.socket:
                continue

            price = product.discount_price or product.price or 0
            if price > budget or price <= 0:
                continue

            # Score: higher price = better performance (within budget)
            score = price
            # Prefer newer sockets
            if specs.socket in [Socket.AM5, Socket.LGA1700]:
                score *= 1.1

            if score > best_score:
                best_score = score
                best_cpu = product
                best_specs = specs

        if best_cpu:
            price = best_cpu.discount_price or best_cpu.price or 0
            return best_cpu, best_specs, price

        return None, CPUSpecs(), 0

    async def _select_motherboard(
        self,
        budget: int,
        cpu_specs: CPUSpecs,
    ) -> tuple[Optional[any], MotherboardSpecs, int]:
        """Select motherboard that matches CPU socket."""
        component_type, keywords = COMPONENT_TYPES["motherboard"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=int(budget * 1.5),  # Allow some overflow for compatibility
            in_stock_only=True,
            limit=50,
            search_keywords=keywords,
        )

        # Filter and score products
        best_mb = None
        best_specs = MotherboardSpecs()
        best_score = -1

        for product in products:
            # Filter out server motherboards
            name_lower = (product.name or "").lower()
            if "сервер" in name_lower or "server" in name_lower:
                continue

            specs = self.specs_extractor.extract_motherboard_specs(
                product.name or "",
                product.specifications or {},
            )

            # CRITICAL: Must match CPU socket
            if cpu_specs.socket and specs.socket != cpu_specs.socket:
                continue

            # Must have determined RAM type
            if not specs.ram_type:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0:
                continue

            # Score: prefer within budget, higher price = better features
            score = min(price, budget)  # Cap at budget
            if price <= budget:
                score *= 1.2  # Bonus for being within budget

            if score > best_score:
                best_score = score
                best_mb = product
                best_specs = specs

        if best_mb:
            price = best_mb.discount_price or best_mb.price or 0
            return best_mb, best_specs, price

        return None, MotherboardSpecs(), 0

    async def _select_ram(
        self,
        budget: int,
        mb_specs: MotherboardSpecs,
        cpu_specs: CPUSpecs,
    ) -> tuple[Optional[any], Optional[RAMSpecs], int]:
        """Select RAM that matches motherboard type."""
        component_type, keywords = COMPONENT_TYPES["ram"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=50,
            search_keywords=keywords,
        )

        best_ram = None
        best_specs = None
        best_score = -1

        for product in products:
            name_lower = (product.name or "").lower()

            # Filter out notebook RAM
            if "so-dimm" in name_lower or "sodimm" in name_lower or "ноутбук" in name_lower:
                continue

            specs = self.specs_extractor.extract_ram_specs(
                product.name or "",
                product.specifications or {},
            )

            # CRITICAL: Must match motherboard RAM type
            if mb_specs.ram_type and specs.ram_type != mb_specs.ram_type:
                continue

            # Also check CPU compatibility
            if cpu_specs.socket and specs.ram_type:
                from app.services.specs_extractor import SOCKET_RAM_SUPPORT
                supported = SOCKET_RAM_SUPPORT.get(cpu_specs.socket, [])
                if specs.ram_type not in supported:
                    continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            # Score: higher capacity and frequency = better
            score = price
            if specs.size_gb:
                score *= (specs.size_gb / 8)  # Prefer more RAM
            if specs.frequency and specs.frequency > 3200:
                score *= 1.1

            if score > best_score:
                best_score = score
                best_ram = product
                best_specs = specs

        if best_ram:
            price = best_ram.discount_price or best_ram.price or 0
            return best_ram, best_specs, price

        return None, None, 0

    async def _select_gpu(
        self,
        budget: int,
        purpose: str,
    ) -> tuple[Optional[any], Optional[any], int]:
        """Select GPU within budget."""
        component_type, keywords = COMPONENT_TYPES["gpu"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            min_price=int(budget * 0.5),
            max_price=budget,
            in_stock_only=True,
            limit=30,
            search_keywords=keywords,
        )

        if not products:
            products = await self.product_repo.get_by_component_type(
                component_type=component_type,
                max_price=budget,
                in_stock_only=True,
                limit=30,
                search_keywords=keywords,
            )

        best_gpu = None
        best_specs = None
        best_score = -1

        for product in products:
            specs = self.specs_extractor.extract_gpu_specs(
                product.name or "",
                product.specifications or {},
            )

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            score = price
            if best_score < score:
                best_score = score
                best_gpu = product
                best_specs = specs

        if best_gpu:
            price = best_gpu.discount_price or best_gpu.price or 0
            return best_gpu, best_specs, price

        return None, None, 0

    async def _select_psu(
        self,
        budget: int,
        required_tdp: int,
    ) -> tuple[Optional[any], any, int]:
        """Select PSU that can handle the system TDP."""
        component_type, keywords = COMPONENT_TYPES["psu"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=30,
            search_keywords=keywords,
        )

        from app.services.specs_extractor import PSUSpecs

        best_psu = None
        best_specs = PSUSpecs()
        best_score = -1

        for product in products:
            specs = self.specs_extractor.extract_psu_specs(
                product.name or "",
                product.specifications or {},
            )

            # CRITICAL: Must have enough wattage
            if specs.wattage < required_tdp:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            # Score: prefer efficient, adequate power
            score = price
            if specs.efficiency:
                score *= 1.1  # Prefer certified efficiency

            if score > best_score:
                best_score = score
                best_psu = product
                best_specs = specs

        if best_psu:
            price = best_psu.discount_price or best_psu.price or 0
            return best_psu, best_specs, price

        return None, PSUSpecs(), 0

    async def _select_case(
        self,
        budget: int,
        gpu_specs: Optional[any],
    ) -> tuple[Optional[any], int]:
        """Select case that fits the GPU."""
        component_type, keywords = COMPONENT_TYPES["case"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=20,
            search_keywords=keywords,
        )

        # Just select best within budget for now
        best_case = None
        best_price = 0

        for product in products:
            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            if price > best_price:
                best_price = price
                best_case = product

        if best_case:
            return best_case, best_price

        return None, 0

    async def _select_storage(
        self,
        budget: int,
    ) -> tuple[Optional[any], int]:
        """Select storage within budget."""
        component_type, keywords = COMPONENT_TYPES["storage"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=20,
            search_keywords=keywords,
        )

        best_storage = None
        best_price = 0

        for product in products:
            name_lower = (product.name or "").lower()

            # Filter out external drives
            if "внешний" in name_lower or "external" in name_lower or "portable" in name_lower:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            if price > best_price:
                best_price = price
                best_storage = product

        if best_storage:
            return best_storage, best_price

        return None, 0

    async def get_component_alternatives(
        self,
        component_type: str,
        current_build: dict,
        budget: Optional[int] = None,
        preference: Optional[str] = None,
        limit: int = 5,
    ) -> list[ProductSchema]:
        """Get alternative components that are COMPATIBLE with current build."""
        component_config = COMPONENT_TYPES.get(component_type)
        if not component_config:
            return []

        db_component_type, search_keywords = component_config

        # Get current component specs for compatibility check
        current_component = current_build.get(component_type, {})
        current_price = 0
        if current_component:
            current_price = current_component.get("discount_price") or current_component.get("price") or 0

        # Determine budget range based on preference
        preference_lower = (preference or "").lower()
        wants_cheaper = any(word in preference_lower for word in ["дешев", "cheap", "бюджет"])
        wants_better = any(word in preference_lower for word in ["дорож", "лучш", "better", "мощн"])

        if current_price > 0:
            if wants_cheaper:
                min_budget = int(current_price * 0.2)
                max_budget = int(current_price * 0.95)
            elif wants_better:
                min_budget = int(current_price * 1.05)
                max_budget = int(current_price * 3.0)
            else:
                min_budget = int(current_price * 0.5)
                max_budget = int(current_price * 2.0)
        else:
            min_budget = None
            max_budget = budget

        # Get products
        products = await self.product_repo.get_by_component_type(
            component_type=db_component_type,
            min_price=min_budget,
            max_price=max_budget,
            in_stock_only=True,
            limit=50,
            search_keywords=search_keywords,
        )

        if not products:
            return []

        # Get build specs for compatibility checking
        cpu_comp = current_build.get("cpu", {})
        mb_comp = current_build.get("motherboard", {})

        cpu_specs = self.specs_extractor.extract_cpu_specs(
            cpu_comp.get("name", ""),
            cpu_comp.get("specifications", {}),
        ) if cpu_comp else CPUSpecs()

        mb_specs = self.specs_extractor.extract_motherboard_specs(
            mb_comp.get("name", ""),
            mb_comp.get("specifications", {}),
        ) if mb_comp else MotherboardSpecs()

        # Filter for compatibility
        compatible_products = []
        for product in products:
            name_lower = (product.name or "").lower()

            # Common filters
            if "сервер" in name_lower or "xeon" in name_lower or "epyc" in name_lower:
                continue
            if "so-dimm" in name_lower or "sodimm" in name_lower or "ноутбук" in name_lower:
                continue
            if "внешний" in name_lower or "external" in name_lower:
                continue

            # Component-specific compatibility
            if component_type == "motherboard":
                specs = self.specs_extractor.extract_motherboard_specs(product.name or "", {})
                # Must match CPU socket
                if cpu_specs.socket and specs.socket != cpu_specs.socket:
                    continue

            elif component_type == "ram":
                specs = self.specs_extractor.extract_ram_specs(product.name or "", {})
                # Must match motherboard RAM type
                if mb_specs.ram_type and specs.ram_type != mb_specs.ram_type:
                    continue

            elif component_type == "cpu":
                specs = self.specs_extractor.extract_cpu_specs(product.name or "", {})
                # Must match motherboard socket
                if mb_specs.socket and specs.socket != mb_specs.socket:
                    continue

            compatible_products.append(product)

        # Sort by price
        compatible_products.sort(key=lambda p: p.discount_price or p.price or 0)

        return [ProductSchema.model_validate(p) for p in compatible_products[:limit]]
