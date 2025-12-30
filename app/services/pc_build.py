"""PC Build service with TIERED compatibility.

TIERED APPROACH:
- Tier 1: Perfect match (all compatible)
- Tier 2: Good enough (minor warnings)
- Tier 3: Will work (with trade-offs)

Budget is FLEXIBLE: ±20% if it means completing a build.
Compatibility issues are WARNINGS, not hard failures.
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import PresetRepository, ProductRepository
from app.llm.service import LLMService
from app.schemas.chat import PCBuildResult
from app.schemas.common import ProductSchema
from app.schemas.llm import PCBuildParams, PCPurpose
from app.services.compatibility import BUDGET_ALLOCATION, CompatibilityEngine
from app.services.specs_extractor import (
    SpecsExtractor,
    Socket,
    RAMType,
    CPUSpecs,
    MotherboardSpecs,
    RAMSpecs,
)

logger = logging.getLogger(__name__)


# Default budgets by purpose
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

# Component name mapping (Russian -> English)
COMPONENT_NAME_MAPPING = {
    "процессор": "cpu", "проц": "cpu", "cpu": "cpu", "цп": "cpu",
    "видеокарта": "gpu", "видеокарту": "gpu", "видюха": "gpu", "gpu": "gpu",
    "материнская плата": "motherboard", "материнка": "motherboard", "мать": "motherboard",
    "оперативная память": "ram", "оперативка": "ram", "память": "ram", "ram": "ram", "озу": "ram",
    "накопитель": "storage", "ssd": "storage", "диск": "storage",
    "блок питания": "psu", "бп": "psu", "psu": "psu",
    "корпус": "case", "кейс": "case", "case": "case",
    "кулер": "cooler", "охлаждение": "cooler", "cooler": "cooler",
}


def normalize_component_type(component_type: str) -> str:
    """Convert Russian component name to English key."""
    if not component_type:
        return ""
    normalized = component_type.lower().strip()
    return COMPONENT_NAME_MAPPING.get(normalized, normalized)


@dataclass
class BuildTier:
    """Represents build quality tier."""
    tier: int  # 1 = perfect, 2 = good, 3 = will work
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class PCBuildService:
    """Service for PC builds with TIERED compatibility.

    Uses flexible budget (±20%) and warnings instead of hard failures.
    """

    def __init__(self, session: AsyncSession, llm_service: LLMService):
        self.session = session
        self.llm_service = llm_service
        self.product_repo = ProductRepository(session)
        self.preset_repo = PresetRepository(session)
        self.compatibility_engine = CompatibilityEngine()
        self.specs_extractor = SpecsExtractor()

    async def recommend_build(self, params: PCBuildParams) -> PCBuildResult:
        """Recommend a PC build with TIERED approach.

        Tries to build in order:
        1. Perfect build within budget
        2. Good build with flexible budget (±20%)
        3. Best available with trade-offs explained
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
            logger.info(f"Using default budget for {purpose}: {budget}")

        # Try tiered approach
        result = await self._build_tiered(budget, purpose, params)
        return result

    async def _build_tiered(
        self,
        budget: int,
        purpose: str,
        params: PCBuildParams,
    ) -> PCBuildResult:
        """Build PC with tiered approach.

        Tier 1: Exact budget, perfect compatibility
        Tier 2: Budget ±20%, good compatibility
        Tier 3: Best available with warnings
        """
        # Tier 1: Try exact budget
        result = await self._attempt_build(budget, purpose, strict=True)
        if self._is_complete_build(result):
            logger.info(f"[TIER 1] Perfect build at {result.total_price}")
            return result

        # Tier 2: Try with flexible budget (+20%)
        flex_budget = int(budget * 1.2)
        result = await self._attempt_build(flex_budget, purpose, strict=False)
        if self._is_complete_build(result):
            if result.total_price > budget:
                result.warnings.append(
                    f"Сборка превышает бюджет на {result.total_price - budget:,.0f} тг. "
                    f"Для точного бюджета можно подобрать компоненты подешевле."
                )
            logger.info(f"[TIER 2] Good build at {result.total_price}")
            return result

        # Tier 3: Best available with warnings
        result = await self._attempt_build(flex_budget, purpose, strict=False, fallback=True)
        if not self._is_complete_build(result):
            # Fill missing with explanations
            result.warnings.append("Некоторые компоненты не найдены в наличии")
        logger.info(f"[TIER 3] Best available at {result.total_price}")
        return result

    def _is_complete_build(self, result: PCBuildResult) -> bool:
        """Check if build has essential components."""
        build = result.build
        essential = ["cpu", "motherboard", "ram"]
        return all(build.get(comp) for comp in essential)

    async def _attempt_build(
        self,
        budget: int,
        purpose: str,
        strict: bool = True,
        fallback: bool = False,
    ) -> PCBuildResult:
        """Attempt to build PC with given constraints."""
        budget_allocation = self.compatibility_engine.allocate_budget(budget, purpose)

        build = {}
        total_price = 0
        remaining = budget
        warnings = []
        notes = []

        # === CPU ===
        cpu_budget = budget_allocation.get("cpu", 0)
        cpu, cpu_specs, cpu_price = await self._select_cpu(
            min(cpu_budget, remaining), purpose, strict
        )

        is_oem = False
        if cpu:
            build["cpu"] = ProductSchema.model_validate(cpu)
            total_price += cpu_price
            remaining -= cpu_price
            is_oem = "oem" in (cpu.name or "").lower()
        else:
            warnings.append("Процессор: не найден в наличии")
            cpu_specs = CPUSpecs()

        # === COOLER (if OEM) ===
        if is_oem and cpu_specs:
            cooler_budget = min(int(budget * 0.08), remaining, 30000)
            cooler, cooler_price = await self._select_cooler(cooler_budget, cpu_specs)
            if cooler:
                build["cooler"] = ProductSchema.model_validate(cooler)
                total_price += cooler_price
                remaining -= cooler_price
            else:
                warnings.append("Кулер: требуется для OEM процессора")

        # === MOTHERBOARD ===
        mb_budget = min(budget_allocation.get("motherboard", 0), remaining)
        mb, mb_specs, mb_price = await self._select_motherboard(
            mb_budget, cpu_specs, strict
        )

        if mb:
            build["motherboard"] = ProductSchema.model_validate(mb)
            total_price += mb_price
            remaining -= mb_price
        else:
            warnings.append("Материнская плата: не найдена совместимая")
            mb_specs = MotherboardSpecs()

        # === RAM ===
        ram_budget = min(budget_allocation.get("ram", 0), remaining)
        ram, ram_specs, ram_price = await self._select_ram(
            ram_budget, mb_specs, cpu_specs, strict
        )

        if ram:
            build["ram"] = ProductSchema.model_validate(ram)
            total_price += ram_price
            remaining -= ram_price
        else:
            warnings.append("Оперативная память: не найдена совместимая")

        # === GPU (biggest item) ===
        # Reserve 15% for PSU, case, storage
        reserved = int(budget * 0.15)
        gpu_budget = min(budget_allocation.get("gpu", 0), remaining - reserved)
        gpu, gpu_specs, gpu_price = await self._select_gpu(gpu_budget, purpose)

        if gpu:
            build["gpu"] = ProductSchema.model_validate(gpu)
            total_price += gpu_price
            remaining -= gpu_price
        elif purpose == "gaming":
            warnings.append("Видеокарта: не найдена в бюджете (увеличьте бюджет для игровой сборки)")

        # === PSU ===
        psu_budget = min(int(remaining * 0.4), remaining)
        total_tdp = (cpu_specs.tdp if cpu_specs else 65) + (gpu_specs.tdp if gpu_specs else 200) + 100
        psu, psu_specs, psu_price = await self._select_psu(psu_budget, total_tdp)

        if psu:
            build["psu"] = ProductSchema.model_validate(psu)
            total_price += psu_price
            remaining -= psu_price
        else:
            warnings.append("Блок питания: требуется минимум 500W")

        # === CASE ===
        case_budget = min(int(remaining * 0.5), remaining)
        case_product, case_price = await self._select_case(case_budget, gpu_specs)

        if case_product:
            build["case"] = ProductSchema.model_validate(case_product)
            total_price += case_price
            remaining -= case_price

        # === STORAGE ===
        storage, storage_price = await self._select_storage(remaining)
        if storage:
            build["storage"] = ProductSchema.model_validate(storage)
            total_price += storage_price

        # Validate compatibility
        compat_notes = self._validate_build(build, cpu_specs, mb_specs, ram_specs)

        return PCBuildResult(
            build=build,
            total_price=total_price,
            compatibility="ok" if not compat_notes else "warning",
            compatibility_notes=compat_notes,
            warnings=warnings,
        )

    def _validate_build(
        self,
        build: dict,
        cpu_specs: CPUSpecs,
        mb_specs: MotherboardSpecs,
        ram_specs: Optional[RAMSpecs],
    ) -> List[str]:
        """Validate build compatibility - returns warnings, not errors."""
        notes = []

        # CPU-MB socket check
        if cpu_specs.socket and mb_specs.socket:
            if cpu_specs.socket != mb_specs.socket:
                notes.append(
                    f"Внимание: CPU ({cpu_specs.socket}) может быть несовместим с "
                    f"материнской платой ({mb_specs.socket})"
                )

        # MB-RAM type check
        if mb_specs.ram_type and ram_specs and ram_specs.ram_type:
            if mb_specs.ram_type != ram_specs.ram_type:
                notes.append(
                    f"Внимание: RAM ({ram_specs.ram_type}) может быть несовместима с "
                    f"материнской платой ({mb_specs.ram_type})"
                )

        return notes

    async def _select_cpu(
        self,
        budget: int,
        purpose: str,
        strict: bool = True,
    ) -> Tuple[Optional[any], CPUSpecs, int]:
        """Select CPU within budget."""
        component_type, keywords = COMPONENT_TYPES["cpu"]

        # First try: within budget
        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            min_price=int(budget * 0.5) if strict else int(budget * 0.3),
            max_price=budget,
            in_stock_only=True,
            limit=30,
            search_keywords=keywords,
        )

        # Fallback: any within budget
        if not products:
            products = await self.product_repo.get_by_component_type(
                component_type=component_type,
                max_price=budget,
                in_stock_only=True,
                limit=30,
                search_keywords=keywords,
            )

        best_cpu = None
        best_specs = CPUSpecs()
        best_score = -1

        for product in products:
            name_lower = (product.name or "").lower()

            # Skip server CPUs
            if "xeon" in name_lower or "epyc" in name_lower:
                continue

            specs = self.specs_extractor.extract_cpu_specs(
                product.name or "",
                product.specifications or {},
            )

            # Skip if no socket detected
            if not specs.socket:
                continue

            price = product.discount_price or product.price or 0
            if price > budget or price <= 0:
                continue

            # Score based on price (higher = better within budget)
            score = price
            # Bonus for modern sockets
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
        strict: bool = True,
    ) -> Tuple[Optional[any], MotherboardSpecs, int]:
        """Select motherboard matching CPU socket."""
        component_type, keywords = COMPONENT_TYPES["motherboard"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=50,
            search_keywords=keywords,
        )

        best_mb = None
        best_specs = MotherboardSpecs()
        best_score = -1

        for product in products:
            name_lower = (product.name or "").lower()

            # Skip server boards
            if "сервер" in name_lower or "server" in name_lower:
                continue

            specs = self.specs_extractor.extract_motherboard_specs(
                product.name or "",
                product.specifications or {},
            )

            # Socket compatibility (strict or fallback)
            if strict and cpu_specs.socket and specs.socket != cpu_specs.socket:
                continue

            # Must have RAM type
            if not specs.ram_type:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0:
                continue

            score = min(price, budget)
            if price <= budget:
                score *= 1.2  # Bonus for being within budget

            # Bonus for socket match
            if cpu_specs.socket and specs.socket == cpu_specs.socket:
                score *= 1.5

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
        strict: bool = True,
    ) -> Tuple[Optional[any], Optional[RAMSpecs], int]:
        """Select RAM matching motherboard type."""
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

            # Skip notebook RAM
            if "so-dimm" in name_lower or "sodimm" in name_lower or "ноутбук" in name_lower:
                continue

            specs = self.specs_extractor.extract_ram_specs(
                product.name or "",
                product.specifications or {},
            )

            # RAM type compatibility
            if strict and mb_specs.ram_type and specs.ram_type != mb_specs.ram_type:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            # Score: prefer more RAM, higher frequency
            score = price
            if specs.size_gb:
                score *= (specs.size_gb / 8)
            if specs.frequency and specs.frequency > 3200:
                score *= 1.1

            # Bonus for RAM type match
            if mb_specs.ram_type and specs.ram_type == mb_specs.ram_type:
                score *= 1.5

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
    ) -> Tuple[Optional[any], Optional[any], int]:
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

        # Skip workstation GPUs for gaming
        workstation_kw = ["quadro", "firepro", "wx ", "radeon pro", "a2000", "a4000"]

        for product in products:
            name_lower = (product.name or "").lower()

            if purpose == "gaming":
                if any(kw in name_lower for kw in workstation_kw):
                    continue

            # Skip mining/old cards
            if "mining" in name_lower or "gt 710" in name_lower or "gt 730" in name_lower:
                continue

            specs = self.specs_extractor.extract_gpu_specs(
                product.name or "",
                product.specifications or {},
            )

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            score = price
            # Prefer gaming cards
            if any(g in name_lower for g in ["rtx", "rx ", "arc "]):
                score *= 1.2

            if score > best_score:
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
    ) -> Tuple[Optional[any], any, int]:
        """Select PSU with enough power."""
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

            # Must have enough wattage
            if specs.wattage < required_tdp:
                continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            score = price
            if specs.efficiency:
                score *= 1.1

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
    ) -> Tuple[Optional[any], int]:
        """Select case."""
        component_type, keywords = COMPONENT_TYPES["case"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=20,
            search_keywords=keywords,
        )

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
    ) -> Tuple[Optional[any], int]:
        """Select storage."""
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

            # Skip external drives
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

    async def _select_cooler(
        self,
        budget: int,
        cpu_specs: CPUSpecs,
    ) -> Tuple[Optional[any], int]:
        """Select CPU cooler for OEM processors."""
        component_type, keywords = COMPONENT_TYPES["cooler"]

        products = await self.product_repo.get_by_component_type(
            component_type=component_type,
            max_price=budget,
            in_stock_only=True,
            limit=30,
            search_keywords=keywords,
        )

        best_cooler = None
        best_price = 0

        # Socket compatibility keywords
        socket_kw = []
        if cpu_specs.socket:
            socket_str = cpu_specs.socket.value if hasattr(cpu_specs.socket, 'value') else str(cpu_specs.socket)
            if "AM4" in socket_str or "AM5" in socket_str:
                socket_kw = ["am4", "am5", "amd"]
            elif "LGA1700" in socket_str:
                socket_kw = ["lga1700", "lga 1700", "intel"]
            elif "LGA1200" in socket_str:
                socket_kw = ["lga1200", "lga 1200", "intel"]

        for product in products:
            name_lower = (product.name or "").lower()

            # Skip water cooling on low budget
            if budget < 15000:
                if any(w in name_lower for w in ["водян", "liquid", "aio", "water"]):
                    continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            score = price
            # Bonus for socket match
            if socket_kw and any(kw in name_lower for kw in socket_kw):
                score *= 1.3

            if score > best_price:
                best_price = price
                best_cooler = product

        if best_cooler:
            return best_cooler, best_cooler.discount_price or best_cooler.price or 0

        return None, 0

    async def get_component_alternatives(
        self,
        component_type: str,
        current_build: dict,
        budget: Optional[int] = None,
        preference: Optional[str] = None,
        limit: int = 5,
    ) -> Tuple[List[ProductSchema], Optional[str]]:
        """Get alternative components with preference filtering."""
        # Normalize component type
        normalized_type = normalize_component_type(component_type)
        logger.info(f"[ALTERNATIVES] '{component_type}' -> '{normalized_type}'")

        component_config = COMPONENT_TYPES.get(normalized_type)
        if not component_config:
            logger.warning(f"Unknown component type: {component_type}")
            return [], None

        component_type = normalized_type
        db_component_type, search_keywords = component_config

        # Get current component price
        current = current_build.get(component_type, {})
        current_price = 0
        if current:
            current_price = current.get("discount_price") or current.get("price") or 0

        # Parse preference
        pref_lower = (preference or "").lower()
        wants_cheaper = any(w in pref_lower for w in ["дешев", "cheap", "бюджет"])
        wants_better = any(w in pref_lower for w in ["дорож", "лучш", "better", "мощн"])

        # Brand preferences
        preferred_brands = []
        excluded_brands = []
        budget_brands = ["afox", "colorful", "maxsun", "huananzhi"]

        brand_map = {
            "msi": "msi", "asus": "asus", "gigabyte": "gigabyte",
            "palit": "palit", "evga": "evga", "zotac": "zotac",
            "sapphire": "sapphire", "powercolor": "powercolor",
        }

        for key, val in brand_map.items():
            if key in pref_lower:
                preferred_brands.append(val)

        # Check for exclusions
        if any(x in pref_lower for x in ["не устраивает", "не нравится", "не хочу"]):
            for brand in budget_brands:
                if brand in pref_lower:
                    excluded_brands.append(brand)
            if not excluded_brands:
                excluded_brands = budget_brands.copy()

        # "с именем", "известный" = premium brand
        if any(x in pref_lower for x in ["с именем", "известн", "топов", "премиум"]):
            if not preferred_brands:
                preferred_brands = list(brand_map.values())
            excluded_brands = budget_brands.copy()

        # Calculate price range
        if current_price > 0:
            if wants_cheaper:
                min_budget = int(current_price * 0.2)
                max_budget = int(current_price * 0.95)
            elif wants_better or preferred_brands:
                min_budget = int(current_price * 0.3)
                max_budget = int(current_price * 5.0)
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
            return [], None

        # Get build specs for compatibility
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

        # Filter products
        compatible = []
        warning = None

        for product in products:
            name_lower = (product.name or "").lower()

            # Common filters
            if "сервер" in name_lower or "xeon" in name_lower or "epyc" in name_lower:
                continue
            if "so-dimm" in name_lower or "sodimm" in name_lower:
                continue

            # Brand filters
            if excluded_brands and any(b in name_lower for b in excluded_brands):
                continue
            if preferred_brands and not any(b in name_lower for b in preferred_brands):
                continue

            # Component-specific compatibility
            if component_type == "motherboard":
                specs = self.specs_extractor.extract_motherboard_specs(product.name or "", {})
                if cpu_specs.socket and specs.socket != cpu_specs.socket:
                    continue

            elif component_type == "ram":
                specs = self.specs_extractor.extract_ram_specs(product.name or "", {})
                if mb_specs.ram_type and specs.ram_type != mb_specs.ram_type:
                    continue

            elif component_type == "cpu":
                specs = self.specs_extractor.extract_cpu_specs(product.name or "", {})
                if mb_specs.socket and specs.socket != mb_specs.socket:
                    continue

            compatible.append(product)

        # Sort by price
        compatible.sort(key=lambda p: p.discount_price or p.price or 0)

        return [ProductSchema.model_validate(p) for p in compatible[:limit]], warning
