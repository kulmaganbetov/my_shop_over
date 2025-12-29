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

# Mapping from Russian names to English keys
COMPONENT_NAME_MAPPING = {
    # CPU
    "процессор": "cpu",
    "проц": "cpu",
    "cpu": "cpu",
    "цп": "cpu",
    # GPU
    "видеокарта": "gpu",
    "видеокарту": "gpu",
    "видюха": "gpu",
    "gpu": "gpu",
    "графика": "gpu",
    # Motherboard
    "материнская плата": "motherboard",
    "материнка": "motherboard",
    "мать": "motherboard",
    "motherboard": "motherboard",
    "мп": "motherboard",
    # RAM
    "оперативная память": "ram",
    "оперативка": "ram",
    "память": "ram",
    "ram": "ram",
    "озу": "ram",
    # Storage
    "накопитель": "storage",
    "ssd": "storage",
    "диск": "storage",
    "storage": "storage",
    "хранилище": "storage",
    # PSU
    "блок питания": "psu",
    "бп": "psu",
    "psu": "psu",
    "питание": "psu",
    # Case
    "корпус": "case",
    "кейс": "case",
    "case": "case",
    # Cooler
    "кулер": "cooler",
    "охлаждение": "cooler",
    "cooler": "cooler",
}


def normalize_component_type(component_type: str) -> str:
    """Convert Russian component name to English key."""
    if not component_type:
        return ""
    normalized = component_type.lower().strip()
    return COMPONENT_NAME_MAPPING.get(normalized, normalized)


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
        """Build PC with STRICT compatibility and BUDGET checks.

        Pipeline: CPU → MB → RAM → GPU → PSU → Case → Storage
        STRICT: Total must not exceed budget!
        """
        budget_allocation = self.compatibility_engine.allocate_budget(budget, purpose)

        build = {}
        total_price = 0
        remaining_budget = budget  # Track remaining budget strictly
        warnings = []
        compatibility_notes = []

        # ============================================
        # Step 1: SELECT CPU (use remaining budget tracking)
        # ============================================
        cpu_budget = min(budget_allocation.get("cpu", 0), remaining_budget)
        cpu, cpu_specs, cpu_price = await self._select_cpu(cpu_budget, purpose)

        is_oem_cpu = False
        if not cpu:
            warnings.append("cpu: не найден подходящий процессор")
            cpu_specs = CPUSpecs()
        else:
            build["cpu"] = ProductSchema.model_validate(cpu)
            total_price += cpu_price
            remaining_budget -= cpu_price
            # Check if CPU is OEM (needs cooler)
            is_oem_cpu = "oem" in (cpu.name or "").lower()
            logger.info(f"Selected CPU: {cpu.name}, price: {cpu_price}, OEM: {is_oem_cpu}, remaining: {remaining_budget}")

        # ============================================
        # Step 1.5: SELECT COOLER (if OEM CPU)
        # ============================================
        if is_oem_cpu and cpu_specs:
            # Reserve ~5-10% of budget for cooler
            cooler_budget = min(int(budget * 0.08), remaining_budget, 30000)
            cooler, cooler_price = await self._select_cooler(cooler_budget, cpu_specs)

            if cooler:
                build["cooler"] = ProductSchema.model_validate(cooler)
                total_price += cooler_price
                remaining_budget -= cooler_price
                logger.info(f"Selected Cooler: {cooler.name}, price: {cooler_price}, remaining: {remaining_budget}")
            else:
                warnings.append("cooler: не найден кулер для OEM процессора")

        # ============================================
        # Step 2: SELECT MOTHERBOARD (must match CPU socket)
        # ============================================
        mb_budget = min(budget_allocation.get("motherboard", 0), remaining_budget)
        mb, mb_specs, mb_price = await self._select_motherboard(mb_budget, cpu_specs)

        if not mb:
            warnings.append("motherboard: не найдена совместимая материнская плата")
            mb_specs = MotherboardSpecs()
        else:
            build["motherboard"] = ProductSchema.model_validate(mb)
            total_price += mb_price
            remaining_budget -= mb_price
            logger.info(f"Selected MB: {mb.name}, price: {mb_price}, remaining: {remaining_budget}")

        # ============================================
        # Step 3: SELECT RAM (must match MB RAM type)
        # ============================================
        ram_budget = min(budget_allocation.get("ram", 0), remaining_budget)
        ram, ram_specs, ram_price = await self._select_ram(ram_budget, mb_specs, cpu_specs)

        if not ram:
            warnings.append("ram: не найдена совместимая оперативная память")
        else:
            build["ram"] = ProductSchema.model_validate(ram)
            total_price += ram_price
            remaining_budget -= ram_price
            logger.info(f"Selected RAM: {ram.name}, price: {ram_price}, remaining: {remaining_budget}")

        # ============================================
        # Step 4: SELECT GPU (biggest budget item - use what's left wisely)
        # ============================================
        # Reserve budget for PSU, Case, Storage (roughly 15% of original)
        reserved_for_rest = int(budget * 0.15)
        gpu_budget = min(budget_allocation.get("gpu", 0), remaining_budget - reserved_for_rest)
        gpu, gpu_specs, gpu_price = await self._select_gpu(gpu_budget, purpose)

        if not gpu:
            warnings.append("gpu: не найдена видеокарта")
        else:
            build["gpu"] = ProductSchema.model_validate(gpu)
            total_price += gpu_price
            remaining_budget -= gpu_price
            logger.info(f"Selected GPU: {gpu.name}, price: {gpu_price}, remaining: {remaining_budget}")

        # ============================================
        # Step 5: SELECT PSU (must support total TDP)
        # ============================================
        # Divide remaining budget proportionally between PSU, Case, Storage
        # PSU ~40%, Case ~30%, Storage ~30% of remaining
        psu_budget = min(int(remaining_budget * 0.4), remaining_budget)
        total_tdp = (cpu_specs.tdp if cpu_specs else 65) + (gpu_specs.tdp if gpu_specs else 200) + 100
        psu, psu_specs, psu_price = await self._select_psu(psu_budget, total_tdp)

        if not psu:
            warnings.append("psu: не найден подходящий блок питания")
        else:
            build["psu"] = ProductSchema.model_validate(psu)
            total_price += psu_price
            remaining_budget -= psu_price
            logger.info(f"Selected PSU: {psu.name}, wattage: {psu_specs.wattage}W, remaining: {remaining_budget}")

        # ============================================
        # Step 6: SELECT CASE
        # ============================================
        # Use ~50% of remaining budget for case
        case_budget = min(int(remaining_budget * 0.5), remaining_budget)
        case_product, case_price = await self._select_case(case_budget, gpu_specs)

        if not case_product:
            warnings.append("case: не найден корпус")
        else:
            build["case"] = ProductSchema.model_validate(case_product)
            total_price += case_price
            remaining_budget -= case_price
            logger.info(f"Selected Case: {case_product.name}, remaining: {remaining_budget}")

        # ============================================
        # Step 7: SELECT STORAGE
        # ============================================
        # Use all remaining budget for storage
        storage_budget = remaining_budget
        storage, storage_price = await self._select_storage(storage_budget)

        if not storage:
            warnings.append("storage: не найден накопитель")
        else:
            build["storage"] = ProductSchema.model_validate(storage)
            total_price += storage_price
            remaining_budget -= storage_price
            logger.info(f"Selected Storage: {storage.name}, remaining: {remaining_budget}")

        # ============================================
        # STRICT BUDGET CHECK: Total must not exceed budget!
        # ============================================
        if total_price > budget:
            logger.warning(f"Build exceeds budget! {total_price} > {budget}, downgrading...")
            # Try to find cheaper alternatives for flexible components
            build, total_price = await self._downgrade_to_fit_budget(
                build, total_price, budget, purpose
            )

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
            max_price=budget,  # Strict budget - no overflow
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

        # Workstation GPU keywords to exclude for gaming builds
        workstation_keywords = ["quadro", "firepro", "wx ", "radeon pro", "a2000", "a4000", "a5000", "a6000"]

        for product in products:
            name_lower = (product.name or "").lower()

            # Filter out workstation GPUs for gaming builds
            if purpose in ["gaming", "general", "streaming"]:
                if any(kw in name_lower for kw in workstation_keywords):
                    continue

            # Filter out mining cards and very old cards
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
            # Prefer gaming cards (RTX, RX, Arc)
            if any(g in name_lower for g in ["rtx", "rx ", "arc "]):
                score *= 1.2

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

    async def _select_cooler(
        self,
        budget: int,
        cpu_specs: CPUSpecs,
    ) -> tuple[Optional[any], int]:
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
        socket_keywords = {}
        if cpu_specs.socket:
            socket_str = cpu_specs.socket.value if hasattr(cpu_specs.socket, 'value') else str(cpu_specs.socket)
            if "AM4" in socket_str or "AM5" in socket_str:
                socket_keywords = ["am4", "am5", "amd"]
            elif "LGA1700" in socket_str:
                socket_keywords = ["lga1700", "lga 1700", "intel"]
            elif "LGA1200" in socket_str:
                socket_keywords = ["lga1200", "lga 1200", "intel"]

        for product in products:
            name_lower = (product.name or "").lower()

            # Filter out water cooling if budget is low
            if budget < 15000 and any(w in name_lower for w in ["водян", "liquid", "aio", "water"]):
                continue

            price = product.discount_price or product.price or 0
            if price <= 0 or price > budget:
                continue

            # Prefer coolers that mention the socket
            score = price
            if socket_keywords:
                if any(kw in name_lower for kw in socket_keywords):
                    score *= 1.3

            if score > best_price:
                best_price = price
                best_cooler = product

        if best_cooler:
            return best_cooler, best_cooler.discount_price or best_cooler.price or 0

        return None, 0

    async def _downgrade_to_fit_budget(
        self,
        build: dict,
        current_total: int,
        budget: int,
        purpose: str,
    ) -> tuple[dict, int]:
        """Try to downgrade components to fit within budget.

        Priority for downgrade (least impact on performance):
        1. Storage (can get smaller/slower SSD)
        2. Case (aesthetics only)
        3. PSU (if still meets TDP requirements)
        """
        excess = current_total - budget
        if excess <= 0:
            return build, current_total

        logger.info(f"Need to save {excess}₸ to fit budget")

        # Try downgrading storage first
        storage = build.get("storage")
        if storage:
            storage_price = storage.discount_price or storage.price or 0
            target_storage_price = max(storage_price - excess - 5000, 10000)  # Min 10k

            cheaper_storage, cheaper_price = await self._select_storage(target_storage_price)
            if cheaper_storage and cheaper_price < storage_price:
                saved = storage_price - cheaper_price
                build["storage"] = ProductSchema.model_validate(cheaper_storage)
                current_total -= saved
                excess -= saved
                logger.info(f"Downgraded storage, saved {saved}₸")

        if excess <= 0:
            return build, current_total

        # Try downgrading case
        case = build.get("case")
        if case:
            case_price = case.discount_price or case.price or 0
            target_case_price = max(case_price - excess - 5000, 15000)  # Min 15k

            cheaper_case, cheaper_case_price = await self._select_case(target_case_price, None)
            if cheaper_case and cheaper_case_price < case_price:
                saved = case_price - cheaper_case_price
                build["case"] = ProductSchema.model_validate(cheaper_case)
                current_total -= saved
                excess -= saved
                logger.info(f"Downgraded case, saved {saved}₸")

        if excess <= 0:
            return build, current_total

        # Try downgrading PSU (be careful about TDP)
        psu = build.get("psu")
        if psu:
            psu_price = psu.discount_price or psu.price or 0
            target_psu_price = max(psu_price - excess - 3000, 15000)

            # Need to calculate TDP requirement
            cpu = build.get("cpu")
            gpu = build.get("gpu")
            cpu_tdp = 65
            gpu_tdp = 200
            if cpu:
                cpu_specs = self.specs_extractor.extract_cpu_specs(cpu.name or "", {})
                cpu_tdp = cpu_specs.tdp or 65
            if gpu:
                gpu_specs = self.specs_extractor.extract_gpu_specs(gpu.name or "", {})
                gpu_tdp = gpu_specs.tdp if gpu_specs else 200

            total_tdp = cpu_tdp + gpu_tdp + 100

            cheaper_psu, cheaper_psu_specs, cheaper_psu_price = await self._select_psu(
                target_psu_price, total_tdp
            )
            if cheaper_psu and cheaper_psu_price < psu_price:
                saved = psu_price - cheaper_psu_price
                build["psu"] = ProductSchema.model_validate(cheaper_psu)
                current_total -= saved
                logger.info(f"Downgraded PSU, saved {saved}₸")

        return build, current_total

    async def get_component_alternatives(
        self,
        component_type: str,
        current_build: dict,
        budget: Optional[int] = None,
        preference: Optional[str] = None,
        limit: int = 5,
    ) -> tuple[list[ProductSchema], Optional[str]]:
        """Get alternative components.

        Returns:
            tuple: (list of alternatives, optional warning message)
        """
        # Normalize Russian component name to English key
        normalized_type = normalize_component_type(component_type)
        logger.info(f"[ALTERNATIVES] component_type='{component_type}' -> normalized='{normalized_type}'")

        component_config = COMPONENT_TYPES.get(normalized_type)
        if not component_config:
            logger.warning(f"[ALTERNATIVES] Unknown component type: {component_type} (normalized: {normalized_type})")
            return [], None

        # Update component_type to normalized version for the rest of the method
        component_type = normalized_type

        db_component_type, search_keywords = component_config

        # Get current component specs for compatibility check
        current_component = current_build.get(component_type, {})
        current_price = 0
        if current_component:
            current_price = current_component.get("discount_price") or current_component.get("price") or 0

        # Parse preference for manufacturer and price hints
        preference_lower = (preference or "").lower()
        wants_cheaper = any(word in preference_lower for word in ["дешев", "cheap", "бюджет"])
        wants_better = any(word in preference_lower for word in ["дорож", "лучш", "better", "мощн", "больше"])

        # Manufacturer preference (with Russian transliterations)
        wants_intel = any(word in preference_lower for word in [
            "intel", "интел", "core", "кор", "кор ай", "core i"
        ])
        wants_amd = any(word in preference_lower for word in [
            "amd", "амд", "ryzen", "райзен", "ризен"
        ])

        # GPU/Component brand preferences (MSI, ASUS, Gigabyte, etc.)
        preferred_brands = []
        excluded_brands = []

        # Check for brand preferences
        brand_mapping = {
            "msi": "msi", "мси": "msi",
            "asus": "asus", "асус": "asus",
            "gigabyte": "gigabyte", "гигабайт": "gigabyte",
            "palit": "palit", "палит": "palit",
            "evga": "evga",
            "zotac": "zotac", "зотак": "zotac",
            "sapphire": "sapphire", "сапфир": "sapphire",
            "powercolor": "powercolor",
            "xfx": "xfx",
            "pny": "pny",
            "inno3d": "inno3d",
            "kfa2": "kfa2", "galax": "galax",
        }

        # Brands to exclude (budget brands)
        budget_brands = ["afox", "colorful", "maxsun", "huananzhi"]

        for brand_key, brand_val in brand_mapping.items():
            if brand_key in preference_lower:
                preferred_brands.append(brand_val)

        # Check for exclusion patterns: "не устраивает", "не нравится", "без"
        if any(x in preference_lower for x in ["не устраивает", "не нравится", "не хочу", "без "]):
            for brand in budget_brands:
                if brand in preference_lower:
                    excluded_brands.append(brand)
            # If user says "не устраивает AFOX/Colorful" without specific brands, exclude all budget brands
            if not excluded_brands and any(b in preference_lower for b in budget_brands):
                excluded_brands = budget_brands.copy()

        # "с именем", "известный", "топовый" = wants premium brand
        wants_premium_brand = any(x in preference_lower for x in [
            "с именем", "известн", "топов", "премиум", "качественн", "нормальн"
        ])
        if wants_premium_brand and not preferred_brands:
            preferred_brands = ["msi", "asus", "gigabyte", "palit", "evga", "zotac", "sapphire"]
            excluded_brands = budget_brands.copy()

        # Model line preference (i3, i5, i7, i9, Ryzen 3/5/7/9)
        model_filters = []
        # Intel Core models
        if any(x in preference_lower for x in ["i9", "ай 9", "ай9", "core i9", "кор ай 9"]):
            model_filters = ["i9", "core i9"]
        elif any(x in preference_lower for x in ["i7", "ай 7", "ай7", "core i7", "кор ай 7"]):
            model_filters = ["i7", "core i7"]
        elif any(x in preference_lower for x in ["i5", "ай 5", "ай5", "core i5", "кор ай 5"]):
            model_filters = ["i5", "core i5"]
        elif any(x in preference_lower for x in ["i3", "ай 3", "ай3", "core i3", "кор ай 3"]):
            model_filters = ["i3", "core i3"]
        # AMD Ryzen models
        elif any(x in preference_lower for x in ["ryzen 9", "райзен 9", "ризен 9"]):
            model_filters = ["ryzen 9"]
        elif any(x in preference_lower for x in ["ryzen 7", "райзен 7", "ризен 7"]):
            model_filters = ["ryzen 7"]
        elif any(x in preference_lower for x in ["ryzen 5", "райзен 5", "ризен 5"]):
            model_filters = ["ryzen 5"]
        elif any(x in preference_lower for x in ["ryzen 3", "райзен 3", "ризен 3"]):
            model_filters = ["ryzen 3"]

        logger.info(f"[ALTERNATIVES] preference='{preference}' wants_intel={wants_intel} wants_amd={wants_amd} model_filters={model_filters} preferred_brands={preferred_brands} excluded_brands={excluded_brands}")

        if current_price > 0:
            if wants_cheaper:
                min_budget = int(current_price * 0.2)
                max_budget = int(current_price * 0.95)
            elif wants_better or model_filters or preferred_brands:
                # Expand range significantly when specific model/brand requested
                min_budget = int(current_price * 0.3)
                max_budget = int(current_price * 5.0)  # Allow much higher for upgrades
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

        # Filter products
        compatible_products = []
        warning_message = None

        # Check if user wants to change CPU platform
        platform_change_requested = False
        if component_type == "cpu" and (wants_intel or wants_amd):
            current_cpu_name = current_component.get("name", "").lower() if current_component else ""
            current_is_intel = "intel" in current_cpu_name or "core i" in current_cpu_name
            current_is_amd = "amd" in current_cpu_name or "ryzen" in current_cpu_name

            if (wants_intel and current_is_amd) or (wants_amd and current_is_intel):
                platform_change_requested = True
                warning_message = (
                    "⚠️ Смена платформы (Intel ↔ AMD) требует также замены "
                    "материнской платы и возможно оперативной памяти. "
                    "Рекомендую собрать новую сборку с нуля командой: "
                    "'Собери ПК на Intel за [бюджет]'"
                )

        for product in products:
            name_lower = (product.name or "").lower()

            # Common filters
            if "сервер" in name_lower or "xeon" in name_lower or "epyc" in name_lower:
                continue
            if "so-dimm" in name_lower or "sodimm" in name_lower or "ноутбук" in name_lower:
                continue
            if "внешний" in name_lower or "external" in name_lower:
                continue

            # Manufacturer filter (Intel/AMD)
            if wants_intel:
                if "amd" in name_lower or "ryzen" in name_lower:
                    continue
            if wants_amd:
                if "intel" in name_lower or "core i" in name_lower:
                    continue

            # Brand filter (MSI, ASUS, Gigabyte, etc.)
            if excluded_brands:
                if any(brand in name_lower for brand in excluded_brands):
                    continue

            if preferred_brands:
                if not any(brand in name_lower for brand in preferred_brands):
                    continue

            # Model line filter (i3, i5, i7, i9, Ryzen 3/5/7/9)
            if model_filters and component_type == "cpu":
                if not any(model in name_lower for model in model_filters):
                    continue

            # Component-specific compatibility (skip if platform change requested)
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
                # If NOT platform change, must match motherboard socket
                if not platform_change_requested:
                    if mb_specs.socket and specs.socket != mb_specs.socket:
                        continue

            compatible_products.append(product)

        # Sort by price
        compatible_products.sort(key=lambda p: p.discount_price or p.price or 0)

        return [ProductSchema.model_validate(p) for p in compatible_products[:limit]], warning_message
