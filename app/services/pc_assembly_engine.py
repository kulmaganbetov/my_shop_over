"""PC Assembly Engine - Smart PC builder with backtracking.

This module implements deterministic PC assembly with:
1. Chain of Constraints - step-by-step compatible selection
2. Backtracking - if a component fails, try alternative paths
3. Budget Balancer - smart allocation based on purpose
4. Database-driven compatibility - uses repository JSONB queries
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Product
from app.db.repositories.product import ProductRepository
from app.services.specs_extractor import (
    SpecsExtractor, Socket, RAMType, Tier, FormFactor,
    CPUSpecs, MotherboardSpecs, RAMSpecs, GPUSpecs, PSUSpecs, StorageSpecs,
    SOCKET_RAM_SUPPORT, CHIPSET_TIERS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BuildComponent:
    """A component in a PC build."""
    product_id: int
    name: str
    price: int
    discount_price: int
    specs: Any  # CPUSpecs, MotherboardSpecs, etc.
    component_type: str

    def effective_price(self) -> int:
        return self.discount_price if self.discount_price else self.price


@dataclass
class CompatibilityIssue:
    """Compatibility warning or error."""
    severity: str  # "warning" or "error"
    component_a: str
    component_b: str
    message: str


@dataclass
class BuildResult:
    """Result of a PC build attempt."""
    success: bool
    cpu: Optional[BuildComponent] = None
    motherboard: Optional[BuildComponent] = None
    ram: Optional[BuildComponent] = None
    gpu: Optional[BuildComponent] = None
    psu: Optional[BuildComponent] = None
    storage: Optional[BuildComponent] = None
    case: Optional[BuildComponent] = None
    cooler: Optional[BuildComponent] = None
    peripherals: Dict[str, BuildComponent] = field(default_factory=dict)

    issues: List[CompatibilityIssue] = field(default_factory=list)
    error_message: Optional[str] = None
    reasoning: List[str] = field(default_factory=list)  # Chain-of-thought log

    def total_price(self) -> int:
        """Calculate total build price."""
        total = 0
        for comp in [self.cpu, self.motherboard, self.ram, self.gpu,
                     self.psu, self.storage, self.case, self.cooler]:
            if comp:
                total += comp.effective_price()
        for p in self.peripherals.values():
            if p:
                total += p.effective_price()
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert build to dictionary for API response."""
        build_dict = {}
        for comp_type in ["cpu", "motherboard", "ram", "gpu", "psu", "storage", "case", "cooler"]:
            comp = getattr(self, comp_type)
            if comp:
                build_dict[comp_type] = {
                    "id": comp.product_id,
                    "name": comp.name,
                    "price": comp.price,
                    "discount_price": comp.discount_price,
                    "specs": comp.specs.to_dict() if comp.specs and hasattr(comp.specs, 'to_dict') else {},
                }

        return {
            "success": self.success,
            "build": build_dict,
            "peripherals": {
                ptype: {
                    "id": p.product_id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                }
                for ptype, p in self.peripherals.items()
            },
            "total_price": self.total_price(),
            "warnings": [
                {"component": i.component_a, "message": i.message}
                for i in self.issues if i.severity == "warning"
            ],
            "errors": [
                {"component": i.component_a, "message": i.message}
                for i in self.issues if i.severity == "error"
            ],
            "error_message": self.error_message,
            "reasoning": self.reasoning,
        }


@dataclass
class BudgetAllocation:
    """Budget allocation for PC build."""
    cpu: int = 0
    motherboard: int = 0
    ram: int = 0
    gpu: int = 0
    psu: int = 0
    case: int = 0
    storage: int = 0
    cooler: int = 0

    def total(self) -> int:
        return sum([
            self.cpu, self.motherboard, self.ram, self.gpu,
            self.psu, self.case, self.storage, self.cooler
        ])


# ============================================================================
# BUDGET BALANCER
# ============================================================================

class BudgetBalancer:
    """Smart budget allocation based on purpose and total budget."""

    # Budget allocation percentages by purpose
    ALLOCATION_PROFILES = {
        "gaming": {
            "cpu": 0.17, "motherboard": 0.10, "ram": 0.12,
            "gpu": 0.38, "psu": 0.07, "case": 0.06, "storage": 0.08, "cooler": 0.02,
        },
        "work": {
            "cpu": 0.28, "motherboard": 0.12, "ram": 0.18,
            "gpu": 0.12, "psu": 0.08, "case": 0.07, "storage": 0.12, "cooler": 0.03,
        },
        "office": {
            "cpu": 0.22, "motherboard": 0.15, "ram": 0.15,
            "gpu": 0.08, "psu": 0.10, "case": 0.12, "storage": 0.15, "cooler": 0.03,
        },
        "streaming": {
            "cpu": 0.25, "motherboard": 0.10, "ram": 0.14,
            "gpu": 0.33, "psu": 0.06, "case": 0.05, "storage": 0.05, "cooler": 0.02,
        },
    }

    # Budget thresholds for tier selection
    BUDGET_TIERS = {
        (0, 300000): Tier.BUDGET,
        (300000, 500000): Tier.MID,
        (500000, 800000): Tier.HIGH,
        (800000, float('inf')): Tier.ENTHUSIAST,
    }

    def get_tier_for_budget(self, budget: int) -> Tier:
        """Determine appropriate tier for budget."""
        for (low, high), tier in self.BUDGET_TIERS.items():
            if low <= budget < high:
                return tier
        return Tier.MID

    def allocate(self, total_budget: int, purpose: str = "gaming") -> BudgetAllocation:
        """Allocate budget across components."""
        profile = self.ALLOCATION_PROFILES.get(purpose, self.ALLOCATION_PROFILES["gaming"])
        tier = self.get_tier_for_budget(total_budget)

        # Adjust profile for budget tier
        adjusted = profile.copy()
        if tier == Tier.BUDGET:
            # For budget builds, reduce GPU allocation, increase essentials
            adjusted["gpu"] = min(adjusted["gpu"], 0.30)
            adjusted["motherboard"] = max(adjusted["motherboard"], 0.12)
        elif tier == Tier.ENTHUSIAST:
            # For enthusiast, can afford premium GPU
            adjusted["gpu"] = min(adjusted["gpu"] + 0.05, 0.45)

        return BudgetAllocation(
            cpu=int(total_budget * adjusted["cpu"]),
            motherboard=int(total_budget * adjusted["motherboard"]),
            ram=int(total_budget * adjusted["ram"]),
            gpu=int(total_budget * adjusted["gpu"]),
            psu=int(total_budget * adjusted["psu"]),
            case=int(total_budget * adjusted["case"]),
            storage=int(total_budget * adjusted["storage"]),
            cooler=int(total_budget * adjusted["cooler"]),
        )


# ============================================================================
# PC ASSEMBLY ENGINE
# ============================================================================

class PCAssemblyEngine:
    """Smart PC assembly engine with backtracking."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repo = ProductRepository(session)
        self.extractor = SpecsExtractor()
        self.balancer = BudgetBalancer()

    async def build_pc(
        self,
        budget: int,
        purpose: str = "gaming",
        preferences: Optional[Dict[str, str]] = None,
    ) -> BuildResult:
        """Build a PC within budget using chain-of-constraints with backtracking.

        Args:
            budget: Total budget in tenge
            purpose: Build purpose (gaming, work, office, streaming)
            preferences: Optional preferences (brand, platform, etc.)

        Returns:
            BuildResult with components or error message
        """
        result = BuildResult(success=False)
        preferences = preferences or {}

        # Determine tier for this budget
        tier = self.balancer.get_tier_for_budget(budget)
        result.reasoning.append(f"Бюджет {budget:,}₸ → уровень сборки: {tier.value}")

        # Allocate budget
        allocation = self.balancer.allocate(budget, purpose)
        result.reasoning.append(
            f"Распределение: CPU {allocation.cpu:,}₸, MB {allocation.motherboard:,}₸, "
            f"RAM {allocation.ram:,}₸, GPU {allocation.gpu:,}₸"
        )

        # Try to build with preferred platform first, then backtrack
        platforms = self._get_platform_order(preferences, tier)

        for platform_socket in platforms:
            result.reasoning.append(f"Пробуем платформу: {platform_socket}")

            build_attempt = await self._try_build_platform(
                socket=platform_socket,
                budget=budget,
                allocation=allocation,
                purpose=purpose,
                tier=tier,
            )

            if build_attempt.success:
                return build_attempt

            result.reasoning.append(f"Платформа {platform_socket} не подошла: {build_attempt.error_message}")

        # All platforms failed
        result.error_message = "Не удалось собрать ПК: нет совместимых компонентов в наличии"
        result.reasoning.append("Все платформы исчерпаны без успешной сборки")
        return result

    def _get_platform_order(self, preferences: Dict[str, str], tier: Tier) -> List[str]:
        """Get ordered list of platforms to try based on preferences and tier."""
        # Default order based on tier and availability
        if tier in [Tier.BUDGET, Tier.MID]:
            # For budget/mid, prefer Intel LGA1700 then AMD AM4 (usually more stock)
            order = ["LGA1700", "AM4", "AM5", "LGA1200"]
        else:
            # For high/enthusiast, prefer newer platforms
            order = ["LGA1700", "AM5", "AM4", "LGA1851"]

        # Move preferred platform to front
        if "platform" in preferences:
            pref = preferences["platform"].upper()
            if pref in order:
                order.remove(pref)
                order.insert(0, pref)

        if "brand" in preferences:
            brand = preferences["brand"].lower()
            if brand == "amd":
                order = [p for p in order if p.startswith("AM")] + [p for p in order if not p.startswith("AM")]
            elif brand == "intel":
                order = [p for p in order if p.startswith("LGA")] + [p for p in order if not p.startswith("LGA")]

        return order

    async def _try_build_platform(
        self,
        socket: str,
        budget: int,
        allocation: BudgetAllocation,
        purpose: str,
        tier: Tier,
    ) -> BuildResult:
        """Try to build a PC on a specific platform with backtracking."""
        result = BuildResult(success=False)
        remaining = budget

        # Step 1: Select CPU for this socket
        result.reasoning.append(f"[1/7] Ищем CPU для {socket}")
        cpu_candidates = await self._find_cpus(socket, allocation.cpu, tier)

        if not cpu_candidates:
            result.error_message = f"CPU для {socket} не найден в бюджете {allocation.cpu:,}₸"
            return result

        # Try each CPU candidate with backtracking
        for cpu_product, cpu_specs in cpu_candidates:
            cpu_component = BuildComponent(
                product_id=cpu_product.id,
                name=cpu_product.name,
                price=cpu_product.price or 0,
                discount_price=cpu_product.discount_price or 0,
                specs=cpu_specs,
                component_type="cpu",
            )
            result.cpu = cpu_component
            cpu_price = cpu_component.effective_price()
            remaining = budget - cpu_price
            result.reasoning.append(f"CPU: {cpu_product.name} ({cpu_price:,}₸)")

            # Step 2: Find compatible motherboard
            result.reasoning.append(f"[2/7] Ищем материнскую плату для {socket}")
            mb_result = await self._find_motherboard(
                socket=socket,
                max_price=min(allocation.motherboard, int(remaining * 0.2)),
                tier=tier,
            )

            if not mb_result:
                result.reasoning.append(f"Материнская плата для {socket} не найдена, пробуем другой CPU")
                continue

            mb_product, mb_specs = mb_result
            mb_component = BuildComponent(
                product_id=mb_product.id,
                name=mb_product.name,
                price=mb_product.price or 0,
                discount_price=mb_product.discount_price or 0,
                specs=mb_specs,
                component_type="motherboard",
            )
            result.motherboard = mb_component
            mb_price = mb_component.effective_price()
            remaining -= mb_price
            result.reasoning.append(f"MB: {mb_product.name} ({mb_price:,}₸), RAM тип: {mb_specs.ram_type.value if mb_specs.ram_type else 'unknown'}")

            # Step 3: Find compatible RAM
            if not mb_specs.ram_type:
                result.reasoning.append("Тип RAM не определен для материнской платы, пробуем другой CPU")
                continue

            result.reasoning.append(f"[3/7] Ищем RAM типа {mb_specs.ram_type.value}")
            ram_result = await self._find_ram(
                ram_type=mb_specs.ram_type.value,
                max_price=min(int(allocation.ram * 1.5), int(remaining * 0.2)),
                min_size_gb=16 if purpose == "gaming" else 8,
            )

            if not ram_result:
                result.reasoning.append(f"RAM {mb_specs.ram_type.value} не найдена, пробуем другой CPU")
                continue

            ram_product, ram_specs = ram_result
            ram_component = BuildComponent(
                product_id=ram_product.id,
                name=ram_product.name,
                price=ram_product.price or 0,
                discount_price=ram_product.discount_price or 0,
                specs=ram_specs,
                component_type="ram",
            )
            result.ram = ram_component
            ram_price = ram_component.effective_price()
            remaining -= ram_price
            result.reasoning.append(f"RAM: {ram_product.name} ({ram_price:,}₸)")

            # Step 4: GPU (if budget allows and purpose requires)
            if purpose in ["gaming", "streaming"] and remaining > 50000:
                result.reasoning.append("[4/7] Ищем видеокарту")
                gpu_budget = min(allocation.gpu, int(remaining * 0.6))
                gpu_result = await self._find_gpu(max_price=gpu_budget, purpose=purpose)

                if gpu_result:
                    gpu_product, gpu_specs = gpu_result
                    gpu_component = BuildComponent(
                        product_id=gpu_product.id,
                        name=gpu_product.name,
                        price=gpu_product.price or 0,
                        discount_price=gpu_product.discount_price or 0,
                        specs=gpu_specs,
                        component_type="gpu",
                    )
                    result.gpu = gpu_component
                    gpu_price = gpu_component.effective_price()
                    remaining -= gpu_price
                    result.reasoning.append(f"GPU: {gpu_product.name} ({gpu_price:,}₸)")
                else:
                    result.reasoning.append("GPU не найдена в бюджете, пропускаем")

            # Step 5: PSU with enough wattage
            result.reasoning.append("[5/7] Ищем блок питания")
            required_wattage = self.extractor.get_required_psu_wattage(
                cpu_specs,
                result.gpu.specs if result.gpu else None
            )
            psu_result = await self._find_psu(
                min_wattage=required_wattage,
                max_price=min(allocation.psu, int(remaining * 0.3)),
            )

            if psu_result:
                psu_product, psu_specs = psu_result
                psu_component = BuildComponent(
                    product_id=psu_product.id,
                    name=psu_product.name,
                    price=psu_product.price or 0,
                    discount_price=psu_product.discount_price or 0,
                    specs=psu_specs,
                    component_type="psu",
                )
                result.psu = psu_component
                psu_price = psu_component.effective_price()
                remaining -= psu_price
                result.reasoning.append(f"PSU: {psu_product.name} ({psu_price:,}₸, {psu_specs.wattage}W)")
            else:
                result.reasoning.append(f"PSU {required_wattage}W не найден")

            # Step 6: Storage
            result.reasoning.append("[6/7] Ищем накопитель")
            storage_result = await self._find_storage(
                max_price=min(allocation.storage, int(remaining * 0.4)),
            )

            if storage_result:
                storage_product, storage_specs = storage_result
                storage_component = BuildComponent(
                    product_id=storage_product.id,
                    name=storage_product.name,
                    price=storage_product.price or 0,
                    discount_price=storage_product.discount_price or 0,
                    specs=storage_specs,
                    component_type="storage",
                )
                result.storage = storage_component
                storage_price = storage_component.effective_price()
                remaining -= storage_price
                result.reasoning.append(f"Storage: {storage_product.name} ({storage_price:,}₸)")

            # Step 7: Case and Cooler
            result.reasoning.append("[7/7] Ищем корпус и кулер")
            case_result = await self._find_case(max_price=min(allocation.case, int(remaining * 0.5)))
            if case_result:
                case_product = case_result
                case_component = BuildComponent(
                    product_id=case_product.id,
                    name=case_product.name,
                    price=case_product.price or 0,
                    discount_price=case_product.discount_price or 0,
                    specs=None,
                    component_type="case",
                )
                result.case = case_component
                case_price = case_component.effective_price()
                remaining -= case_price
                result.reasoning.append(f"Case: {case_product.name} ({case_price:,}₸)")

            cooler_result = await self._find_cooler(
                socket=socket,
                max_price=min(allocation.cooler, remaining),
            )
            if cooler_result:
                cooler_product = cooler_result
                cooler_component = BuildComponent(
                    product_id=cooler_product.id,
                    name=cooler_product.name,
                    price=cooler_product.price or 0,
                    discount_price=cooler_product.discount_price or 0,
                    specs=None,
                    component_type="cooler",
                )
                result.cooler = cooler_component
                cooler_price = cooler_component.effective_price()
                remaining -= cooler_price
                result.reasoning.append(f"Cooler: {cooler_product.name} ({cooler_price:,}₸)")

            # Build succeeded
            result.success = True
            result.reasoning.append(f"Сборка завершена! Итого: {result.total_price():,}₸, осталось: {remaining:,}₸")
            return result

        # All CPU candidates failed
        result.error_message = f"Не удалось собрать ПК на платформе {socket}"
        return result

    # =========================================================================
    # COMPONENT SELECTION METHODS
    # =========================================================================

    async def _find_cpus(
        self, socket: str, max_price: int, tier: Tier
    ) -> List[Tuple[Product, CPUSpecs]]:
        """Find CPU candidates for a given socket."""
        products = await self.repo.find_cpus(
            max_price=max_price,
            min_price=int(max_price * 0.4),
            limit=15,
        )

        candidates = []
        for p in products:
            specs = self.extractor.extract_cpu_specs(p.name, p.specifications)
            if specs and specs.socket and specs.socket.value == socket:
                candidates.append((p, specs))

        # Sort by price descending (best value for budget)
        candidates.sort(key=lambda x: x[0].discount_price or x[0].price or 0, reverse=True)
        return candidates[:5]

    async def _find_motherboard(
        self, socket: str, max_price: int, tier: Tier
    ) -> Optional[Tuple[Product, MotherboardSpecs]]:
        """Find a compatible motherboard."""
        products = await self.repo.find_compatible_motherboards(
            socket=socket,
            max_price=max_price,
            tier=tier.value if tier == Tier.BUDGET else None,
            limit=20,
        )

        for p in products:
            specs = self.extractor.extract_motherboard_specs(p.name, p.specifications)
            if specs and specs.is_complete:
                if specs.socket and specs.socket.value == socket:
                    return (p, specs)

        return None

    async def _find_ram(
        self, ram_type: str, max_price: int, min_size_gb: int = 16
    ) -> Optional[Tuple[Product, RAMSpecs]]:
        """Find compatible RAM."""
        products = await self.repo.find_compatible_ram(
            ram_type=ram_type,
            max_price=max_price,
            min_size_gb=min_size_gb,
            limit=20,
        )

        best = None
        best_score = -1

        for p in products:
            specs = self.extractor.extract_ram_specs(p.name, p.specifications)
            if not specs or not specs.is_complete:
                continue

            if specs.ram_type and specs.ram_type.value != ram_type:
                continue

            price = p.discount_price or p.price or 0
            if price > max_price:
                continue

            score = price
            if specs.size_gb:
                if specs.size_gb >= 32:
                    score *= 2.0
                elif specs.size_gb >= 16:
                    score *= 1.5
                elif specs.size_gb <= 8:
                    score *= 0.3

            if specs.frequency:
                if ram_type == "DDR5" and specs.frequency >= 6000:
                    score *= 1.2
                elif ram_type == "DDR4" and specs.frequency >= 3600:
                    score *= 1.2

            if score > best_score:
                best_score = score
                best = (p, specs)

        return best

    async def _find_gpu(
        self, max_price: int, purpose: str
    ) -> Optional[Tuple[Product, GPUSpecs]]:
        """Find a GPU."""
        products = await self.repo.find_gpus(
            max_price=max_price,
            min_price=int(max_price * 0.3),
            limit=15,
        )

        best = None
        best_score = -1

        for p in products:
            specs = self.extractor.extract_gpu_specs(p.name, p.specifications)
            if not specs or not specs.is_complete:
                continue

            price = p.discount_price or p.price or 0
            if price > max_price:
                continue

            score = price
            if specs.vram_gb and specs.vram_gb >= 8:
                score *= 1.2
            if specs.vram_gb and specs.vram_gb >= 12:
                score *= 1.3

            if score > best_score:
                best_score = score
                best = (p, specs)

        return best

    async def _find_psu(
        self, min_wattage: int, max_price: int
    ) -> Optional[Tuple[Product, PSUSpecs]]:
        """Find a PSU with sufficient wattage."""
        products = await self.repo.find_psus(
            min_wattage=min_wattage,
            max_price=max_price,
            limit=10,
        )

        for p in products:
            specs = self.extractor.extract_psu_specs(p.name, p.specifications)
            if specs and specs.is_complete and specs.wattage >= min_wattage:
                return (p, specs)

        if min_wattage > 500:
            products = await self.repo.find_psus(
                min_wattage=500,
                max_price=max_price,
                limit=10,
            )
            for p in products:
                specs = self.extractor.extract_psu_specs(p.name, p.specifications)
                if specs and specs.is_complete:
                    return (p, specs)

        return None

    async def _find_storage(
        self, max_price: int
    ) -> Optional[Tuple[Product, StorageSpecs]]:
        """Find storage."""
        products = await self.repo.find_storage(
            max_price=max_price,
            storage_type="SSD",
            limit=10,
        )

        for p in products:
            specs = self.extractor.extract_storage_specs(p.name, p.specifications)
            if specs and specs.is_complete:
                return (p, specs)

        return None

    async def _find_case(self, max_price: int) -> Optional[Product]:
        """Find a case."""
        products = await self.repo.find_cases(max_price=max_price, limit=5)
        return products[0] if products else None

    async def _find_cooler(self, socket: str, max_price: int) -> Optional[Product]:
        """Find a CPU cooler."""
        products = await self.repo.find_coolers(
            socket=socket,
            max_price=max_price,
            limit=5,
        )
        return products[0] if products else None

    # =========================================================================
    # COMPONENT REPLACEMENT
    # =========================================================================

    async def get_alternatives(
        self,
        current_build: BuildResult,
        component_type: str,
        max_price: Optional[int] = None,
        preference: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get alternative components compatible with current build."""
        alternatives = []

        if component_type == "cpu":
            if current_build.motherboard and current_build.motherboard.specs:
                socket = current_build.motherboard.specs.socket
                if socket:
                    products = await self.repo.find_cpus(max_price=max_price, limit=10)
                    for p in products:
                        specs = self.extractor.extract_cpu_specs(p.name, p.specifications)
                        if specs and specs.socket == socket:
                            alternatives.append({
                                "id": p.id,
                                "name": p.name,
                                "price": p.price,
                                "discount_price": p.discount_price,
                                "socket": socket.value,
                            })

        elif component_type == "motherboard":
            if current_build.cpu and current_build.cpu.specs:
                socket = current_build.cpu.specs.socket
                if socket:
                    products = await self.repo.find_compatible_motherboards(
                        socket=socket.value,
                        max_price=max_price,
                        limit=10,
                    )
                    for p in products:
                        specs = self.extractor.extract_motherboard_specs(p.name, p.specifications)
                        if specs and specs.is_complete:
                            alternatives.append({
                                "id": p.id,
                                "name": p.name,
                                "price": p.price,
                                "discount_price": p.discount_price,
                                "chipset": specs.chipset,
                                "ram_type": specs.ram_type.value if specs.ram_type else None,
                            })

        elif component_type == "ram":
            if current_build.motherboard and current_build.motherboard.specs:
                ram_type = current_build.motherboard.specs.ram_type
                if ram_type:
                    products = await self.repo.find_compatible_ram(
                        ram_type=ram_type.value,
                        max_price=max_price,
                        limit=10,
                    )
                    for p in products:
                        specs = self.extractor.extract_ram_specs(p.name, p.specifications)
                        if specs and specs.is_complete and specs.ram_type == ram_type:
                            alternatives.append({
                                "id": p.id,
                                "name": p.name,
                                "price": p.price,
                                "discount_price": p.discount_price,
                                "size_gb": specs.size_gb,
                                "frequency": specs.frequency,
                            })

        elif component_type == "gpu":
            products = await self.repo.find_gpus(max_price=max_price, limit=10)
            for p in products:
                specs = self.extractor.extract_gpu_specs(p.name, p.specifications)
                if specs and specs.is_complete:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "vram_gb": specs.vram_gb,
                    })

        elif component_type == "storage":
            products = await self.repo.find_storage(max_price=max_price, limit=10)
            for p in products:
                specs = self.extractor.extract_storage_specs(p.name, p.specifications)
                if specs and specs.is_complete:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "capacity_gb": specs.capacity_gb,
                        "type": specs.type,
                    })

        elif component_type == "psu":
            cpu_tdp = current_build.cpu.specs.tdp if current_build.cpu and current_build.cpu.specs else 65
            gpu_tdp = current_build.gpu.specs.tdp if current_build.gpu and current_build.gpu.specs else 150
            min_wattage = int((cpu_tdp + gpu_tdp) * 1.25) + 100

            products = await self.repo.find_psus(
                min_wattage=min_wattage,
                max_price=max_price,
                limit=10,
            )
            for p in products:
                specs = self.extractor.extract_psu_specs(p.name, p.specifications)
                if specs and specs.is_complete:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "wattage": specs.wattage,
                        "efficiency": specs.efficiency,
                    })

        elif component_type in ["case", "cooler"]:
            if component_type == "case":
                products = await self.repo.find_cases(max_price=max_price, limit=10)
            else:
                socket = None
                if current_build.cpu and current_build.cpu.specs:
                    socket = current_build.cpu.specs.socket
                products = await self.repo.find_coolers(
                    socket=socket.value if socket else None,
                    max_price=max_price,
                    limit=10,
                )

            for p in products:
                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                })

        alternatives.sort(key=lambda x: x.get("discount_price") or x.get("price") or 0)

        if preference:
            pref_lower = preference.lower()
            if "дешев" in pref_lower or "cheap" in pref_lower:
                alternatives = alternatives[:5]
            elif "дорог" in pref_lower or "лучш" in pref_lower:
                alternatives = alternatives[-5:]
            else:
                alternatives = [a for a in alternatives if preference.lower() in a["name"].lower()][:5]

        return alternatives[:10]

    # =========================================================================
    # PERIPHERALS
    # =========================================================================

    async def find_peripherals(
        self,
        peripheral_type: str,
        max_price: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find peripherals by type."""
        products = await self.repo.find_peripherals(
            peripheral_type=peripheral_type,
            max_price=max_price,
            limit=limit,
        )

        return [
            {
                "id": p.id,
                "name": p.name,
                "price": p.price,
                "discount_price": p.discount_price,
                "type": peripheral_type,
            }
            for p in products
        ]

    async def add_peripheral_to_build(
        self,
        build: BuildResult,
        peripheral_type: str,
        product_id: int,
    ) -> BuildResult:
        """Add a peripheral to the build."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(Product).where(Product.id == product_id)
        )
        product = result.scalar_one_or_none()

        if product:
            build.peripherals[peripheral_type] = BuildComponent(
                product_id=product.id,
                name=product.name,
                price=product.price or 0,
                discount_price=product.discount_price or 0,
                specs=None,
                component_type=peripheral_type,
            )

        return build
