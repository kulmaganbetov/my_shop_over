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
    CPUSpecs, MotherboardSpecs, RAMSpecs, GPUSpecs, PSUSpecs, StorageSpecs, CoolerSpecs,
    SOCKET_RAM_SUPPORT, CHIPSET_TIERS, STOCK_COOLER_TDP,
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
    # CRITICAL: These percentages should add up to ~1.0 and GPU should be ~45% for gaming
    ALLOCATION_PROFILES = {
        "gaming": {
            "cpu": 0.18, "motherboard": 0.10, "ram": 0.10,
            "gpu": 0.45, "psu": 0.06, "case": 0.05, "storage": 0.04, "cooler": 0.02,
        },
        "work": {
            "cpu": 0.30, "motherboard": 0.12, "ram": 0.20,
            "gpu": 0.10, "psu": 0.08, "case": 0.07, "storage": 0.10, "cooler": 0.03,
        },
        "office": {
            "cpu": 0.25, "motherboard": 0.15, "ram": 0.15,
            "gpu": 0.05, "psu": 0.10, "case": 0.12, "storage": 0.15, "cooler": 0.03,
        },
        "streaming": {
            "cpu": 0.22, "motherboard": 0.10, "ram": 0.12,
            "gpu": 0.40, "psu": 0.06, "case": 0.04, "storage": 0.04, "cooler": 0.02,
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

    @staticmethod
    def _select_by_target_price(
        products: List[Product],
        target_price: int,
        min_price: Optional[int] = None,
    ) -> List[Product]:
        """Select products closest to target price (not cheapest!).

        CRITICAL FIX: If budget is 550,000₸, we should find components that
        USE that budget, not the cheapest ones available.

        Args:
            products: List of candidate products
            target_price: The ideal price to target
            min_price: Optional minimum price (defaults to 50% of target)

        Returns:
            Products sorted by closeness to target price
        """
        if not products:
            return []

        min_price = min_price or int(target_price * 0.5)

        # Filter products within acceptable range
        candidates = []
        for p in products:
            price = p.discount_price or p.price or 0
            if price >= min_price:  # No upper limit - allow exceeding target slightly
                candidates.append(p)

        if not candidates:
            # Fallback: return what we have
            return products

        # Sort by distance from target price (closest first)
        # Prefer products slightly above target over those far below
        def price_distance(p: Product) -> float:
            price = p.discount_price or p.price or 0
            distance = abs(price - target_price)
            # Penalize being far below target more than being slightly above
            if price < target_price * 0.7:
                distance *= 1.5  # Heavy penalty for being too cheap
            return distance

        candidates.sort(key=price_distance)
        return candidates

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

        # Step 1: Select CPU for this socket - TARGET the allocated budget
        result.reasoning.append(f"[1/7] Ищем CPU для {socket}, целевой бюджет: {allocation.cpu:,}₸")
        cpu_candidates = await self._find_cpus(socket, target_price=allocation.cpu, tier=tier)

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

            # Step 2: Find compatible motherboard - TARGET the allocated budget
            result.reasoning.append(f"[2/7] Ищем материнскую плату для {socket}, целевой бюджет: {allocation.motherboard:,}₸")
            mb_result = await self._find_motherboard(
                socket=socket,
                target_price=allocation.motherboard,
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

            result.reasoning.append(f"[3/7] Ищем RAM типа {mb_specs.ram_type.value}, целевой бюджет: {allocation.ram:,}₸")
            ram_result = await self._find_ram(
                ram_type=mb_specs.ram_type.value,
                target_price=allocation.ram,
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

            # Step 4: GPU (if purpose requires) - TARGET the allocated budget!
            # CRITICAL: For gaming, GPU should get 45% of budget
            if purpose in ["gaming", "streaming"]:
                result.reasoning.append(f"[4/7] Ищем видеокарту, целевой бюджет: {allocation.gpu:,}₸")
                gpu_result = await self._find_gpu(target_price=allocation.gpu, purpose=purpose)

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
            # CRITICAL: Minimum SSD capacity based on total budget
            # Budget < 300k: 256GB min
            # Budget 300k-500k: 512GB min
            # Budget > 500k: 1TB min
            if budget >= 500000:
                min_storage_gb = 1000  # 1TB
            elif budget >= 300000:
                min_storage_gb = 512
            else:
                min_storage_gb = 256

            result.reasoning.append(f"[6/7] Ищем накопитель (минимум {min_storage_gb}GB)")
            storage_result = await self._find_storage(
                max_price=min(allocation.storage * 2, int(remaining * 0.5)),  # Allow more budget for storage
                min_capacity_gb=min_storage_gb,
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

            # Get CPU TDP for cooler selection
            cpu_tdp = cpu_specs.tdp if cpu_specs and cpu_specs.tdp else 65
            # K/KF chips need more cooling
            if "k" in cpu_product.name.lower() or "kf" in cpu_product.name.lower():
                cpu_tdp = max(cpu_tdp, 125)  # Minimum 125W for K chips
            if "i9" in cpu_product.name.lower():
                cpu_tdp = max(cpu_tdp, 150)  # i9 needs serious cooling
            if "12900" in cpu_product.name or "13900" in cpu_product.name or "14900" in cpu_product.name:
                cpu_tdp = max(cpu_tdp, 241)  # Top-tier chips

            cooler_result = await self._find_cooler(
                socket=socket,
                max_price=min(allocation.cooler * 3, remaining),  # Allow higher budget for adequate cooling
                cpu_tdp=cpu_tdp,
            )
            if cooler_result:
                cooler_product, cooler_specs = cooler_result
                cooler_component = BuildComponent(
                    product_id=cooler_product.id,
                    name=cooler_product.name,
                    price=cooler_product.price or 0,
                    discount_price=cooler_product.discount_price or 0,
                    specs=cooler_specs,
                    component_type="cooler",
                )
                result.cooler = cooler_component
                cooler_price = cooler_component.effective_price()
                remaining -= cooler_price
                result.reasoning.append(f"Cooler: {cooler_product.name} ({cooler_price:,}₸, max TDP: {cooler_specs.max_tdp}W)")
            else:
                # Warn if no cooler found in budget
                result.issues.append(CompatibilityIssue(
                    severity="warning",
                    component_a="cooler",
                    component_b="cpu",
                    message=f"Требуется кулер для CPU. Добавьте кулер командой 'добавь кулер'."
                ))

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
        self, socket: str, target_price: int, tier: Tier
    ) -> List[Tuple[Product, CPUSpecs]]:
        """Find CPU candidates for a given socket, targeting the allocated budget.

        CRITICAL: Uses target_price selection - picks CPUs closest to budget allocation,
        NOT the cheapest ones!
        """
        # Fetch wide range of CPUs (no min_price restriction in DB query)
        products = await self.repo.find_cpus(
            max_price=int(target_price * 1.5),  # Allow some flexibility above target
            limit=30,
        )

        # Filter by socket and extract specs
        valid_products = []
        for p in products:
            specs = self.extractor.extract_cpu_specs(p.name, p.specifications)
            if specs and specs.socket and specs.socket.value == socket:
                valid_products.append((p, specs))

        if not valid_products:
            return []

        # Use target price selection - sort by closeness to target
        sorted_products = self._select_by_target_price(
            [p for p, _ in valid_products],
            target_price=target_price,
            min_price=int(target_price * 0.4),
        )

        # Rebuild candidates list maintaining order
        product_to_specs = {p.id: specs for p, specs in valid_products}
        candidates = [(p, product_to_specs[p.id]) for p in sorted_products if p.id in product_to_specs]

        logger.info(f"[CPU] Target: {target_price:,}₸, found {len(candidates)} candidates for {socket}")
        return candidates[:5]

    async def _find_motherboard(
        self, socket: str, target_price: int, tier: Tier
    ) -> Optional[Tuple[Product, MotherboardSpecs]]:
        """Find a compatible motherboard, targeting the allocated budget."""
        products = await self.repo.find_compatible_motherboards(
            socket=socket,
            max_price=int(target_price * 1.5),  # Allow flexibility above target
            tier=tier.value if tier == Tier.BUDGET else None,
            limit=30,
        )

        # Filter valid motherboards
        valid_products = []
        for p in products:
            specs = self.extractor.extract_motherboard_specs(p.name, p.specifications)
            if specs and specs.is_complete and specs.socket and specs.socket.value == socket:
                valid_products.append((p, specs))

        if not valid_products:
            return None

        # Use target price selection
        sorted_products = self._select_by_target_price(
            [p for p, _ in valid_products],
            target_price=target_price,
            min_price=int(target_price * 0.4),
        )

        if not sorted_products:
            return None

        # Return the best match (closest to target price)
        best = sorted_products[0]
        for p, specs in valid_products:
            if p.id == best.id:
                logger.info(f"[MB] Target: {target_price:,}₸, selected: {p.name} @ {(p.discount_price or p.price):,}₸")
                return (p, specs)

        return None

    async def _find_ram(
        self, ram_type: str, target_price: int, min_size_gb: int = 16
    ) -> Optional[Tuple[Product, RAMSpecs]]:
        """Find compatible RAM, targeting the allocated budget."""
        products = await self.repo.find_compatible_ram(
            ram_type=ram_type,
            max_price=int(target_price * 1.5),  # Allow flexibility
            min_size_gb=min_size_gb,
            limit=30,
        )

        # Filter valid RAM
        valid_products = []
        for p in products:
            specs = self.extractor.extract_ram_specs(p.name, p.specifications)
            if not specs or not specs.is_complete:
                continue
            if specs.ram_type and specs.ram_type.value != ram_type:
                continue
            valid_products.append((p, specs))

        if not valid_products:
            return None

        # Sort by target price first
        sorted_products = self._select_by_target_price(
            [p for p, _ in valid_products],
            target_price=target_price,
            min_price=int(target_price * 0.3),
        )

        if not sorted_products:
            return None

        # Find best RAM from target-price-sorted list with quality scoring
        best = None
        best_score = -1

        for p in sorted_products[:10]:  # Check top 10 closest to target
            for prod, specs in valid_products:
                if prod.id != p.id:
                    continue

                price = p.discount_price or p.price or 0
                # Base score from closeness to target (inverted distance)
                distance = abs(price - target_price)
                score = 1000000 - distance  # Higher is better

                # Bonus for good specs
                if specs.size_gb:
                    if specs.size_gb >= 32:
                        score += 50000
                    elif specs.size_gb >= 16:
                        score += 30000

                if specs.frequency:
                    if ram_type == "DDR5" and specs.frequency >= 6000:
                        score += 20000
                    elif ram_type == "DDR4" and specs.frequency >= 3600:
                        score += 20000

                if score > best_score:
                    best_score = score
                    best = (prod, specs)

        if best:
            logger.info(f"[RAM] Target: {target_price:,}₸, selected: {best[0].name} @ {(best[0].discount_price or best[0].price):,}₸")

        return best

    async def _find_gpu(
        self, target_price: int, purpose: str
    ) -> Optional[Tuple[Product, GPUSpecs]]:
        """Find a GPU, targeting the allocated budget.

        CRITICAL: For gaming builds, GPU gets 45% of budget.
        If budget is 550,000₸, GPU target is ~247,500₸.
        We should find a GPU CLOSE to that price, not the cheapest one!
        """
        products = await self.repo.find_gpus(
            max_price=int(target_price * 1.3),  # Allow slightly above target
            limit=30,
        )

        # Filter valid GPUs
        valid_products = []
        for p in products:
            specs = self.extractor.extract_gpu_specs(p.name, p.specifications)
            if specs and specs.is_complete:
                valid_products.append((p, specs))

        if not valid_products:
            return None

        # Sort by target price
        sorted_products = self._select_by_target_price(
            [p for p, _ in valid_products],
            target_price=target_price,
            min_price=int(target_price * 0.5),  # At least 50% of target
        )

        if not sorted_products:
            return None

        # Find best GPU from target-price-sorted list with quality scoring
        best = None
        best_score = -1

        for p in sorted_products[:10]:
            for prod, specs in valid_products:
                if prod.id != p.id:
                    continue

                price = p.discount_price or p.price or 0
                # Base score from closeness to target
                distance = abs(price - target_price)
                score = 1000000 - distance

                # Bonus for good VRAM
                if specs.vram_gb:
                    if specs.vram_gb >= 16:
                        score += 50000
                    elif specs.vram_gb >= 12:
                        score += 40000
                    elif specs.vram_gb >= 8:
                        score += 20000

                if score > best_score:
                    best_score = score
                    best = (prod, specs)

        if best:
            logger.info(f"[GPU] Target: {target_price:,}₸, selected: {best[0].name} @ {(best[0].discount_price or best[0].price):,}₸")

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
        self, max_price: int, min_capacity_gb: int = 256
    ) -> Optional[Tuple[Product, StorageSpecs]]:
        """Find storage with minimum capacity requirement.

        CRITICAL: min_capacity_gb should be based on total budget:
        - Budget < 300k: 256GB minimum
        - Budget 300k-500k: 512GB minimum
        - Budget > 500k: 1TB minimum
        """
        products = await self.repo.find_storage(
            max_price=max_price,
            storage_type="SSD",
            min_capacity_gb=min_capacity_gb,
            limit=20,
        )

        # Filter by minimum capacity and find best option
        valid_products = []
        for p in products:
            specs = self.extractor.extract_storage_specs(p.name, p.specifications)
            if specs and specs.is_complete and specs.capacity_gb >= min_capacity_gb:
                valid_products.append((p, specs))
                logger.debug(f"[STORAGE] Valid: {p.name} ({specs.capacity_gb}GB)")

        if not valid_products:
            # CRITICAL FIX: Don't fallback to tiny SSDs!
            # Try with lower budget constraint but keep capacity requirement
            logger.warning(f"[STORAGE] No SSD >= {min_capacity_gb}GB found in budget, trying wider search")

            # Get more products with higher price limit
            products_extended = await self.repo.find_storage(
                max_price=max_price * 2,  # Allow double the budget for storage
                storage_type="SSD",
                min_capacity_gb=min_capacity_gb,
                limit=30,
            )

            for p in products_extended:
                specs = self.extractor.extract_storage_specs(p.name, p.specifications)
                if specs and specs.is_complete and specs.capacity_gb >= min_capacity_gb:
                    logger.info(f"[STORAGE] Found with extended budget: {p.name} ({specs.capacity_gb}GB)")
                    return (p, specs)

            # Last resort: get AT LEAST 256GB (never go below)
            if min_capacity_gb > 256:
                logger.warning(f"[STORAGE] No {min_capacity_gb}GB found, falling back to 256GB minimum")
                for p in products_extended:
                    specs = self.extractor.extract_storage_specs(p.name, p.specifications)
                    if specs and specs.is_complete and specs.capacity_gb >= 256:
                        return (p, specs)

            return None

        # Sort by capacity (prefer larger within budget)
        valid_products.sort(key=lambda x: x[1].capacity_gb, reverse=True)

        best = valid_products[0]
        logger.info(f"[STORAGE] Selected: {best[0].name} ({best[1].capacity_gb}GB)")
        return best

    async def _find_case(self, max_price: int) -> Optional[Product]:
        """Find a case."""
        products = await self.repo.find_cases(max_price=max_price, limit=5)
        return products[0] if products else None

    async def _find_cooler(
        self, socket: str, max_price: int, cpu_tdp: int = 65
    ) -> Optional[Tuple[Product, CoolerSpecs]]:
        """Find a CPU cooler compatible with the socket.

        NOTE: TDP validation removed per user request - select any cooler by socket/price.
        """
        products = await self.repo.find_coolers(
            socket=socket,
            max_price=max_price,
            limit=20,
        )

        if not products:
            return None

        logger.info(f"[COOLER] Looking for {socket} cooler, budget: {max_price:,}₸")

        # Select cheapest cooler that fits the budget (no TDP filtering)
        valid_coolers = []
        for p in products:
            cooler_specs = self.extractor.extract_cooler_specs(p.name)
            price = p.discount_price or p.price or 0
            valid_coolers.append((p, cooler_specs, price))

        if not valid_coolers:
            logger.warning(f"[COOLER] No cooler found for {socket} in budget {max_price:,}₸")
            return None

        # Sort by price (cheapest first)
        valid_coolers.sort(key=lambda x: x[2])
        best_product, best_specs, _ = valid_coolers[0]

        logger.info(f"[COOLER] Selected: {best_product.name}")
        return (best_product, best_specs)

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
        """Get alternative components compatible with current build.

        Price corridor: 0.5x - 2.0x of current component price (unless user specifies budget).
        """
        alternatives = []

        # Helper to get current component price
        def get_current_price(comp) -> int:
            if comp:
                return comp.effective_price()
            return 0

        # Apply price corridor if no explicit max_price given
        def apply_price_corridor(current_price: int, user_max: Optional[int]) -> tuple[Optional[int], Optional[int]]:
            """Returns (min_price, max_price) based on corridor or user preference."""
            if user_max:
                return (None, user_max)  # User specified budget takes priority
            if current_price > 0:
                return (int(current_price * 0.5), int(current_price * 2.0))
            return (None, None)

        if component_type == "cpu":
            if current_build.motherboard and current_build.motherboard.specs:
                socket = current_build.motherboard.specs.socket
                if socket:
                    current_price = get_current_price(current_build.cpu)
                    min_price, effective_max = apply_price_corridor(current_price, max_price)

                    products = await self.repo.find_cpus(
                        min_price=min_price,
                        max_price=effective_max,
                        limit=20,
                    )
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
                    current_price = get_current_price(current_build.motherboard)
                    min_price, effective_max = apply_price_corridor(current_price, max_price)

                    products = await self.repo.find_compatible_motherboards(
                        socket=socket.value,
                        min_price=min_price,
                        max_price=effective_max,
                        limit=20,
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
                    current_price = get_current_price(current_build.ram)
                    min_price, effective_max = apply_price_corridor(current_price, max_price)

                    products = await self.repo.find_compatible_ram(
                        ram_type=ram_type.value,
                        min_price=min_price,
                        max_price=effective_max,
                        limit=20,
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
            current_price = get_current_price(current_build.gpu)
            min_price, effective_max = apply_price_corridor(current_price, max_price)

            products = await self.repo.find_gpus(
                min_price=min_price,
                max_price=effective_max,
                limit=20,
            )
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
            # CRITICAL FIX: Smart storage alternatives with price range and capacity parsing
            current_price = 0
            current_capacity = 0
            if current_build.storage:
                current_price = current_build.storage.effective_price()
                if current_build.storage.specs:
                    current_capacity = current_build.storage.specs.capacity_gb or 0

            # Parse requested capacity from preference (e.g., "512 GB", "1TB", "1 ТБ")
            requested_capacity = None
            if preference:
                pref_lower = preference.lower()
                # Parse TB
                import re
                tb_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:tb|тб)", pref_lower)
                gb_match = re.search(r"(\d+)\s*(?:gb|гб)", pref_lower)
                if tb_match:
                    requested_capacity = int(float(tb_match.group(1)) * 1000)
                elif gb_match:
                    requested_capacity = int(gb_match.group(1))

            # Price range: 0.3x - 5x of current price (allow upgrades)
            min_price = int(current_price * 0.3) if current_price > 0 else None
            if max_price is None and current_price > 0:
                max_price = int(current_price * 5)

            # Capacity filter: use requested or at least current capacity
            min_capacity = requested_capacity or current_capacity or 256

            logger.info(f"[STORAGE ALT] Current: {current_price:,}₸/{current_capacity}GB, "
                        f"Requested: {requested_capacity}GB, Price range: {min_price}-{max_price}")

            products = await self.repo.find_storage(
                min_price=min_price,
                max_price=max_price,
                storage_type="SSD",
                min_capacity_gb=min_capacity,
                limit=30,  # Get more to filter
            )

            # Filter and score products
            valid_products = []
            for p in products:
                name_lower = (p.name or "").lower()

                # CRITICAL: Skip enterprise/server drives
                if any(x in name_lower for x in ["enterprise", "server", "hpe", "sedc", "datacenter"]):
                    continue

                specs = self.extractor.extract_storage_specs(p.name, p.specifications)
                if not specs or not specs.is_complete:
                    continue

                # Must meet capacity requirement
                if specs.capacity_gb < min_capacity:
                    continue

                # Score by closeness to requested capacity
                price = p.discount_price or p.price or 0
                capacity_diff = abs(specs.capacity_gb - min_capacity) if requested_capacity else 0
                price_diff = abs(price - current_price) if current_price else price

                # Lower score is better
                score = capacity_diff * 10 + price_diff / 1000

                valid_products.append((p, specs, score))

            # Sort by score and take top 5
            valid_products.sort(key=lambda x: x[2])

            for p, specs, _ in valid_products[:5]:
                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                    "capacity_gb": specs.capacity_gb,
                    "type": specs.type,
                })

            logger.info(f"[STORAGE ALT] Found {len(alternatives)} alternatives")

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

        elif component_type == "case":
            current_price = get_current_price(current_build.case) if current_build.case else 0
            min_price, effective_max = apply_price_corridor(current_price, max_price)

            products = await self.repo.find_cases(
                min_price=min_price,
                max_price=effective_max,
                limit=20,
            )
            for p in products:
                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                })

        elif component_type == "cooler":
            current_price = get_current_price(current_build.cooler) if current_build.cooler else 0
            min_price, effective_max = apply_price_corridor(current_price, max_price)

            products = await self.repo.find_coolers(
                min_price=min_price,
                max_price=effective_max,
                limit=20,
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
