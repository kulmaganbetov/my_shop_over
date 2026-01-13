#!/usr/bin/env python3
"""Seed script for Smart Presets system.

Creates expert PC build presets for various budget segments.
Each preset contains real SKUs from the products table.

Usage:
    python scripts/seed_presets.py

Budget segments: 300k, 400k, 500k, 600k, 700k, 800k, 900k, 1M, 1.2M, 1.5M, 2M, 4M tenge
For each segment: 7-8 variants (3+ Intel, 3+ AMD, 1-2 Workstation)
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, func, delete, and_, or_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Product, BuildPreset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PRESET TEMPLATES
# ============================================================================

# Component allocation percentages by category
ALLOCATION = {
    "gaming": {"cpu": 0.18, "mb": 0.10, "ram": 0.10, "gpu": 0.45, "psu": 0.06, "case": 0.05, "storage": 0.04, "cooler": 0.02},
    "office": {"cpu": 0.25, "mb": 0.15, "ram": 0.15, "gpu": 0.05, "psu": 0.10, "case": 0.12, "storage": 0.15, "cooler": 0.03},
    "workstation": {"cpu": 0.30, "mb": 0.12, "ram": 0.20, "gpu": 0.10, "psu": 0.08, "case": 0.07, "storage": 0.10, "cooler": 0.03},
}

# Budget segments in tenge
BUDGET_SEGMENTS = [
    300_000, 400_000, 500_000, 600_000, 700_000, 800_000,
    900_000, 1_000_000, 1_200_000, 1_500_000, 2_000_000, 4_000_000
]

# Platform preferences by budget tier
PLATFORM_TIERS = {
    "budget": ["LGA1700", "AM4"],  # 300k-500k - mainstream platforms
    "mid": ["LGA1700", "AM5", "AM4"],  # 500k-800k
    "high": ["LGA1700", "AM5"],  # 800k-1.5M
    "enthusiast": ["LGA1700", "AM5", "LGA1851"],  # 1.5M+
}


class PresetBuilder:
    """Builds preset configurations from real products in database."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.products_cache: Dict[str, List[Product]] = {}

    async def load_products_cache(self):
        """Load all products into cache for faster queries."""
        logger.info("Loading products cache...")

        # Load by category
        categories = [
            ("cpu", ["Процессоры"]),
            ("motherboard", ["Материнские платы"]),
            ("ram", ["Оперативная память"]),
            ("gpu", ["Видеокарты"]),
            ("psu", ["Блоки питания"]),
            ("case", ["Корпуса"]),
            ("storage", ["SSD накопители", "SSD M.2 накопители", "SSD"]),
            ("cooler", ["Кулеры и охлаждение", "Охлаждение"]),
        ]

        for comp_type, cat_names in categories:
            query = select(Product).where(
                and_(
                    Product.is_active == True,
                    Product.stock > 0,
                    func.coalesce(Product.discount_price, Product.price) >= 100,
                    or_(*[Product.category.ilike(f"%{cat}%") for cat in cat_names])
                )
            ).order_by(func.coalesce(Product.discount_price, Product.price))

            result = await self.session.execute(query)
            products = result.scalars().all()
            self.products_cache[comp_type] = list(products)
            logger.info(f"  {comp_type}: {len(products)} products")

    def _find_product(
        self,
        comp_type: str,
        target_price: int,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
    ) -> Optional[Product]:
        """Find a product matching criteria."""
        products = self.products_cache.get(comp_type, [])

        # Apply price filters
        min_p = min_price or int(target_price * 0.5)
        max_p = max_price or int(target_price * 1.5)

        candidates = []
        for p in products:
            price = p.discount_price or p.price or 0
            if price < min_p or price > max_p:
                continue

            name_lower = p.name.lower()

            # Keyword filtering
            if keywords:
                if not any(kw.lower() in name_lower for kw in keywords):
                    continue

            if exclude_keywords:
                if any(kw.lower() in name_lower for kw in exclude_keywords):
                    continue

            candidates.append((p, abs(price - target_price)))

        if not candidates:
            return None

        # Sort by distance to target price
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def _find_intel_cpu(self, target_price: int, tier: str) -> Optional[Product]:
        """Find Intel CPU for budget tier."""
        keywords = ["Intel", "Core"]
        if tier == "budget":
            keywords.extend(["i3", "i5"])
        elif tier == "mid":
            keywords.extend(["i5", "i7"])
        elif tier in ["high", "enthusiast"]:
            keywords.extend(["i7", "i9"])

        return self._find_product("cpu", target_price, keywords=keywords)

    def _find_amd_cpu(self, target_price: int, tier: str) -> Optional[Product]:
        """Find AMD CPU for budget tier."""
        keywords = ["AMD", "Ryzen"]
        if tier == "budget":
            keywords.extend(["Ryzen 5", "Ryzen 3", "5600", "5500"])
        elif tier == "mid":
            keywords.extend(["Ryzen 5", "Ryzen 7", "7600", "5700"])
        elif tier in ["high", "enthusiast"]:
            keywords.extend(["Ryzen 7", "Ryzen 9", "7800", "7900", "7950"])

        return self._find_product("cpu", target_price, keywords=keywords)

    def _find_motherboard(self, target_price: int, platform: str) -> Optional[Product]:
        """Find motherboard for platform."""
        socket_keywords = {
            "LGA1700": ["LGA1700", "1700", "Z690", "B660", "H670", "Z790", "B760"],
            "AM4": ["AM4", "B450", "B550", "X570"],
            "AM5": ["AM5", "B650", "X670", "B650E", "X670E"],
            "LGA1851": ["LGA1851", "Z890", "B860"],
        }
        keywords = socket_keywords.get(platform, [platform])
        return self._find_product("motherboard", target_price, keywords=keywords)

    def _find_ram(self, target_price: int, platform: str) -> Optional[Product]:
        """Find RAM for platform."""
        if platform in ["AM5", "LGA1851"]:
            keywords = ["DDR5"]
        else:
            keywords = ["DDR4"]
        # Exclude single module kits
        return self._find_product("ram", target_price, keywords=keywords, exclude_keywords=["8GB x1", "8 GB x1"])

    def _find_gpu(self, target_price: int, tier: str, variant: str = "nvidia") -> Optional[Product]:
        """Find GPU for tier and variant."""
        if variant == "nvidia":
            if tier == "budget":
                keywords = ["GeForce", "RTX 4060", "RTX 3060", "GTX 1660"]
            elif tier == "mid":
                keywords = ["GeForce", "RTX 4060 Ti", "RTX 4070", "RTX 3070"]
            elif tier == "high":
                keywords = ["GeForce", "RTX 4070", "RTX 4080", "RTX 4070 Ti"]
            else:
                keywords = ["GeForce", "RTX 4080", "RTX 4090"]
        else:  # AMD
            if tier == "budget":
                keywords = ["Radeon", "RX 7600", "RX 6600", "RX 6650"]
            elif tier == "mid":
                keywords = ["Radeon", "RX 7700", "RX 7800", "RX 6750"]
            elif tier == "high":
                keywords = ["Radeon", "RX 7800 XT", "RX 7900"]
            else:
                keywords = ["Radeon", "RX 7900 XT", "RX 7900 XTX"]

        return self._find_product("gpu", target_price, keywords=keywords)

    def _find_psu(self, target_price: int, min_wattage: int = 500) -> Optional[Product]:
        """Find PSU with minimum wattage."""
        keywords = [f"{min_wattage}W", f"{min_wattage + 50}W", f"{min_wattage + 100}W"]
        return self._find_product("psu", target_price, keywords=keywords) or \
               self._find_product("psu", target_price)

    def _find_storage(self, target_price: int, min_capacity: int = 512) -> Optional[Product]:
        """Find SSD with minimum capacity."""
        if min_capacity >= 1000:
            keywords = ["1TB", "1000GB", "2TB"]
        elif min_capacity >= 512:
            keywords = ["512GB", "500GB", "1TB"]
        else:
            keywords = ["256GB", "512GB"]

        return self._find_product("storage", target_price, keywords=keywords, exclude_keywords=["Enterprise", "Server"])

    def _find_case(self, target_price: int) -> Optional[Product]:
        """Find case."""
        return self._find_product("case", target_price, exclude_keywords=["mini", "Mini ITX"])

    def _find_cooler(self, target_price: int, is_high_tdp: bool = False) -> Optional[Product]:
        """Find CPU cooler."""
        if is_high_tdp:
            keywords = ["Tower", "башенн", "AIO", "жидкостн"]
        else:
            keywords = None

        return self._find_product(
            "cooler", target_price,
            keywords=keywords,
            exclude_keywords=["для корпуса", "корпусн", "RGB Fan", "Case Fan"]
        )

    def _get_tier(self, budget: int) -> str:
        """Get tier name for budget."""
        if budget <= 500_000:
            return "budget"
        elif budget <= 800_000:
            return "mid"
        elif budget <= 1_500_000:
            return "high"
        else:
            return "enthusiast"

    async def build_preset(
        self,
        budget: int,
        category_tag: str,
        purpose: str = "gaming",
        variant_num: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Build a single preset configuration."""
        tier = self._get_tier(budget)
        alloc = ALLOCATION.get(purpose, ALLOCATION["gaming"])

        # Determine platform
        is_intel = "Intel" in category_tag
        is_amd = "AMD" in category_tag

        # Find CPU
        cpu_budget = int(budget * alloc["cpu"])
        if is_intel:
            cpu = self._find_intel_cpu(cpu_budget, tier)
            platform = "LGA1700"
        elif is_amd:
            cpu = self._find_amd_cpu(cpu_budget, tier)
            # Determine platform from CPU
            if cpu and any(x in cpu.name for x in ["7600", "7700", "7800", "7900", "7950", "9"]):
                platform = "AM5"
            else:
                platform = "AM4"
        else:
            # Workstation - prefer high-core count
            cpu = self._find_amd_cpu(int(cpu_budget * 1.2), tier) or self._find_intel_cpu(cpu_budget, tier)
            platform = "AM5" if cpu and "Ryzen" in cpu.name else "LGA1700"

        if not cpu:
            logger.warning(f"No CPU found for {category_tag} @ {budget:,}")
            return None

        # Find motherboard
        mb_budget = int(budget * alloc["mb"])
        motherboard = self._find_motherboard(mb_budget, platform)
        if not motherboard:
            logger.warning(f"No motherboard for {platform}")
            return None

        # Find RAM
        ram_budget = int(budget * alloc["ram"])
        ram = self._find_ram(ram_budget, platform)
        if not ram:
            logger.warning(f"No RAM for {platform}")
            return None

        # Find GPU (skip for office builds)
        gpu = None
        if purpose != "office":
            gpu_budget = int(budget * alloc["gpu"])
            gpu_variant = "nvidia" if variant_num % 2 == 1 else "amd"
            if "Workstation" in category_tag:
                gpu_variant = "nvidia"  # Workstations prefer Nvidia
            gpu = self._find_gpu(gpu_budget, tier, gpu_variant)

        # Find PSU
        psu_budget = int(budget * alloc["psu"])
        min_wattage = 650 if gpu else 450
        if tier in ["high", "enthusiast"]:
            min_wattage = 750
        psu = self._find_psu(psu_budget, min_wattage)

        # Find Storage
        storage_budget = int(budget * alloc["storage"])
        min_capacity = 1000 if budget >= 500_000 else 512 if budget >= 300_000 else 256
        storage = self._find_storage(storage_budget, min_capacity)

        # Find Case
        case_budget = int(budget * alloc["case"])
        case = self._find_case(case_budget)

        # Find Cooler
        cooler_budget = int(budget * alloc["cooler"])
        is_high_tdp = "i9" in cpu.name or "i7" in cpu.name or "Ryzen 9" in cpu.name or "Ryzen 7" in cpu.name
        cooler = self._find_cooler(cooler_budget * 2, is_high_tdp)  # Allow more for cooler

        # Build components dict with SKUs
        components = {}
        if cpu:
            components["cpu"] = cpu.sku
        if motherboard:
            components["motherboard"] = motherboard.sku
        if ram:
            components["ram"] = ram.sku
        if gpu:
            components["gpu"] = gpu.sku
        if psu:
            components["psu"] = psu.sku
        if storage:
            components["storage"] = storage.sku
        if case:
            components["case"] = case.sku
        if cooler:
            components["cooler"] = cooler.sku

        # Calculate actual total
        total = sum([
            (cpu.discount_price or cpu.price or 0) if cpu else 0,
            (motherboard.discount_price or motherboard.price or 0) if motherboard else 0,
            (ram.discount_price or ram.price or 0) if ram else 0,
            (gpu.discount_price or gpu.price or 0) if gpu else 0,
            (psu.discount_price or psu.price or 0) if psu else 0,
            (storage.discount_price or storage.price or 0) if storage else 0,
            (case.discount_price or case.price or 0) if case else 0,
            (cooler.discount_price or cooler.price or 0) if cooler else 0,
        ])

        # Generate name and description
        cpu_name = cpu.name.split()[2:4] if cpu else ["Unknown"]
        gpu_name = gpu.name.split()[2:5] if gpu else ["Integrated"]

        name = f"{category_tag} Build {budget // 1000}K v{variant_num}"

        # Description based on purpose and tier
        descriptions = {
            ("gaming", "budget"): f"Бюджетная игровая сборка на базе {' '.join(cpu_name)}. Отлично для 1080p гейминга.",
            ("gaming", "mid"): f"Сбалансированная игровая сборка. {' '.join(gpu_name)} тянет AAA-игры в 1080p/1440p.",
            ("gaming", "high"): f"Мощная игровая система. {' '.join(gpu_name)} для 1440p и 4K гейминга.",
            ("gaming", "enthusiast"): f"Топовая игровая станция. Максимальная производительность в 4K.",
            ("office", "budget"): f"Компактная офисная сборка. Идеально для работы с документами.",
            ("office", "mid"): f"Производительная офисная система для многозадачности.",
            ("workstation", "mid"): f"Рабочая станция для 3D/видео. Много RAM и быстрый CPU.",
            ("workstation", "high"): f"Профессиональная рабочая станция для рендеринга и CAD.",
            ("workstation", "enthusiast"): f"Топовая рабочая станция для студий и продакшена.",
        }

        description = descriptions.get((purpose, tier), f"Сборка {category_tag} на {budget:,}₸")

        return {
            "name": name,
            "target_budget": budget,
            "category_tag": category_tag,
            "purpose": purpose,
            "components": components,
            "description": description,
            "actual_total": total,
        }


async def seed_presets():
    """Main function to seed presets."""
    logger.info("=" * 60)
    logger.info("Smart Presets Seeder")
    logger.info("=" * 60)

    # Create database connection
    engine = create_async_engine(settings.database_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        builder = PresetBuilder(session)
        await builder.load_products_cache()

        # Check if we have enough products
        total_products = sum(len(v) for v in builder.products_cache.values())
        if total_products < 50:
            logger.error(f"Not enough products in database ({total_products}). Run FTP sync first.")
            return

        # Clear existing presets
        logger.info("Clearing existing presets...")
        await session.execute(delete(BuildPreset))
        await session.commit()

        presets_created = 0

        for budget in BUDGET_SEGMENTS:
            tier = builder._get_tier(budget)
            logger.info(f"\n{'='*40}")
            logger.info(f"Budget: {budget:,}₸ (Tier: {tier})")
            logger.info(f"{'='*40}")

            # Gaming presets
            variants = [
                ("Intel Gaming", "gaming", 1),
                ("Intel Gaming", "gaming", 2),
                ("Intel Gaming", "gaming", 3),
                ("AMD Gaming", "gaming", 1),
                ("AMD Gaming", "gaming", 2),
                ("AMD Gaming", "gaming", 3),
            ]

            # Add workstation for mid+ budgets
            if budget >= 500_000:
                variants.extend([
                    ("Intel Workstation", "workstation", 1),
                    ("AMD Workstation", "workstation", 1),
                ])

            # Add office for budget tier
            if budget <= 400_000:
                variants.extend([
                    ("Intel Office", "office", 1),
                    ("AMD Office", "office", 1),
                ])

            for category_tag, purpose, variant_num in variants:
                preset_data = await builder.build_preset(
                    budget=budget,
                    category_tag=category_tag,
                    purpose=purpose,
                    variant_num=variant_num,
                )

                if preset_data and len(preset_data["components"]) >= 5:  # At least 5 components
                    preset = BuildPreset(
                        name=preset_data["name"],
                        target_budget=preset_data["target_budget"],
                        category_tag=category_tag,
                        purpose=purpose,
                        components=preset_data["components"],
                        description=preset_data["description"],
                        priority=variant_num,
                        is_active=True,
                    )
                    session.add(preset)
                    presets_created += 1
                    logger.info(f"  + {preset.name}: {preset_data['actual_total']:,}₸ ({len(preset_data['components'])} components)")
                else:
                    logger.warning(f"  - Failed: {category_tag} {purpose} v{variant_num}")

        await session.commit()

        logger.info(f"\n{'='*60}")
        logger.info(f"Created {presets_created} presets successfully!")
        logger.info(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(seed_presets())
