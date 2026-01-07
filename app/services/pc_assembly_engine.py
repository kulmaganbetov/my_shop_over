"""PC Assembly Engine - Zero-hallucination PC builder.

This module implements deterministic PC assembly with:
1. Chain of Constraints - step-by-step compatible selection
2. Tier System - prevents mismatched component levels
3. Budget Balancer - smart allocation based on purpose
4. Swap & Validation - cascade compatibility checks
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any

from sqlalchemy import and_, or_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Product
from app.schemas.common import ProductSchema

logger = logging.getLogger(__name__)


# ============================================================================
# TIER SYSTEM - Component classification by performance level
# ============================================================================

class Tier(str, Enum):
    """Component performance tiers."""
    BUDGET = "budget"      # Entry level
    MID = "mid"           # Mid-range
    HIGH = "high"         # High-end
    ENTHUSIAST = "enthusiast"  # Top tier


class Socket(str, Enum):
    """CPU Sockets."""
    AM5 = "AM5"
    AM4 = "AM4"
    LGA1700 = "LGA1700"
    LGA1200 = "LGA1200"
    LGA1151 = "LGA1151"


class RAMType(str, Enum):
    """RAM Types."""
    DDR5 = "DDR5"
    DDR4 = "DDR4"


class FormFactor(str, Enum):
    """Motherboard form factors."""
    ATX = "ATX"
    MATX = "mATX"
    ITX = "ITX"


# Chipset tier mapping
CHIPSET_TIERS = {
    # Intel - High
    "Z790": Tier.HIGH, "Z690": Tier.HIGH,
    # Intel - Mid
    "B760": Tier.MID, "B660": Tier.MID, "H770": Tier.MID, "H670": Tier.MID,
    # Intel - Budget
    "H610": Tier.BUDGET, "H510": Tier.BUDGET, "B560": Tier.BUDGET,
    # AMD AM5 - High
    "X670E": Tier.ENTHUSIAST, "X670": Tier.HIGH,
    # AMD AM5 - Mid/Budget
    "B650E": Tier.HIGH, "B650": Tier.MID, "A620": Tier.BUDGET,
    # AMD AM4 - High
    "X570": Tier.HIGH,
    # AMD AM4 - Mid
    "B550": Tier.MID, "X470": Tier.MID,
    # AMD AM4 - Budget
    "B450": Tier.BUDGET, "A520": Tier.BUDGET, "A320": Tier.BUDGET,
}

# Chipset to socket mapping
CHIPSET_SOCKET_MAP = {
    # Intel LGA1700
    "Z790": Socket.LGA1700, "B760": Socket.LGA1700, "H770": Socket.LGA1700,
    "H610": Socket.LGA1700, "Z690": Socket.LGA1700, "B660": Socket.LGA1700,
    "H670": Socket.LGA1700,
    # Intel LGA1200
    "Z590": Socket.LGA1200, "B560": Socket.LGA1200, "H570": Socket.LGA1200,
    "H510": Socket.LGA1200, "Z490": Socket.LGA1200, "B460": Socket.LGA1200,
    # AMD AM5
    "X670E": Socket.AM5, "X670": Socket.AM5, "B650E": Socket.AM5,
    "B650": Socket.AM5, "A620": Socket.AM5,
    # AMD AM4
    "X570": Socket.AM4, "B550": Socket.AM4, "A520": Socket.AM4,
    "X470": Socket.AM4, "B450": Socket.AM4, "A320": Socket.AM4,
}

# Chipset to RAM type mapping
CHIPSET_RAM_MAP = {
    # DDR5 only
    "Z790": RAMType.DDR5, "H770": RAMType.DDR5, "X670E": RAMType.DDR5,
    "X670": RAMType.DDR5, "B650E": RAMType.DDR5, "B650": RAMType.DDR5,
    "A620": RAMType.DDR5,
    # DDR4/DDR5 (depends on board variant)
    "B760": None, "H610": None, "Z690": None, "B660": None, "H670": None,
    # DDR4 only
    "Z590": RAMType.DDR4, "B560": RAMType.DDR4, "H570": RAMType.DDR4,
    "H510": RAMType.DDR4, "Z490": RAMType.DDR4, "B460": RAMType.DDR4,
    "X570": RAMType.DDR4, "B550": RAMType.DDR4, "A520": RAMType.DDR4,
    "X470": RAMType.DDR4, "B450": RAMType.DDR4, "A320": RAMType.DDR4,
}

# CPU tier by model patterns
CPU_TIER_PATTERNS = {
    Tier.ENTHUSIAST: [
        r"i9-1[234]\d{3}", r"ryzen\s*9\s*(7|9)\d{3}",
    ],
    Tier.HIGH: [
        r"i7-1[234]\d{3}", r"ryzen\s*7\s*(5|7|9)\d{3}",
        r"i9-1[01]\d{3}",
    ],
    Tier.MID: [
        r"i5-1[234]\d{3}", r"ryzen\s*5\s*(5|7)\d{3}",
        r"i7-1[01]\d{3}",
    ],
    Tier.BUDGET: [
        r"i3-1[234]\d{3}", r"ryzen\s*[35]\s*(3|4|5)\d{3}",
        r"i5-1[01]\d{3}", r"pentium", r"celeron", r"athlon",
    ],
}

# GPU tier by model patterns
GPU_TIER_PATTERNS = {
    Tier.ENTHUSIAST: [
        r"rtx\s*4090", r"rtx\s*4080", r"rx\s*7900\s*xtx",
    ],
    Tier.HIGH: [
        r"rtx\s*4070\s*ti", r"rtx\s*4070", r"rtx\s*3080", r"rtx\s*3090",
        r"rx\s*7900\s*xt", r"rx\s*7800\s*xt", r"rx\s*6800\s*xt",
    ],
    Tier.MID: [
        r"rtx\s*4060\s*ti", r"rtx\s*4060", r"rtx\s*3070", r"rtx\s*3060\s*ti",
        r"rx\s*7700\s*xt", r"rx\s*7600", r"rx\s*6700\s*xt", r"rx\s*6600\s*xt",
    ],
    Tier.BUDGET: [
        r"rtx\s*3060\b", r"rtx\s*3050", r"gtx\s*1660", r"gtx\s*1650",
        r"rx\s*6600\b", r"rx\s*6500", r"rx\s*6400",
        r"arc\s*a7", r"arc\s*a5", r"arc\s*a3",
    ],
}

# GPU TDP reference (watts)
GPU_TDP_MAP = {
    # NVIDIA RTX 40
    "4090": 450, "4080 super": 320, "4080": 320, "4070 ti super": 285,
    "4070 ti": 285, "4070 super": 220, "4070": 200, "4060 ti": 160, "4060": 115,
    # NVIDIA RTX 30
    "3090 ti": 450, "3090": 350, "3080 ti": 350, "3080": 320,
    "3070 ti": 290, "3070": 220, "3060 ti": 200, "3060": 170, "3050": 130,
    # AMD RX 7000
    "7900 xtx": 355, "7900 xt": 300, "7900 gre": 260,
    "7800 xt": 263, "7700 xt": 245, "7600 xt": 190, "7600": 165,
    # AMD RX 6000
    "6950 xt": 335, "6900 xt": 300, "6800 xt": 300, "6800": 250,
    "6750 xt": 250, "6700 xt": 230, "6650 xt": 180, "6600 xt": 160, "6600": 132,
}


# ============================================================================
# SPECS EXTRACTION - Parse component details from product names
# ============================================================================

@dataclass
class CPUSpecs:
    """Extracted CPU specifications."""
    socket: Optional[Socket] = None
    tier: Tier = Tier.MID
    brand: Optional[str] = None
    model: Optional[str] = None
    tdp: int = 65
    is_oem: bool = False


@dataclass
class MotherboardSpecs:
    """Extracted motherboard specifications."""
    socket: Optional[Socket] = None
    chipset: Optional[str] = None
    ram_type: Optional[RAMType] = None
    tier: Tier = Tier.MID
    form_factor: FormFactor = FormFactor.ATX


@dataclass
class RAMSpecs:
    """Extracted RAM specifications."""
    ram_type: Optional[RAMType] = None
    frequency: int = 3200
    size_gb: int = 16
    modules: int = 1


@dataclass
class GPUSpecs:
    """Extracted GPU specifications."""
    brand: Optional[str] = None
    model: Optional[str] = None
    tier: Tier = Tier.MID
    vram_gb: int = 8
    tdp: int = 200
    length_mm: int = 300


@dataclass
class PSUSpecs:
    """Extracted PSU specifications."""
    wattage: int = 500
    efficiency: Optional[str] = None
    tier: Tier = Tier.MID


@dataclass
class CaseSpecs:
    """Extracted case specifications."""
    form_factor: FormFactor = FormFactor.ATX
    max_gpu_length: int = 350
    max_cooler_height: int = 160


class SpecsParser:
    """Parse component specifications from product names."""

    # CPU socket patterns
    CPU_SOCKET_PATTERNS = {
        Socket.AM5: [r"ryzen\s*(9|7|5)\s*(9|8|7)\d{3}", r"\bam5\b"],
        Socket.AM4: [r"ryzen\s*(9|7|5|3)\s*(5|4|3|2|1)\d{3}", r"\bam4\b"],
        Socket.LGA1700: [r"i[3579]-1[234]\d{3}", r"\blga\s*1700\b", r"-1[234]\d00"],
        Socket.LGA1200: [r"i[3579]-1[01]\d{3}", r"\blga\s*1200\b"],
        Socket.LGA1151: [r"i[3579]-[89]\d{3}", r"\blga\s*1151\b"],
    }

    def parse_cpu(self, name: str, specs: dict = None) -> CPUSpecs:
        """Extract CPU specs from product name."""
        result = CPUSpecs()
        name_lower = name.lower()
        specs = specs or {}

        # Brand detection
        if "amd" in name_lower or "ryzen" in name_lower:
            result.brand = "AMD"
        elif "intel" in name_lower or "core" in name_lower:
            result.brand = "Intel"

        # Socket detection
        for socket, patterns in self.CPU_SOCKET_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    result.socket = socket
                    break
            if result.socket:
                break

        # OEM detection
        result.is_oem = "oem" in name_lower or "tray" in name_lower

        # Tier detection
        for tier, patterns in CPU_TIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    result.tier = tier
                    break

        # TDP estimation
        if result.tier == Tier.ENTHUSIAST:
            result.tdp = 125
        elif result.tier == Tier.HIGH:
            result.tdp = 105
        elif result.tier == Tier.MID:
            result.tdp = 65
        else:
            result.tdp = 65

        # Extract model
        model_match = re.search(r"(i[3579]-\d{4,5}\w*|ryzen\s*[3579]\s*\d{4}\w*)", name_lower)
        if model_match:
            result.model = model_match.group(1)

        return result

    def parse_motherboard(self, name: str, specs: dict = None) -> MotherboardSpecs:
        """Extract motherboard specs from product name."""
        result = MotherboardSpecs()
        name_upper = name.upper()
        name_lower = name.lower()

        # Chipset detection
        for chipset in CHIPSET_SOCKET_MAP.keys():
            if chipset in name_upper:
                result.chipset = chipset
                result.socket = CHIPSET_SOCKET_MAP[chipset]
                result.tier = CHIPSET_TIERS.get(chipset, Tier.MID)
                break

        # RAM type detection
        if result.chipset:
            result.ram_type = CHIPSET_RAM_MAP.get(result.chipset)

        # If chipset supports both DDR4/DDR5, check name
        if result.ram_type is None:
            if "DDR5" in name_upper or "D5" in name_upper:
                result.ram_type = RAMType.DDR5
            elif "DDR4" in name_upper or "D4" in name_upper:
                result.ram_type = RAMType.DDR4

        # Fallback RAM type from socket
        if result.ram_type is None and result.socket:
            if result.socket == Socket.AM5:
                result.ram_type = RAMType.DDR5
            else:
                result.ram_type = RAMType.DDR4

        # Form factor detection
        if "mini-itx" in name_lower or "itx" in name_lower:
            result.form_factor = FormFactor.ITX
        elif "micro" in name_lower or "matx" in name_lower or "m-atx" in name_lower:
            result.form_factor = FormFactor.MATX
        else:
            result.form_factor = FormFactor.ATX

        return result

    def parse_ram(self, name: str, specs: dict = None) -> RAMSpecs:
        """Extract RAM specs from product name."""
        result = RAMSpecs()
        name_upper = name.upper()
        name_lower = name.lower()

        # Type detection
        if "DDR5" in name_upper:
            result.ram_type = RAMType.DDR5
        elif "DDR4" in name_upper:
            result.ram_type = RAMType.DDR4

        # Frequency detection
        freq_match = re.search(r"(\d{4,5})\s*(?:mhz|мгц)?", name_upper)
        if freq_match:
            result.frequency = int(freq_match.group(1))

        # Size detection
        size_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)
        if size_match:
            result.size_gb = int(size_match.group(1))

        # Module count (2x8GB = 2 modules)
        kit_match = re.search(r"(\d)\s*x\s*\d+\s*(?:gb|гб)", name_lower)
        if kit_match:
            result.modules = int(kit_match.group(1))

        return result

    def parse_gpu(self, name: str, specs: dict = None) -> GPUSpecs:
        """Extract GPU specs from product name."""
        result = GPUSpecs()
        name_lower = name.lower()

        # Brand detection
        if "nvidia" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
            result.brand = "NVIDIA"
        elif "amd" in name_lower or "radeon" in name_lower or "rx" in name_lower:
            result.brand = "AMD"
        elif "intel" in name_lower or "arc" in name_lower:
            result.brand = "Intel"

        # VRAM detection
        vram_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)
        if vram_match:
            result.vram_gb = int(vram_match.group(1))

        # Model and TDP detection
        for model, tdp in GPU_TDP_MAP.items():
            if model in name_lower:
                result.model = model
                result.tdp = tdp
                break

        # Tier detection
        for tier, patterns in GPU_TIER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    result.tier = tier
                    break

        return result

    def parse_psu(self, name: str, specs: dict = None) -> PSUSpecs:
        """Extract PSU specs from product name."""
        result = PSUSpecs()
        name_upper = name.upper()

        # Wattage detection
        watt_match = re.search(r"(\d{3,4})\s*(?:W|ВТ)?", name_upper)
        if watt_match:
            result.wattage = int(watt_match.group(1))

        # Efficiency detection
        if "PLATINUM" in name_upper:
            result.efficiency = "80+ Platinum"
            result.tier = Tier.HIGH
        elif "GOLD" in name_upper:
            result.efficiency = "80+ Gold"
            result.tier = Tier.MID
        elif "BRONZE" in name_upper:
            result.efficiency = "80+ Bronze"
            result.tier = Tier.BUDGET
        elif "80+" in name_upper:
            result.efficiency = "80+"
            result.tier = Tier.BUDGET

        return result


# ============================================================================
# BUDGET BALANCER - Smart budget allocation
# ============================================================================

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


class BudgetBalancer:
    """Smart budget allocation based on purpose and total budget."""

    # Budget allocation percentages by purpose
    ALLOCATION_PROFILES = {
        "gaming": {
            "cpu": 0.18, "motherboard": 0.10, "ram": 0.10,
            "gpu": 0.40, "psu": 0.07, "case": 0.05, "storage": 0.08, "cooler": 0.02,
        },
        "work": {
            "cpu": 0.28, "motherboard": 0.12, "ram": 0.15,
            "gpu": 0.15, "psu": 0.08, "case": 0.07, "storage": 0.12, "cooler": 0.03,
        },
        "office": {
            "cpu": 0.22, "motherboard": 0.15, "ram": 0.15,
            "gpu": 0.10, "psu": 0.10, "case": 0.10, "storage": 0.15, "cooler": 0.03,
        },
        "streaming": {
            "cpu": 0.25, "motherboard": 0.10, "ram": 0.12,
            "gpu": 0.35, "psu": 0.06, "case": 0.04, "storage": 0.06, "cooler": 0.02,
        },
    }

    # Maximum allocation percentages for low budgets
    LOW_BUDGET_CAPS = {
        "cpu": 0.25,
        "gpu": 0.40,
        "motherboard": 0.15,
    }

    LOW_BUDGET_THRESHOLD = 300000  # тенге

    def allocate(self, total_budget: int, purpose: str = "gaming") -> BudgetAllocation:
        """Allocate budget across components."""
        profile = self.ALLOCATION_PROFILES.get(purpose, self.ALLOCATION_PROFILES["gaming"])

        # Apply low budget caps
        if total_budget < self.LOW_BUDGET_THRESHOLD:
            for comp, cap in self.LOW_BUDGET_CAPS.items():
                if profile.get(comp, 0) > cap:
                    profile[comp] = cap

        return BudgetAllocation(
            cpu=int(total_budget * profile["cpu"]),
            motherboard=int(total_budget * profile["motherboard"]),
            ram=int(total_budget * profile["ram"]),
            gpu=int(total_budget * profile["gpu"]),
            psu=int(total_budget * profile["psu"]),
            case=int(total_budget * profile["case"]),
            storage=int(total_budget * profile["storage"]),
            cooler=int(total_budget * profile["cooler"]),
        )

    def validate_tier_match(self, cpu_tier: Tier, mb_tier: Tier) -> Tuple[bool, str]:
        """Validate that CPU and motherboard tiers are compatible.

        Rules:
        - Enthusiast CPU requires High/Enthusiast motherboard
        - High CPU requires Mid+ motherboard
        - Budget motherboard shouldn't have High+ CPU
        """
        tier_order = [Tier.BUDGET, Tier.MID, Tier.HIGH, Tier.ENTHUSIAST]
        cpu_idx = tier_order.index(cpu_tier)
        mb_idx = tier_order.index(mb_tier)

        # Enthusiast CPU needs at least High motherboard
        if cpu_tier == Tier.ENTHUSIAST and mb_tier in [Tier.BUDGET, Tier.MID]:
            return False, f"Процессор уровня {cpu_tier.value} требует материнскую плату уровня High или выше"

        # High CPU needs at least Mid motherboard
        if cpu_tier == Tier.HIGH and mb_tier == Tier.BUDGET:
            return False, f"Процессор уровня {cpu_tier.value} требует материнскую плату уровня Mid или выше"

        # Warn about mismatch (budget MB with high CPU)
        if cpu_idx - mb_idx > 1:
            return True, f"Предупреждение: материнская плата ({mb_tier.value}) может ограничивать производительность CPU ({cpu_tier.value})"

        return True, ""


# ============================================================================
# COMPATIBILITY ENGINE - Chain of Constraints
# ============================================================================

@dataclass
class CompatibilityIssue:
    """Compatibility issue description."""
    severity: str  # "error" or "warning"
    component_a: str
    component_b: str
    message: str


@dataclass
class BuildComponent:
    """Component in a PC build."""
    product_id: int
    name: str
    price: int
    discount_price: int
    specs: Any
    component_type: str


@dataclass
class PCBuild:
    """Complete PC build."""
    cpu: Optional[BuildComponent] = None
    motherboard: Optional[BuildComponent] = None
    ram: Optional[BuildComponent] = None
    gpu: Optional[BuildComponent] = None
    psu: Optional[BuildComponent] = None
    case: Optional[BuildComponent] = None
    storage: Optional[BuildComponent] = None
    cooler: Optional[BuildComponent] = None

    compatibility_issues: List[CompatibilityIssue] = field(default_factory=list)

    def total_price(self) -> int:
        """Calculate total build price."""
        total = 0
        for comp in [self.cpu, self.motherboard, self.ram, self.gpu,
                     self.psu, self.case, self.storage, self.cooler]:
            if comp:
                total += comp.discount_price or comp.price
        return total

    def is_valid(self) -> bool:
        """Check if build has no critical errors."""
        return not any(issue.severity == "error" for issue in self.compatibility_issues)

    def to_dict(self) -> dict:
        """Convert build to dictionary for API response."""
        build_dict = {}
        for comp_type in ["cpu", "motherboard", "ram", "gpu", "psu", "case", "storage", "cooler"]:
            comp = getattr(self, comp_type)
            if comp:
                build_dict[comp_type] = {
                    "id": comp.product_id,
                    "name": comp.name,
                    "price": comp.price,
                    "discount_price": comp.discount_price,
                }

        return {
            "build": build_dict,
            "total_price": self.total_price(),
            "warnings": [
                {"component": i.component_a, "message": i.message}
                for i in self.compatibility_issues if i.severity == "warning"
            ],
            "errors": [
                {"component": i.component_a, "message": i.message}
                for i in self.compatibility_issues if i.severity == "error"
            ],
        }


class CompatibilityEngine:
    """Validate component compatibility."""

    def __init__(self):
        self.parser = SpecsParser()

    def check_cpu_motherboard(self, cpu: CPUSpecs, mb: MotherboardSpecs) -> List[CompatibilityIssue]:
        """Check CPU and motherboard compatibility."""
        issues = []

        # Socket must match
        if cpu.socket and mb.socket and cpu.socket != mb.socket:
            issues.append(CompatibilityIssue(
                severity="error",
                component_a="CPU",
                component_b="Motherboard",
                message=f"Сокет CPU ({cpu.socket.value}) не совместим с материнской платой ({mb.socket.value})"
            ))

        # Tier validation
        balancer = BudgetBalancer()
        valid, msg = balancer.validate_tier_match(cpu.tier, mb.tier)
        if msg:
            issues.append(CompatibilityIssue(
                severity="warning" if valid else "error",
                component_a="CPU",
                component_b="Motherboard",
                message=msg
            ))

        return issues

    def check_motherboard_ram(self, mb: MotherboardSpecs, ram: RAMSpecs) -> List[CompatibilityIssue]:
        """Check motherboard and RAM compatibility."""
        issues = []

        if mb.ram_type and ram.ram_type and mb.ram_type != ram.ram_type:
            issues.append(CompatibilityIssue(
                severity="error",
                component_a="Motherboard",
                component_b="RAM",
                message=f"Материнская плата поддерживает {mb.ram_type.value}, выбрана память {ram.ram_type.value}"
            ))

        return issues

    def check_psu_power(self, psu: PSUSpecs, cpu: CPUSpecs, gpu: Optional[GPUSpecs]) -> List[CompatibilityIssue]:
        """Check if PSU has enough power."""
        issues = []

        # Calculate required power: (CPU TDP + GPU TDP) * 1.25 + 100W overhead
        cpu_tdp = cpu.tdp if cpu else 65
        gpu_tdp = gpu.tdp if gpu else 0
        required = int((cpu_tdp + gpu_tdp) * 1.25) + 100

        if psu.wattage < required:
            issues.append(CompatibilityIssue(
                severity="error",
                component_a="PSU",
                component_b="System",
                message=f"Мощность БП ({psu.wattage}W) недостаточна. Требуется минимум {required}W"
            ))
        elif psu.wattage < required * 1.1:
            issues.append(CompatibilityIssue(
                severity="warning",
                component_a="PSU",
                component_b="System",
                message=f"Мощность БП ({psu.wattage}W) близка к минимуму. Рекомендуется {int(required * 1.2)}W"
            ))

        return issues

    def validate_build(self, build: PCBuild) -> List[CompatibilityIssue]:
        """Validate entire build compatibility."""
        issues = []

        # CPU-Motherboard
        if build.cpu and build.motherboard:
            issues.extend(self.check_cpu_motherboard(
                build.cpu.specs, build.motherboard.specs
            ))

        # Motherboard-RAM
        if build.motherboard and build.ram:
            issues.extend(self.check_motherboard_ram(
                build.motherboard.specs, build.ram.specs
            ))

        # PSU power
        if build.psu:
            issues.extend(self.check_psu_power(
                build.psu.specs,
                build.cpu.specs if build.cpu else None,
                build.gpu.specs if build.gpu else None
            ))

        return issues


# ============================================================================
# PC ASSEMBLY ENGINE - Main build service
# ============================================================================

class PCAssemblyEngine:
    """Zero-hallucination PC assembly engine.

    Uses chain of constraints to build compatible PCs:
    1. Select CPU within budget
    2. Select Motherboard matching CPU socket
    3. Select RAM matching motherboard type
    4. Select GPU with remaining budget
    5. Select PSU with enough power
    6. Fill remaining components
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.parser = SpecsParser()
        self.compatibility = CompatibilityEngine()
        self.balancer = BudgetBalancer()

    async def build_pc(
        self,
        budget: int,
        purpose: str = "gaming",
    ) -> PCBuild:
        """Build a PC within budget constraints.

        Args:
            budget: Total budget in tenge
            purpose: gaming, work, office, streaming

        Returns:
            PCBuild with selected components and compatibility status
        """
        logger.info(f"[BUILD] Starting PC build: budget={budget}, purpose={purpose}")

        build = PCBuild()
        allocation = self.balancer.allocate(budget, purpose)

        # Track remaining budget
        remaining = budget

        # Step 1: Select CPU
        cpu_result = await self._select_cpu(allocation.cpu, purpose)
        if cpu_result:
            build.cpu = cpu_result
            remaining -= cpu_result.discount_price or cpu_result.price
            logger.info(f"[BUILD] CPU: {cpu_result.name} ({cpu_result.specs.socket})")

            # Add cooler if OEM
            if cpu_result.specs.is_oem:
                cooler_budget = min(allocation.cooler + 15000, remaining)
                cooler = await self._select_cooler(cooler_budget, cpu_result.specs)
                if cooler:
                    build.cooler = cooler
                    remaining -= cooler.discount_price or cooler.price
        else:
            build.compatibility_issues.append(CompatibilityIssue(
                severity="error", component_a="CPU", component_b="",
                message="Процессор не найден в наличии в указанном бюджете"
            ))
            return build

        # Step 2: Select Motherboard (MUST match CPU socket)
        mb_result = await self._select_motherboard(
            min(allocation.motherboard, remaining),
            build.cpu.specs.socket,
            build.cpu.specs.tier
        )
        if mb_result:
            build.motherboard = mb_result
            remaining -= mb_result.discount_price or mb_result.price
            logger.info(f"[BUILD] MB: {mb_result.name} ({mb_result.specs.ram_type})")
        else:
            build.compatibility_issues.append(CompatibilityIssue(
                severity="error", component_a="Motherboard", component_b="CPU",
                message=f"Материнская плата под сокет {build.cpu.specs.socket.value} не найдена в наличии"
            ))
            return build

        # Step 3: Select RAM (MUST match motherboard type)
        ram_result = await self._select_ram(
            min(allocation.ram, remaining),
            build.motherboard.specs.ram_type
        )
        if ram_result:
            build.ram = ram_result
            remaining -= ram_result.discount_price or ram_result.price
            logger.info(f"[BUILD] RAM: {ram_result.name}")
        else:
            build.compatibility_issues.append(CompatibilityIssue(
                severity="error", component_a="RAM", component_b="Motherboard",
                message=f"Память {build.motherboard.specs.ram_type.value} не найдена в наличии"
            ))
            return build

        # Step 4: Select GPU
        # IMPORTANT: Reserve budget for PSU, storage, case BEFORE GPU selection
        reserved_for_essentials = allocation.psu + allocation.storage + allocation.case + 10000  # +10k buffer
        max_gpu_budget = remaining - reserved_for_essentials

        if purpose == "gaming" and max_gpu_budget > 50000:
            # GPU gets its allocation + some extra, but NOT everything
            gpu_budget = min(allocation.gpu + int(max_gpu_budget * 0.3), max_gpu_budget)
            gpu_result = await self._select_gpu(gpu_budget, purpose)
            if gpu_result:
                build.gpu = gpu_result
                remaining -= gpu_result.discount_price or gpu_result.price
                logger.info(f"[BUILD] GPU: {gpu_result.name}")

        # Step 5: Select PSU (based on system power requirements)
        cpu_tdp = build.cpu.specs.tdp if build.cpu else 65
        gpu_tdp = build.gpu.specs.tdp if build.gpu else 150
        required_wattage = int((cpu_tdp + gpu_tdp) * 1.25) + 100

        # PSU budget: use allocation or more if needed
        psu_budget = max(allocation.psu, min(int(remaining * 0.35), 60000))
        psu_result = await self._select_psu(psu_budget, required_wattage)
        if psu_result:
            build.psu = psu_result
            remaining -= psu_result.discount_price or psu_result.price
            logger.info(f"[BUILD] PSU: {psu_result.name}")

        # Step 6: Fill remaining - Storage
        storage_budget = max(allocation.storage, min(int(remaining * 0.5), 50000))
        storage_result = await self._select_storage(storage_budget)
        if storage_result:
            build.storage = storage_result
            remaining -= storage_result.discount_price or storage_result.price
            logger.info(f"[BUILD] Storage: {storage_result.name}")

        # Step 7: Fill remaining - Case
        case_budget = max(allocation.case, remaining)
        case_result = await self._select_case(case_budget)
        if case_result:
            build.case = case_result
            remaining -= case_result.discount_price or case_result.price
            logger.info(f"[BUILD] Case: {case_result.name}")

        # Final validation
        build.compatibility_issues.extend(
            self.compatibility.validate_build(build)
        )

        logger.info(f"[BUILD] Complete: total={build.total_price()}, remaining={remaining}")
        return build

    async def _select_cpu(self, budget: int, purpose: str) -> Optional[BuildComponent]:
        """Select CPU within budget."""
        products = await self._query_products(
            categories=["Процессоры"],
            max_price=budget,
            min_price=int(budget * 0.4),
        )

        if not products:
            # Fallback: any CPU in budget
            products = await self._query_products(
                categories=["Процессоры"],
                max_price=budget,
            )

        best = None
        best_score = -1

        for p in products:
            name_lower = p.name.lower()

            # Skip server CPUs
            if "xeon" in name_lower or "epyc" in name_lower:
                continue

            specs = self.parser.parse_cpu(p.name)
            if not specs.socket:
                continue

            price = p.discount_price or p.price or 0
            score = price

            # Prefer modern sockets
            if specs.socket in [Socket.AM5, Socket.LGA1700]:
                score *= 1.15

            # Prefer matching tier for purpose
            if purpose == "gaming" and specs.tier in [Tier.MID, Tier.HIGH]:
                score *= 1.1

            if score > best_score:
                best_score = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="cpu"
                )

        return best

    async def _select_motherboard(
        self,
        budget: int,
        socket: Socket,
        cpu_tier: Tier
    ) -> Optional[BuildComponent]:
        """Select motherboard matching CPU socket."""
        products = await self._query_products(
            categories=["Материнские платы"],
            max_price=budget,
        )

        best = None
        best_score = -1

        for p in products:
            name_lower = p.name.lower()

            # Skip server boards
            if "server" in name_lower or "сервер" in name_lower:
                continue

            specs = self.parser.parse_motherboard(p.name)

            # MUST match socket
            if specs.socket != socket:
                continue

            # Must have RAM type
            if not specs.ram_type:
                continue

            price = p.discount_price or p.price or 0
            if price > budget:
                continue

            score = price

            # Prefer matching tier
            if specs.tier == cpu_tier:
                score *= 1.2
            elif abs(list(Tier).index(specs.tier) - list(Tier).index(cpu_tier)) <= 1:
                score *= 1.1

            if score > best_score:
                best_score = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="motherboard"
                )

        return best

    async def _select_ram(self, budget: int, ram_type: RAMType) -> Optional[BuildComponent]:
        """Select RAM matching motherboard type.

        For gaming, STRONGLY prefer 16GB+ (8GB is inadequate for modern games).
        """
        products = await self._query_products(
            categories=["Оперативная память"],
            max_price=budget,
        )

        best = None
        best_score = -1
        best_16gb = None
        best_16gb_score = -1

        for p in products:
            name_lower = p.name.lower()

            # Skip laptop RAM
            if "so-dimm" in name_lower or "sodimm" in name_lower or "ноутбук" in name_lower:
                continue

            # Skip server RAM (ECC, RDIMM, LRDIMM, Registered)
            if any(x in name_lower for x in ["сервер", "server", "ecc", "rdimm", "lrdimm", "registered"]):
                continue

            specs = self.parser.parse_ram(p.name)

            # MUST match RAM type
            if specs.ram_type != ram_type:
                continue

            price = p.discount_price or p.price or 0
            if price > budget:
                continue

            score = price

            # STRONGLY prefer 16GB+ (gaming requirement)
            if specs.size_gb >= 32:
                score *= 2.0  # Strong bonus for 32GB
            elif specs.size_gb >= 16:
                score *= 1.5  # Good bonus for 16GB
            elif specs.size_gb <= 8:
                score *= 0.3  # Heavy penalty for 8GB or less

            # Prefer higher frequency
            if specs.frequency >= 6000 and ram_type == RAMType.DDR5:
                score *= 1.15
            elif specs.frequency >= 3600 and ram_type == RAMType.DDR4:
                score *= 1.15

            # Track best 16GB+ option separately
            if specs.size_gb >= 16 and score > best_16gb_score:
                best_16gb_score = score
                best_16gb = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="ram"
                )

            if score > best_score:
                best_score = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="ram"
                )

        # Return 16GB+ if available, otherwise best available
        return best_16gb if best_16gb else best

    async def _select_gpu(self, budget: int, purpose: str) -> Optional[BuildComponent]:
        """Select GPU within budget."""
        products = await self._query_products(
            categories=["Видеокарты"],
            max_price=budget,
            min_price=int(budget * 0.3),
        )

        if not products:
            products = await self._query_products(
                categories=["Видеокарты"],
                max_price=budget,
            )

        best = None
        best_score = -1

        # Skip workstation cards for gaming
        workstation_kw = ["quadro", "firepro", "radeon pro", "a2000", "a4000", "a5000", "a6000"]

        for p in products:
            name_lower = p.name.lower()

            # Skip workstation/mining cards
            if any(kw in name_lower for kw in workstation_kw):
                continue
            if "mining" in name_lower:
                continue

            specs = self.parser.parse_gpu(p.name)
            price = p.discount_price or p.price or 0

            if price > budget:
                continue

            score = price

            # Prefer gaming cards
            if any(kw in name_lower for kw in ["rtx", "rx 7", "rx 6"]):
                score *= 1.15

            if score > best_score:
                best_score = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="gpu"
                )

        return best

    async def _select_psu(self, budget: int, required_wattage: int) -> Optional[BuildComponent]:
        """Select PSU with enough power."""
        products = await self._query_products(
            categories=["Блоки питания"],
            max_price=budget,
        )

        best = None
        best_score = -1

        for p in products:
            specs = self.parser.parse_psu(p.name)

            # Must have enough wattage
            if specs.wattage < required_wattage:
                continue

            price = p.discount_price or p.price or 0
            if price > budget:
                continue

            score = price

            # Prefer efficiency
            if specs.efficiency and "gold" in specs.efficiency.lower():
                score *= 1.15
            elif specs.efficiency and "platinum" in specs.efficiency.lower():
                score *= 1.2

            if score > best_score:
                best_score = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=specs,
                    component_type="psu"
                )

        return best

    async def _select_storage(self, budget: int) -> Optional[BuildComponent]:
        """Select storage within budget."""
        products = await self._query_products(
            categories=["Твердотельные диски (SSD)", "SSD накопители"],
            max_price=budget,
        )

        best = None
        best_price = 0

        for p in products:
            name_lower = p.name.lower()

            # Skip external drives
            if "внешний" in name_lower or "external" in name_lower or "portable" in name_lower:
                continue

            price = p.discount_price or p.price or 0
            if price > budget:
                continue

            # Prefer higher capacity (by price)
            if price > best_price:
                best_price = price
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=None,
                    component_type="storage"
                )

        return best

    async def _select_case(self, budget: int) -> Optional[BuildComponent]:
        """Select case within budget."""
        products = await self._query_products(
            categories=["Корпуса"],
            max_price=budget,
        )

        best = None
        best_price = 0

        for p in products:
            price = p.discount_price or p.price or 0
            if price > budget:
                continue

            if price > best_price:
                best_price = price
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=None,
                    component_type="case"
                )

        return best

    async def _select_cooler(self, budget: int, cpu_specs: CPUSpecs) -> Optional[BuildComponent]:
        """Select CPU cooler."""
        products = await self._query_products(
            categories=["Кулеры для процессоров", "Кулеры"],
            max_price=budget,
        )

        best = None
        best_price = 0

        # Socket keywords for filtering
        socket_kw = []
        if cpu_specs.socket:
            if cpu_specs.socket in [Socket.AM4, Socket.AM5]:
                socket_kw = ["am4", "am5", "amd"]
            elif cpu_specs.socket == Socket.LGA1700:
                socket_kw = ["lga1700", "lga 1700", "1700"]

        for p in products:
            name_lower = p.name.lower()
            price = p.discount_price or p.price or 0

            if price > budget:
                continue

            # Skip water cooling on low budget
            if budget < 15000 and any(w in name_lower for w in ["водян", "liquid", "aio"]):
                continue

            score = price
            # Bonus for socket match
            if socket_kw and any(kw in name_lower for kw in socket_kw):
                score *= 1.2

            if score > best_price:
                best_price = score
                best = BuildComponent(
                    product_id=p.id,
                    name=p.name,
                    price=p.price or 0,
                    discount_price=p.discount_price or 0,
                    specs=None,
                    component_type="cooler"
                )

        return best

    async def _query_products(
        self,
        categories: List[str],
        max_price: Optional[int] = None,
        min_price: Optional[int] = None,
    ) -> List[Product]:
        """Query products from database with stock > 0."""
        # Build category conditions
        cat_conditions = []
        for cat in categories:
            cat_conditions.append(Product.category == cat)
            cat_conditions.append(Product.category.ilike(f"%{cat}%"))

        conditions = [
            or_(*cat_conditions),
            Product.stock > 0,  # SAFETY: Always require stock
            Product.is_active == True,
        ]

        if max_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) <= max_price
            )
        if min_price:
            conditions.append(
                func.coalesce(Product.discount_price, Product.price) >= min_price
            )

        result = await self.session.execute(
            select(Product)
            .where(and_(*conditions))
            .order_by(func.coalesce(Product.discount_price, Product.price).desc())
            .limit(100)
        )

        return list(result.scalars().all())

    # ========================================================================
    # SWAP & VALIDATION - Replace component with cascade check
    # ========================================================================

    async def validate_and_replace(
        self,
        current_build: PCBuild,
        new_product_id: int,
        component_type: str,
    ) -> Tuple[PCBuild, List[str]]:
        """Replace a component and validate compatibility.

        Returns:
            Tuple of (updated_build, list_of_components_to_also_replace)
        """
        # Fetch new product
        result = await self.session.execute(
            select(Product).where(Product.id == new_product_id)
        )
        new_product = result.scalar_one_or_none()

        if not new_product:
            return current_build, ["Товар не найден"]

        if new_product.stock <= 0:
            return current_build, ["Товар не в наличии"]

        cascade_replacements = []

        # Parse new component specs
        if component_type == "cpu":
            new_specs = self.parser.parse_cpu(new_product.name)
            new_component = BuildComponent(
                product_id=new_product.id,
                name=new_product.name,
                price=new_product.price or 0,
                discount_price=new_product.discount_price or 0,
                specs=new_specs,
                component_type="cpu"
            )

            # Check if motherboard needs replacement (socket mismatch)
            if current_build.motherboard:
                mb_specs = current_build.motherboard.specs
                if mb_specs.socket != new_specs.socket:
                    cascade_replacements.append(
                        f"Материнская плата ({mb_specs.socket.value}) несовместима с новым CPU ({new_specs.socket.value}). Нужно заменить."
                    )

            # Check if RAM needs replacement (socket → RAM type change)
            if current_build.ram and new_specs.socket:
                if new_specs.socket == Socket.AM5:
                    # AM5 requires DDR5
                    if current_build.ram.specs.ram_type != RAMType.DDR5:
                        cascade_replacements.append(
                            f"Новый CPU (AM5) требует DDR5 память. Текущая память несовместима."
                        )

            current_build.cpu = new_component

        elif component_type == "motherboard":
            new_specs = self.parser.parse_motherboard(new_product.name)
            new_component = BuildComponent(
                product_id=new_product.id,
                name=new_product.name,
                price=new_product.price or 0,
                discount_price=new_product.discount_price or 0,
                specs=new_specs,
                component_type="motherboard"
            )

            # Check CPU compatibility
            if current_build.cpu:
                cpu_specs = current_build.cpu.specs
                if cpu_specs.socket != new_specs.socket:
                    cascade_replacements.append(
                        f"CPU ({cpu_specs.socket.value}) несовместим с новой материнской платой ({new_specs.socket.value}). Нужно заменить CPU."
                    )

            # Check RAM compatibility
            if current_build.ram:
                ram_specs = current_build.ram.specs
                if ram_specs.ram_type != new_specs.ram_type:
                    cascade_replacements.append(
                        f"Память ({ram_specs.ram_type.value}) несовместима с новой материнской платой ({new_specs.ram_type.value}). Нужно заменить RAM."
                    )

            current_build.motherboard = new_component

        elif component_type == "ram":
            new_specs = self.parser.parse_ram(new_product.name)
            new_component = BuildComponent(
                product_id=new_product.id,
                name=new_product.name,
                price=new_product.price or 0,
                discount_price=new_product.discount_price or 0,
                specs=new_specs,
                component_type="ram"
            )

            # Check motherboard compatibility
            if current_build.motherboard:
                mb_specs = current_build.motherboard.specs
                if mb_specs.ram_type != new_specs.ram_type:
                    cascade_replacements.append(
                        f"Материнская плата ({mb_specs.ram_type.value}) несовместима с новой памятью ({new_specs.ram_type.value}). Нужно заменить материнскую плату."
                    )

            current_build.ram = new_component

        elif component_type == "gpu":
            new_specs = self.parser.parse_gpu(new_product.name)
            new_component = BuildComponent(
                product_id=new_product.id,
                name=new_product.name,
                price=new_product.price or 0,
                discount_price=new_product.discount_price or 0,
                specs=new_specs,
                component_type="gpu"
            )

            # Check PSU power
            if current_build.psu:
                psu_specs = current_build.psu.specs
                cpu_tdp = current_build.cpu.specs.tdp if current_build.cpu else 65
                required = int((cpu_tdp + new_specs.tdp) * 1.25) + 100

                if psu_specs.wattage < required:
                    cascade_replacements.append(
                        f"Блок питания ({psu_specs.wattage}W) недостаточен для новой видеокарты. Требуется минимум {required}W."
                    )

            current_build.gpu = new_component

        elif component_type == "psu":
            new_specs = self.parser.parse_psu(new_product.name)
            new_component = BuildComponent(
                product_id=new_product.id,
                name=new_product.name,
                price=new_product.price or 0,
                discount_price=new_product.discount_price or 0,
                specs=new_specs,
                component_type="psu"
            )

            # Check if enough power
            cpu_tdp = current_build.cpu.specs.tdp if current_build.cpu else 65
            gpu_tdp = current_build.gpu.specs.tdp if current_build.gpu else 0
            required = int((cpu_tdp + gpu_tdp) * 1.25) + 100

            if new_specs.wattage < required:
                cascade_replacements.append(
                    f"Новый БП ({new_specs.wattage}W) недостаточен для текущей системы. Требуется минимум {required}W."
                )

            current_build.psu = new_component

        # Re-validate entire build
        current_build.compatibility_issues = self.compatibility.validate_build(current_build)

        return current_build, cascade_replacements

    async def get_compatible_alternatives(
        self,
        current_build: PCBuild,
        component_type: str,
        budget: Optional[int] = None,
        preference: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get alternative components compatible with current build.

        Args:
            current_build: Current PC build
            component_type: Component to find alternatives for
            budget: Max budget for alternative
            preference: "cheaper", "better", or brand name

        Returns:
            List of compatible alternatives
        """
        alternatives = []

        if component_type == "motherboard" and current_build.cpu:
            # Must match CPU socket
            socket = current_build.cpu.specs.socket
            products = await self._query_products(
                categories=["Материнские платы"],
                max_price=budget,
            )

            for p in products:
                specs = self.parser.parse_motherboard(p.name)
                if specs.socket == socket:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "socket": specs.socket.value if specs.socket else None,
                        "ram_type": specs.ram_type.value if specs.ram_type else None,
                    })

        elif component_type == "ram" and current_build.motherboard:
            # Must match motherboard RAM type
            ram_type = current_build.motherboard.specs.ram_type
            products = await self._query_products(
                categories=["Оперативная память"],
                max_price=budget,
            )

            for p in products:
                name_lower = p.name.lower()
                # Skip laptop RAM
                if "so-dimm" in name_lower or "sodimm" in name_lower:
                    continue
                # Skip server RAM
                if any(x in name_lower for x in ["сервер", "server", "ecc", "rdimm", "lrdimm", "registered"]):
                    continue

                specs = self.parser.parse_ram(p.name)
                if specs.ram_type == ram_type:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "ram_type": specs.ram_type.value if specs.ram_type else None,
                        "size_gb": specs.size_gb,
                    })

        elif component_type == "cpu":
            # Get CPUs, filter by motherboard socket if exists
            required_socket = None
            if current_build.motherboard:
                required_socket = current_build.motherboard.specs.socket

            products = await self._query_products(
                categories=["Процессоры"],
                max_price=budget,
            )

            for p in products:
                name_lower = p.name.lower()
                if "xeon" in name_lower or "epyc" in name_lower:
                    continue

                specs = self.parser.parse_cpu(p.name)
                if required_socket and specs.socket != required_socket:
                    continue

                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                    "socket": specs.socket.value if specs.socket else None,
                    "tier": specs.tier.value,
                })

        elif component_type == "gpu":
            products = await self._query_products(
                categories=["Видеокарты"],
                max_price=budget,
            )

            for p in products:
                name_lower = p.name.lower()
                if "quadro" in name_lower or "mining" in name_lower:
                    continue

                specs = self.parser.parse_gpu(p.name)
                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                    "vram_gb": specs.vram_gb,
                    "tdp": specs.tdp,
                })

        elif component_type == "psu":
            # Calculate minimum required wattage
            cpu_tdp = current_build.cpu.specs.tdp if current_build.cpu else 65
            gpu_tdp = current_build.gpu.specs.tdp if current_build.gpu else 200
            min_wattage = int((cpu_tdp + gpu_tdp) * 1.25) + 100

            products = await self._query_products(
                categories=["Блоки питания"],
                max_price=budget,
            )

            for p in products:
                specs = self.parser.parse_psu(p.name)
                if specs.wattage >= min_wattage:
                    alternatives.append({
                        "id": p.id,
                        "name": p.name,
                        "price": p.price,
                        "discount_price": p.discount_price,
                        "wattage": specs.wattage,
                        "efficiency": specs.efficiency,
                    })

        elif component_type == "storage":
            # Storage has no compatibility requirements
            products = await self._query_products(
                categories=["Твердотельные диски (SSD)", "SSD накопители", "SSD"],
                max_price=budget,
            )

            for p in products:
                name_lower = p.name.lower()
                # Skip external drives
                if "внешний" in name_lower or "external" in name_lower or "portable" in name_lower:
                    continue

                # Extract capacity from name
                capacity = "N/A"
                import re
                cap_match = re.search(r"(\d+)\s*(tb|тб|gb|гб)", name_lower)
                if cap_match:
                    size = int(cap_match.group(1))
                    unit = cap_match.group(2).lower()
                    if unit in ["tb", "тб"]:
                        capacity = f"{size}TB"
                    else:
                        capacity = f"{size}GB"

                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                    "capacity": capacity,
                })

        elif component_type == "case":
            # Case has no compatibility requirements (simplified)
            products = await self._query_products(
                categories=["Корпуса"],
                max_price=budget,
            )

            for p in products:
                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                })

        elif component_type == "cooler":
            # Cooler - filter by socket if CPU exists
            products = await self._query_products(
                categories=["Кулеры для процессоров", "Кулеры"],
                max_price=budget,
            )

            socket_kw = []
            if current_build.cpu and current_build.cpu.specs.socket:
                if current_build.cpu.specs.socket in [Socket.AM4, Socket.AM5]:
                    socket_kw = ["am4", "am5", "amd"]
                elif current_build.cpu.specs.socket == Socket.LGA1700:
                    socket_kw = ["lga1700", "lga 1700", "1700", "intel"]

            for p in products:
                name_lower = p.name.lower()
                # Prefer socket-compatible coolers
                if socket_kw and not any(kw in name_lower for kw in socket_kw):
                    continue

                alternatives.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "discount_price": p.discount_price,
                })

        # Sort by price
        alternatives.sort(key=lambda x: x.get("discount_price") or x.get("price") or 0)

        # Apply preference filter
        if preference:
            pref_lower = preference.lower()
            if "дешев" in pref_lower or "cheap" in pref_lower:
                alternatives = alternatives[:5]
            elif "дорог" in pref_lower or "лучш" in pref_lower:
                alternatives = alternatives[-5:]
            else:
                # Brand filter
                alternatives = [a for a in alternatives if preference.lower() in a["name"].lower()][:5]

        return alternatives[:10]
