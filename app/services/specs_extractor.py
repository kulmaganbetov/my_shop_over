"""Component specifications extractor with validation.

Extracts structured specs from product names and specifications.
Includes validation to ensure critical parameters are present.
"""

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class Socket(str, Enum):
    """CPU Sockets."""
    AM5 = "AM5"
    AM4 = "AM4"
    LGA1700 = "LGA1700"
    LGA1851 = "LGA1851"  # Intel Arrow Lake
    LGA1200 = "LGA1200"
    LGA1151 = "LGA1151"


class RAMType(str, Enum):
    """RAM Types."""
    DDR5 = "DDR5"
    DDR4 = "DDR4"
    DDR3 = "DDR3"


class FormFactor(str, Enum):
    """Motherboard form factors."""
    EATX = "E-ATX"
    ATX = "ATX"
    MATX = "mATX"
    ITX = "ITX"


class Tier(str, Enum):
    """Component performance tier."""
    BUDGET = "budget"
    MID = "mid"
    HIGH = "high"
    ENTHUSIAST = "enthusiast"


# CPU socket detection patterns - more comprehensive
CPU_SOCKET_PATTERNS = {
    # AMD AM5 - Ryzen 7000/8000/9000 series
    Socket.AM5: [
        r"ryzen\s*(9|7|5|3)\s*(9\d{3}|8\d{3}|7\d{3})",
        r"ryzen.*\bam5\b",
        r"7\d{3}x3d",  # 7800X3D etc
        r"9\d{3}x3d",  # 9800X3D
    ],
    # AMD AM4 - Ryzen 1000-5000 series
    Socket.AM4: [
        r"ryzen\s*(9|7|5|3)\s*(5\d{3}|3\d{3}|2\d{3}|1\d{3})",
        r"ryzen.*\bam4\b",
        r"5\d{3}x3d",  # 5800X3D
        r"5\d{3}g",    # 5600G etc
    ],
    # Intel LGA1851 - Arrow Lake (Core Ultra 200)
    Socket.LGA1851: [
        r"core\s*ultra\s*\d",
        r"ultra\s*(9|7|5)\s*2\d{2}",
        r"\blga\s*1851\b",
    ],
    # Intel LGA1700 - 12th-14th gen
    Socket.LGA1700: [
        r"core\s*i[3579]-1[234]\d{3}",
        r"i[3579]-1[234]\d{3}",
        r"\b1[234]\d{3}[a-z]?[fk]?\b",
        r"\blga\s*1700\b",
    ],
    # Intel LGA1200 - 10th-11th gen
    Socket.LGA1200: [
        r"core\s*i[3579]-1[01]\d{3}",
        r"i[3579]-1[01]\d{3}",
        r"\b1[01]\d{3}[a-z]?[fk]?\b",
        r"\blga\s*1200\b",
    ],
    # Intel LGA1151 - 8th-9th gen
    Socket.LGA1151: [
        r"core\s*i[3579]-[89]\d{3}",
        r"i[3579]-[89]\d{3}",
        r"\b[89]\d{3}[a-z]?[fk]?\b",
        r"\blga\s*1151\b",
    ],
}

# Motherboard chipset to socket mapping
CHIPSET_SOCKET_MAP = {
    # Intel LGA1851 (Arrow Lake)
    "Z890": Socket.LGA1851,
    "B860": Socket.LGA1851,
    "H810": Socket.LGA1851,
    # Intel LGA1700
    "Z790": Socket.LGA1700,
    "B760": Socket.LGA1700,
    "H770": Socket.LGA1700,
    "H610": Socket.LGA1700,
    "Z690": Socket.LGA1700,
    "B660": Socket.LGA1700,
    "H670": Socket.LGA1700,
    # Intel LGA1200
    "Z590": Socket.LGA1200,
    "B560": Socket.LGA1200,
    "H570": Socket.LGA1200,
    "H510": Socket.LGA1200,
    "Z490": Socket.LGA1200,
    "B460": Socket.LGA1200,
    # AMD AM5
    "X870E": Socket.AM5,
    "X870": Socket.AM5,
    "X670E": Socket.AM5,
    "X670": Socket.AM5,
    "B650E": Socket.AM5,
    "B650": Socket.AM5,
    "A620": Socket.AM5,
    # AMD AM4
    "X570": Socket.AM4,
    "B550": Socket.AM4,
    "A520": Socket.AM4,
    "X470": Socket.AM4,
    "B450": Socket.AM4,
    "A320": Socket.AM4,
}

# Chipset tiers for budget matching
CHIPSET_TIERS = {
    # Intel Enthusiast
    "Z890": Tier.ENTHUSIAST, "Z790": Tier.ENTHUSIAST, "Z690": Tier.ENTHUSIAST,
    # Intel High
    "H770": Tier.HIGH, "H670": Tier.HIGH,
    # Intel Mid
    "B860": Tier.MID, "B760": Tier.MID, "B660": Tier.MID, "B560": Tier.MID,
    # Intel Budget
    "H810": Tier.BUDGET, "H610": Tier.BUDGET, "H510": Tier.BUDGET,
    # AMD Enthusiast
    "X870E": Tier.ENTHUSIAST, "X670E": Tier.ENTHUSIAST,
    # AMD High
    "X870": Tier.HIGH, "X670": Tier.HIGH, "X570": Tier.HIGH,
    # AMD Mid
    "B650E": Tier.HIGH, "B650": Tier.MID, "B550": Tier.MID,
    # AMD Budget
    "A620": Tier.BUDGET, "A520": Tier.BUDGET, "B450": Tier.BUDGET, "A320": Tier.BUDGET,
}

# Chipset to RAM type mapping
CHIPSET_RAM_MAP = {
    # Intel LGA1851 = DDR5 only
    "Z890": RAMType.DDR5,
    "B860": RAMType.DDR5,
    "H810": RAMType.DDR5,
    # Intel LGA1700 DDR5 preferred
    "Z790": RAMType.DDR5,
    "H770": RAMType.DDR5,
    # Intel LGA1700 both supported - check board name for D4/D5
    "B760": None,
    "H610": None,
    "Z690": None,
    "B660": None,
    "H670": None,
    # Intel LGA1200 = DDR4 only
    "Z590": RAMType.DDR4,
    "B560": RAMType.DDR4,
    "H570": RAMType.DDR4,
    "H510": RAMType.DDR4,
    "Z490": RAMType.DDR4,
    "B460": RAMType.DDR4,
    # AMD AM5 = DDR5 only
    "X870E": RAMType.DDR5,
    "X870": RAMType.DDR5,
    "X670E": RAMType.DDR5,
    "X670": RAMType.DDR5,
    "B650E": RAMType.DDR5,
    "B650": RAMType.DDR5,
    "A620": RAMType.DDR5,
    # AMD AM4 = DDR4 only
    "X570": RAMType.DDR4,
    "B550": RAMType.DDR4,
    "A520": RAMType.DDR4,
    "X470": RAMType.DDR4,
    "B450": RAMType.DDR4,
    "A320": RAMType.DDR4,
}

# Socket to RAM type support
SOCKET_RAM_SUPPORT = {
    Socket.AM5: [RAMType.DDR5],
    Socket.AM4: [RAMType.DDR4],
    Socket.LGA1851: [RAMType.DDR5],
    Socket.LGA1700: [RAMType.DDR4, RAMType.DDR5],
    Socket.LGA1200: [RAMType.DDR4],
    Socket.LGA1151: [RAMType.DDR4],
}


@dataclass
class CPUSpecs:
    """Extracted CPU specifications."""
    socket: Optional[Socket] = None
    brand: Optional[str] = None
    series: Optional[str] = None
    model: Optional[str] = None
    tdp: int = 65
    has_igpu: bool = False
    tier: Tier = Tier.MID
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "socket": self.socket.value if self.socket else None,
            "brand": self.brand,
            "series": self.series,
            "model": self.model,
            "tdp": self.tdp,
            "has_igpu": self.has_igpu,
            "tier": self.tier.value,
        }


@dataclass
class MotherboardSpecs:
    """Extracted motherboard specifications."""
    socket: Optional[Socket] = None
    chipset: Optional[str] = None
    ram_type: Optional[RAMType] = None
    form_factor: Optional[FormFactor] = None
    brand: Optional[str] = None
    tier: Tier = Tier.MID
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "socket": self.socket.value if self.socket else None,
            "chipset": self.chipset,
            "ram_type": self.ram_type.value if self.ram_type else None,
            "form_factor": self.form_factor.value if self.form_factor else None,
            "brand": self.brand,
            "tier": self.tier.value,
        }


@dataclass
class RAMSpecs:
    """Extracted RAM specifications."""
    ram_type: Optional[RAMType] = None
    frequency: Optional[int] = None
    size_gb: Optional[int] = None
    modules: int = 1  # Number of modules (1 for single, 2 for kit)
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ram_type": self.ram_type.value if self.ram_type else None,
            "frequency": self.frequency,
            "size_gb": self.size_gb,
            "modules": self.modules,
        }


@dataclass
class GPUSpecs:
    """Extracted GPU specifications."""
    brand: Optional[str] = None
    model: Optional[str] = None
    vram_gb: Optional[int] = None
    tdp: int = 200
    length_mm: int = 300
    tier: Tier = Tier.MID
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "vram_gb": self.vram_gb,
            "tdp": self.tdp,
            "length_mm": self.length_mm,
            "tier": self.tier.value,
        }


@dataclass
class PSUSpecs:
    """Extracted PSU specifications."""
    wattage: int = 500
    efficiency: Optional[str] = None
    modular: bool = False
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wattage": self.wattage,
            "efficiency": self.efficiency,
            "modular": self.modular,
        }


@dataclass
class StorageSpecs:
    """Extracted storage specifications."""
    type: str = "SSD"  # SSD, HDD, NVMe
    capacity_gb: int = 0
    interface: Optional[str] = None  # SATA, NVMe, M.2
    is_complete: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "capacity_gb": self.capacity_gb,
            "interface": self.interface,
        }


class SpecsExtractor:
    """Extracts structured specs from product data with validation."""

    def extract_cpu_specs(self, name: str, specs: dict = None) -> Optional[CPUSpecs]:
        """Extract CPU specifications from name and specs.

        Returns None if critical parameters (socket) cannot be determined.
        """
        result = CPUSpecs()
        name_lower = name.lower()
        specs = specs or {}

        # Detect brand
        if "amd" in name_lower or "ryzen" in name_lower:
            result.brand = "AMD"
        elif "intel" in name_lower or "core i" in name_lower or "core ultra" in name_lower:
            result.brand = "Intel"

        # Detect socket from patterns
        for socket, patterns in CPU_SOCKET_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    result.socket = socket
                    break
            if result.socket:
                break

        # Check specs dict for socket
        if not result.socket and specs.get("socket"):
            socket_str = str(specs["socket"]).upper()
            for socket in Socket:
                if socket.value in socket_str:
                    result.socket = socket
                    break

        # Extract model name
        model_patterns = [
            r"(ryzen\s*[3579]\s*\d{4}[a-z0-9]*)",
            r"(core\s*i[3579]-\d{4,5}[a-z]*)",
            r"(core\s*ultra\s*[579]\s*\d{3}[a-z]*)",
        ]
        for pattern in model_patterns:
            match = re.search(pattern, name_lower)
            if match:
                result.model = match.group(1)
                break

        # Detect integrated GPU
        if result.brand == "AMD":
            if re.search(r"\d{4}g\b", name_lower):
                result.has_igpu = True
        elif result.brand == "Intel":
            # F suffix = no iGPU
            if not re.search(r"\d{4,5}f\b", name_lower):
                result.has_igpu = True

        # Determine tier based on model
        if result.model:
            model_lower = result.model.lower()
            if any(x in model_lower for x in ["i9", "ryzen 9", "ultra 9"]):
                result.tier = Tier.ENTHUSIAST
            elif any(x in model_lower for x in ["i7", "ryzen 7", "ultra 7"]):
                result.tier = Tier.HIGH
            elif any(x in model_lower for x in ["i5", "ryzen 5", "ultra 5"]):
                result.tier = Tier.MID
            else:
                result.tier = Tier.BUDGET

        # Extract TDP from specs
        if specs.get("tdp"):
            try:
                result.tdp = int(re.search(r"\d+", str(specs["tdp"])).group())
            except (AttributeError, ValueError):
                pass

        # Estimate TDP based on model if not in specs
        if result.tier == Tier.ENTHUSIAST:
            result.tdp = max(result.tdp, 125)
        elif result.tier == Tier.HIGH:
            result.tdp = max(result.tdp, 105)

        # VALIDATION: Socket is critical
        if result.socket:
            result.is_complete = True
        else:
            logger.warning(f"CPU specs incomplete - no socket detected: {name}")
            return None  # Return None for incomplete specs

        return result

    def extract_motherboard_specs(self, name: str, specs: dict = None) -> Optional[MotherboardSpecs]:
        """Extract motherboard specifications from name and specs.

        Returns None if critical parameters (socket, ram_type) cannot be determined.
        """
        result = MotherboardSpecs()
        name_upper = name.upper()
        name_lower = name.lower()
        specs = specs or {}

        # Detect chipset - order matters (check longer names first)
        chipset_order = sorted(CHIPSET_SOCKET_MAP.keys(), key=len, reverse=True)
        for chipset in chipset_order:
            if chipset in name_upper:
                result.chipset = chipset
                result.socket = CHIPSET_SOCKET_MAP[chipset]
                result.ram_type = CHIPSET_RAM_MAP.get(chipset)
                result.tier = CHIPSET_TIERS.get(chipset, Tier.MID)
                break

        # Detect RAM type from name if not set by chipset or if chipset supports both
        if result.ram_type is None:
            if "D5" in name_upper or "DDR5" in name_upper:
                result.ram_type = RAMType.DDR5
            elif "D4" in name_upper or "DDR4" in name_upper:
                result.ram_type = RAMType.DDR4

        # Infer RAM type from socket if still not set
        if not result.ram_type and result.socket:
            socket_ram = SOCKET_RAM_SUPPORT.get(result.socket, [])
            if len(socket_ram) == 1:
                result.ram_type = socket_ram[0]

        # Detect form factor
        if "mini-itx" in name_lower or "-itx" in name_lower:
            result.form_factor = FormFactor.ITX
        elif "micro-atx" in name_lower or "matx" in name_lower or "m-atx" in name_lower or name_upper.endswith("M"):
            result.form_factor = FormFactor.MATX
        elif "e-atx" in name_lower or "eatx" in name_lower:
            result.form_factor = FormFactor.EATX
        elif "atx" in name_lower:
            result.form_factor = FormFactor.ATX

        # Detect brand
        brands = ["asus", "gigabyte", "msi", "asrock", "colorful", "biostar"]
        for brand in brands:
            if brand in name_lower:
                result.brand = brand.upper()
                break

        # Check specs dict
        if not result.socket and specs.get("socket"):
            socket_str = str(specs["socket"]).upper()
            for socket in Socket:
                if socket.value in socket_str:
                    result.socket = socket
                    break

        if not result.ram_type and specs.get("ram_type"):
            for ram_type in RAMType:
                if ram_type.value in str(specs["ram_type"]).upper():
                    result.ram_type = ram_type
                    break

        # VALIDATION: Socket and RAM type are critical
        if result.socket and result.ram_type:
            result.is_complete = True
        else:
            missing = []
            if not result.socket:
                missing.append("socket")
            if not result.ram_type:
                missing.append("ram_type")
            logger.warning(f"Motherboard specs incomplete - missing {missing}: {name}")
            return None

        return result

    def extract_ram_specs(self, name: str, specs: dict = None) -> Optional[RAMSpecs]:
        """Extract RAM specifications from name and specs.

        Returns None if critical parameters (ram_type) cannot be determined.
        """
        result = RAMSpecs()
        name_upper = name.upper()
        name_lower = name.lower()
        specs = specs or {}

        # Skip laptop RAM
        if any(x in name_lower for x in ["so-dimm", "sodimm", "ноутбук", "laptop"]):
            return None

        # Skip server RAM
        if any(x in name_lower for x in ["сервер", "server", "ecc", "rdimm", "lrdimm", "registered"]):
            return None

        # Detect RAM type
        if "DDR5" in name_upper:
            result.ram_type = RAMType.DDR5
        elif "DDR4" in name_upper:
            result.ram_type = RAMType.DDR4
        elif "DDR3" in name_upper:
            result.ram_type = RAMType.DDR3

        # Extract frequency (e.g., 3200, 5600, 6000)
        freq_match = re.search(r"(\d{4,5})\s*(?:mhz|мгц)?", name_upper)
        if freq_match:
            result.frequency = int(freq_match.group(1))

        # Extract size - handle kits like "2x8GB" or "32GB (2x16)"
        kit_match = re.search(r"(\d)x(\d+)\s*(?:gb|гб)", name_lower)
        if kit_match:
            result.modules = int(kit_match.group(1))
            result.size_gb = int(kit_match.group(1)) * int(kit_match.group(2))
        else:
            size_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)
            if size_match:
                result.size_gb = int(size_match.group(1))

        # Check specs dict
        if not result.ram_type and specs.get("type"):
            for ram_type in RAMType:
                if ram_type.value in str(specs["type"]).upper():
                    result.ram_type = ram_type
                    break

        if not result.size_gb and specs.get("capacity"):
            try:
                result.size_gb = int(re.search(r"\d+", str(specs["capacity"])).group())
            except (AttributeError, ValueError):
                pass

        # VALIDATION: RAM type is critical
        if result.ram_type:
            result.is_complete = True
        else:
            logger.warning(f"RAM specs incomplete - no type detected: {name}")
            return None

        return result

    def extract_gpu_specs(self, name: str, specs: dict = None) -> Optional[GPUSpecs]:
        """Extract GPU specifications from name and specs."""
        result = GPUSpecs()
        name_lower = name.lower()
        name_upper = name.upper()
        specs = specs or {}

        # Skip workstation cards
        if any(x in name_lower for x in ["quadro", "tesla", "a100", "h100", "workstation"]):
            return None

        # Detect brand
        if any(x in name_lower for x in ["nvidia", "rtx", "gtx", "geforce"]):
            result.brand = "NVIDIA"
        elif any(x in name_lower for x in ["amd", "radeon", "rx "]):
            result.brand = "AMD"
        elif "intel" in name_lower and "arc" in name_lower:
            result.brand = "Intel"

        # Extract VRAM
        vram_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)
        if vram_match:
            result.vram_gb = int(vram_match.group(1))

        # GPU model and tier/TDP estimates
        gpu_data = {
            # NVIDIA RTX 50 series
            "5090": {"tdp": 575, "tier": Tier.ENTHUSIAST},
            "5080": {"tdp": 360, "tier": Tier.ENTHUSIAST},
            "5070 ti": {"tdp": 300, "tier": Tier.HIGH},
            "5070": {"tdp": 250, "tier": Tier.HIGH},
            "5060 ti": {"tdp": 180, "tier": Tier.MID},
            "5060": {"tdp": 150, "tier": Tier.MID},
            "5050": {"tdp": 115, "tier": Tier.BUDGET},
            # NVIDIA RTX 40 series
            "4090": {"tdp": 450, "tier": Tier.ENTHUSIAST},
            "4080 super": {"tdp": 320, "tier": Tier.ENTHUSIAST},
            "4080": {"tdp": 320, "tier": Tier.ENTHUSIAST},
            "4070 ti super": {"tdp": 285, "tier": Tier.HIGH},
            "4070 ti": {"tdp": 285, "tier": Tier.HIGH},
            "4070 super": {"tdp": 220, "tier": Tier.HIGH},
            "4070": {"tdp": 200, "tier": Tier.HIGH},
            "4060 ti": {"tdp": 165, "tier": Tier.MID},
            "4060": {"tdp": 115, "tier": Tier.MID},
            # NVIDIA RTX 30 series
            "3090 ti": {"tdp": 450, "tier": Tier.ENTHUSIAST},
            "3090": {"tdp": 350, "tier": Tier.ENTHUSIAST},
            "3080 ti": {"tdp": 350, "tier": Tier.ENTHUSIAST},
            "3080": {"tdp": 320, "tier": Tier.HIGH},
            "3070 ti": {"tdp": 290, "tier": Tier.HIGH},
            "3070": {"tdp": 220, "tier": Tier.HIGH},
            "3060 ti": {"tdp": 200, "tier": Tier.MID},
            "3060": {"tdp": 170, "tier": Tier.MID},
            "3050": {"tdp": 130, "tier": Tier.BUDGET},
            # AMD RX 7000 series
            "7900 xtx": {"tdp": 355, "tier": Tier.ENTHUSIAST},
            "7900 xt": {"tdp": 300, "tier": Tier.ENTHUSIAST},
            "7900 gre": {"tdp": 260, "tier": Tier.HIGH},
            "7800 xt": {"tdp": 263, "tier": Tier.HIGH},
            "7700 xt": {"tdp": 245, "tier": Tier.MID},
            "7600 xt": {"tdp": 190, "tier": Tier.MID},
            "7600": {"tdp": 165, "tier": Tier.MID},
            # AMD RX 6000 series
            "6950 xt": {"tdp": 335, "tier": Tier.ENTHUSIAST},
            "6900 xt": {"tdp": 300, "tier": Tier.ENTHUSIAST},
            "6800 xt": {"tdp": 300, "tier": Tier.HIGH},
            "6800": {"tdp": 250, "tier": Tier.HIGH},
            "6700 xt": {"tdp": 230, "tier": Tier.MID},
            "6600 xt": {"tdp": 160, "tier": Tier.MID},
            "6600": {"tdp": 132, "tier": Tier.BUDGET},
            "6500 xt": {"tdp": 107, "tier": Tier.BUDGET},
        }

        for model, data in gpu_data.items():
            if model in name_lower:
                result.model = model.upper()
                result.tdp = data["tdp"]
                result.tier = data["tier"]
                break

        # Get length from specs
        if specs.get("length"):
            try:
                result.length_mm = int(re.search(r"\d+", str(specs["length"])).group())
            except (AttributeError, ValueError):
                pass

        # Mark as complete if we detected brand and have basic info
        if result.brand:
            result.is_complete = True
            return result

        return None

    def extract_psu_specs(self, name: str, specs: dict = None) -> Optional[PSUSpecs]:
        """Extract PSU specifications from name and specs."""
        result = PSUSpecs()
        name_upper = name.upper()
        name_lower = name.lower()
        specs = specs or {}

        # Extract wattage
        watt_match = re.search(r"(\d{3,4})\s*(?:W|ВТ|WATT)?", name_upper)
        if watt_match:
            result.wattage = int(watt_match.group(1))

        # Detect efficiency rating
        if "TITANIUM" in name_upper:
            result.efficiency = "80+ Titanium"
        elif "PLATINUM" in name_upper:
            result.efficiency = "80+ Platinum"
        elif "GOLD" in name_upper:
            result.efficiency = "80+ Gold"
        elif "BRONZE" in name_upper:
            result.efficiency = "80+ Bronze"
        elif "80+" in name_upper or "80 PLUS" in name_upper:
            result.efficiency = "80+"

        # Detect modular
        if any(x in name_lower for x in ["full modular", "fully modular", "modular"]):
            result.modular = True

        # Check specs dict
        if specs.get("wattage"):
            try:
                result.wattage = int(re.search(r"\d+", str(specs["wattage"])).group())
            except (AttributeError, ValueError):
                pass

        # Mark as complete if we have wattage
        if result.wattage >= 300:
            result.is_complete = True
            return result

        return None

    def extract_storage_specs(self, name: str, specs: dict = None) -> Optional[StorageSpecs]:
        """Extract storage specifications from name and specs."""
        result = StorageSpecs()
        name_lower = name.lower()
        specs = specs or {}

        # Skip external drives
        if any(x in name_lower for x in ["внешний", "external", "portable"]):
            return None

        # Detect type and interface
        if "nvme" in name_lower or "m.2 pci" in name_lower or "pcie" in name_lower:
            result.type = "NVMe"
            result.interface = "M.2 NVMe"
        elif "m.2" in name_lower:
            result.type = "SSD"
            result.interface = "M.2 SATA"
        elif "ssd" in name_lower:
            result.type = "SSD"
            result.interface = "SATA"
        elif "hdd" in name_lower or "жесткий" in name_lower:
            result.type = "HDD"
            result.interface = "SATA"

        # Extract capacity
        tb_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:tb|тб)", name_lower)
        gb_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)

        if tb_match:
            result.capacity_gb = int(float(tb_match.group(1)) * 1000)
        elif gb_match:
            result.capacity_gb = int(gb_match.group(1))

        if result.capacity_gb > 0:
            result.is_complete = True
            return result

        return None

    # Compatibility check methods
    def is_compatible_cpu_mb(self, cpu: CPUSpecs, mb: MotherboardSpecs) -> bool:
        """Check if CPU and motherboard are compatible."""
        if not cpu or not mb:
            return False
        if not cpu.socket or not mb.socket:
            return False
        return cpu.socket == mb.socket

    def is_compatible_mb_ram(self, mb: MotherboardSpecs, ram: RAMSpecs) -> bool:
        """Check if motherboard and RAM are compatible."""
        if not mb or not ram:
            return False
        if not mb.ram_type or not ram.ram_type:
            return False
        return mb.ram_type == ram.ram_type

    def is_compatible_cpu_ram(self, cpu: CPUSpecs, ram: RAMSpecs) -> bool:
        """Check if CPU supports the RAM type."""
        if not cpu or not ram:
            return False
        if not cpu.socket or not ram.ram_type:
            return False
        supported = SOCKET_RAM_SUPPORT.get(cpu.socket, [])
        return ram.ram_type in supported

    def check_psu_power(self, psu: PSUSpecs, cpu: CPUSpecs, gpu: GPUSpecs) -> bool:
        """Check if PSU has enough power for the system."""
        if not psu:
            return False

        cpu_tdp = cpu.tdp if cpu else 65
        gpu_tdp = gpu.tdp if gpu else 150

        # Required = (CPU + GPU) * 1.25 + 100W overhead
        required = int((cpu_tdp + gpu_tdp) * 1.25) + 100

        return psu.wattage >= required

    def get_required_psu_wattage(self, cpu: CPUSpecs, gpu: GPUSpecs) -> int:
        """Calculate required PSU wattage for given components."""
        cpu_tdp = cpu.tdp if cpu else 65
        gpu_tdp = gpu.tdp if gpu else 150
        return int((cpu_tdp + gpu_tdp) * 1.25) + 100
