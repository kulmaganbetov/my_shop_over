"""Component specifications extractor.

Extracts structured specs from product names and specifications.
This is necessary because DB often doesn't have structured data.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
    DDR3 = "DDR3"


class FormFactor(str, Enum):
    """Motherboard form factors."""
    ATX = "ATX"
    MATX = "mATX"
    ITX = "ITX"


# CPU socket detection patterns
CPU_SOCKET_PATTERNS = {
    # AMD
    Socket.AM5: [
        r"ryzen\s*(9|7|5|3)\s*(9\d{3}|8\d{3}|7\d{3})",  # Ryzen 7000-9000 series
        r"ryzen.*\bam5\b",
    ],
    Socket.AM4: [
        r"ryzen\s*(9|7|5|3)\s*(5\d{3}|3\d{3}|2\d{3}|1\d{3})",  # Ryzen 1000-5000 series
        r"ryzen.*\bam4\b",
    ],
    # Intel
    Socket.LGA1700: [
        r"core\s*i[3579]-1[234]\d{3}",  # 12th-14th gen
        r"\b(12|13|14)\d{3}[a-z]?\b",  # i5-12400, i7-13700, etc.
        r"\blga\s*1700\b",
    ],
    Socket.LGA1200: [
        r"core\s*i[3579]-1[01]\d{3}",  # 10th-11th gen
        r"\b(10|11)\d{3}[a-z]?\b",
        r"\blga\s*1200\b",
    ],
    Socket.LGA1151: [
        r"core\s*i[3579]-[89]\d{3}",  # 8th-9th gen
        r"\b[89]\d{3}[a-z]?\b",
        r"\blga\s*1151\b",
    ],
}

# Motherboard chipset to socket mapping
CHIPSET_SOCKET_MAP = {
    # Intel LGA1700
    "Z790": Socket.LGA1700,
    "B760": Socket.LGA1700,
    "H770": Socket.LGA1700,
    "H610": Socket.LGA1700,
    "H810": Socket.LGA1700,
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

# Chipset to RAM type mapping
CHIPSET_RAM_MAP = {
    # Intel DDR5 only
    "Z790": RAMType.DDR5,
    "H770": RAMType.DDR5,
    # Intel DDR4/DDR5 (we'll prefer DDR5 for newer)
    "B760": None,  # Both supported
    "H610": None,  # Both supported
    "H810": RAMType.DDR5,  # Newer H810 is DDR5
    "Z690": None,  # Both supported
    "B660": None,  # Both supported
    # Intel DDR4 only
    "Z590": RAMType.DDR4,
    "B560": RAMType.DDR4,
    "H570": RAMType.DDR4,
    "H510": RAMType.DDR4,
    "Z490": RAMType.DDR4,
    "B460": RAMType.DDR4,
    # AMD AM5 = DDR5 only
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

# Socket to RAM type (for CPU selection)
SOCKET_RAM_SUPPORT = {
    Socket.AM5: [RAMType.DDR5],
    Socket.AM4: [RAMType.DDR4],
    Socket.LGA1700: [RAMType.DDR4, RAMType.DDR5],  # Depends on motherboard
    Socket.LGA1200: [RAMType.DDR4],
    Socket.LGA1151: [RAMType.DDR4],
}


@dataclass
class CPUSpecs:
    """Extracted CPU specifications."""
    socket: Optional[Socket] = None
    brand: Optional[str] = None  # AMD, Intel
    series: Optional[str] = None  # Ryzen 5, Core i7
    model: Optional[str] = None
    tdp: int = 65
    has_igpu: bool = False


@dataclass
class MotherboardSpecs:
    """Extracted motherboard specifications."""
    socket: Optional[Socket] = None
    chipset: Optional[str] = None
    ram_type: Optional[RAMType] = None
    form_factor: Optional[FormFactor] = None
    brand: Optional[str] = None


@dataclass
class RAMSpecs:
    """Extracted RAM specifications."""
    ram_type: Optional[RAMType] = None
    frequency: Optional[int] = None
    size_gb: Optional[int] = None


@dataclass
class GPUSpecs:
    """Extracted GPU specifications."""
    brand: Optional[str] = None  # NVIDIA, AMD
    model: Optional[str] = None
    vram_gb: Optional[int] = None
    tdp: int = 200
    length_mm: int = 300


@dataclass
class PSUSpecs:
    """Extracted PSU specifications."""
    wattage: int = 500
    efficiency: Optional[str] = None  # 80+ Bronze, Gold, etc.


class SpecsExtractor:
    """Extracts structured specs from product data."""

    def extract_cpu_specs(self, name: str, specs: dict = None) -> CPUSpecs:
        """Extract CPU specifications from name and specs."""
        result = CPUSpecs()
        name_lower = name.lower()
        specs = specs or {}

        # Detect brand
        if "amd" in name_lower or "ryzen" in name_lower:
            result.brand = "AMD"
        elif "intel" in name_lower or "core i" in name_lower:
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
            socket_str = specs["socket"].upper()
            for socket in Socket:
                if socket.value in socket_str:
                    result.socket = socket
                    break

        # Detect integrated GPU
        if "g" in name_lower and result.brand == "AMD":
            # Ryzen with G suffix has iGPU (e.g., Ryzen 5 5600G)
            if re.search(r"ryzen.*\d{4}g", name_lower):
                result.has_igpu = True

        # Extract TDP from specs
        if specs.get("tdp"):
            try:
                result.tdp = int(re.search(r"\d+", str(specs["tdp"])).group())
            except (AttributeError, ValueError):
                pass

        return result

    def extract_motherboard_specs(self, name: str, specs: dict = None) -> MotherboardSpecs:
        """Extract motherboard specifications from name and specs."""
        result = MotherboardSpecs()
        name_upper = name.upper()
        specs = specs or {}

        # Detect chipset
        for chipset in CHIPSET_SOCKET_MAP.keys():
            if chipset in name_upper:
                result.chipset = chipset
                result.socket = CHIPSET_SOCKET_MAP[chipset]
                result.ram_type = CHIPSET_RAM_MAP.get(chipset)
                break

        # Detect RAM type from name if not set by chipset
        if not result.ram_type:
            if "D5" in name_upper or "DDR5" in name_upper:
                result.ram_type = RAMType.DDR5
            elif "D4" in name_upper or "DDR4" in name_upper:
                result.ram_type = RAMType.DDR4

        # Infer RAM type from socket if still not set
        if not result.ram_type and result.socket:
            if result.socket == Socket.AM5:
                result.ram_type = RAMType.DDR5
            elif result.socket in [Socket.AM4, Socket.LGA1200, Socket.LGA1151]:
                result.ram_type = RAMType.DDR4

        # Detect form factor
        name_lower = name.lower()
        if "mini-itx" in name_lower or "itx" in name_lower:
            result.form_factor = FormFactor.ITX
        elif "micro-atx" in name_lower or "matx" in name_lower or "m-atx" in name_lower:
            result.form_factor = FormFactor.MATX
        elif "atx" in name_lower:
            result.form_factor = FormFactor.ATX

        # Check specs dict
        if not result.socket and specs.get("socket"):
            socket_str = specs["socket"].upper()
            for socket in Socket:
                if socket.value in socket_str:
                    result.socket = socket
                    break

        if not result.ram_type and specs.get("ram_type"):
            for ram_type in RAMType:
                if ram_type.value in specs["ram_type"].upper():
                    result.ram_type = ram_type
                    break

        return result

    def extract_ram_specs(self, name: str, specs: dict = None) -> RAMSpecs:
        """Extract RAM specifications from name and specs."""
        result = RAMSpecs()
        name_upper = name.upper()
        specs = specs or {}

        # Detect RAM type
        if "DDR5" in name_upper:
            result.ram_type = RAMType.DDR5
        elif "DDR4" in name_upper:
            result.ram_type = RAMType.DDR4
        elif "DDR3" in name_upper:
            result.ram_type = RAMType.DDR3

        # Extract frequency (e.g., 3200, 5600)
        freq_match = re.search(r"(\d{4,5})\s*(?:mhz|мгц)?", name_upper)
        if freq_match:
            result.frequency = int(freq_match.group(1))

        # Extract size
        size_match = re.search(r"(\d+)\s*(?:gb|гб)", name.lower())
        if size_match:
            result.size_gb = int(size_match.group(1))

        # Check specs dict
        if specs.get("type"):
            for ram_type in RAMType:
                if ram_type.value in specs["type"].upper():
                    result.ram_type = ram_type
                    break

        return result

    def extract_gpu_specs(self, name: str, specs: dict = None) -> GPUSpecs:
        """Extract GPU specifications from name and specs."""
        result = GPUSpecs()
        name_lower = name.lower()
        name_upper = name.upper()
        specs = specs or {}

        # Detect brand
        if "nvidia" in name_lower or "rtx" in name_lower or "gtx" in name_lower:
            result.brand = "NVIDIA"
        elif "amd" in name_lower or "radeon" in name_lower or "rx " in name_lower:
            result.brand = "AMD"

        # Extract VRAM
        vram_match = re.search(r"(\d+)\s*(?:gb|гб)", name_lower)
        if vram_match:
            result.vram_gb = int(vram_match.group(1))

        # Estimate TDP based on model
        tdp_estimates = {
            "4090": 450, "4080": 320, "4070 ti": 285, "4070": 200,
            "4060 ti": 160, "4060": 115, "3090": 350, "3080": 320,
            "3070": 220, "3060": 170, "7900 xtx": 355, "7900 xt": 300,
            "7800 xt": 263, "7700 xt": 245, "7600": 165,
        }
        for model, tdp in tdp_estimates.items():
            if model in name_lower:
                result.tdp = tdp
                result.model = model
                break

        # Get length from specs
        if specs.get("length"):
            try:
                result.length_mm = int(re.search(r"\d+", str(specs["length"])).group())
            except (AttributeError, ValueError):
                pass

        return result

    def extract_psu_specs(self, name: str, specs: dict = None) -> PSUSpecs:
        """Extract PSU specifications from name and specs."""
        result = PSUSpecs()
        name_upper = name.upper()
        specs = specs or {}

        # Extract wattage
        watt_match = re.search(r"(\d{3,4})\s*(?:W|ВТ|WATT)?", name_upper)
        if watt_match:
            result.wattage = int(watt_match.group(1))

        # Detect efficiency rating
        if "PLATINUM" in name_upper:
            result.efficiency = "80+ Platinum"
        elif "GOLD" in name_upper:
            result.efficiency = "80+ Gold"
        elif "BRONZE" in name_upper:
            result.efficiency = "80+ Bronze"
        elif "80+" in name_upper or "80 PLUS" in name_upper:
            result.efficiency = "80+"

        # Check specs dict
        if specs.get("wattage"):
            try:
                result.wattage = int(re.search(r"\d+", str(specs["wattage"])).group())
            except (AttributeError, ValueError):
                pass

        return result

    def is_compatible_cpu_mb(self, cpu: CPUSpecs, mb: MotherboardSpecs) -> bool:
        """Check if CPU and motherboard are compatible."""
        if not cpu.socket or not mb.socket:
            return False
        return cpu.socket == mb.socket

    def is_compatible_mb_ram(self, mb: MotherboardSpecs, ram: RAMSpecs) -> bool:
        """Check if motherboard and RAM are compatible."""
        if not mb.ram_type or not ram.ram_type:
            return False
        return mb.ram_type == ram.ram_type

    def is_compatible_cpu_ram(self, cpu: CPUSpecs, ram: RAMSpecs) -> bool:
        """Check if CPU supports the RAM type."""
        if not cpu.socket or not ram.ram_type:
            return False
        supported = SOCKET_RAM_SUPPORT.get(cpu.socket, [])
        return ram.ram_type in supported
