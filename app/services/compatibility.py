"""PC component compatibility rules engine.

This module contains DETERMINISTIC rules for checking PC component compatibility.
LLM is NOT involved in compatibility decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CompatibilityStatus(str, Enum):
    """Compatibility check status."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class CompatibilityResult:
    """Result of a compatibility check."""

    status: CompatibilityStatus
    message: str
    component_a: str
    component_b: str


class CPUSockets(str, Enum):
    """Known CPU sockets."""

    LGA1700 = "LGA1700"  # Intel 12th-14th gen
    LGA1200 = "LGA1200"  # Intel 10th-11th gen
    LGA1151 = "LGA1151"  # Intel 8th-9th gen
    AM5 = "AM5"  # AMD Ryzen 7000+
    AM4 = "AM4"  # AMD Ryzen 1000-5000


class RAMTypes(str, Enum):
    """Memory types."""

    DDR4 = "DDR4"
    DDR5 = "DDR5"


# Socket to memory type mapping
SOCKET_MEMORY_MAP = {
    CPUSockets.LGA1700: [RAMTypes.DDR4, RAMTypes.DDR5],
    CPUSockets.LGA1200: [RAMTypes.DDR4],
    CPUSockets.LGA1151: [RAMTypes.DDR4],
    CPUSockets.AM5: [RAMTypes.DDR5],
    CPUSockets.AM4: [RAMTypes.DDR4],
}

# Recommended PSU wattage by GPU tier
GPU_POWER_REQUIREMENTS = {
    # NVIDIA
    "RTX 4090": 850,
    "RTX 4080": 750,
    "RTX 4070 Ti": 700,
    "RTX 4070": 650,
    "RTX 4060 Ti": 550,
    "RTX 4060": 500,
    "RTX 3090": 850,
    "RTX 3080": 750,
    "RTX 3070": 650,
    "RTX 3060": 550,
    # AMD
    "RX 7900 XTX": 850,
    "RX 7900 XT": 750,
    "RX 7800 XT": 650,
    "RX 7700 XT": 600,
    "RX 7600": 500,
}

# Budget allocation percentages for different purposes
BUDGET_ALLOCATION = {
    "gaming": {
        "cpu": 0.18,
        "gpu": 0.38,
        "motherboard": 0.12,
        "ram": 0.12,  # DDR5 needs more budget
        "storage": 0.08,
        "psu": 0.06,
        "case": 0.06,
    },
    "work": {
        "cpu": 0.30,
        "gpu": 0.15,
        "motherboard": 0.12,
        "ram": 0.15,
        "storage": 0.12,
        "psu": 0.08,
        "case": 0.08,
    },
    "office": {
        "cpu": 0.25,
        "gpu": 0.10,
        "motherboard": 0.15,
        "ram": 0.15,
        "storage": 0.15,
        "psu": 0.10,
        "case": 0.10,
    },
    "streaming": {
        "cpu": 0.25,
        "gpu": 0.35,
        "motherboard": 0.10,
        "ram": 0.10,
        "storage": 0.08,
        "psu": 0.07,
        "case": 0.05,
    },
    "content_creation": {
        "cpu": 0.30,
        "gpu": 0.25,
        "motherboard": 0.10,
        "ram": 0.12,
        "storage": 0.10,
        "psu": 0.07,
        "case": 0.06,
    },
    "general": {
        "cpu": 0.22,
        "gpu": 0.25,
        "motherboard": 0.12,
        "ram": 0.12,
        "storage": 0.12,
        "psu": 0.09,
        "case": 0.08,
    },
}


class CompatibilityEngine:
    """Engine for checking PC component compatibility.

    All rules are DETERMINISTIC - no LLM involvement.
    """

    def check_cpu_motherboard(
        self,
        cpu_socket: str,
        motherboard_socket: str,
    ) -> CompatibilityResult:
        """Check CPU and motherboard socket compatibility."""
        if cpu_socket == motherboard_socket:
            return CompatibilityResult(
                status=CompatibilityStatus.OK,
                message="CPU и материнская плата совместимы",
                component_a="CPU",
                component_b="Motherboard",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.ERROR,
            message=f"Несовместимые сокеты: CPU {cpu_socket} не подходит к материнской плате {motherboard_socket}",
            component_a="CPU",
            component_b="Motherboard",
        )

    def check_motherboard_ram(
        self,
        motherboard_ram_type: str,
        ram_type: str,
    ) -> CompatibilityResult:
        """Check motherboard and RAM type compatibility."""
        if motherboard_ram_type == ram_type:
            return CompatibilityResult(
                status=CompatibilityStatus.OK,
                message="Оперативная память совместима с материнской платой",
                component_a="Motherboard",
                component_b="RAM",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.ERROR,
            message=f"Несовместимый тип памяти: материнская плата поддерживает {motherboard_ram_type}, выбрана {ram_type}",
            component_a="Motherboard",
            component_b="RAM",
        )

    def check_psu_power(
        self,
        psu_wattage: int,
        gpu_model: Optional[str] = None,
        total_system_tdp: int = 200,
    ) -> CompatibilityResult:
        """Check if PSU has enough power for the system."""
        required_power = total_system_tdp

        # Add GPU power requirement
        if gpu_model:
            for gpu_key, power in GPU_POWER_REQUIREMENTS.items():
                if gpu_key.lower() in gpu_model.lower():
                    required_power = max(required_power, power)
                    break

        if psu_wattage >= required_power:
            return CompatibilityResult(
                status=CompatibilityStatus.OK,
                message=f"Блок питания {psu_wattage}W достаточен для системы",
                component_a="PSU",
                component_b="System",
            )
        elif psu_wattage >= required_power * 0.9:
            return CompatibilityResult(
                status=CompatibilityStatus.WARNING,
                message=f"Блок питания {psu_wattage}W близок к минимуму. Рекомендуется {required_power}W",
                component_a="PSU",
                component_b="System",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.ERROR,
            message=f"Недостаточная мощность БП: {psu_wattage}W, требуется минимум {required_power}W",
            component_a="PSU",
            component_b="System",
        )

    def check_case_gpu_length(
        self,
        case_max_gpu_length: int,
        gpu_length: int,
    ) -> CompatibilityResult:
        """Check if GPU fits in the case."""
        if gpu_length <= case_max_gpu_length:
            return CompatibilityResult(
                status=CompatibilityStatus.OK,
                message="Видеокарта помещается в корпус",
                component_a="Case",
                component_b="GPU",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.ERROR,
            message=f"Видеокарта ({gpu_length}мм) не помещается в корпус (макс. {case_max_gpu_length}мм)",
            component_a="Case",
            component_b="GPU",
        )

    def check_case_cooler_height(
        self,
        case_max_cooler_height: int,
        cooler_height: int,
    ) -> CompatibilityResult:
        """Check if CPU cooler fits in the case."""
        if cooler_height <= case_max_cooler_height:
            return CompatibilityResult(
                status=CompatibilityStatus.OK,
                message="Кулер помещается в корпус",
                component_a="Case",
                component_b="CPU Cooler",
            )
        return CompatibilityResult(
            status=CompatibilityStatus.ERROR,
            message=f"Кулер ({cooler_height}мм) не помещается в корпус (макс. {case_max_cooler_height}мм)",
            component_a="Case",
            component_b="CPU Cooler",
        )

    def check_all_compatibility(
        self,
        build_specs: dict,
    ) -> list[CompatibilityResult]:
        """Run all compatibility checks on a build."""
        results = []

        # CPU-Motherboard check
        if "cpu_socket" in build_specs and "motherboard_socket" in build_specs:
            results.append(
                self.check_cpu_motherboard(
                    build_specs["cpu_socket"],
                    build_specs["motherboard_socket"],
                )
            )

        # Motherboard-RAM check
        if "motherboard_ram_type" in build_specs and "ram_type" in build_specs:
            results.append(
                self.check_motherboard_ram(
                    build_specs["motherboard_ram_type"],
                    build_specs["ram_type"],
                )
            )

        # PSU check
        if "psu_wattage" in build_specs:
            results.append(
                self.check_psu_power(
                    build_specs["psu_wattage"],
                    build_specs.get("gpu_model"),
                    build_specs.get("total_tdp", 200),
                )
            )

        # Case-GPU check
        if "case_max_gpu_length" in build_specs and "gpu_length" in build_specs:
            results.append(
                self.check_case_gpu_length(
                    build_specs["case_max_gpu_length"],
                    build_specs["gpu_length"],
                )
            )

        # Case-Cooler check
        if "case_max_cooler_height" in build_specs and "cooler_height" in build_specs:
            results.append(
                self.check_case_cooler_height(
                    build_specs["case_max_cooler_height"],
                    build_specs["cooler_height"],
                )
            )

        return results

    def get_overall_status(
        self,
        results: list[CompatibilityResult],
    ) -> CompatibilityStatus:
        """Get overall compatibility status from multiple checks."""
        if any(r.status == CompatibilityStatus.ERROR for r in results):
            return CompatibilityStatus.ERROR
        if any(r.status == CompatibilityStatus.WARNING for r in results):
            return CompatibilityStatus.WARNING
        return CompatibilityStatus.OK

    def allocate_budget(
        self,
        total_budget: int,
        purpose: str,
    ) -> dict[str, int]:
        """Allocate budget across components based on purpose."""
        allocation = BUDGET_ALLOCATION.get(purpose, BUDGET_ALLOCATION["general"])
        return {
            component: int(total_budget * percentage)
            for component, percentage in allocation.items()
        }
