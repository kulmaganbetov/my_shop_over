"""Tests for PC compatibility engine."""

import pytest

from app.services.compatibility import (
    CompatibilityEngine,
    CompatibilityStatus,
    BUDGET_ALLOCATION,
)


class TestCompatibilityEngine:
    """Test suite for compatibility engine."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engine = CompatibilityEngine()

    def test_cpu_motherboard_compatible(self):
        """Test compatible CPU and motherboard."""
        result = self.engine.check_cpu_motherboard("LGA1700", "LGA1700")
        assert result.status == CompatibilityStatus.OK

    def test_cpu_motherboard_incompatible(self):
        """Test incompatible CPU and motherboard."""
        result = self.engine.check_cpu_motherboard("LGA1700", "AM5")
        assert result.status == CompatibilityStatus.ERROR

    def test_motherboard_ram_compatible(self):
        """Test compatible motherboard and RAM."""
        result = self.engine.check_motherboard_ram("DDR5", "DDR5")
        assert result.status == CompatibilityStatus.OK

    def test_motherboard_ram_incompatible(self):
        """Test incompatible motherboard and RAM."""
        result = self.engine.check_motherboard_ram("DDR5", "DDR4")
        assert result.status == CompatibilityStatus.ERROR

    def test_psu_power_sufficient(self):
        """Test sufficient PSU power."""
        result = self.engine.check_psu_power(850, "RTX 4090", 300)
        assert result.status == CompatibilityStatus.OK

    def test_psu_power_insufficient(self):
        """Test insufficient PSU power."""
        result = self.engine.check_psu_power(500, "RTX 4090", 300)
        assert result.status == CompatibilityStatus.ERROR

    def test_psu_power_warning(self):
        """Test borderline PSU power."""
        result = self.engine.check_psu_power(780, "RTX 4090", 300)
        assert result.status == CompatibilityStatus.WARNING

    def test_case_gpu_fits(self):
        """Test GPU fits in case."""
        result = self.engine.check_case_gpu_length(350, 300)
        assert result.status == CompatibilityStatus.OK

    def test_case_gpu_too_long(self):
        """Test GPU too long for case."""
        result = self.engine.check_case_gpu_length(300, 350)
        assert result.status == CompatibilityStatus.ERROR

    def test_budget_allocation_gaming(self):
        """Test budget allocation for gaming."""
        allocation = self.engine.allocate_budget(500000, "gaming")
        assert allocation["gpu"] > allocation["cpu"]  # GPU priority for gaming
        assert sum(allocation.values()) <= 500000

    def test_budget_allocation_work(self):
        """Test budget allocation for work."""
        allocation = self.engine.allocate_budget(500000, "work")
        assert allocation["cpu"] > allocation["gpu"]  # CPU priority for work

    def test_overall_status_ok(self):
        """Test overall status when all checks pass."""
        results = [
            self.engine.check_cpu_motherboard("LGA1700", "LGA1700"),
            self.engine.check_motherboard_ram("DDR5", "DDR5"),
        ]
        status = self.engine.get_overall_status(results)
        assert status == CompatibilityStatus.OK

    def test_overall_status_error(self):
        """Test overall status when one check fails."""
        results = [
            self.engine.check_cpu_motherboard("LGA1700", "LGA1700"),
            self.engine.check_motherboard_ram("DDR5", "DDR4"),  # Error
        ]
        status = self.engine.get_overall_status(results)
        assert status == CompatibilityStatus.ERROR

    def test_check_all_compatibility(self):
        """Test full compatibility check."""
        build_specs = {
            "cpu_socket": "LGA1700",
            "motherboard_socket": "LGA1700",
            "motherboard_ram_type": "DDR5",
            "ram_type": "DDR5",
            "psu_wattage": 750,
            "gpu_model": "RTX 4070",
            "total_tdp": 300,
        }
        results = self.engine.check_all_compatibility(build_specs)
        assert len(results) == 3  # CPU-MB, MB-RAM, PSU
        assert all(r.status == CompatibilityStatus.OK for r in results)
