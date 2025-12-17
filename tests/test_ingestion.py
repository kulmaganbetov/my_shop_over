"""Tests for data ingestion."""

import pytest
import pandas as pd

from app.tasks.ingestion import (
    parse_csv,
    normalize_product,
    extract_specifications,
)


class TestCSVParsing:
    """Test suite for CSV parsing."""

    def test_parse_csv_valid(self):
        """Test parsing valid CSV content."""
        csv_content = """SKU;КодКаспи;Номенклатура;Поставщик;Остаток;Производитель;КредитРассрочка;БонуснаяЦена;Категория
12345;K123;Intel Core i7-13700K;Supplier1;10;Intel;150000;145000;Процессоры
67890;K456;NVIDIA RTX 4070;Supplier2;5;NVIDIA;250000;240000;Видеокарты"""

        df = parse_csv(csv_content)
        assert len(df) == 2
        assert "sku" in df.columns
        assert "name" in df.columns
        assert df.iloc[0]["sku"] == "12345"

    def test_parse_csv_empty(self):
        """Test parsing empty CSV."""
        csv_content = ""
        df = parse_csv(csv_content)
        assert df.empty


class TestProductNormalization:
    """Test suite for product normalization."""

    def test_normalize_product_full(self):
        """Test normalizing a complete product row."""
        row = pd.Series({
            "sku": "12345",
            "kaspi_code": "K123",
            "name": "Intel Core i7-13700K LGA1700",
            "supplier": "Supplier1",
            "stock": "10",
            "manufacturer": "Intel",
            "price": "150000",
            "discount_price": "145000",
            "category": "Процессоры",
        })

        product = normalize_product(row)

        assert product["sku"] == "12345"
        assert product["kaspi_code"] == "K123"
        assert product["name"] == "Intel Core i7-13700K LGA1700"
        assert product["stock"] == 10
        assert product["price"] == 150000.0
        assert product["discount_price"] == 145000.0
        assert product["component_type"] == "Процессоры"

    def test_normalize_product_with_na(self):
        """Test normalizing product with NA values."""
        row = pd.Series({
            "sku": "12345",
            "kaspi_code": pd.NA,
            "name": "Test Product",
            "supplier": pd.NA,
            "stock": pd.NA,
            "manufacturer": pd.NA,
            "price": pd.NA,
            "discount_price": pd.NA,
            "category": pd.NA,
        })

        product = normalize_product(row)

        assert product["sku"] == "12345"
        assert product["kaspi_code"] is None
        assert product["stock"] == 0
        assert product["price"] is None


class TestSpecificationExtraction:
    """Test suite for specification extraction."""

    def test_extract_cpu_socket_intel(self):
        """Test extracting Intel CPU socket."""
        specs = extract_specifications("Intel Core i7-13700K LGA1700", "Процессоры")
        assert specs.get("socket") == "LGA1700"

    def test_extract_cpu_socket_amd(self):
        """Test extracting AMD CPU socket."""
        specs = extract_specifications("AMD Ryzen 9 7950X AM5", "Процессоры")
        assert specs.get("socket") == "AM5"

    def test_extract_ram_type_ddr5(self):
        """Test extracting DDR5 RAM type."""
        specs = extract_specifications("Kingston Fury DDR5 32GB 6000MHz", "Оперативная память")
        assert specs.get("ram_type") == "DDR5"
        assert specs.get("type") == "DDR5"

    def test_extract_ram_type_ddr4(self):
        """Test extracting DDR4 RAM type."""
        specs = extract_specifications("Corsair Vengeance DDR4 16GB", "Оперативная память")
        assert specs.get("ram_type") == "DDR4"

    def test_extract_psu_wattage(self):
        """Test extracting PSU wattage."""
        specs = extract_specifications("Seasonic Focus GX-850 850W Gold", "Блоки питания")
        assert specs.get("wattage") == 850

    def test_extract_gpu_vram(self):
        """Test extracting GPU VRAM."""
        specs = extract_specifications("NVIDIA RTX 4070 12GB", "Видеокарты")
        assert specs.get("vram_gb") == 12

    def test_extract_storage_tb(self):
        """Test extracting storage capacity in TB."""
        specs = extract_specifications("Samsung 990 Pro 2TB NVMe", "SSD накопители")
        assert specs.get("capacity_gb") == 2048

    def test_extract_storage_gb(self):
        """Test extracting storage capacity in GB."""
        specs = extract_specifications("Kingston A400 480GB SSD", "SSD накопители")
        assert specs.get("capacity_gb") == 480
