"""Product ingestion tasks from FTP CSV files."""

import ftplib
import io
import logging
import re
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Product
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Category to component type mapping
COMPONENT_TYPE_MAPPING = {
    "Процессоры": "Процессоры",
    "CPU": "Процессоры",
    "Видеокарты": "Видеокарты",
    "GPU": "Видеокарты",
    "Материнские платы": "Материнские платы",
    "Motherboard": "Материнские платы",
    "Оперативная память": "Оперативная память",
    "RAM": "Оперативная память",
    "SSD": "SSD накопители",
    "SSD накопители": "SSD накопители",
    "Жесткие диски": "Жесткие диски",
    "HDD": "Жесткие диски",
    "Блоки питания": "Блоки питания",
    "PSU": "Блоки питания",
    "Корпуса": "Корпуса",
    "Case": "Корпуса",
    "Кулеры": "Кулеры",
    "Cooler": "Кулеры",
    "Охлаждение": "Кулеры",
}


def download_csv_from_ftp() -> Optional[str]:
    """Download CSV file from FTP server."""
    try:
        ftp = ftplib.FTP()
        ftp.connect(settings.ftp_host, settings.ftp_port)
        ftp.login(settings.ftp_user, settings.ftp_password)

        # Download file to memory
        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {settings.ftp_csv_filename}", buffer.write)
        ftp.quit()

        buffer.seek(0)
        content = buffer.read().decode("utf-8-sig")
        logger.info(f"Downloaded {len(content)} bytes from FTP")
        return content

    except Exception as e:
        logger.error(f"FTP download failed: {e}")
        return None


def parse_csv(csv_content: str) -> pd.DataFrame:
    """Parse CSV content into DataFrame."""
    try:
        df = pd.read_csv(
            io.StringIO(csv_content),
            sep=";",
            encoding="utf-8",
            on_bad_lines="skip",
        )

        # Rename columns to English
        column_mapping = {
            "SKU": "sku",
            "КодКаспи": "kaspi_code",
            "Номенклатура": "name",
            "Поставщик": "supplier",
            "Остаток": "stock",
            "Производитель": "manufacturer",
            "КредитРассрочка": "price",
            "БонуснаяЦена": "discount_price",
            "Категория": "category",
        }

        df = df.rename(columns=column_mapping)
        logger.info(f"Parsed {len(df)} products from CSV")
        return df

    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        return pd.DataFrame()


def normalize_product(row: pd.Series) -> dict:
    """Normalize a product row into database format."""
    # Clean and convert values
    def clean_number(val, default=0):
        if pd.isna(val):
            return default
        try:
            return float(str(val).replace(",", ".").replace(" ", ""))
        except (ValueError, TypeError):
            return default

    def clean_string(val, default=""):
        if pd.isna(val):
            return default
        return str(val).strip()

    product = {
        "sku": clean_string(row.get("sku")),
        "kaspi_code": clean_string(row.get("kaspi_code")) or None,
        "name": clean_string(row.get("name")),
        "supplier": clean_string(row.get("supplier")) or None,
        "stock": int(clean_number(row.get("stock"), 0)),
        "manufacturer": clean_string(row.get("manufacturer")) or None,
        "price": clean_number(row.get("price")) or None,
        "discount_price": clean_number(row.get("discount_price")) or None,
        "category": clean_string(row.get("category")) or None,
        "is_active": True,
    }

    # Determine component type from category
    category = product.get("category", "")
    for cat_pattern, component_type in COMPONENT_TYPE_MAPPING.items():
        if cat_pattern.lower() in category.lower():
            product["component_type"] = component_type
            break

    # Extract specifications from name (basic extraction)
    product["specifications"] = extract_specifications(product["name"], product.get("category", ""))

    return product


def extract_specifications(name: str, category: str) -> dict:
    """Extract technical specifications from product name.

    This is a basic extraction - in production, you'd want more sophisticated parsing.
    """
    specs = {}
    name_lower = name.lower()

    # CPU Socket detection
    socket_patterns = [
        (r"lga\s*1700", "LGA1700"),
        (r"lga\s*1200", "LGA1200"),
        (r"lga\s*1151", "LGA1151"),
        (r"am5", "AM5"),
        (r"am4", "AM4"),
    ]
    for pattern, socket in socket_patterns:
        if re.search(pattern, name_lower):
            specs["socket"] = socket
            break

    # RAM type detection
    if "ddr5" in name_lower:
        specs["ram_type"] = "DDR5"
        specs["type"] = "DDR5"
    elif "ddr4" in name_lower:
        specs["ram_type"] = "DDR4"
        specs["type"] = "DDR4"

    # PSU wattage detection
    wattage_match = re.search(r"(\d{3,4})\s*w", name_lower)
    if wattage_match:
        specs["wattage"] = int(wattage_match.group(1))

    # GPU memory detection
    vram_match = re.search(r"(\d{1,2})\s*gb", name_lower)
    if vram_match and "видеокарт" in category.lower():
        specs["vram_gb"] = int(vram_match.group(1))

    # Storage capacity detection
    storage_patterns = [
        (r"(\d+)\s*tb", lambda m: int(m.group(1)) * 1024),
        (r"(\d{3,4})\s*gb", lambda m: int(m.group(1))),
    ]
    for pattern, converter in storage_patterns:
        match = re.search(pattern, name_lower)
        if match:
            specs["capacity_gb"] = converter(match)
            break

    return specs


@celery_app.task(bind=True, max_retries=3)
def sync_products_from_ftp(self):
    """Sync products from FTP CSV file."""
    logger.info("Starting product sync from FTP")

    # Download CSV
    csv_content = download_csv_from_ftp()
    if not csv_content:
        self.retry(countdown=60)
        return {"status": "error", "message": "Failed to download CSV"}

    # Parse CSV
    df = parse_csv(csv_content)
    if df.empty:
        return {"status": "error", "message": "Empty or invalid CSV"}

    # Create sync database session
    engine = create_engine(settings.database_url_sync)

    created = 0
    updated = 0
    errors = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            try:
                product_data = normalize_product(row)

                if not product_data["sku"]:
                    continue

                # Check if product exists
                existing = session.query(Product).filter(
                    Product.sku == product_data["sku"]
                ).first()

                if existing:
                    # Update existing product
                    for key, value in product_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    updated += 1
                else:
                    # Create new product
                    product = Product(**product_data)
                    session.add(product)
                    created += 1

            except Exception as e:
                logger.error(f"Error processing product: {e}")
                errors += 1

        session.commit()

    result = {
        "status": "success",
        "created": created,
        "updated": updated,
        "errors": errors,
        "total": len(df),
    }
    logger.info(f"Product sync completed: {result}")
    return result


@celery_app.task
def import_products_from_csv(csv_path: str):
    """Import products from a local CSV file."""
    logger.info(f"Importing products from {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            csv_content = f.read()
    except FileNotFoundError:
        return {"status": "error", "message": f"File not found: {csv_path}"}

    df = parse_csv(csv_content)
    if df.empty:
        return {"status": "error", "message": "Empty or invalid CSV"}

    engine = create_engine(settings.database_url_sync)

    created = 0
    updated = 0

    with Session(engine) as session:
        for _, row in df.iterrows():
            product_data = normalize_product(row)

            if not product_data["sku"]:
                continue

            existing = session.query(Product).filter(
                Product.sku == product_data["sku"]
            ).first()

            if existing:
                for key, value in product_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                updated += 1
            else:
                product = Product(**product_data)
                session.add(product)
                created += 1

        session.commit()

    return {"status": "success", "created": created, "updated": updated}
