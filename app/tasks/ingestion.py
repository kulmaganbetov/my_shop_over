"""Product ingestion tasks from FTP CSV files.

DATA PIPELINE FLOW:
==================
1. FTP Connection -> Download CSV file from over-shop.kz FTP server
2. CSV Parsing -> Convert CSV to pandas DataFrame
3. Data Normalization -> Clean and transform each row
4. Database Upsert -> Insert new products or update existing ones
5. Trigger Embeddings -> Queue embedding generation for new products

This module is triggered by:
- Celery Beat scheduler (every hour)
- Manual API call: POST /api/v1/admin/sync/products
"""

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

# Configure logger for this module
logger = logging.getLogger(__name__)

# Category to component type mapping for PC parts
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
    """
    STEP 1: Download CSV file from FTP server.

    Connects to: ftp://over-shop.kz
    Downloads: Dealer.csv
    Returns: CSV content as string or None if failed
    """
    logger.info("=" * 60)
    logger.info("STEP 1: DOWNLOADING CSV FROM FTP")
    logger.info("=" * 60)
    logger.info(f"  FTP Host: {settings.ftp_host}")
    logger.info(f"  FTP Port: {settings.ftp_port}")
    logger.info(f"  FTP User: {settings.ftp_user}")
    logger.info(f"  CSV File: {settings.ftp_csv_filename}")

    try:
        logger.info("  Connecting to FTP server...")
        ftp = ftplib.FTP()
        ftp.connect(settings.ftp_host, settings.ftp_port)
        logger.info("  ✓ Connected to FTP server")

        logger.info("  Authenticating...")
        ftp.login(settings.ftp_user, settings.ftp_password)
        logger.info("  ✓ Authentication successful")

        # List files in directory (for debugging)
        logger.info("  Listing FTP directory contents...")
        files = ftp.nlst()
        logger.info(f"  Found {len(files)} files: {files[:10]}{'...' if len(files) > 10 else ''}")

        # Download file to memory
        logger.info(f"  Downloading {settings.ftp_csv_filename}...")
        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {settings.ftp_csv_filename}", buffer.write)
        ftp.quit()

        buffer.seek(0)
        content = buffer.read().decode("utf-8-sig")

        logger.info(f"  ✓ Downloaded successfully!")
        logger.info(f"  File size: {len(content):,} bytes")
        logger.info(f"  First 200 chars: {content[:200]}...")

        return content

    except ftplib.error_perm as e:
        logger.error(f"  ✗ FTP permission error: {e}")
        return None
    except ftplib.error_temp as e:
        logger.error(f"  ✗ FTP temporary error: {e}")
        return None
    except ConnectionRefusedError:
        logger.error(f"  ✗ Connection refused - FTP server may be down")
        return None
    except Exception as e:
        logger.error(f"  ✗ FTP download failed: {type(e).__name__}: {e}")
        return None


def parse_csv(csv_content: str) -> pd.DataFrame:
    """
    STEP 2: Parse CSV content into pandas DataFrame.

    CSV Format (semicolon-separated):
    - SKU: Product SKU
    - КодКаспи: Kaspi marketplace code
    - Номенклатура: Product name
    - Поставщик: Supplier
    - Остаток: Stock quantity
    - Производитель: Manufacturer
    - КредитРассрочка: Main price
    - БонуснаяЦена: Discount price
    - Категория: Category
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: PARSING CSV")
    logger.info("=" * 60)

    try:
        logger.info("  Reading CSV with pandas...")
        df = pd.read_csv(
            io.StringIO(csv_content),
            sep=";",
            encoding="utf-8",
            on_bad_lines="skip",
        )

        logger.info(f"  ✓ Read {len(df)} rows")
        logger.info(f"  Original columns: {list(df.columns)}")

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
        logger.info(f"  Renamed columns: {list(df.columns)}")

        # Show sample data
        logger.info("")
        logger.info("  Sample data (first 3 rows):")
        for i, row in df.head(3).iterrows():
            logger.info(f"    Row {i}: SKU={row.get('sku')}, Name={str(row.get('name'))[:50]}...")

        # Show statistics
        logger.info("")
        logger.info("  Statistics:")
        logger.info(f"    Total products: {len(df)}")
        logger.info(f"    Unique categories: {df['category'].nunique() if 'category' in df.columns else 'N/A'}")
        logger.info(f"    Products with price: {df['price'].notna().sum() if 'price' in df.columns else 'N/A'}")

        return df

    except pd.errors.EmptyDataError:
        logger.error("  ✗ CSV file is empty")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"  ✗ CSV parsing error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"  ✗ CSV parsing failed: {type(e).__name__}: {e}")
        return pd.DataFrame()


def normalize_product(row: pd.Series) -> dict:
    """
    STEP 3: Normalize a single product row.

    - Cleans string values
    - Converts numeric values
    - Extracts specifications from product name
    - Maps category to component type
    """
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

    # Extract specifications from name
    product["specifications"] = extract_specifications(
        product["name"],
        product.get("category", "")
    )

    return product


def extract_specifications(name: str, category: str) -> dict:
    """Extract technical specifications from product name using regex."""
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
    """
    MAIN TASK: Sync products from FTP to PostgreSQL.

    This is the main entry point for product synchronization.
    Called by:
    - Celery Beat scheduler (hourly)
    - POST /api/v1/admin/sync/products

    Flow:
    1. Download CSV from FTP
    2. Parse CSV to DataFrame
    3. For each product:
       - Normalize data
       - Insert or update in PostgreSQL
    4. Return statistics
    """
    logger.info("")
    logger.info("*" * 60)
    logger.info("*  PRODUCT SYNC TASK STARTED")
    logger.info("*" * 60)
    logger.info("")

    # STEP 1: Download CSV
    csv_content = download_csv_from_ftp()
    if not csv_content:
        logger.error("Failed to download CSV, will retry in 60 seconds")
        self.retry(countdown=60)
        return {"status": "error", "message": "Failed to download CSV"}

    # STEP 2: Parse CSV
    df = parse_csv(csv_content)
    if df.empty:
        logger.error("CSV is empty or invalid")
        return {"status": "error", "message": "Empty or invalid CSV"}

    # STEP 3 & 4: Normalize and save to database
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 & 4: SAVING TO POSTGRESQL")
    logger.info("=" * 60)
    logger.info(f"  Database URL: {settings.database_url_sync[:50]}...")

    engine = create_engine(settings.database_url_sync)

    created = 0
    updated = 0
    errors = 0

    logger.info(f"  Processing {len(df)} products...")

    with Session(engine) as session:
        for idx, row in df.iterrows():
            try:
                product_data = normalize_product(row)

                if not product_data["sku"]:
                    logger.warning(f"    Skipping row {idx}: no SKU")
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
                    if updated <= 3:
                        logger.info(f"    Updated: SKU={product_data['sku']}, Name={product_data['name'][:30]}...")
                else:
                    # Create new product
                    product = Product(**product_data)
                    session.add(product)
                    created += 1
                    if created <= 3:
                        logger.info(f"    Created: SKU={product_data['sku']}, Name={product_data['name'][:30]}...")

                # Progress log every 100 items
                if (idx + 1) % 100 == 0:
                    logger.info(f"    Progress: {idx + 1}/{len(df)} processed...")

            except Exception as e:
                logger.error(f"    Error processing row {idx}: {type(e).__name__}: {e}")
                errors += 1

        logger.info("  Committing transaction...")
        session.commit()
        logger.info("  ✓ Transaction committed")

    # Summary
    result = {
        "status": "success",
        "created": created,
        "updated": updated,
        "errors": errors,
        "total": len(df),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("SYNC COMPLETED")
    logger.info("=" * 60)
    logger.info(f"  ✓ Created: {created} new products")
    logger.info(f"  ✓ Updated: {updated} existing products")
    logger.info(f"  ✗ Errors: {errors}")
    logger.info(f"  Total processed: {len(df)}")
    logger.info("")
    logger.info("*" * 60)
    logger.info("*  PRODUCT SYNC TASK FINISHED")
    logger.info("*" * 60)

    return result


@celery_app.task
def import_products_from_csv(csv_path: str):
    """Import products from a local CSV file (for testing)."""
    logger.info(f"Importing products from local file: {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            csv_content = f.read()
        logger.info(f"  Read {len(csv_content)} bytes from {csv_path}")
    except FileNotFoundError:
        logger.error(f"  File not found: {csv_path}")
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

    logger.info(f"Import completed: created={created}, updated={updated}")
    return {"status": "success", "created": created, "updated": updated}
