"""Product ingestion tasks from FTP CSV files.

DATA PIPELINE FLOW:
==================
1. FTP Connection -> Download CSV file from over-shop.kz FTP server
2. CSV Parsing -> Convert CSV to pandas DataFrame
3. Smart Sync -> Only update price/stock if name unchanged (preserve specs!)
4. Zero Missing -> Products NOT in FTP file get stock = 0
5. Specs Enrichment -> Extract specs for products with empty specs field
6. Trigger Embeddings -> Queue embedding generation for new products

SMART SYNC RULES:
- If product exists AND name matches: only update price, discount_price, stock
- If product exists AND name changed: update all fields, reset specs for re-parsing
- If product is new: insert with all fields
- Products NOT in FTP file: set stock = 0 (so bot doesn't offer them)

This module is triggered by:
- Celery Beat scheduler (every hour)
- Manual API call: POST /api/v1/admin/sync/products
"""

import ftplib
import io
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Set

import pandas as pd
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Product, SyncStatus
from app.tasks.celery_app import celery_app

# Configure logger for this module
logger = logging.getLogger(__name__)

# Minimum valid price (filter out admin errors)
MIN_VALID_PRICE = 100

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
    "Твердотельные диски (SSD)": "SSD накопители",
    "Жесткие диски": "Жесткие диски",
    "HDD": "Жесткие диски",
    "Блоки питания": "Блоки питания",
    "PSU": "Блоки питания",
    "Корпуса": "Корпуса",
    "Case": "Корпуса",
    "Кулеры": "Кулеры",
    "Кулеры для процессоров": "Кулеры",
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
        raw_bytes = buffer.read()

        # Try different encodings (cp1251 is common for Russian/Kazakh files)
        encodings = ["cp1251", "utf-8-sig", "utf-8", "windows-1251", "koi8-r"]
        content = None

        for encoding in encodings:
            try:
                content = raw_bytes.decode(encoding)
                logger.info(f"  ✓ Decoded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.debug(f"  Failed to decode with {encoding}, trying next...")
                continue

        if content is None:
            logger.error("  ✗ Could not decode file with any known encoding")
            return None

        logger.info(f"  ✓ Downloaded successfully!")
        logger.info(f"  File size: {len(content):,} characters")
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
    - Maps category to component type
    NOTE: Does NOT extract specs here - that's done separately for efficiency
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

    return product


def extract_specifications(name: str, category: str) -> dict:
    """Extract technical specifications from product name using regex.

    CRITICAL: Used for specs enrichment after sync.
    """
    specs = {}
    name_lower = name.lower()

    # CPU Socket detection
    socket_patterns = [
        (r"lga\s*1851", "LGA1851"),
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
    if vram_match and ("видеокарт" in category.lower() or "gpu" in category.lower()):
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


def update_sync_status(session: Session, sync_type: str, **kwargs):
    """Update or create sync status record."""
    existing = session.query(SyncStatus).filter(
        SyncStatus.sync_type == sync_type
    ).first()

    if existing:
        for key, value in kwargs.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
    else:
        sync_status = SyncStatus(sync_type=sync_type, **kwargs)
        session.add(sync_status)

    session.commit()


def get_sync_status(session: Session, sync_type: str) -> Optional[SyncStatus]:
    """Get sync status record."""
    return session.query(SyncStatus).filter(
        SyncStatus.sync_type == sync_type
    ).first()


@celery_app.task(bind=True, max_retries=3)
def sync_products_from_ftp(self):
    """
    MAIN TASK: Sync products from FTP to PostgreSQL with SMART UPDATE.

    SMART SYNC RULES:
    1. If SKU exists and name matches: only update price, discount_price, stock (PRESERVE SPECS!)
    2. If SKU exists and name changed: update all fields, reset specs
    3. If SKU is new: insert all fields
    4. All SKUs NOT in FTP file: set stock = 0

    This is the main entry point for product synchronization.
    Called by:
    - Celery Beat scheduler (hourly)
    - POST /api/v1/admin/sync/products
    """
    logger.info("")
    logger.info("*" * 60)
    logger.info("*  PRODUCT SYNC TASK STARTED (SMART MODE)")
    logger.info("*" * 60)
    logger.info("")

    engine = create_engine(settings.database_url_sync)

    # Mark sync as started
    with Session(engine) as session:
        update_sync_status(
            session, "ftp_products",
            status="running",
            last_sync_at=datetime.now(timezone.utc),
            error_message=None,
        )

    # STEP 1: Download CSV
    csv_content = download_csv_from_ftp()
    if not csv_content:
        logger.error("Failed to download CSV, will retry in 60 seconds")
        with Session(engine) as session:
            update_sync_status(
                session, "ftp_products",
                status="failed",
                error_message="Failed to download CSV from FTP",
            )
        self.retry(countdown=60)
        return {"status": "error", "message": "Failed to download CSV"}

    # STEP 2: Parse CSV
    df = parse_csv(csv_content)
    if df.empty:
        logger.error("CSV is empty or invalid")
        with Session(engine) as session:
            update_sync_status(
                session, "ftp_products",
                status="failed",
                error_message="CSV is empty or invalid",
            )
        return {"status": "error", "message": "Empty or invalid CSV"}

    # STEP 3 & 4: SMART SYNC to database
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 & 4: SMART SYNC TO POSTGRESQL")
    logger.info("=" * 60)
    logger.info(f"  Database URL: {settings.database_url_sync[:50]}...")

    created = 0
    updated_price_only = 0
    updated_full = 0
    zeroed = 0
    skipped_invalid_price = 0
    errors = 0

    # Collect all SKUs from FTP file
    ftp_skus: Set[str] = set()

    logger.info(f"  Processing {len(df)} products...")

    with Session(engine) as session:
        for idx, row in df.iterrows():
            try:
                product_data = normalize_product(row)

                if not product_data["sku"]:
                    logger.warning(f"    Skipping row {idx}: no SKU")
                    continue

                sku = product_data["sku"]
                ftp_skus.add(sku)

                # Skip products with invalid price (admin errors)
                price = product_data.get("price") or product_data.get("discount_price") or 0
                if price < MIN_VALID_PRICE:
                    skipped_invalid_price += 1
                    continue

                # Check if product exists
                existing = session.query(Product).filter(
                    Product.sku == sku
                ).first()

                if existing:
                    # SMART UPDATE: Check if name changed
                    old_name = (existing.name or "").strip()
                    new_name = (product_data["name"] or "").strip()

                    if old_name == new_name:
                        # Name unchanged: only update price and stock (PRESERVE SPECS!)
                        existing.price = product_data["price"]
                        existing.discount_price = product_data["discount_price"]
                        existing.stock = product_data["stock"]
                        existing.is_active = True
                        updated_price_only += 1
                        if updated_price_only <= 3:
                            logger.info(f"    [PRICE] SKU={sku}: price={product_data['price']}, stock={product_data['stock']}")
                    else:
                        # Name changed: full update + reset specs for re-parsing
                        for key, value in product_data.items():
                            if hasattr(existing, key):
                                setattr(existing, key, value)
                        # CRITICAL: Reset specs so enrichment will re-parse
                        existing.specifications = {}
                        updated_full += 1
                        if updated_full <= 3:
                            logger.info(f"    [FULL] SKU={sku}: name changed, specs reset")
                else:
                    # Create new product (specs will be enriched later)
                    product_data["specifications"] = {}  # Empty, will be enriched
                    product = Product(**product_data)
                    session.add(product)
                    created += 1
                    if created <= 3:
                        logger.info(f"    [NEW] SKU={sku}, Name={product_data['name'][:30]}...")

                # Progress log every 500 items
                if (idx + 1) % 500 == 0:
                    logger.info(f"    Progress: {idx + 1}/{len(df)} processed...")

            except Exception as e:
                logger.error(f"    Error processing row {idx}: {type(e).__name__}: {e}")
                errors += 1

        # STEP 5: Zero out products NOT in FTP file
        logger.info("")
        logger.info("  Setting stock=0 for products not in FTP file...")

        # Find products that exist in DB but not in FTP file
        existing_skus_result = session.execute(
            select(Product.sku).where(
                Product.stock > 0,
                ~Product.sku.in_(ftp_skus) if ftp_skus else True
            )
        )
        missing_skus = [row[0] for row in existing_skus_result.all()]

        if missing_skus:
            # Update in batches for efficiency
            for i in range(0, len(missing_skus), 1000):
                batch = missing_skus[i:i+1000]
                session.execute(
                    update(Product)
                    .where(Product.sku.in_(batch))
                    .values(stock=0)
                )
            zeroed = len(missing_skus)
            logger.info(f"  ✓ Zeroed {zeroed} products not in FTP file")

        logger.info("  Committing transaction...")
        session.commit()
        logger.info("  ✓ Transaction committed")

        # Update sync status
        update_sync_status(
            session, "ftp_products",
            status="success",
            last_success_at=datetime.now(timezone.utc),
            products_created=created,
            products_updated=updated_price_only + updated_full,
            products_zeroed=zeroed,
            errors=errors,
            error_message=None,
        )

    # STEP 6: Trigger specs enrichment for products with empty specs
    logger.info("")
    logger.info("  Triggering specs enrichment task...")
    try:
        enrich_product_specs.delay()
        logger.info("  ✓ Specs enrichment task queued")
    except Exception as e:
        logger.warning(f"  Could not queue enrichment task: {e}")

    # Summary
    result = {
        "status": "success",
        "created": created,
        "updated_price_only": updated_price_only,
        "updated_full": updated_full,
        "zeroed": zeroed,
        "skipped_invalid_price": skipped_invalid_price,
        "errors": errors,
        "total_processed": len(df),
    }

    logger.info("")
    logger.info("=" * 60)
    logger.info("SYNC COMPLETED")
    logger.info("=" * 60)
    logger.info(f"  ✓ Created: {created} new products")
    logger.info(f"  ✓ Updated (price only): {updated_price_only}")
    logger.info(f"  ✓ Updated (full): {updated_full}")
    logger.info(f"  ✓ Zeroed (not in FTP): {zeroed}")
    logger.info(f"  ⚠ Skipped (invalid price): {skipped_invalid_price}")
    logger.info(f"  ✗ Errors: {errors}")
    logger.info(f"  Total processed: {len(df)}")
    logger.info("")
    logger.info("*" * 60)
    logger.info("*  PRODUCT SYNC TASK FINISHED")
    logger.info("*" * 60)

    return result


@celery_app.task
def enrich_product_specs():
    """
    SPECS ENRICHMENT: Parse specs for products with empty specifications.

    Run after sync to extract specs from product names.
    Critical for: CPUs (socket), Motherboards (socket, RAM type), GPUs (VRAM).
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("SPECS ENRICHMENT TASK")
    logger.info("=" * 60)

    engine = create_engine(settings.database_url_sync)
    enriched = 0
    marked_invalid = 0

    # Categories that MUST have specs for build compatibility
    critical_categories = [
        "Процессоры",
        "Материнские платы",
        "Видеокарты",
        "Оперативная память",
        "Блоки питания",
    ]

    with Session(engine) as session:
        # Find products with empty or null specifications
        products = session.query(Product).filter(
            Product.stock > 0,
            Product.is_active == True,
            (Product.specifications == None) | (Product.specifications == {})
        ).limit(1000).all()

        logger.info(f"  Found {len(products)} products needing specs enrichment")

        for product in products:
            try:
                specs = extract_specifications(
                    product.name or "",
                    product.category or ""
                )

                if specs:
                    product.specifications = specs
                    enriched += 1
                    if enriched <= 5:
                        logger.info(f"    Enriched: {product.sku} -> {specs}")
                else:
                    # Check if this is a critical category without specs
                    is_critical = any(
                        cat.lower() in (product.category or "").lower()
                        for cat in critical_categories
                    )
                    if is_critical:
                        # Mark as invalid for build (missing critical specs)
                        product.specifications = {"invalid_for_build": True}
                        marked_invalid += 1
                        if marked_invalid <= 5:
                            logger.info(f"    Marked invalid: {product.sku} (no specs found)")

            except Exception as e:
                logger.error(f"    Error enriching {product.sku}: {e}")

        session.commit()

    logger.info("")
    logger.info(f"  ✓ Enriched: {enriched} products")
    logger.info(f"  ⚠ Marked invalid for build: {marked_invalid} products")
    logger.info("=" * 60)

    return {"enriched": enriched, "marked_invalid": marked_invalid}


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

            # Skip invalid prices
            price = product_data.get("price") or product_data.get("discount_price") or 0
            if price < MIN_VALID_PRICE:
                continue

            existing = session.query(Product).filter(
                Product.sku == product_data["sku"]
            ).first()

            if existing:
                # Smart update
                if existing.name == product_data["name"]:
                    # Name unchanged: only price/stock
                    existing.price = product_data["price"]
                    existing.discount_price = product_data["discount_price"]
                    existing.stock = product_data["stock"]
                else:
                    # Name changed: full update
                    for key, value in product_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    existing.specifications = {}
                updated += 1
            else:
                product = Product(**product_data)
                session.add(product)
                created += 1

        session.commit()

    logger.info(f"Import completed: created={created}, updated={updated}")
    return {"status": "success", "created": created, "updated": updated}
