#!/usr/bin/env python3
"""
TEST DATA PIPELINE SCRIPT
=========================

Этот скрипт показывает как работает весь пайплайн данных:

1. СКАЧИВАНИЕ CSV с FTP сервера over-shop.kz
2. ПАРСИНГ CSV и нормализация данных
3. СОХРАНЕНИЕ в PostgreSQL
4. ГЕНЕРАЦИЯ EMBEDDINGS через OpenAI API
5. СОХРАНЕНИЕ EMBEDDINGS в pgvector

Запуск:
    python scripts/test_pipeline.py

Требования:
    - Docker контейнеры должны быть запущены (PostgreSQL, Redis)
    - Переменные окружения должны быть настроены (.env файл)
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_header(text: str):
    """Print a formatted header."""
    print("\n")
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num: int, text: str):
    """Print a step header."""
    print(f"\n{'─' * 70}")
    print(f"  STEP {step_num}: {text}")
    print(f"{'─' * 70}\n")


def test_database_connection():
    """Test PostgreSQL connection."""
    print_step(1, "TESTING DATABASE CONNECTION")

    from sqlalchemy import create_engine, text
    from app.core.config import settings

    print(f"  Database URL: {settings.database_url_sync[:50]}...")

    try:
        engine = create_engine(settings.database_url_sync)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"  ✓ Connected to PostgreSQL!")
            print(f"  Version: {version[:50]}...")

            # Check pgvector extension
            result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
            if result.scalar():
                print(f"  ✓ pgvector extension installed")
            else:
                print(f"  ✗ pgvector extension NOT installed!")
                return False

        return True
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


def test_ftp_connection():
    """Test FTP connection and download."""
    print_step(2, "TESTING FTP CONNECTION")

    from app.tasks.ingestion import download_csv_from_ftp

    csv_content = download_csv_from_ftp()

    if csv_content:
        print(f"  ✓ Downloaded CSV successfully!")
        print(f"  Size: {len(csv_content):,} bytes")
        lines = csv_content.split('\n')
        print(f"  Lines: {len(lines)}")
        print(f"  Header: {lines[0][:100]}...")
        return csv_content
    else:
        print(f"  ✗ FTP download failed!")
        return None


def test_csv_parsing(csv_content: str):
    """Test CSV parsing."""
    print_step(3, "TESTING CSV PARSING")

    from app.tasks.ingestion import parse_csv

    df = parse_csv(csv_content)

    if not df.empty:
        print(f"  ✓ Parsed {len(df)} products")
        print(f"\n  Columns: {list(df.columns)}")
        print(f"\n  Sample product:")
        sample = df.iloc[0]
        for col in ['sku', 'name', 'category', 'price', 'stock']:
            if col in df.columns:
                print(f"    {col}: {sample.get(col)}")
        return df
    else:
        print(f"  ✗ CSV parsing failed!")
        return None


def test_product_save(df):
    """Test saving products to database."""
    print_step(4, "TESTING PRODUCT SAVE TO POSTGRESQL")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    from app.db.models import Product
    from app.tasks.ingestion import normalize_product

    engine = create_engine(settings.database_url_sync)

    # Test with first 5 products
    test_count = min(5, len(df))
    created = 0
    updated = 0

    print(f"  Testing with {test_count} products...")

    with Session(engine) as session:
        for i, (_, row) in enumerate(df.head(test_count).iterrows()):
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
                print(f"    Updated: {product_data['sku']} - {product_data['name'][:40]}...")
            else:
                product = Product(**product_data)
                session.add(product)
                created += 1
                print(f"    Created: {product_data['sku']} - {product_data['name'][:40]}...")

        session.commit()

    print(f"\n  ✓ Created: {created}, Updated: {updated}")

    # Show database stats
    with Session(engine) as session:
        total = session.query(Product).count()
        print(f"  Total products in database: {total}")

    return True


def test_embedding_generation():
    """Test embedding generation."""
    print_step(5, "TESTING EMBEDDING GENERATION")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    from app.db.models import Product, ProductEmbedding
    from app.tasks.embeddings import create_product_text
    from app.llm.client import get_llm_client
    import asyncio

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        # Get first product without embedding
        product = session.query(Product).first()

        if not product:
            print("  ✗ No products in database!")
            return False

        print(f"  Testing with product: {product.name[:50]}...")

        # Create text for embedding
        text = create_product_text(product)
        print(f"  Embedding text: {text[:100]}...")

        # Call OpenAI API
        print(f"  Calling OpenAI API for embedding...")
        client = get_llm_client()

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            embedding = loop.run_until_complete(client.get_embedding(text))
            loop.close()

            print(f"  ✓ Got embedding vector!")
            print(f"  Dimensions: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")

            # Save to database
            existing = session.query(ProductEmbedding).filter(
                ProductEmbedding.product_id == product.id
            ).first()

            if existing:
                existing.embedding = embedding
                existing.embedding_text = text
                print(f"  Updated existing embedding")
            else:
                pe = ProductEmbedding(
                    product_id=product.id,
                    embedding=embedding,
                    embedding_text=text
                )
                session.add(pe)
                print(f"  Created new embedding")

            session.commit()
            print(f"  ✓ Saved to pgvector!")

            # Show stats
            total_embeddings = session.query(ProductEmbedding).count()
            print(f"  Total embeddings in database: {total_embeddings}")

            return True

        except Exception as e:
            print(f"  ✗ OpenAI API error: {e}")
            return False


def test_vector_search():
    """Test vector similarity search."""
    print_step(6, "TESTING VECTOR SEARCH")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.core.config import settings
    from app.db.models import Product, ProductEmbedding
    from app.llm.client import get_llm_client
    import asyncio

    engine = create_engine(settings.database_url_sync)

    # Check if we have embeddings
    with Session(engine) as session:
        embedding_count = session.query(ProductEmbedding).count()
        if embedding_count == 0:
            print("  ✗ No embeddings in database! Run embedding generation first.")
            return False

        print(f"  Found {embedding_count} embeddings")

    # Test search query
    test_query = "видеокарта для игр"
    print(f"  Search query: '{test_query}'")

    # Get embedding for query
    client = get_llm_client()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        query_embedding = loop.run_until_complete(client.get_embedding(test_query))
        print(f"  ✓ Got query embedding")
    except Exception as e:
        print(f"  ✗ Failed to get query embedding: {e}")
        return False
    finally:
        loop.close()

    # Perform vector search
    with Session(engine) as session:
        from sqlalchemy import text

        # Convert embedding to string format for PostgreSQL
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        query = text("""
            SELECT p.id, p.name, p.category, p.price,
                   1 - (pe.embedding <=> :embedding::vector) as similarity
            FROM products p
            JOIN product_embeddings pe ON p.id = pe.product_id
            ORDER BY pe.embedding <=> :embedding::vector
            LIMIT 5
        """)

        results = session.execute(query, {"embedding": embedding_str}).fetchall()

        if results:
            print(f"\n  ✓ Found {len(results)} similar products:\n")
            for i, row in enumerate(results, 1):
                print(f"    {i}. {row.name[:50]}...")
                print(f"       Category: {row.category}")
                print(f"       Price: {row.price}")
                print(f"       Similarity: {row.similarity:.4f}")
                print()
            return True
        else:
            print("  No results found")
            return False


def show_database_stats():
    """Show current database statistics."""
    print_step(7, "DATABASE STATISTICS")

    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session
    from app.core.config import settings
    from app.db.models import Product, ProductEmbedding, FAQDocument

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        products = session.query(Product).count()
        active_products = session.query(Product).filter(Product.is_active == True).count()
        embeddings = session.query(ProductEmbedding).count()
        faqs = session.query(FAQDocument).count()

        print(f"  Products:")
        print(f"    Total: {products}")
        print(f"    Active: {active_products}")
        print(f"    With embeddings: {embeddings}")
        print(f"    Missing embeddings: {active_products - embeddings}")
        print(f"\n  FAQ Documents: {faqs}")

        # Categories breakdown
        from sqlalchemy import func
        categories = session.query(
            Product.category,
            func.count(Product.id)
        ).group_by(Product.category).order_by(func.count(Product.id).desc()).limit(10).all()

        print(f"\n  Top 10 Categories:")
        for cat, count in categories:
            print(f"    {cat}: {count}")


def main():
    """Run all pipeline tests."""
    print_header("DATA PIPELINE TEST")
    print("""
    Этот скрипт тестирует весь пайплайн данных:

    FTP Server (over-shop.kz)
           │
           ▼
    ┌─────────────────┐
    │  Download CSV   │  ← Скачивание Dealer.csv
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Parse CSV     │  ← Парсинг и нормализация
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   PostgreSQL    │  ← Сохранение продуктов
    │   (products)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   OpenAI API    │  ← Генерация embeddings
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    pgvector     │  ← Vector search
    │  (embeddings)   │
    └─────────────────┘
    """)

    # Test database connection
    if not test_database_connection():
        print("\n✗ Database connection failed. Make sure Docker containers are running.")
        return

    # Test FTP
    csv_content = test_ftp_connection()
    if not csv_content:
        print("\n⚠ FTP test failed. Continuing with other tests...")
        csv_content = None

    # Test CSV parsing
    df = None
    if csv_content:
        df = test_csv_parsing(csv_content)

    # Test product save
    if df is not None and not df.empty:
        test_product_save(df)

    # Test embedding generation
    print("\n⚠ Embedding test requires OPENAI_API_KEY to be set")
    try:
        test_embedding_generation()
    except Exception as e:
        print(f"  ✗ Embedding test failed: {e}")

    # Test vector search
    try:
        test_vector_search()
    except Exception as e:
        print(f"  ✗ Vector search test failed: {e}")

    # Show stats
    show_database_stats()

    print_header("TEST COMPLETE")


if __name__ == "__main__":
    main()
