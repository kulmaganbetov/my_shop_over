"""Embedding generation tasks for pgvector semantic search.

EMBEDDING PIPELINE FLOW:
========================
1. Get product/FAQ data from PostgreSQL
2. Create text representation for embedding
3. Call OpenAI API to generate embedding vector (1536 dimensions)
4. Store embedding in PostgreSQL (pgvector)

Embeddings enable:
- Semantic product search (find similar products by meaning)
- FAQ answer retrieval (RAG)

This module is triggered by:
- After product sync (automatically)
- Manual API call: POST /api/v1/admin/sync/embeddings
- Celery Beat scheduler (daily)
"""

import asyncio
import logging
from typing import Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Product, ProductEmbedding, FAQDocument
from app.llm.client import get_llm_client
from app.llm.prompts import EMBEDDING_TEXT_TEMPLATE
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def create_product_text(product: Product) -> str:
    """
    Create text representation of product for embedding.

    The text combines:
    - Product name
    - Category
    - Manufacturer
    - Price
    - Technical specifications

    This text is what gets converted to a vector.
    """
    specs_str = ""
    if product.specifications:
        specs_str = ", ".join(f"{k}: {v}" for k, v in product.specifications.items())

    return EMBEDDING_TEXT_TEMPLATE.format(
        name=product.name or "",
        category=product.category or "",
        manufacturer=product.manufacturer or "",
        price=product.discount_price or product.price or "",
        specifications=specs_str,
    )


@celery_app.task(bind=True, max_retries=3)
def generate_product_embedding(self, product_id: int):
    """
    Generate embedding for a single product.

    Flow:
    1. Load product from PostgreSQL
    2. Create text representation
    3. Call OpenAI API for embedding
    4. Store in product_embeddings table
    """
    logger.info(f"[Embedding] Generating embedding for product_id={product_id}")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        # Step 1: Load product
        product = session.query(Product).filter(Product.id == product_id).first()
        if not product:
            logger.error(f"[Embedding] Product {product_id} not found")
            return {"status": "error", "message": f"Product {product_id} not found"}

        logger.info(f"[Embedding] Product: {product.name[:50]}...")

        # Step 2: Create text for embedding
        text = create_product_text(product)
        logger.debug(f"[Embedding] Text: {text[:100]}...")

        # Step 3: Get embedding from OpenAI API
        logger.info(f"[Embedding] Calling OpenAI API...")
        client = get_llm_client()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding = loop.run_until_complete(client.get_embedding(text))
            logger.info(f"[Embedding] ✓ Got embedding vector (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"[Embedding] ✗ OpenAI API error: {e}")
            raise
        finally:
            loop.close()

        # Step 4: Save to product_embeddings table
        existing = session.query(ProductEmbedding).filter(
            ProductEmbedding.product_id == product_id
        ).first()

        if existing:
            existing.embedding = embedding
            existing.embedding_text = text
            logger.info(f"[Embedding] Updated existing embedding")
        else:
            product_embedding = ProductEmbedding(
                product_id=product_id,
                embedding=embedding,
                embedding_text=text,
            )
            session.add(product_embedding)
            logger.info(f"[Embedding] Created new embedding")

        session.commit()
        logger.info(f"[Embedding] ✓ Saved to PostgreSQL (pgvector)")

    return {"status": "success", "product_id": product_id}


@celery_app.task
def update_missing_embeddings(batch_size: int = 100):
    """
    Find products without embeddings and generate them in BATCH.

    Uses OpenAI batch API to generate multiple embeddings in one request.
    Much faster than individual requests!
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("EMBEDDING UPDATE: Batch processing")
    logger.info("=" * 60)
    logger.info(f"  Batch size: {batch_size}")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        # Count totals
        total_products = session.query(Product).filter(Product.is_active == True).count()
        total_embeddings = session.query(ProductEmbedding).count()
        missing = total_products - total_embeddings

        logger.info(f"  Total active products: {total_products}")
        logger.info(f"  Products with embeddings: {total_embeddings}")
        logger.info(f"  Products missing embeddings: {missing}")

        if missing == 0:
            logger.info("  ✓ All products have embeddings!")
            return {"status": "success", "processed": 0}

        # Find products without embeddings
        subquery = select(ProductEmbedding.product_id)
        products = (
            session.query(Product)
            .filter(~Product.id.in_(subquery))
            .filter(Product.is_active == True)
            .limit(batch_size)
            .all()
        )

        if not products:
            logger.info("  ✓ All products have embeddings!")
            return {"status": "success", "processed": 0}

        logger.info(f"  Processing {len(products)} products in ONE API call...")

        # Create texts for all products
        texts = []
        for product in products:
            text = create_product_text(product)
            texts.append(text)

        # Get ALL embeddings in ONE API call
        logger.info(f"  Calling OpenAI Batch Embedding API...")
        client = get_llm_client()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embeddings = loop.run_until_complete(client.get_embeddings_batch(texts))
            logger.info(f"  ✓ Got {len(embeddings)} embeddings in one request!")
        except Exception as e:
            logger.error(f"  ✗ Batch API error: {e}")
            raise
        finally:
            loop.close()

        # Save all embeddings to database
        logger.info(f"  Saving to PostgreSQL...")
        saved = 0
        for product, embedding, text in zip(products, embeddings, texts):
            product_embedding = ProductEmbedding(
                product_id=product.id,
                embedding=embedding,
                embedding_text=text,
            )
            session.add(product_embedding)
            saved += 1

        session.commit()
        logger.info(f"  ✓ Saved {saved} embeddings to PostgreSQL")

    remaining = missing - saved
    logger.info(f"  Remaining without embeddings: {remaining}")
    logger.info("=" * 60)

    return {"status": "success", "processed": saved, "remaining": remaining}


@celery_app.task
def regenerate_all_embeddings(batch_size: int = 50):
    """Regenerate embeddings for ALL products (useful after model change)."""
    logger.info("")
    logger.info("*" * 60)
    logger.info("*  FULL EMBEDDING REGENERATION")
    logger.info("*" * 60)

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        products = (
            session.query(Product.id)
            .filter(Product.is_active == True)
            .all()
        )
        product_ids = [p.id for p in products]

    logger.info(f"  Total products to process: {len(product_ids)}")
    logger.info(f"  Batch size: {batch_size}")

    # Process in batches
    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        for product_id in batch:
            generate_product_embedding.delay(product_id)
        logger.info(f"  Queued batch {i // batch_size + 1}: {len(batch)} products")

    logger.info(f"  ✓ Total queued: {len(product_ids)} products")
    return {"status": "success", "total_queued": len(product_ids)}


@celery_app.task
def generate_faq_embedding(faq_id: int):
    """Generate embedding for a FAQ document (for RAG)."""
    logger.info(f"[FAQ Embedding] Generating for faq_id={faq_id}")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        faq = session.query(FAQDocument).filter(FAQDocument.id == faq_id).first()
        if not faq:
            logger.error(f"[FAQ Embedding] FAQ {faq_id} not found")
            return {"status": "error", "message": f"FAQ {faq_id} not found"}

        logger.info(f"[FAQ Embedding] Title: {faq.title}")

        # Create text for embedding
        text = f"{faq.title}: {faq.content}"

        # Get embedding
        client = get_llm_client()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding = loop.run_until_complete(client.get_embedding(text))
            logger.info(f"[FAQ Embedding] ✓ Got embedding (dim={len(embedding)})")
        except Exception as e:
            logger.error(f"[FAQ Embedding] ✗ API error: {e}")
            raise
        finally:
            loop.close()

        # Update FAQ with embedding
        faq.embedding = embedding
        session.commit()
        logger.info(f"[FAQ Embedding] ✓ Saved to PostgreSQL")

    return {"status": "success", "faq_id": faq_id}


@celery_app.task
def update_all_faq_embeddings():
    """Update embeddings for all FAQ documents."""
    logger.info("[FAQ Embedding] Updating all FAQ embeddings")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        faqs = session.query(FAQDocument.id, FAQDocument.title).all()
        faq_ids = [f.id for f in faqs]

    logger.info(f"[FAQ Embedding] Found {len(faq_ids)} FAQ documents")

    for faq_id in faq_ids:
        generate_faq_embedding.delay(faq_id)

    logger.info(f"[FAQ Embedding] ✓ Queued {len(faq_ids)} tasks")
    return {"status": "success", "total_queued": len(faq_ids)}
