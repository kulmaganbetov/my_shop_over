"""Embedding generation tasks."""

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
    """Create text representation of product for embedding."""
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
    """Generate embedding for a single product."""
    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        product = session.query(Product).filter(Product.id == product_id).first()
        if not product:
            return {"status": "error", "message": f"Product {product_id} not found"}

        # Create text for embedding
        text = create_product_text(product)

        # Get embedding from LLM
        import asyncio
        client = get_llm_client()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding = loop.run_until_complete(client.get_embedding(text))
        finally:
            loop.close()

        # Save or update embedding
        existing = session.query(ProductEmbedding).filter(
            ProductEmbedding.product_id == product_id
        ).first()

        if existing:
            existing.embedding = embedding
            existing.embedding_text = text
        else:
            product_embedding = ProductEmbedding(
                product_id=product_id,
                embedding=embedding,
                embedding_text=text,
            )
            session.add(product_embedding)

        session.commit()

    return {"status": "success", "product_id": product_id}


@celery_app.task
def update_missing_embeddings(batch_size: int = 100):
    """Update embeddings for products that don't have them."""
    logger.info("Starting missing embeddings update")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        # Find products without embeddings
        subquery = select(ProductEmbedding.product_id)
        products_without_embeddings = (
            session.query(Product.id)
            .filter(~Product.id.in_(subquery))
            .filter(Product.is_active == True)
            .limit(batch_size)
            .all()
        )

        product_ids = [p.id for p in products_without_embeddings]

    if not product_ids:
        logger.info("No products without embeddings found")
        return {"status": "success", "processed": 0}

    # Queue embedding generation for each product
    for product_id in product_ids:
        generate_product_embedding.delay(product_id)

    logger.info(f"Queued {len(product_ids)} products for embedding generation")
    return {"status": "success", "queued": len(product_ids)}


@celery_app.task
def regenerate_all_embeddings(batch_size: int = 50):
    """Regenerate embeddings for all products."""
    logger.info("Starting full embedding regeneration")

    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        products = (
            session.query(Product.id)
            .filter(Product.is_active == True)
            .all()
        )
        product_ids = [p.id for p in products]

    # Process in batches
    for i in range(0, len(product_ids), batch_size):
        batch = product_ids[i:i + batch_size]
        for product_id in batch:
            generate_product_embedding.delay(product_id)

    return {"status": "success", "total_queued": len(product_ids)}


@celery_app.task
def generate_faq_embedding(faq_id: int):
    """Generate embedding for a FAQ document."""
    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        faq = session.query(FAQDocument).filter(FAQDocument.id == faq_id).first()
        if not faq:
            return {"status": "error", "message": f"FAQ {faq_id} not found"}

        # Create text for embedding
        text = f"{faq.title}: {faq.content}"

        # Get embedding
        import asyncio
        client = get_llm_client()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            embedding = loop.run_until_complete(client.get_embedding(text))
        finally:
            loop.close()

        # Update FAQ with embedding
        faq.embedding = embedding
        session.commit()

    return {"status": "success", "faq_id": faq_id}


@celery_app.task
def update_all_faq_embeddings():
    """Update embeddings for all FAQ documents."""
    engine = create_engine(settings.database_url_sync)

    with Session(engine) as session:
        faqs = session.query(FAQDocument.id).all()
        faq_ids = [f.id for f in faqs]

    for faq_id in faq_ids:
        generate_faq_embedding.delay(faq_id)

    return {"status": "success", "total_queued": len(faq_ids)}
