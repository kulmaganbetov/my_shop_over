"""Database models for the e-commerce AI assistant."""

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base


class Product(Base):
    """Product model representing items from the store catalog."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sku: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    kaspi_code: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    supplier: Mapped[Optional[str]] = mapped_column(String(255))
    stock: Mapped[int] = mapped_column(Integer, default=0)
    manufacturer: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    price: Mapped[Optional[float]] = mapped_column(Float)
    discount_price: Mapped[Optional[float]] = mapped_column(Float)
    category: Mapped[Optional[str]] = mapped_column(String(255), index=True)

    # PC component specific fields
    component_type: Mapped[Optional[str]] = mapped_column(String(50), index=True)
    specifications: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationship to embedding
    embedding: Mapped[Optional["ProductEmbedding"]] = relationship(
        "ProductEmbedding", back_populates="product", uselist=False
    )

    __table_args__ = (
        Index("idx_products_name_search", "name"),
        Index("idx_products_category_stock", "category", "stock"),
    )


class ProductEmbedding(Base):
    """Vector embeddings for products."""

    __tablename__ = "product_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    product_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("products.id", ondelete="CASCADE"), unique=True
    )
    embedding: Mapped[list] = mapped_column(Vector(settings.embedding_dimension))
    embedding_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    product: Mapped["Product"] = relationship("Product", back_populates="embedding")


class FAQDocument(Base):
    """FAQ documents for RAG-based answers."""

    __tablename__ = "faq_documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(settings.embedding_dimension))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PCPreset(Base):
    """Predefined PC build presets."""

    __tablename__ = "pc_presets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    purpose: Mapped[str] = mapped_column(String(100), index=True)
    budget_min: Mapped[int] = mapped_column(Integer)
    budget_max: Mapped[int] = mapped_column(Integer)
    description: Mapped[Optional[str]] = mapped_column(Text)
    components: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatSession(Base):
    """Chat session for conversation tracking."""

    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    context: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ChatMessage(Base):
    """Individual chat messages."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(100), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[Optional[str]] = mapped_column(String(50))
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
