"""Common Pydantic schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(from_attributes=True)


class ProductSchema(BaseSchema):
    """Product response schema."""

    id: int
    sku: str
    kaspi_code: Optional[str] = None
    name: str
    supplier: Optional[str] = None
    stock: int
    manufacturer: Optional[str] = None
    price: Optional[float] = None  # Installment price (КредитРассрочка)
    discount_price: Optional[float] = None  # Card payment price (БонуснаяЦена)
    category: Optional[str] = None
    component_type: Optional[str] = None
    specifications: Optional[dict] = None


class ProductSearchResult(BaseSchema):
    """Product search result with relevance score."""

    product: ProductSchema
    score: float


class FAQDocumentSchema(BaseSchema):
    """FAQ document schema."""

    id: int
    title: str
    content: str
    category: Optional[str] = None


class PCPresetSchema(BaseSchema):
    """PC preset schema."""

    id: int
    name: str
    purpose: str
    budget_min: int
    budget_max: int
    description: Optional[str] = None
    components: dict


class MessageSchema(BaseSchema):
    """Chat message schema."""

    role: str
    content: str
    timestamp: Optional[datetime] = None
