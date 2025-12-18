"""Common Pydantic schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, computed_field


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
    price: Optional[float] = None
    discount_price: Optional[float] = None
    category: Optional[str] = None
    component_type: Optional[str] = None
    specifications: Optional[dict] = None

    @computed_field
    @property
    def installment_price(self) -> Optional[float]:
        """Calculate 12-month installment price (no interest)."""
        base_price = self.discount_price or self.price
        if base_price:
            return round(base_price / 12, 0)
        return None

    @computed_field
    @property
    def display_price(self) -> Optional[float]:
        """Price for display (discount or regular)."""
        return self.discount_price or self.price


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
