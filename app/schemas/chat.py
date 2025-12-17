"""Chat-related schemas."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.common import ProductSchema
from app.schemas.llm import Intent


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None


class PCBuildComponent(BaseModel):
    """A component in a PC build."""

    component_type: str
    product: Optional[ProductSchema] = None
    note: Optional[str] = None


class PCBuildResult(BaseModel):
    """Result of a PC build recommendation."""

    build: dict[str, Optional[ProductSchema]]
    total_price: float
    compatibility: str
    compatibility_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ProductSearchResponse(BaseModel):
    """Response for product search."""

    products: list[ProductSchema]
    total_count: int
    query: str
    filters_applied: dict = Field(default_factory=dict)


class FAQResponse(BaseModel):
    """Response for FAQ queries."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    message: str
    intent: Intent
    session_id: str
    data: Optional[Any] = None
    suggestions: list[str] = Field(default_factory=list)
