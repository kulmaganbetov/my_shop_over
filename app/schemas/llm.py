"""Schemas for LLM inputs and outputs."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Supported user intents."""

    PC_BUILD = "pc_build"
    PRODUCT_SEARCH = "product_search"
    FAQ = "faq"
    GENERAL = "general"
    UNKNOWN = "unknown"


class PCPurpose(str, Enum):
    """PC build purposes."""

    GAMING = "gaming"
    WORK = "work"
    OFFICE = "office"
    STREAMING = "streaming"
    CONTENT_CREATION = "content_creation"
    GENERAL = "general"


class IntentDetectionResult(BaseModel):
    """Result of intent detection."""

    intent: Intent
    confidence: float = Field(ge=0, le=1)
    params: dict = Field(default_factory=dict)


class PCBuildParams(BaseModel):
    """Parameters for PC build intent."""

    budget: Optional[int] = None
    budget_min: Optional[int] = None
    budget_max: Optional[int] = None
    purpose: PCPurpose = PCPurpose.GENERAL
    resolution: Optional[str] = None
    noise_preference: Optional[str] = None
    specific_games: list[str] = Field(default_factory=list)
    specific_components: dict = Field(default_factory=dict)


class ProductSearchParams(BaseModel):
    """Parameters for product search intent."""

    query: str
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    manufacturer: Optional[str] = None
    in_stock_only: bool = True
    limit: int = 10


class FAQParams(BaseModel):
    """Parameters for FAQ intent."""

    question: str
    topic: Optional[str] = None


class LLMResponse(BaseModel):
    """Generic LLM response wrapper."""

    success: bool
    data: Any
    error: Optional[str] = None
