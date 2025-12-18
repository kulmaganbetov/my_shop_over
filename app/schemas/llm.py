"""Schemas for LLM inputs and outputs."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """Supported user intents."""

    PC_BUILD = "pc_build"
    COMPONENT_REPLACE = "component_replace"
    SELECT_ALTERNATIVE = "select_alternative"
    ADD_PERIPHERAL = "add_peripheral"
    PRODUCT_SEARCH = "product_search"
    SHOW_SPECS = "show_specs"
    FILTER_PRICE = "filter_price"
    DELIVERY_INFO = "delivery_info"
    CALL_MANAGER = "call_manager"
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


class ComponentReplaceParams(BaseModel):
    """Parameters for component replacement intent."""

    component_type: str  # cpu, gpu, ram, etc.
    budget: Optional[int] = None
    preference: Optional[str] = None  # cheaper, better, specific brand


class SelectAlternativeParams(BaseModel):
    """Parameters for selecting an alternative component."""

    selection: int  # 1, 2, 3, etc.
    component_type: Optional[str] = None


class AddPeripheralParams(BaseModel):
    """Parameters for adding peripherals."""

    peripheral_type: str  # monitor, mouse, keyboard, headset
    budget: Optional[int] = None


class FilterPriceParams(BaseModel):
    """Parameters for price filtering."""

    min_price: Optional[float] = None
    max_price: Optional[float] = None


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
