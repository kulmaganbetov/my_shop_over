"""LLM service layer for structured interactions."""

import logging
from typing import Any, Optional

from app.llm.client import BaseLLMClient, get_llm_client
from app.llm.prompts import (
    FAQ_RESPONSE_SYSTEM_PROMPT,
    FAQ_RESPONSE_USER_PROMPT,
    GENERAL_RESPONSE_SYSTEM_PROMPT,
    GENERAL_RESPONSE_USER_PROMPT,
    INTENT_DETECTION_SYSTEM_PROMPT,
    INTENT_DETECTION_USER_PROMPT,
    PC_BUILD_RESPONSE_SYSTEM_PROMPT,
    PC_BUILD_RESPONSE_USER_PROMPT,
    PRODUCT_SEARCH_RESPONSE_SYSTEM_PROMPT,
    PRODUCT_SEARCH_RESPONSE_USER_PROMPT,
    EMBEDDING_TEXT_TEMPLATE,
)
from app.schemas.llm import (
    FAQParams,
    Intent,
    IntentDetectionResult,
    PCBuildParams,
    ProductSearchParams,
)

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based operations."""

    def __init__(self, client: Optional[BaseLLMClient] = None):
        self.client = client or get_llm_client()

    async def detect_intent(self, message: str, chat_history: str = "") -> IntentDetectionResult:
        """Detect user intent from message."""
        try:
            user_prompt = INTENT_DETECTION_USER_PROMPT.format(
                message=message,
                chat_history=chat_history,
            )
            result = await self.client.complete_json(
                system_prompt=INTENT_DETECTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
            )

            intent_str = result.get("intent", "unknown")
            try:
                intent = Intent(intent_str)
            except ValueError:
                intent = Intent.UNKNOWN

            return IntentDetectionResult(
                intent=intent,
                confidence=result.get("confidence", 0.5),
                params=result.get("params", {}),
            )
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return IntentDetectionResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                params={},
            )

    def parse_pc_build_params(self, params: dict) -> PCBuildParams:
        """Parse PC build parameters from intent detection."""
        return PCBuildParams(
            budget=params.get("budget"),
            budget_min=params.get("budget_min"),
            budget_max=params.get("budget_max"),
            purpose=params.get("purpose", "general"),
            resolution=params.get("resolution"),
            noise_preference=params.get("noise_preference"),
            specific_games=params.get("specific_games", []),
            specific_components=params.get("specific_components", {}),
        )

    def parse_product_search_params(self, params: dict) -> ProductSearchParams:
        """Parse product search parameters from intent detection."""
        return ProductSearchParams(
            query=params.get("query", ""),
            category=params.get("category"),
            min_price=params.get("min_price"),
            max_price=params.get("max_price"),
            manufacturer=params.get("manufacturer"),
            in_stock_only=params.get("in_stock_only", True),
            limit=params.get("limit", 10),
        )

    def parse_faq_params(self, params: dict) -> FAQParams:
        """Parse FAQ parameters from intent detection."""
        return FAQParams(
            question=params.get("question", ""),
            topic=params.get("topic"),
        )

    async def generate_pc_build_response(
        self,
        build_data: dict,
        user_request: str,
        chat_history: str = "",
    ) -> str:
        """Generate a natural language response for PC build."""
        import json
        user_prompt = PC_BUILD_RESPONSE_USER_PROMPT.format(
            build_data=json.dumps(build_data, ensure_ascii=False, indent=2),
            user_request=user_request,
            chat_history=chat_history,
        )
        return await self.client.complete(
            system_prompt=PC_BUILD_RESPONSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.7,
        )

    async def generate_product_search_response(
        self,
        query: str,
        results: list[dict],
        user_message: str,
        chat_history: str = "",
    ) -> str:
        """Generate a natural language response for product search."""
        import json
        user_prompt = PRODUCT_SEARCH_RESPONSE_USER_PROMPT.format(
            query=query,
            results=json.dumps(results, ensure_ascii=False, indent=2),
            user_message=user_message,
            chat_history=chat_history,
        )
        return await self.client.complete(
            system_prompt=PRODUCT_SEARCH_RESPONSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.7,
        )

    async def generate_faq_response(
        self,
        context: str,
        question: str,
    ) -> str:
        """Generate a natural language response for FAQ."""
        user_prompt = FAQ_RESPONSE_USER_PROMPT.format(
            context=context,
            question=question,
        )
        return await self.client.complete(
            system_prompt=FAQ_RESPONSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.5,
        )

    async def generate_general_response(self, message: str, chat_history: str = "") -> str:
        """Generate a general conversational response."""
        user_prompt = GENERAL_RESPONSE_USER_PROMPT.format(
            message=message,
            chat_history=chat_history,
        )
        return await self.client.complete(
            system_prompt=GENERAL_RESPONSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.8,
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        return await self.client.get_embedding(text)

    def create_product_embedding_text(self, product: dict) -> str:
        """Create text representation of product for embedding."""
        return EMBEDDING_TEXT_TEMPLATE.format(
            name=product.get("name", ""),
            category=product.get("category", ""),
            manufacturer=product.get("manufacturer", ""),
            price=product.get("price", ""),
            specifications=product.get("specifications", ""),
        )
