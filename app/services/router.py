"""Intent router that directs requests to appropriate services."""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import ChatRepository
from app.llm.service import LLMService
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.llm import Intent, IntentDetectionResult
from app.services.faq import FAQService
from app.services.pc_build import PCBuildService
from app.services.product_search import ProductSearchService

logger = logging.getLogger(__name__)


class IntentRouter:
    """Routes user intents to appropriate services."""

    def __init__(
        self,
        session: AsyncSession,
        llm_service: LLMService,
    ):
        self.session = session
        self.llm_service = llm_service
        self.chat_repo = ChatRepository(session)

        # Initialize services
        self.product_search_service = ProductSearchService(session, llm_service)
        self.pc_build_service = PCBuildService(session, llm_service)
        self.faq_service = FAQService(session, llm_service)

    async def process_message(
        self,
        request: ChatRequest,
        session_id: str,
    ) -> ChatResponse:
        """Process a user message and return a response."""
        # Ensure session exists first (creates if needed)
        await self.chat_repo.get_or_create_session(session_id)

        # Store user message
        await self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=request.message,
        )

        # Detect intent
        intent_result = await self.llm_service.detect_intent(request.message)
        logger.info(f"Detected intent: {intent_result.intent} with confidence {intent_result.confidence}")

        # Route to appropriate service
        response = await self._route_intent(
            intent_result=intent_result,
            user_message=request.message,
            session_id=session_id,
        )

        # Store assistant response
        await self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=response.message,
            intent=response.intent.value,
            extra_data={"data": response.data} if response.data else None,
        )

        return response

    async def _route_intent(
        self,
        intent_result: IntentDetectionResult,
        user_message: str,
        session_id: str,
    ) -> ChatResponse:
        """Route to the appropriate service based on intent."""
        intent = intent_result.intent
        params = intent_result.params

        try:
            if intent == Intent.PC_BUILD:
                return await self._handle_pc_build(params, user_message, session_id)
            elif intent == Intent.PRODUCT_SEARCH:
                return await self._handle_product_search(params, user_message, session_id)
            elif intent == Intent.FAQ:
                return await self._handle_faq(params, user_message, session_id)
            else:
                return await self._handle_general(user_message, session_id)
        except Exception as e:
            logger.error(f"Error processing intent {intent}: {e}")
            return await self._handle_error(session_id, str(e))

    async def _handle_pc_build(
        self,
        params: dict,
        user_message: str,
        session_id: str,
    ) -> ChatResponse:
        """Handle PC build intent."""
        build_params = self.llm_service.parse_pc_build_params(params)

        # Get build recommendation
        build_result = await self.pc_build_service.recommend_build(build_params)

        # Generate response
        response_text = await self.llm_service.generate_pc_build_response(
            build_data=build_result.model_dump(),
            user_request=user_message,
        )

        suggestions = [
            "Показать альтернативные комплектующие",
            "Изменить бюджет",
            "Добавить периферию",
        ]

        return ChatResponse(
            message=response_text,
            intent=Intent.PC_BUILD,
            session_id=session_id,
            data=build_result.model_dump(),
            suggestions=suggestions,
        )

    async def _handle_product_search(
        self,
        params: dict,
        user_message: str,
        session_id: str,
    ) -> ChatResponse:
        """Handle product search intent."""
        search_params = self.llm_service.parse_product_search_params(params)

        # Search products
        search_result = await self.product_search_service.search(search_params)

        # Generate response
        products_data = [p.model_dump() for p in search_result.products]
        response_text = await self.llm_service.generate_product_search_response(
            query=search_params.query,
            results=products_data,
            user_message=user_message,
        )

        suggestions = [
            "Показать другие модели",
            "Фильтровать по цене",
            "Показать характеристики",
        ]

        return ChatResponse(
            message=response_text,
            intent=Intent.PRODUCT_SEARCH,
            session_id=session_id,
            data=search_result.model_dump(),
            suggestions=suggestions,
        )

    async def _handle_faq(
        self,
        params: dict,
        user_message: str,
        session_id: str,
    ) -> ChatResponse:
        """Handle FAQ intent."""
        faq_params = self.llm_service.parse_faq_params(params)

        # Get FAQ answer
        faq_result = await self.faq_service.answer_question(
            question=faq_params.question or user_message,
            topic=faq_params.topic,
        )

        suggestions = [
            "Узнать о доставке",
            "Условия гарантии",
            "Способы оплаты",
        ]

        return ChatResponse(
            message=faq_result.answer,
            intent=Intent.FAQ,
            session_id=session_id,
            data=faq_result.model_dump(),
            suggestions=suggestions,
        )

    async def _handle_general(
        self,
        user_message: str,
        session_id: str,
    ) -> ChatResponse:
        """Handle general conversation."""
        response_text = await self.llm_service.generate_general_response(user_message)

        suggestions = [
            "Собрать игровой ПК",
            "Найти товар",
            "Информация о доставке",
        ]

        return ChatResponse(
            message=response_text,
            intent=Intent.GENERAL,
            session_id=session_id,
            suggestions=suggestions,
        )

    async def _handle_error(
        self,
        session_id: str,
        error: str,
    ) -> ChatResponse:
        """Handle errors gracefully."""
        return ChatResponse(
            message="Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте еще раз.",
            intent=Intent.UNKNOWN,
            session_id=session_id,
            suggestions=[
                "Попробовать снова",
                "Связаться с поддержкой",
            ],
        )
