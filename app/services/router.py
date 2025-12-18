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

        # Get chat history for context
        chat_history = await self._get_chat_history(session_id)

        # Store user message
        await self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=request.message,
        )

        # Detect intent with chat history context
        intent_result = await self.llm_service.detect_intent(request.message, chat_history)
        logger.info(f"Detected intent: {intent_result.intent} with confidence {intent_result.confidence}")

        # Route to appropriate service
        response = await self._route_intent(
            intent_result=intent_result,
            user_message=request.message,
            session_id=session_id,
            chat_history=chat_history,
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

    async def _get_chat_history(self, session_id: str, limit: int = 6) -> str:
        """Get formatted chat history for LLM context."""
        messages = await self.chat_repo.get_session_messages(session_id, limit=limit)
        if not messages:
            return ""

        history_lines = ["Recent chat history:"]
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            history_lines.append(f"  {role}: {content}")
        history_lines.append("")
        return "\n".join(history_lines)

    async def _route_intent(
        self,
        intent_result: IntentDetectionResult,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Route to the appropriate service based on intent."""
        intent = intent_result.intent
        params = intent_result.params

        try:
            if intent == Intent.PC_BUILD:
                return await self._handle_pc_build(params, user_message, session_id, chat_history)
            elif intent == Intent.COMPONENT_REPLACE:
                return await self._handle_component_replace(params, user_message, session_id, chat_history)
            elif intent == Intent.PRODUCT_SEARCH:
                return await self._handle_product_search(params, user_message, session_id, chat_history)
            elif intent == Intent.FAQ:
                return await self._handle_faq(params, user_message, session_id)
            else:
                return await self._handle_general(user_message, session_id, chat_history)
        except Exception as e:
            logger.error(f"Error processing intent {intent}: {e}")
            return await self._handle_error(session_id, str(e))

    async def _handle_pc_build(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle PC build intent."""
        build_params = self.llm_service.parse_pc_build_params(params)

        # Get build recommendation
        build_result = await self.pc_build_service.recommend_build(build_params)

        # Store build in session context for later replacement
        build_data = build_result.model_dump()
        await self.chat_repo.update_session_context(
            session_id,
            {"current_build": build_data}
        )

        # Generate response with chat history
        response_text = await self.llm_service.generate_pc_build_response(
            build_data=build_data,
            user_request=user_message,
            chat_history=chat_history,
        )

        suggestions = [
            "Заменить видеокарту",
            "Заменить процессор",
            "Добавить клавиатуру/мышь",
        ]

        return ChatResponse(
            message=response_text,
            intent=Intent.PC_BUILD,
            session_id=session_id,
            data=build_data,
            suggestions=suggestions,
        )

    async def _handle_component_replace(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle component replacement intent."""
        replace_params = self.llm_service.parse_component_replace_params(params)

        # Get current build from session context
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        current_build = {}
        if chat_session and chat_session.context:
            current_build = chat_session.context.get("current_build", {})

        if not current_build or not current_build.get("build"):
            return ChatResponse(
                message="Сначала давайте соберём ПК. Скажите, какой бюджет и для чего нужен компьютер?",
                intent=Intent.COMPONENT_REPLACE,
                session_id=session_id,
                suggestions=[
                    "Собрать игровой ПК за 500000 ₸",
                    "Собрать офисный ПК за 200000 ₸",
                ],
            )

        component_type = replace_params.component_type
        preference = replace_params.preference

        # Get alternatives
        alternatives = await self.pc_build_service.get_component_alternatives(
            component_type=component_type,
            current_build=current_build.get("build", {}),
            budget=replace_params.budget,
            limit=5,
        )

        alternatives_data = [alt.model_dump() for alt in alternatives]

        # Generate response
        response_text = await self.llm_service.generate_component_replace_response(
            component_type=component_type,
            preference=preference,
            alternatives=alternatives_data,
            current_build=current_build,
            chat_history=chat_history,
        )

        suggestions = [
            "Выбрать первый вариант",
            "Показать ещё варианты",
            "Заменить другой компонент",
        ]

        return ChatResponse(
            message=response_text,
            intent=Intent.COMPONENT_REPLACE,
            session_id=session_id,
            data={"alternatives": alternatives_data, "component_type": component_type},
            suggestions=suggestions,
        )

    async def _handle_product_search(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle product search intent."""
        search_params = self.llm_service.parse_product_search_params(params)

        # Search products
        search_result = await self.product_search_service.search(search_params)

        # Generate response with chat history
        products_data = [p.model_dump() for p in search_result.products]
        response_text = await self.llm_service.generate_product_search_response(
            query=search_params.query,
            results=products_data,
            user_message=user_message,
            chat_history=chat_history,
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
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle general conversation."""
        response_text = await self.llm_service.generate_general_response(user_message, chat_history)

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
