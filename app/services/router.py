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

        # Get chat history for context (increased limit for better memory)
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

    async def _get_chat_history(self, session_id: str, limit: int = 10) -> str:
        """Get formatted chat history for LLM context."""
        messages = await self.chat_repo.get_session_messages(session_id, limit=limit)
        if not messages:
            return ""

        history_lines = ["Chat history:"]
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            # Truncate long messages
            content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
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
            elif intent == Intent.SELECT_ALTERNATIVE:
                return await self._handle_select_alternative(params, user_message, session_id, chat_history)
            elif intent == Intent.ADD_PERIPHERAL:
                return await self._handle_add_peripheral(params, user_message, session_id, chat_history)
            elif intent == Intent.PRODUCT_SEARCH:
                return await self._handle_product_search(params, user_message, session_id, chat_history)
            elif intent == Intent.CLARIFY_SEARCH:
                return await self._handle_clarify_search(session_id)
            elif intent == Intent.SHOW_SPECS:
                return await self._handle_show_specs(params, user_message, session_id, chat_history)
            elif intent == Intent.SHOW_BUILD:
                return await self._handle_show_build(session_id, chat_history)
            elif intent == Intent.FILTER_PRICE:
                return await self._handle_filter_price(params, user_message, session_id, chat_history)
            elif intent == Intent.DELIVERY_INFO:
                return await self._handle_delivery_info(session_id)
            elif intent == Intent.CALL_MANAGER:
                return await self._handle_call_manager(params, session_id)
            elif intent == Intent.CANCEL_MANAGER:
                return await self._handle_cancel_manager(session_id)
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

        return ChatResponse(
            message=response_text,
            intent=Intent.PC_BUILD,
            session_id=session_id,
            data=build_data,
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
            )

        component_type = replace_params.component_type
        preference = replace_params.preference

        # Get alternatives
        alternatives = await self.pc_build_service.get_component_alternatives(
            component_type=component_type,
            current_build=current_build.get("build", {}),
            budget=replace_params.budget,
            preference=preference,
            limit=5,
        )

        alternatives_data = [alt.model_dump() for alt in alternatives]

        # Store alternatives for later selection
        await self.chat_repo.update_session_context(
            session_id,
            {
                "last_alternatives": alternatives_data,
                "last_component_type": component_type,
            }
        )

        # Generate response
        response_text = await self.llm_service.generate_component_replace_response(
            component_type=component_type,
            preference=preference,
            alternatives=alternatives_data,
            current_build=current_build,
            chat_history=chat_history,
        )

        return ChatResponse(
            message=response_text,
            intent=Intent.COMPONENT_REPLACE,
            session_id=session_id,
            data={"alternatives": alternatives_data, "component_type": component_type},
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

        # If query is too vague or is "другие модели", check context
        query = search_params.query.lower()
        if query in ["другие модели", "другие", "еще", "ещё"]:
            # Try to get last search context
            chat_session = await self.chat_repo.get_session_by_id(session_id)
            if chat_session and chat_session.context:
                last_query = chat_session.context.get("last_search_query", "")
                last_category = chat_session.context.get("last_search_category", "")
                if last_query:
                    search_params.query = last_query
                elif last_category:
                    search_params.query = last_category
                else:
                    return ChatResponse(
                        message="Какие именно товары вас интересуют? Назовите категорию или конкретный товар.",
                        intent=Intent.PRODUCT_SEARCH,
                        session_id=session_id,
                    )

        # Search products
        search_result = await self.product_search_service.search(search_params)

        # Store search context for "show more" functionality
        await self.chat_repo.update_session_context(
            session_id,
            {
                "last_search_query": search_params.query,
                "last_search_category": search_params.category,
                "last_search_results": [p.model_dump() for p in search_result.products],
            }
        )

        # Generate response with chat history
        products_data = [p.model_dump() for p in search_result.products]
        response_text = await self.llm_service.generate_product_search_response(
            query=search_params.query,
            results=products_data,
            user_message=user_message,
            chat_history=chat_history,
        )

        return ChatResponse(
            message=response_text,
            intent=Intent.PRODUCT_SEARCH,
            session_id=session_id,
            data=search_result.model_dump(),
        )

    async def _handle_clarify_search(self, session_id: str) -> ChatResponse:
        """Handle vague search requests that need clarification."""
        return ChatResponse(
            message="Какой товар вас интересует? Например:\n"
                    "- Клавиатуры и мыши\n"
                    "- Видеокарты\n"
                    "- Мониторы\n"
                    "- Процессоры\n"
                    "Или назовите конкретную модель.",
            intent=Intent.CLARIFY_SEARCH,
            session_id=session_id,
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

        return ChatResponse(
            message=faq_result.answer,
            intent=Intent.FAQ,
            session_id=session_id,
            data=faq_result.model_dump(),
        )

    async def _handle_general(
        self,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle general conversation."""
        response_text = await self.llm_service.generate_general_response(user_message, chat_history)

        return ChatResponse(
            message=response_text,
            intent=Intent.GENERAL,
            session_id=session_id,
        )

    async def _handle_select_alternative(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle selecting an alternative component."""
        selection = params.get("selection", 1) - 1  # Convert to 0-indexed

        # Get session context
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        if not chat_session or not chat_session.context:
            return ChatResponse(
                message="Нет доступных вариантов для выбора. Что бы вы хотели найти?",
                intent=Intent.SELECT_ALTERNATIVE,
                session_id=session_id,
            )

        context = chat_session.context

        # Check for alternatives (from component replace) or peripherals
        alternatives = context.get("last_alternatives", [])
        peripherals = context.get("last_peripherals", [])
        search_results = context.get("last_search_results", [])

        # Determine which list to select from
        items_list = alternatives or peripherals or search_results

        if not items_list:
            return ChatResponse(
                message="Нет вариантов для выбора. Попробуйте сначала найти товар или собрать ПК.",
                intent=Intent.SELECT_ALTERNATIVE,
                session_id=session_id,
            )

        if selection < 0 or selection >= len(items_list):
            return ChatResponse(
                message=f"Пожалуйста, выберите номер от 1 до {len(items_list)}.",
                intent=Intent.SELECT_ALTERNATIVE,
                session_id=session_id,
            )

        selected = items_list[selection]

        # If this was a component replacement, update the build
        if alternatives:
            component_type = context.get("last_component_type", "")
            current_build = context.get("current_build", {})

            if current_build.get("build") and component_type:
                current_build["build"][component_type] = selected
                # Recalculate total price
                total = sum(
                    (c.get("discount_price") or c.get("price") or 0)
                    for c in current_build["build"].values()
                    if c
                )
                current_build["total_price"] = total

                await self.chat_repo.update_session_context(
                    session_id,
                    {"current_build": current_build, "last_alternatives": []}
                )

        name = selected.get("name", "товар")
        price = selected.get("price", 0)
        discount_price = selected.get("discount_price", 0)

        return ChatResponse(
            message=f"Выбран: {name}\n"
                    f"Рассрочка: {price:,.0f} ₸ | Картой: {discount_price:,.0f} ₸",
            intent=Intent.SELECT_ALTERNATIVE,
            session_id=session_id,
            data=selected,
        )

    async def _handle_add_peripheral(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle adding peripherals (monitor, mouse, keyboard, etc.)."""
        peripheral_type = params.get("peripheral_type", "mouse")
        budget = params.get("budget")

        # Map peripheral types to categories
        peripheral_categories = {
            "monitor": ("Мониторы", ["Мониторы"]),
            "mouse": ("Мыши", ["Мыши"]),
            "keyboard": ("Клавиатуры", ["Клавиатуры"]),
            "headset": ("Гарнитуры", ["Гарнитуры", "Наушники"]),
            "mousepad": ("Коврики для мыши", ["Коврики"]),
            "webcam": ("Веб-камеры", ["Веб-камеры"]),
        }

        category_config = peripheral_categories.get(peripheral_type, ("Мыши", ["Мыши"]))
        category_name, search_keywords = category_config

        # Search for peripherals
        from app.db.repositories import ProductRepository
        product_repo = ProductRepository(self.session)

        products = await product_repo.get_by_component_type(
            component_type=category_name,
            max_price=budget,
            in_stock_only=True,
            limit=5,
            search_keywords=search_keywords,
        )

        if not products:
            return ChatResponse(
                message=f"К сожалению, {category_name.lower()} не найдены в наличии. "
                        f"Попробуйте изменить параметры поиска.",
                intent=Intent.ADD_PERIPHERAL,
                session_id=session_id,
            )

        from app.schemas.common import ProductSchema
        products_data = [ProductSchema.model_validate(p).model_dump() for p in products]

        # Store for selection
        await self.chat_repo.update_session_context(
            session_id,
            {"last_peripherals": products_data, "last_peripheral_type": peripheral_type}
        )

        # Format response
        lines = [f"Вот {category_name.lower()} в наличии:\n"]
        for i, p in enumerate(products_data, 1):
            price = p.get("price", 0)
            discount = p.get("discount_price", 0)
            lines.append(f"{i}. {p['name'][:60]}")
            lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")

        return ChatResponse(
            message="\n".join(lines),
            intent=Intent.ADD_PERIPHERAL,
            session_id=session_id,
            data={"peripherals": products_data},
        )

    async def _handle_show_specs(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle showing product specifications."""
        # Get context to find what to show specs for
        chat_session = await self.chat_repo.get_session_by_id(session_id)

        if not chat_session or not chat_session.context:
            return ChatResponse(
                message="Чтобы показать характеристики, сначала найдите товар или соберите ПК.",
                intent=Intent.SHOW_SPECS,
                session_id=session_id,
            )

        context = chat_session.context
        current_build = context.get("current_build", {})
        last_search = context.get("last_search_results", [])
        last_peripherals = context.get("last_peripherals", [])

        # Priority: build > search results > peripherals
        if current_build.get("build"):
            # Show specs for all components in build
            lines = ["Характеристики сборки:\n"]
            for comp_type, comp in current_build["build"].items():
                if comp:
                    specs = comp.get("specifications", {})
                    lines.append(f"**{comp_type.upper()}**: {comp.get('name', '')[:50]}")
                    if specs:
                        for key, val in list(specs.items())[:5]:  # Limit specs shown
                            lines.append(f"  • {key}: {val}")
                    lines.append("")

            return ChatResponse(
                message="\n".join(lines),
                intent=Intent.SHOW_SPECS,
                session_id=session_id,
            )
        elif last_search:
            # Show specs for last search results
            lines = ["Характеристики найденных товаров:\n"]
            for i, p in enumerate(last_search[:3], 1):  # First 3
                specs = p.get("specifications", {})
                lines.append(f"{i}. {p.get('name', '')[:50]}")
                if specs:
                    for key, val in list(specs.items())[:3]:
                        lines.append(f"   • {key}: {val}")
                lines.append("")

            return ChatResponse(
                message="\n".join(lines),
                intent=Intent.SHOW_SPECS,
                session_id=session_id,
            )
        elif last_peripherals:
            # Show specs for peripherals
            lines = ["Характеристики:\n"]
            for i, p in enumerate(last_peripherals[:3], 1):
                specs = p.get("specifications", {})
                lines.append(f"{i}. {p.get('name', '')[:50]}")
                if specs:
                    for key, val in list(specs.items())[:3]:
                        lines.append(f"   • {key}: {val}")
                lines.append("")

            return ChatResponse(
                message="\n".join(lines),
                intent=Intent.SHOW_SPECS,
                session_id=session_id,
            )
        else:
            return ChatResponse(
                message="Сначала найдите товары или соберите ПК, чтобы посмотреть характеристики.",
                intent=Intent.SHOW_SPECS,
                session_id=session_id,
            )

    async def _handle_show_build(
        self,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle showing current PC build."""
        chat_session = await self.chat_repo.get_session_by_id(session_id)

        if not chat_session or not chat_session.context:
            return ChatResponse(
                message="У вас пока нет сборки. Скажите, какой бюджет и для чего нужен компьютер?",
                intent=Intent.SHOW_BUILD,
                session_id=session_id,
            )

        current_build = chat_session.context.get("current_build", {})

        if not current_build.get("build"):
            return ChatResponse(
                message="У вас пока нет сборки. Скажите, какой бюджет и для чего нужен компьютер?",
                intent=Intent.SHOW_BUILD,
                session_id=session_id,
            )

        # Format build
        lines = ["Ваша текущая сборка:\n"]
        total = 0
        for comp_type, comp in current_build["build"].items():
            if comp:
                name = comp.get("name", "")[:50]
                price = comp.get("price", 0)
                discount = comp.get("discount_price", 0)
                total += discount or price
                lines.append(f"• {comp_type.upper()}: {name}")
                lines.append(f"  Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")

        lines.append(f"**Итого картой: {total:,.0f} ₸**")

        return ChatResponse(
            message="\n".join(lines),
            intent=Intent.SHOW_BUILD,
            session_id=session_id,
            data=current_build,
        )

    async def _handle_filter_price(
        self,
        params: dict,
        user_message: str,
        session_id: str,
        chat_history: str = "",
    ) -> ChatResponse:
        """Handle price filtering."""
        min_price = params.get("min_price")
        max_price = params.get("max_price")

        # Get last search context
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        last_query = ""
        if chat_session and chat_session.context:
            last_query = chat_session.context.get("last_search_query", "")

        if not last_query:
            return ChatResponse(
                message=f"Какой товар искать в диапазоне от {min_price or 0:,.0f} до {max_price or '∞'} ₸?",
                intent=Intent.FILTER_PRICE,
                session_id=session_id,
            )

        # Re-run search with price filter
        from app.schemas.llm import ProductSearchParams
        search_params = ProductSearchParams(
            query=last_query,
            min_price=min_price,
            max_price=max_price,
            limit=5,
        )
        search_result = await self.product_search_service.search(search_params)

        products_data = [p.model_dump() for p in search_result.products]
        response_text = await self.llm_service.generate_product_search_response(
            query=last_query,
            results=products_data,
            user_message=user_message,
            chat_history=chat_history,
        )

        return ChatResponse(
            message=response_text,
            intent=Intent.FILTER_PRICE,
            session_id=session_id,
            data=search_result.model_dump(),
        )

    async def _handle_delivery_info(self, session_id: str) -> ChatResponse:
        """Handle delivery information request."""
        delivery_text = """**Доставка по Казахстану**

**Алматы:**
• Бесплатная доставка при заказе от 50,000 ₸
• Доставка 1-2 рабочих дня
• Самовывоз из магазина

**Другие города:**
• Доставка через Казпочту или курьерские службы
• Срок: 3-7 рабочих дней
• Стоимость зависит от веса и города

**Оплата:**
• Картой онлайн
• Рассрочка 0-0-12 от Kaspi
• Наличными при получении"""

        return ChatResponse(
            message=delivery_text,
            intent=Intent.DELIVERY_INFO,
            session_id=session_id,
        )

    async def _handle_call_manager(
        self,
        params: dict,
        session_id: str,
    ) -> ChatResponse:
        """Handle manager call request with confirmation."""
        reason = params.get("reason", "запрос клиента")
        confirmed = params.get("confirmed", False)

        # Check if this is a confirmation
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        awaiting_confirmation = False
        if chat_session and chat_session.context:
            awaiting_confirmation = chat_session.context.get("awaiting_manager_confirmation", False)

        if confirmed or awaiting_confirmation:
            # Confirmation received - register the request
            await self.chat_repo.update_session_context(
                session_id,
                {
                    "awaiting_manager_confirmation": False,
                    "manager_requested": True,
                    "manager_request_reason": reason,
                }
            )
            logger.warning(f"[MANAGER_REQUEST] session={session_id} reason={reason}")

            return ChatResponse(
                message="Заявка принята! Менеджер свяжется с вами в ближайшее время.\n"
                        "Время работы: Пн-Пт 9:00-18:00\n\n"
                        "Пока можете продолжить пользоваться ботом.",
                intent=Intent.CALL_MANAGER,
                session_id=session_id,
            )
        else:
            # First request - ask for confirmation
            await self.chat_repo.update_session_context(
                session_id,
                {"awaiting_manager_confirmation": True, "manager_reason": reason}
            )
            return ChatResponse(
                message="Вы хотите связаться с менеджером? "
                        "Менеджер свяжется с вами в ближайшее время.\n\n"
                        "Напишите 'Да' для подтверждения или 'Нет' чтобы продолжить с ботом.",
                intent=Intent.CALL_MANAGER,
                session_id=session_id,
            )

    async def _handle_cancel_manager(self, session_id: str) -> ChatResponse:
        """Handle canceling manager request."""
        # Clear the awaiting confirmation flag
        await self.chat_repo.update_session_context(
            session_id,
            {"awaiting_manager_confirmation": False}
        )

        return ChatResponse(
            message="Хорошо, продолжаем! Чем могу помочь?",
            intent=Intent.CANCEL_MANAGER,
            session_id=session_id,
        )

    async def _handle_error(
        self,
        session_id: str,
        error: str,
    ) -> ChatResponse:
        """Handle errors gracefully."""
        logger.error(f"[SYSTEM_ERROR] session={session_id} error={error}")
        return ChatResponse(
            message="Извините, произошла ошибка. Попробуйте переформулировать вопрос.",
            intent=Intent.UNKNOWN,
            session_id=session_id,
        )
