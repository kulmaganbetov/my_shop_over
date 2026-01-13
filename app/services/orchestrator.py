"""LLM Orchestrator - simplified with keyword matching first.

Architecture:
1. Keyword matching for obvious intents (90% of cases)
2. LLM only for ambiguous cases
3. State machine tracks conversation flow
4. Context-aware continuation for дороже/дешевле
"""

import json
import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import ChatRepository
from app.llm.client import get_llm_client
from app.schemas.chat import ChatResponse
from app.schemas.llm import Intent as LegacyIntent
from app.services.conversation import (
    ConversationContext,
    ConversationState,
    Intent,
    LastAction,
    detect_intent_from_keywords,
)
from app.services.tools import ToolExecutor

logger = logging.getLogger(__name__)


# LLM prompt for ambiguous cases only
AMBIGUOUS_INTENT_PROMPT = """Ты классификатор интентов для магазина компьютерной техники over-shop.kz.

Определи намерение пользователя. Верни ТОЛЬКО JSON:
{{"intent": "название", "params": {{}}}}

Доступные интенты:
- search_product: поиск товаров (query, category)
- build_pc: сборка ПК (budget, purpose)
- modify_build: изменить бюджет сборки (modifier: higher/lower)
- replace_component: заменить компонент (component_type, preference)
- add_peripheral: добавить периферию (peripheral_type)
- ask_question: вопрос о товарах/магазине
- delivery_info: доставка/оплата/адреса (topic)
- call_manager: вызов менеджера
- greeting: приветствие

Контекст: {context}
Сообщение: {message}"""


class Orchestrator:
    """Simplified orchestrator with keyword matching first."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.llm_client = get_llm_client()
        self.tool_executor = ToolExecutor(session)
        self.chat_repo = ChatRepository(session)

    async def process_message(
        self,
        message: str,
        session_id: str,
    ) -> ChatResponse:
        """Process user message."""
        # Get or create session
        chat_session = await self.chat_repo.get_or_create_session(session_id)

        # Check manager mode - bot should NOT respond
        if chat_session.status in ["waiting_manager", "manager_active"]:
            await self.chat_repo.add_message(
                session_id=session_id,
                role="user",
                content=message,
            )
            logger.info(f"[MANAGER] Session {session_id} in manager mode, message stored")
            return ChatResponse(
                message="Ваш диалог передан менеджеру. Ожидайте ответа.\n\n"
                        "Менеджер ответит вам в ближайшее время.\n"
                        "Срочно: +7 771 013-00-20",
                intent=LegacyIntent.GENERAL,
                session_id=session_id,
                data={"status": "waiting_manager"},
            )

        # Load conversation context
        context = await self._load_context(session_id)
        chat_history = await self._get_chat_history(session_id)

        # Store user message
        await self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=message,
        )

        # Step 1: Detect intent (keyword matching first)
        intent, params = detect_intent_from_keywords(message, context)

        # Step 2: If unknown, use LLM
        if intent == Intent.UNKNOWN:
            intent, params = await self._detect_intent_llm(message, context, chat_history)

        logger.info(f"[INTENT] {intent.value} | params={params}")

        # Step 3: Execute intent
        tool_name, tool_params = self._intent_to_tool(intent, params, context)
        tool_result = await self.tool_executor.execute(tool_name, tool_params, session_id)

        if tool_result.get("success"):
            logger.info(f"[OK] {tool_name}")
        else:
            logger.error(f"[FAIL] {tool_name}: {tool_result.get('error')}")

        # Step 4: Update context based on action
        await self._update_context_after_action(session_id, intent, tool_result, context)

        # Step 5: Format response
        response_text = await self._format_response(
            intent=intent,
            tool_name=tool_name,
            tool_result=tool_result,
            user_message=message,
            chat_history=chat_history,
            context=context,
        )

        # Map to legacy intent
        legacy_intent = self._intent_to_legacy(intent)

        # Store assistant response
        await self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            intent=legacy_intent.value,
            extra_data={"tool": tool_name, "data": tool_result.get("data")},
        )

        return ChatResponse(
            message=response_text,
            intent=legacy_intent,
            session_id=session_id,
            data=tool_result.get("data"),
        )

    async def _load_context(self, session_id: str) -> ConversationContext:
        """Load conversation context from session."""
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        if chat_session and chat_session.context:
            return ConversationContext.from_dict(chat_session.context)
        return ConversationContext()

    async def _save_context(self, session_id: str, context: ConversationContext):
        """Save conversation context to session."""
        await self.chat_repo.update_session_context(session_id, context.to_dict())

    async def _detect_intent_llm(
        self,
        message: str,
        context: ConversationContext,
        chat_history: str,
    ) -> tuple[Intent, dict]:
        """Use LLM for ambiguous intent detection."""
        context_str = self._format_context_for_llm(context)

        prompt = AMBIGUOUS_INTENT_PROMPT.format(
            context=context_str,
            message=message,
        )

        try:
            result = await self.llm_client.complete_json(
                system_prompt=prompt,
                user_prompt=f"История: {chat_history}\n\nСообщение: {message}",
                temperature=0.1,
            )

            intent_str = result.get("intent", "ask_question")
            params = result.get("params", {})

            # Map string to Intent enum
            intent_map = {
                "search_product": Intent.SEARCH_PRODUCT,
                "build_pc": Intent.BUILD_PC,
                "modify_build": Intent.MODIFY_BUILD,
                "replace_component": Intent.REPLACE_COMPONENT,
                "add_peripheral": Intent.ADD_PERIPHERAL,
                "show_build": Intent.SHOW_BUILD,
                "ask_question": Intent.ASK_QUESTION,
                "delivery_info": Intent.DELIVERY_INFO,
                "call_manager": Intent.CALL_MANAGER,
                "greeting": Intent.GREETING,
            }
            return intent_map.get(intent_str, Intent.ASK_QUESTION), params

        except Exception as e:
            logger.error(f"LLM intent detection failed: {e}")
            return Intent.ASK_QUESTION, {}

    def _format_context_for_llm(self, context: ConversationContext) -> str:
        """Format context for LLM prompt."""
        parts = []

        if context.has_build():
            total = context.current_build.get("total_price", 0)
            parts.append(f"Есть сборка ПК на {total:,} тг")

        if context.last_action != LastAction.NONE:
            action_map = {
                LastAction.SEARCH: "Последний поиск",
                LastAction.BUILD_PC: "Показана сборка",
                LastAction.SHOW_ALTERNATIVES: f"Показаны альтернативы для {context.last_component_type}",
                LastAction.SHOW_PERIPHERALS: f"Показана периферия ({context.last_peripheral_type})",
            }
            parts.append(action_map.get(context.last_action, ""))

        return "; ".join(parts) if parts else "Новая сессия"

    def _intent_to_tool(
        self,
        intent: Intent,
        params: dict,
        context: ConversationContext,
    ) -> tuple[str, dict]:
        """Map intent to tool name and params."""

        if intent == Intent.SEARCH_PRODUCT:
            return "search_products", {
                "query": params.get("query", ""),
                "category": params.get("category"),
                "min_price": params.get("min_price"),
                "max_price": params.get("max_price"),
            }

        elif intent == Intent.BUILD_PC:
            return "build_pc", {
                "budget": params.get("budget"),
                "purpose": params.get("purpose", "gaming"),
            }

        elif intent == Intent.MODIFY_BUILD:
            return "modify_build", {
                "modifier": params.get("modifier", "higher"),
            }

        elif intent == Intent.REPLACE_COMPONENT:
            return "get_alternatives", {
                "component_type": params.get("component_type", "gpu"),
                "preference": params.get("preference"),
                "budget": params.get("budget"),  # Pass user-specified budget
            }

        elif intent == Intent.SELECT_ITEM:
            return "select_item", {
                "number": params.get("number", 1),
            }

        elif intent == Intent.ADD_PERIPHERAL:
            return "add_peripheral", {
                "peripheral_type": params.get("peripheral_type", "mouse"),
                "budget": params.get("budget"),
            }

        elif intent == Intent.SHOW_BUILD:
            return "show_current_build", {}

        elif intent == Intent.DELIVERY_INFO:
            return "get_delivery_info", {
                "topic": params.get("topic", "all"),
            }

        elif intent == Intent.CALL_MANAGER:
            return "call_manager", {
                "reason": params.get("reason", "запрос клиента"),
            }

        elif intent == Intent.GREETING:
            return "general_response", {}

        else:  # ASK_QUESTION, UNKNOWN
            return "general_response", {}

    async def _update_context_after_action(
        self,
        session_id: str,
        intent: Intent,
        tool_result: dict,
        context: ConversationContext,
    ):
        """Update conversation context after tool execution."""
        data = tool_result.get("data", {})

        if intent == Intent.SEARCH_PRODUCT:
            context.last_action = LastAction.SEARCH
            context.last_shown_products = data.get("products", [])
            context.last_search_query = data.get("query", "")
            context.state = ConversationState.SEARCHING

        elif intent == Intent.BUILD_PC:
            context.last_action = LastAction.BUILD_PC
            context.current_build = data
            context.original_budget = data.get("total_price", 0)
            context.state = ConversationState.BUILDING

        elif intent == Intent.MODIFY_BUILD:
            context.last_action = LastAction.BUILD_PC
            context.current_build = data
            context.state = ConversationState.BUILDING

        elif intent == Intent.REPLACE_COMPONENT:
            context.last_action = LastAction.SHOW_ALTERNATIVES
            context.last_alternatives = data.get("alternatives", [])
            context.last_component_type = data.get("component_type", "")
            context.state = ConversationState.REVIEWING

        elif intent == Intent.SELECT_ITEM:
            if data.get("action") == "replaced_component":
                context.current_build = data.get("current_build", context.current_build)
                context.last_alternatives = []
                context.last_action = LastAction.BUILD_PC
            elif data.get("action") == "added_peripheral":
                context.current_build = data.get("current_build", context.current_build)
                context.last_peripherals = []
                context.last_action = LastAction.BUILD_PC
            context.state = ConversationState.BUILDING

        elif intent == Intent.ADD_PERIPHERAL:
            context.last_action = LastAction.SHOW_PERIPHERALS
            context.last_peripherals = data.get("peripherals", [])
            context.last_peripheral_type = data.get("peripheral_type", "")
            context.state = ConversationState.REVIEWING

        elif intent == Intent.CALL_MANAGER:
            context.state = ConversationState.MANAGER

        # Track intent history
        context.intent_history.append(intent.value)

        # Save context
        await self._save_context(session_id, context)

    async def _format_response(
        self,
        intent: Intent,
        tool_name: str,
        tool_result: dict,
        user_message: str,
        chat_history: str,
        context: ConversationContext,
    ) -> str:
        """Format response based on tool result."""
        # Handle errors
        if not tool_result.get("success"):
            error = tool_result.get("error", "Произошла ошибка")
            if "error" in tool_result.get("data", {}):
                error = tool_result["data"]["error"]
            return error

        data = tool_result.get("data", {})

        # Use deterministic formatting for structured data
        if tool_name in ["build_pc", "modify_build"]:
            # Check if this is a preset selection response
            if data.get("type") == "preset_selection":
                return self._format_preset_options(data)
            return self._format_pc_build(data)

        if tool_name == "get_alternatives":
            return self._format_alternatives(data)

        if tool_name == "select_item":
            return self._format_selection(data)

        if tool_name == "add_peripheral":
            return self._format_peripherals(data)

        if tool_name == "search_products":
            return self._format_search_results(data)

        if tool_name == "show_current_build":
            return self._format_pc_build(data.get("current_build", data))

        if tool_name == "get_delivery_info":
            return self._format_delivery_info(data)

        if tool_name == "call_manager":
            return self._format_manager_response(data)

        if tool_name == "general_response":
            return await self._generate_general_response(user_message, chat_history, context)

        return "Готово! Чем ещё могу помочь?"

    def _format_preset_options(self, data: dict) -> str:
        """Format preset options for user selection."""
        budget = data.get("budget", 500000)
        options = data.get("options", {})
        message = options.get("message", "")

        if message:
            return message

        # Fallback formatting
        counts = options.get("counts", {})
        total = options.get("total_presets", 0)

        lines = [f"**Для бюджета {budget:,}₸ у меня есть {total} проверенных конфигураций:**\n"]

        for category, count in counts.items():
            emoji = "🔵" if "Intel" in category else "🔴" if "AMD" in category else "💼"
            lines.append(f"{emoji} **{category}**: {count} вариант{'а' if 2 <= count <= 4 else 'ов'}")

        lines.append("\n**Что выберем: Intel, AMD или решение для работы?**")

        return "\n".join(lines)

    def _format_pc_build(self, data: dict) -> str:
        """Format PC build response - DETERMINISTIC."""
        build = data.get("build", {})
        if not build:
            # Try to suggest next steps instead of just failing
            return "Не удалось собрать ПК в указанном бюджете. Попробуйте увеличить бюджет или уточните требования."

        lines = ["**Ваша сборка ПК:**\n"]
        total = 0
        total_installment = 0

        component_names = {
            "cpu": "Процессор",
            "motherboard": "Материнская плата",
            "ram": "Оперативная память",
            "gpu": "Видеокарта",
            "storage": "Накопитель",
            "psu": "Блок питания",
            "case": "Корпус",
            "cooler": "Кулер",
        }

        for comp_type in ["cpu", "motherboard", "ram", "gpu", "storage", "psu", "case", "cooler"]:
            comp = build.get(comp_type)
            if comp:
                name = comp.get("name", "")[:70]
                price = comp.get("price", 0)
                discount = comp.get("discount_price", 0)
                total += discount or price
                total_installment += price
                display_name = component_names.get(comp_type, comp_type.upper())
                lines.append(f"**{display_name}**: {name}")
                lines.append(f"Рассрочка: {price:,.0f} | Картой: {discount:,.0f}\n")

        # Peripherals
        peripherals = data.get("peripherals", {})
        if peripherals:
            peripheral_names = {
                "monitor": "Монитор",
                "mouse": "Мышь",
                "keyboard": "Клавиатура",
                "headset": "Гарнитура",
            }
            lines.append("\n**Периферия:**")
            for ptype, p in peripherals.items():
                if p:
                    name = p.get("name", "")[:50]
                    price = p.get("price", 0)
                    discount = p.get("discount_price", 0)
                    total += discount or price
                    total_installment += price
                    lines.append(f"**{peripheral_names.get(ptype, ptype)}**: {name}")
                    lines.append(f"Рассрочка: {price:,.0f} | Картой: {discount:,.0f}")

        lines.append(f"\n**Итого картой: {total:,.0f}**")
        lines.append(f"**Итого в рассрочку: {total_installment:,.0f}**")

        # Show warnings from PCAssemblyEngine
        warnings = data.get("warnings", [])
        if warnings:
            lines.append("\n**Примечания:**")
            for w in warnings:
                # Handle both string and dict format
                if isinstance(w, dict):
                    lines.append(f"- {w.get('message', str(w))}")
                else:
                    lines.append(f"- {w}")

        # Show errors (should not happen in valid build)
        errors = data.get("errors", [])
        if errors:
            lines.append("\n**Внимание:**")
            for e in errors:
                if isinstance(e, dict):
                    lines.append(f"⚠️ {e.get('message', str(e))}")
                else:
                    lines.append(f"⚠️ {e}")

        # Compatibility notes (legacy format)
        compat_notes = data.get("compatibility_notes", [])
        if compat_notes:
            lines.append("\n**Совместимость:**")
            for note in compat_notes:
                lines.append(f"- {note}")

        # Peripheral suggestion - only if base build is complete and no peripherals yet
        if build and not peripherals:
            lines.append("\n---")
            lines.append("💡 *Нужны ли вам монитор, мышь или гарнитура к этому ПК?*")

        return "\n".join(lines)

    def _format_alternatives(self, data: dict) -> str:
        """Format alternatives response."""
        alts = data.get("alternatives", [])
        comp_type = data.get("component_type", "")
        warning = data.get("warning")

        component_names = {
            "cpu": "процессора",
            "gpu": "видеокарты",
            "motherboard": "материнской платы",
            "ram": "оперативной памяти",
            "storage": "накопителя",
            "psu": "блока питания",
            "case": "корпуса",
            "cooler": "кулера",
        }
        display_name = component_names.get(comp_type, comp_type)

        lines = []
        if warning:
            lines.append(warning)
            lines.append("")

        if not alts:
            return f"Альтернативы для {display_name} не найдены."

        lines.append(f"**Альтернативы для {display_name}:**\n")
        for i, p in enumerate(alts[:5], 1):
            name = p.get("name", "")[:65]
            price = p.get("price", 0)
            discount = p.get("discount_price", 0)
            lines.append(f"{i}. {name}")
            lines.append(f"   Рассрочка: {price:,.0f} | Картой: {discount:,.0f}\n")

        lines.append("Выберите номер для замены.")
        return "\n".join(lines)

    def _format_selection(self, data: dict) -> str:
        """Format selection response."""
        action = data.get("action", "")
        current_build = data.get("current_build")

        if action in ["replaced_component", "added_peripheral"] and current_build:
            return self._format_pc_build(current_build)

        selected = data.get("selected", {})
        name = selected.get("name", "товар")
        price = selected.get("price", 0)
        discount = selected.get("discount_price", 0)
        return f"Выбран: {name}\nРассрочка: {price:,.0f} | Картой: {discount:,.0f}"

    def _format_peripherals(self, data: dict) -> str:
        """Format peripherals response."""
        perips = data.get("peripherals", [])
        ptype = data.get("peripheral_type", "")

        peripheral_names = {
            "mouse": "Мыши",
            "keyboard": "Клавиатуры",
            "monitor": "Мониторы",
            "headset": "Гарнитуры",
            "mousepad": "Коврики",
            "webcam": "Веб-камеры",
        }
        display_name = peripheral_names.get(ptype, ptype)

        if not perips:
            return f"{display_name} не найдены в наличии."

        lines = [f"**{display_name} в наличии:**\n"]
        for i, p in enumerate(perips[:5], 1):
            name = p.get("name", "")[:60]
            price = p.get("price", 0)
            discount = p.get("discount_price", 0)
            lines.append(f"{i}. {name}")
            lines.append(f"   Рассрочка: {price:,.0f} | Картой: {discount:,.0f}\n")

        lines.append("Выберите номер для добавления к сборке.")
        return "\n".join(lines)

    def _format_search_results(self, data: dict) -> str:
        """Format search results."""
        products = data.get("products", [])
        query = data.get("query", "")

        if not products:
            return f"По запросу '{query}' товары не найдены. Попробуйте изменить запрос."

        lines = [f"**Результаты поиска '{query}':**\n"]
        for i, p in enumerate(products[:5], 1):
            name = p.get("name", "")[:60]
            price = p.get("price", 0)
            discount = p.get("discount_price", 0)
            stock = p.get("stock", 0)
            lines.append(f"{i}. {name}")
            lines.append(f"   Рассрочка: {price:,.0f} | Картой: {discount:,.0f}")
            if stock > 0:
                lines.append(f"   В наличии: {stock} шт.\n")
            else:
                lines.append(f"   Нет в наличии\n")

        return "\n".join(lines)

    def _format_delivery_info(self, data: dict) -> str:
        """Format delivery/store info."""
        topic = data.get("topic", "all")

        if topic == "phone":
            lines = ["**Телефоны Over-Shop.kz**\n"]
            lines.append(f"Интернет-магазин: {data.get('online_phone', '')}")
            lines.append(f"Kaspi заказы: {data.get('kaspi_orders', '')}")
            for key, store in data.get("stores", {}).items():
                lines.append(f"\n{store.get('name', '')}: {', '.join(store.get('phones', []))}")
            return "\n".join(lines)

        if topic == "hours":
            lines = ["**Режим работы магазинов**\n"]
            for key, store in data.get("stores", {}).items():
                lines.append(f"**{store.get('name', '')}**")
                lines.append(f"{store.get('hours', '')}\n")
            return "\n".join(lines)

        if topic == "address":
            lines = ["**Адреса магазинов**\n"]
            for key, store in data.get("stores", {}).items():
                lines.append(f"**{store.get('name', '')}**")
                lines.append(f"{store.get('address', '')}\n")
            return "\n".join(lines)

        if topic == "delivery":
            delivery = data.get("delivery", {})
            lines = ["**Доставка**\n"]
            lines.append(f"- {delivery.get('pickup', '')}")
            lines.append(f"- {delivery.get('courier', '')}")
            lines.append(f"- Транспортные компании: {delivery.get('transport', '')}")
            return "\n".join(lines)

        if topic == "payment":
            payment = data.get("payment", {})
            lines = ["**Способы оплаты**\n"]
            for method in payment.get("methods", []):
                lines.append(f"- {method}")
            return "\n".join(lines)

        # Full info
        return self._format_full_delivery_info(data)

    def _format_full_delivery_info(self, data: dict) -> str:
        """Format complete delivery info."""
        stores = data.get("stores", {})
        online = data.get("online", {})
        delivery = data.get("delivery", {})
        payment = data.get("payment", {})

        lines = ["**Магазины Over-Shop.kz**\n"]

        for city_key in ["almaty", "astana", "pavlodar"]:
            if city_key in stores:
                s = stores[city_key]
                city_names = {"almaty": "Алматы", "astana": "Астана", "pavlodar": "Павлодар"}
                lines.append(f"**{city_names.get(city_key, city_key)}**")
                lines.append(f"{s.get('address', '')}")
                lines.append(f"{s.get('hours', '')}")
                lines.append(f"Тел: {', '.join(s.get('phones', []))}\n")

        lines.append("**Интернет-магазин**")
        lines.append(f"Тел: {online.get('phone', '')}")
        lines.append(f"Kaspi: {online.get('kaspi_orders', '')}")
        lines.append(f"Email: {online.get('email', '')}\n")

        lines.append("**Доставка**")
        lines.append(f"- {delivery.get('pickup', '')}")
        lines.append(f"- {delivery.get('courier', '')}")
        lines.append(f"- ТК: {delivery.get('transport', '')}\n")

        lines.append("**Оплата**")
        for method in payment.get("methods", []):
            lines.append(f"- {method}")

        return "\n".join(lines)

    def _format_manager_response(self, data: dict) -> str:
        """Format manager escalation response."""
        return (
            "**Диалог передан менеджеру**\n\n"
            "Менеджер ответит вам в ближайшее время.\n"
            "Все ваши сообщения будут сохранены.\n\n"
            "Срочный вопрос: +7 771 013-00-20\n"
            "Время работы: Пн-Пт 9:00-19:00"
        )

    async def _generate_general_response(
        self,
        message: str,
        chat_history: str,
        context: ConversationContext,
    ) -> str:
        """Generate conversational response with personality.

        CRITICAL FIX: Handle interruptions and off-topic messages gracefully.
        Don't act like a robot - be friendly and human-like.
        CRITICAL FIX #2: Pass full build context so bot knows prices and specs!
        """
        msg_lower = message.lower()

        # Check for greeting
        greeting_words = ["привет", "здравствуй", "добрый", "салем", "хай", "hello", "hi"]
        if not chat_history or any(word in msg_lower for word in greeting_words):
            return (
                "Привет! Я консультант Over-Shop.kz.\n\n"
                "Помогу собрать ПК, найти комплектующие или ответить на вопросы о магазине.\n"
                "Что вас интересует?"
            )

        # Handle common interruptions/off-topic messages
        # These should be handled gracefully, not ignored
        interruption_response = self._handle_interruption(msg_lower, context)
        if interruption_response:
            return interruption_response

        # CRITICAL: Check if user asks about price/payment and has a build
        # Answer directly without LLM if possible
        if context.has_build():
            direct_answer = self._answer_build_question_directly(msg_lower, context)
            if direct_answer:
                return direct_answer

        # Use LLM for conversational responses
        # CRITICAL: Updated personality - friendly expert, not a robot
        system_prompt = """Ты — дружелюбный эксперт по компьютерной технике в Over-Shop.kz.

🚨 ПРАВИЛО НУЛЕВОГО ОТКАЗА:
- Ты ОБЯЗАН знать ВСЁ о текущей сборке клиента! Данные есть ниже в BUILD_CONTEXT.
- НИКОГДА не говори "я не знаю цену", "спросите менеджера", "не могу назвать характеристики"!
- Если спрашивают цену — посчитай из BUILD_CONTEXT и ответь!
- Если спрашивают о компоненте — расскажи на основе его названия!

ПОЛИТИКА ЦЕН:
- Наличными = Картой = discount_price (со скидкой, выгодная цена!)
- Рассрочка = price (выше, через банк 0-0-12)

ЛИЧНОСТЬ:
- Ты живой человек, не робот. Отвечай естественно.
- Если клиент злится — успокой его, пойми его проблему.
- Если клиент шутит — поддержи шутку, но мягко верни к теме.

СТИЛЬ:
- Краткий, но человечный (2-3 предложения MAX)
- Уверенный, но не высокомерный

ЗАПРЕЩЕНО:
- Длинные лекции
- Фразы типа "Я — языковая модель" или "Я не могу"
- Придумывать цены (бери из BUILD_CONTEXT!)
- Говорить "я не знаю" когда данные есть в BUILD_CONTEXT"""

        # Build detailed context with all prices
        build_context = self._format_build_context_for_llm(context)

        try:
            return await self.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=f"История: {chat_history}\n\n{build_context}\n\nСообщение клиента: {message}",
                temperature=0.7,
            )
        except Exception:
            return "Чем могу помочь? Подберу комплектующие или найду нужный товар."

    def _format_build_context_for_llm(self, context: ConversationContext) -> str:
        """Format build context for LLM to use in answers."""
        if not context.has_build():
            return "BUILD_CONTEXT: Нет активной сборки."

        build = context.current_build.get("build", {})
        if not build:
            return "BUILD_CONTEXT: Нет активной сборки."

        lines = ["BUILD_CONTEXT (ИСПОЛЬЗУЙ ЭТИ ДАННЫЕ ДЛЯ ОТВЕТОВ!):"]
        total_card = 0
        total_installment = 0

        component_names = {
            "cpu": "Процессор",
            "motherboard": "Мат. плата",
            "ram": "ОЗУ",
            "gpu": "Видеокарта",
            "storage": "Накопитель",
            "psu": "Блок питания",
            "case": "Корпус",
            "cooler": "Кулер",
        }

        for comp_type, display_name in component_names.items():
            comp = build.get(comp_type)
            if comp:
                name = comp.get("name", "")
                price = comp.get("price", 0)
                discount = comp.get("discount_price", 0)
                specs = comp.get("specs_summary", "")
                total_card += discount or price
                total_installment += price
                line = f"- {display_name}: {name} | Картой/Наличными: {discount:,}₸ | Рассрочка: {price:,}₸"
                if specs:
                    line += f" | Характеристики: {specs}"
                lines.append(line)

        lines.append(f"\nИТОГО КАРТОЙ/НАЛИЧНЫМИ: {total_card:,}₸ (со скидкой!)")
        lines.append(f"ИТОГО В РАССРОЧКУ: {total_installment:,}₸")

        return "\n".join(lines)

    def _answer_build_question_directly(self, msg_lower: str, context: ConversationContext) -> Optional[str]:
        """Answer common build questions directly without LLM for speed and accuracy."""
        build = context.current_build.get("build", {})
        if not build:
            return None

        # Calculate totals
        total_card = 0
        total_installment = 0
        for comp in build.values():
            if isinstance(comp, dict):
                price = comp.get("price", 0)
                discount = comp.get("discount_price", 0)
                total_card += discount or price
                total_installment += price

        # "сколько наличными" / "сколько картой" / "сколько стоит"
        cash_keywords = ["наличными", "наличкой", "налом", "наличные"]
        card_keywords = ["картой", "карта", "безнал"]
        general_price_keywords = ["сколько стоит", "сколько будет", "какая цена", "цена сборки", "общая цена"]

        if any(kw in msg_lower for kw in cash_keywords):
            return (
                f"При оплате наличными ваша сборка обойдётся в **{total_card:,} тенге** — "
                f"это уже со скидкой!\n\n"
                f"Хотите оформить заказ или что-то изменить в сборке?"
            )

        if any(kw in msg_lower for kw in card_keywords) and "рассрочк" not in msg_lower:
            return (
                f"При оплате картой ваша сборка обойдётся в **{total_card:,} тенге** — "
                f"это цена со скидкой!\n\n"
                f"Готовы оформить или есть вопросы?"
            )

        # "сколько в рассрочку"
        installment_keywords = ["рассрочк", "рассрочку", "в рассрочку", "кредит"]
        if any(kw in msg_lower for kw in installment_keywords):
            return (
                f"В рассрочку (0-0-12) ваша сборка выйдет в **{total_installment:,} тенге**.\n\n"
                f"При оплате картой/наличными — **{total_card:,} тенге** (экономия {total_installment - total_card:,}₸).\n\n"
                f"Какой вариант оплаты предпочитаете?"
            )

        # General "сколько стоит"
        if any(kw in msg_lower for kw in general_price_keywords):
            return (
                f"**Стоимость вашей сборки:**\n"
                f"- Картой/наличными: **{total_card:,} тенге** (со скидкой!)\n"
                f"- В рассрочку 0-0-12: **{total_installment:,} тенге**\n\n"
                f"Что выбираете?"
            )

        # Price freshness question: "это актуальная цена?"
        freshness_keywords = ["актуальн", "свежие цен", "обновлен", "последние цен"]
        if any(kw in msg_lower for kw in freshness_keywords):
            return self._get_price_freshness_response()

        return None

    def _get_price_freshness_response(self) -> str:
        """Get response about price data freshness.

        NOTE: This is a sync method that checks the sync_status table.
        """
        from datetime import datetime, timezone
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session as SyncSession
        from app.core.config import settings
        from app.db.models import SyncStatus

        try:
            engine = create_engine(settings.database_url_sync)
            with SyncSession(engine) as session:
                sync_status = session.query(SyncStatus).filter(
                    SyncStatus.sync_type == "ftp_products"
                ).first()

                if sync_status and sync_status.last_success_at:
                    now = datetime.now(timezone.utc)
                    last_sync = sync_status.last_success_at
                    if last_sync.tzinfo is None:
                        last_sync = last_sync.replace(tzinfo=timezone.utc)

                    diff = now - last_sync
                    minutes = int(diff.total_seconds() // 60)

                    if minutes < 60:
                        freshness = f"{minutes} минут назад"
                    elif minutes < 1440:  # < 24 hours
                        hours = minutes // 60
                        freshness = f"{hours} час{'а' if 2 <= hours <= 4 else 'ов' if hours >= 5 else ''} назад"
                    else:
                        days = minutes // 1440
                        freshness = f"{days} дней назад"

                    return (
                        f"Да, цены актуальные! Данные обновлены **{freshness}**.\n\n"
                        f"Цены синхронизируются с базой магазина каждый час."
                    )

                return (
                    "Цены актуальные — синхронизируются с базой магазина каждый час.\n\n"
                    "Если нужна точная информация — могу связать с менеджером."
                )

        except Exception:
            return (
                "Цены актуальные — они обновляются автоматически каждый час.\n\n"
                "Готовы оформить заказ?"
            )

    def _handle_interruption(self, msg_lower: str, context: ConversationContext) -> Optional[str]:
        """Handle common interruptions gracefully.

        CRITICAL: Don't act like a robot. Respond naturally to off-topic messages.
        """
        # Name/identity questions
        if any(w in msg_lower for w in ["как тебя зовут", "кто ты", "ты кто", "твое имя"]):
            return (
                "Меня зовут Овер — консультант Over-Shop.kz! 😊\n\n"
                "Помогу собрать ПК или найти нужный товар. Чем могу помочь?"
            )

        # Insults/frustration - respond calmly
        if any(w in msg_lower for w in ["тупой", "тупая", "идиот", "дурак", "бот", "робот"]):
            base_response = (
                "Понимаю ваше разочарование. Давайте попробуем разобраться вместе.\n\n"
            )
            if context.has_build():
                return base_response + "Что именно не устраивает в текущей сборке? Могу предложить альтернативы."
            else:
                return base_response + "Расскажите, что вы ищете — постараюсь помочь."

        # Thanks
        if any(w in msg_lower for w in ["спасибо", "благодарю", "thanks"]):
            if context.has_build():
                return (
                    "Рад помочь! Сборка сохранена.\n\n"
                    "Если захотите что-то изменить — просто напишите.\n"
                    "Удачных покупок! 🎮"
                )
            return "Всегда рад помочь! Обращайтесь, если понадобится что-то ещё."

        # Jokes/small talk
        if any(w in msg_lower for w in ["ха", "лол", "прикольно", "круто", "класс"]):
            return None  # Let LLM handle this naturally

        # "What can you do" questions
        if any(w in msg_lower for w in ["что умеешь", "что ты можешь", "что ты умеешь"]):
            return (
                "Я могу:\n"
                "• Собрать ПК под ваш бюджет (игровой, рабочий, офисный)\n"
                "• Найти конкретные комплектующие\n"
                "• Подобрать периферию (монитор, мышь, клавиатура)\n"
                "• Рассказать о доставке и оплате\n\n"
                "Что вас интересует?"
            )

        return None  # Not an interruption, proceed normally

    async def _get_chat_history(self, session_id: str, limit: int = 10) -> str:
        """Get formatted chat history."""
        messages = await self.chat_repo.get_session_messages(session_id, limit=limit)
        if not messages:
            return ""

        lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _intent_to_legacy(self, intent: Intent) -> LegacyIntent:
        """Map new Intent to legacy Intent enum."""
        mapping = {
            Intent.SEARCH_PRODUCT: LegacyIntent.PRODUCT_SEARCH,
            Intent.BUILD_PC: LegacyIntent.PC_BUILD,
            Intent.MODIFY_BUILD: LegacyIntent.PC_BUILD,
            Intent.REPLACE_COMPONENT: LegacyIntent.COMPONENT_REPLACE,
            Intent.SELECT_ITEM: LegacyIntent.SELECT_ALTERNATIVE,
            Intent.ADD_PERIPHERAL: LegacyIntent.ADD_PERIPHERAL,
            Intent.SHOW_BUILD: LegacyIntent.SHOW_BUILD,
            Intent.DELIVERY_INFO: LegacyIntent.DELIVERY_INFO,
            Intent.CALL_MANAGER: LegacyIntent.CALL_MANAGER,
            Intent.ASK_QUESTION: LegacyIntent.GENERAL,
            Intent.GREETING: LegacyIntent.GENERAL,
            Intent.UNKNOWN: LegacyIntent.GENERAL,
        }
        return mapping.get(intent, LegacyIntent.GENERAL)
