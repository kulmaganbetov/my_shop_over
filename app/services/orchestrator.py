"""LLM Orchestrator - uses LLM to decide which tools to call.

Architecture:
1. User message + context → LLM decides which tool to call
2. Tool is executed → returns structured data
3. Data + user message → LLM formats nice response
"""

import json
import logging
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import ChatRepository
from app.llm.client import get_llm_client
from app.llm.tools import TOOLS, get_tools_for_prompt
from app.schemas.chat import ChatResponse
from app.schemas.llm import Intent
from app.services.tools import ToolExecutor

logger = logging.getLogger(__name__)

# System prompt for the orchestrator
ORCHESTRATOR_SYSTEM_PROMPT = """Ты AI-ассистент интернет-магазина компьютерной техники over-shop.kz.
Твоя задача - понять запрос пользователя и выбрать нужную функцию для выполнения.

{tools}

## Правила выбора функции:

1. **build_pc** - когда пользователь хочет собрать компьютер:
   - "собери пк", "хочу сборку", "нужен компьютер", "пк для игр"
   - "соберите мне компьютер за 500000"

2. **modify_build** - когда нужно изменить бюджет СУЩЕСТВУЮЩЕЙ сборки:
   - "дороже", "дешевле", "более дорогую", "подешевле"
   - ТОЛЬКО если уже есть сборка в контексте!

3. **search_products** - поиск конкретных товаров:
   - "покажи видеокарты", "найди мышку", "ноутбуки"
   - "RTX 4070", "мониторы до 200000"

4. **get_alternatives** - замена компонента в сборке:
   - "замени видеокарту", "другой процессор", "поменяй на Intel"
   - ТОЛЬКО если есть сборка!

5. **select_item** - выбор из списка:
   - "1", "2", "первый", "выбираю второй", "беру третий"

6. **add_peripheral** - добавление периферии:
   - "добавь мышку", "нужен монитор", "клавиатура"

7. **show_current_build** - показать текущую сборку:
   - "покажи сборку", "что выбрано", "моя конфигурация"

8. **get_delivery_info** - информация о доставке:
   - "доставка", "как получить", "оплата"

9. **call_manager** - вызов менеджера:
   - "позвоните", "нужен менеджер", "хочу поговорить"

10. **general_response** - приветствия и общие вопросы:
    - "привет", "спасибо", "пока"

## Формат ответа (ТОЛЬКО JSON):

```json
{{
  "tool": "имя_функции",
  "params": {{параметры}},
  "reasoning": "почему выбрана эта функция"
}}
```

## Контекст сессии:
{context}

## Важно:
- Если нет сборки и пользователь хочет "дороже/дешевле" - сначала нужно собрать ПК (build_pc)
- Если запрос непонятен - используй general_response
- Для замены компонентов нужна существующая сборка
"""

# Response formatting prompt
RESPONSE_FORMAT_PROMPT = """Ты AI-ассистент магазина over-shop.kz.
Сформулируй красивый, понятный ответ пользователю на основе данных.

## Правила:
- Пиши кратко и по делу
- Используй форматирование: **жирный** для важного
- Цены показывай в формате: Рассрочка: X ₸ | Картой: Y ₸
- Для списков используй нумерацию 1. 2. 3.
- Не добавляй то, чего нет в данных
- Отвечай на русском языке

## Данные:
{data}

## Запрос пользователя:
{user_message}

## История чата:
{chat_history}
"""


class Orchestrator:
    """LLM-based orchestrator that routes requests to backend tools."""

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
        """Process user message using LLM orchestration."""
        # Ensure session exists
        await self.chat_repo.get_or_create_session(session_id)

        # Get chat history and context
        chat_history = await self._get_chat_history(session_id)
        context = await self._get_context(session_id)

        # Store user message
        await self.chat_repo.add_message(
            session_id=session_id,
            role="user",
            content=message,
        )

        # Step 1: LLM decides which tool to call
        tool_decision = await self._decide_tool(message, context, chat_history)
        logger.info(f"Tool decision: {tool_decision}")

        tool_name = tool_decision.get("tool", "general_response")
        params = tool_decision.get("params", {})

        # Step 2: Execute the tool
        tool_result = await self.tool_executor.execute(tool_name, params, session_id)
        logger.info(f"Tool result success: {tool_result.get('success')}")

        # Step 3: Format the response
        response_text = await self._format_response(
            tool_name=tool_name,
            tool_result=tool_result,
            user_message=message,
            chat_history=chat_history,
        )

        # Map tool to intent for compatibility
        intent = self._tool_to_intent(tool_name)

        # Store assistant response
        await self.chat_repo.add_message(
            session_id=session_id,
            role="assistant",
            content=response_text,
            intent=intent.value,
            extra_data={"tool": tool_name, "data": tool_result.get("data")},
        )

        return ChatResponse(
            message=response_text,
            intent=intent,
            session_id=session_id,
            data=tool_result.get("data"),
        )

    async def _decide_tool(
        self,
        message: str,
        context: dict,
        chat_history: str,
    ) -> dict:
        """Use LLM to decide which tool to call."""
        # Format context for the prompt
        context_str = self._format_context(context)
        tools_str = get_tools_for_prompt()

        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            tools=tools_str,
            context=context_str,
        )

        user_prompt = f"Сообщение пользователя: {message}\n\nИстория чата:\n{chat_history}"

        try:
            result = await self.llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,  # Low temperature for consistent decisions
            )
            return result
        except Exception as e:
            logger.error(f"Tool decision failed: {e}")
            return {"tool": "general_response", "params": {}}

    async def _format_response(
        self,
        tool_name: str,
        tool_result: dict,
        user_message: str,
        chat_history: str,
    ) -> str:
        """Use LLM to format a nice response."""
        # Handle errors
        if not tool_result.get("success"):
            error = tool_result.get("error", "Произошла ошибка")
            if "error" in tool_result.get("data", {}):
                error = tool_result["data"]["error"]
            return error

        data = tool_result.get("data", {})

        # Some responses can be formatted without LLM
        if tool_name == "get_delivery_info":
            return self._format_delivery_info(data)

        if tool_name == "call_manager":
            return self._format_manager_response(data)

        if tool_name == "general_response":
            # Use LLM for general responses
            return await self._generate_general_response(user_message, chat_history)

        # Use LLM to format complex responses
        user_prompt = RESPONSE_FORMAT_PROMPT.format(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            user_message=user_message,
            chat_history=chat_history,
        )

        try:
            return await self.llm_client.complete(
                system_prompt="Ты помощник магазина. Форматируй ответ красиво и понятно.",
                user_prompt=user_prompt,
                temperature=0.7,
            )
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return self._fallback_format(tool_name, data)

    def _format_delivery_info(self, data: dict) -> str:
        """Format delivery info without LLM."""
        return """**Доставка по Казахстану**

**Алматы:**
• Бесплатная доставка при заказе от 50,000 ₸
• Доставка 1-2 рабочих дня
• Самовывоз из магазина

**Другие города:**
• Доставка через Казпочту или курьерские службы
• Срок: 3-7 рабочих дней

**Оплата:**
• Картой онлайн
• Рассрочка 0-0-12 от Kaspi
• Наличными при получении"""

    def _format_manager_response(self, data: dict) -> str:
        """Format manager callback response."""
        status = data.get("status")
        if status == "awaiting_confirmation":
            return ("Вы хотите связаться с менеджером? "
                    "Менеджер свяжется с вами в ближайшее время.\n\n"
                    "Напишите 'Да' для подтверждения или 'Нет' чтобы продолжить с ботом.")
        elif status == "confirmed":
            return ("Заявка принята! Менеджер свяжется с вами в ближайшее время.\n"
                    "Время работы: Пн-Пт 9:00-18:00\n\n"
                    "Пока можете продолжить пользоваться ботом.")
        return "Чем могу помочь?"

    async def _generate_general_response(self, message: str, chat_history: str) -> str:
        """Generate general conversational response."""
        system_prompt = """Ты AI-ассистент интернет-магазина компьютерной техники over-shop.kz.
Отвечай дружелюбно и кратко. Предлагай помощь с:
- Подбором комплектующих для сборки ПК
- Поиском товаров
- Информацией о доставке и оплате

Не выдумывай информацию о товарах или ценах."""

        user_prompt = f"История: {chat_history}\n\nСообщение: {message}"

        try:
            return await self.llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,
            )
        except Exception:
            return "Привет! Чем могу помочь? Могу подобрать комплектующие для ПК или найти нужный товар."

    def _fallback_format(self, tool_name: str, data: dict) -> str:
        """Fallback formatting without LLM."""
        if "products" in data:
            products = data["products"]
            if not products:
                return "Товары не найдены. Попробуйте изменить запрос."
            lines = ["Найденные товары:\n"]
            for i, p in enumerate(products[:5], 1):
                name = p.get("name", "")[:60]
                price = p.get("price", 0)
                discount = p.get("discount_price", 0)
                lines.append(f"{i}. {name}")
                lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")
            return "\n".join(lines)

        if "build" in data or "current_build" in data:
            build = data.get("build") or data.get("current_build", {}).get("build", {})
            if not build:
                return "Сборка не найдена."
            lines = ["Ваша сборка:\n"]
            total = 0
            for comp_type, comp in build.items():
                if comp:
                    name = comp.get("name", "")[:50]
                    price = comp.get("price", 0)
                    discount = comp.get("discount_price", 0)
                    total += discount or price
                    lines.append(f"• {comp_type.upper()}: {name}")
                    lines.append(f"  Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")
            lines.append(f"**Итого картой: {total:,.0f} ₸**")
            return "\n".join(lines)

        if "alternatives" in data:
            alts = data["alternatives"]
            if not alts:
                return "Альтернативы не найдены."
            lines = [f"Альтернативы для {data.get('component_type', 'компонента')}:\n"]
            for i, p in enumerate(alts[:5], 1):
                name = p.get("name", "")[:60]
                price = p.get("price", 0)
                discount = p.get("discount_price", 0)
                lines.append(f"{i}. {name}")
                lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")
            return "\n".join(lines)

        if "peripherals" in data:
            perips = data["peripherals"]
            if not perips:
                return "Периферия не найдена."
            lines = ["Периферия:\n"]
            for i, p in enumerate(perips[:5], 1):
                name = p.get("name", "")[:60]
                price = p.get("price", 0)
                discount = p.get("discount_price", 0)
                lines.append(f"{i}. {name}")
                lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")
            return "\n".join(lines)

        return "Готово! Чем ещё могу помочь?"

    def _format_context(self, context: dict) -> str:
        """Format context for the prompt."""
        if not context:
            return "Контекст пуст (новая сессия)"

        lines = []
        if context.get("current_build"):
            build = context["current_build"]
            total = build.get("total_price", 0)
            components = list(build.get("build", {}).keys())
            lines.append(f"- Есть сборка ПК: {', '.join(components)} на сумму {total:,} ₸")

        if context.get("last_search_results"):
            count = len(context["last_search_results"])
            lines.append(f"- Последний поиск: {count} товаров")

        if context.get("last_alternatives"):
            count = len(context["last_alternatives"])
            comp = context.get("last_component_type", "")
            lines.append(f"- Показаны альтернативы для {comp}: {count} вариантов")

        if context.get("last_peripherals"):
            count = len(context["last_peripherals"])
            ptype = context.get("last_peripheral_type", "")
            lines.append(f"- Показана периферия ({ptype}): {count} вариантов")

        if context.get("awaiting_manager_confirmation"):
            lines.append("- Ожидается подтверждение вызова менеджера")

        return "\n".join(lines) if lines else "Контекст пуст"

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

    async def _get_context(self, session_id: str) -> dict:
        """Get session context."""
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        if chat_session and chat_session.context:
            return chat_session.context
        return {}

    def _tool_to_intent(self, tool_name: str) -> Intent:
        """Map tool name to Intent enum for compatibility."""
        mapping = {
            "search_products": Intent.PRODUCT_SEARCH,
            "build_pc": Intent.PC_BUILD,
            "modify_build": Intent.PC_BUILD,
            "get_alternatives": Intent.COMPONENT_REPLACE,
            "select_item": Intent.SELECT_ALTERNATIVE,
            "add_peripheral": Intent.ADD_PERIPHERAL,
            "show_current_build": Intent.SHOW_BUILD,
            "show_specs": Intent.SHOW_SPECS,
            "get_delivery_info": Intent.DELIVERY_INFO,
            "call_manager": Intent.CALL_MANAGER,
            "answer_faq": Intent.FAQ,
            "general_response": Intent.GENERAL,
        }
        return mapping.get(tool_name, Intent.GENERAL)
