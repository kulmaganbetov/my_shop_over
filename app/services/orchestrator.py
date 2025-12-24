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
   - "добавил 50000 к бюджету" → build_pc с budget = предыдущий_бюджет + 50000
   - "другую сборку за X" → build_pc с budget = X

2. **modify_build** - когда нужно ОТНОСИТЕЛЬНО изменить бюджет:
   - "дороже", "дешевле", "более дорогую", "подешевле" (без конкретной суммы)
   - ТОЛЬКО если уже есть сборка в контексте!

3. **search_products** - поиск/покупка товаров ОТДЕЛЬНО от сборки:
   - "покажи видеокарты", "найди мышку", "ноутбуки"
   - "RTX 4070", "мониторы до 200000"
   - ВАЖНО: Если пользователь говорит "хочу ЕЩЕ купить", "ДОПОЛНИТЕЛЬНО", "ОТДЕЛЬНО" → это search_products!
   - Примеры search_products:
     - "хочу еще купить процессор amd" → search_products(query="процессор amd")
     - "дополнительно купить видеокарту" → search_products(query="видеокарта")
     - "отдельно нужен монитор" → search_products(query="монитор")
     - "купить процессор" → search_products(query="процессор")

4. **get_alternatives** - ЗАМЕНА компонента ВНУТРИ текущей сборки:
   - ТОЛЬКО слова: "замени", "поменяй", "смени", "обнови В СБОРКЕ"
   - ТОЛЬКО если есть сборка!
   - НЕ используй если пользователь говорит "купить", "еще", "дополнительно", "отдельно"!
   - Примеры get_alternatives:
     - "замени процессор" → get_alternatives(component_type="cpu")
     - "поменяй видеокарту на nvidia" → get_alternatives(component_type="gpu", preference="nvidia")
     - "смени материнку" → get_alternatives(component_type="motherboard")

5. **clarify_intent** - когда НЕЯСНО что хочет пользователь:
   - Если есть сборка И пользователь говорит про компонент без явного "замени" или "купить"
   - Пример: "процессор intel" (неясно - заменить в сборке или купить отдельно?)
   - Задай вопрос: "Вы хотите заменить процессор в сборке или купить отдельно?"

6. **select_item** - выбор из списка:
   - "1", "2", "первый", "выбираю второй", "беру третий"

7. **add_peripheral** - добавление периферии К СБОРКЕ:
   - "добавь мышку к сборке", "добавь монитор"

8. **show_current_build** - показать текущую сборку:
   - "покажи сборку", "что выбрано", "моя конфигурация"

9. **get_delivery_info** - информация о доставке:
   - "доставка", "как получить", "оплата"

10. **call_manager** - вызов менеджера:
    - "позвоните", "нужен менеджер", "хочу поговорить"

11. **general_response** - приветствия и общие вопросы:
    - "привет", "спасибо", "пока"

## КРИТИЧЕСКИ ВАЖНО - разделение интентов:

### Покупка ОТДЕЛЬНО (search_products):
- Слова-маркеры: "купить", "еще", "дополнительно", "отдельно", "нужен", "хочу"
- Пример: "хочу еще купить процессор amd ryzen" → search_products!

### Замена В СБОРКЕ (get_alternatives):
- Слова-маркеры: "замени", "поменяй", "смени", "обнови", "в сборке"
- Пример: "замени процессор на intel" → get_alternatives!

### Правило приоритета:
Если есть сборка И пользователь упоминает компонент И есть слова "еще/дополнительно/отдельно/купить":
→ ВСЕГДА выбирай search_products, НЕ get_alternatives!

## КРИТИЧЕСКИ ВАЖНО - расчёт бюджета:
- Если пользователь говорит "добавил X к бюджету" → новый budget = предыдущий бюджет + X
- Если в контексте есть сборка за 500000 и пользователь говорит "добавил 50000" → budget = 550000
- "Сделай другую сборку" без указания бюджета → используй предыдущий бюджет из контекста

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
- ВСЕГДА рассчитывай бюджет из контекста если пользователь говорит об изменении
- Если нет сборки и пользователь хочет "дороже/дешевле" - сначала нужно собрать ПК (build_pc)
- Если запрос непонятен - используй clarify_intent или general_response
- Для замены компонентов ВНУТРИ сборки нужна существующая сборка
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

        tool_name = tool_decision.get("tool", "general_response")
        params = tool_decision.get("params", {})
        reasoning = tool_decision.get("reasoning", "")

        logger.info(f"[CHAT] User: {message[:50]}...")
        logger.info(f"[TOOL] {tool_name} | params={params}")
        if reasoning:
            logger.debug(f"[WHY] {reasoning}")

        # Step 2: Execute the tool
        tool_result = await self.tool_executor.execute(tool_name, params, session_id)

        if tool_result.get("success"):
            logger.info(f"[OK] {tool_name} executed successfully")
        else:
            logger.error(f"[FAIL] {tool_name}: {tool_result.get('error')}")

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

        # CRITICAL: Use deterministic formatting for PC builds to prevent hallucination
        if tool_name in ["build_pc", "modify_build"]:
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

        if tool_name == "clarify_intent":
            return self._format_clarification(data)

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
        """Format delivery and store info - DETERMINISTIC."""
        stores = data.get("stores", {})
        online = data.get("online", {})
        delivery = data.get("delivery", {})
        payment = data.get("payment", {})

        lines = ["**Магазины Over-Shop.kz**\n"]

        # Almaty
        if "almaty" in stores:
            s = stores["almaty"]
            lines.append("📍 **Алматы**")
            lines.append(f"   {s.get('address', '')}")
            lines.append(f"   🕐 {s.get('hours', '')}")
            lines.append(f"   📞 {', '.join(s.get('phones', []))}")
            lines.append("")

        # Astana
        if "astana" in stores:
            s = stores["astana"]
            lines.append("📍 **Астана**")
            lines.append(f"   {s.get('address', '')}")
            lines.append(f"   🕐 {s.get('hours', '')}")
            lines.append(f"   📞 {', '.join(s.get('phones', []))}")
            lines.append("")

        # Pavlodar
        if "pavlodar" in stores:
            s = stores["pavlodar"]
            lines.append("📍 **Павлодар**")
            lines.append(f"   {s.get('address', '')}")
            lines.append(f"   🕐 {s.get('hours', '')}")
            lines.append(f"   📞 {', '.join(s.get('phones', []))}")
            if s.get("service_center"):
                lines.append(f"   🔧 Сервис: {s.get('service_center')}")
            lines.append("")

        # Online contacts
        lines.append("**Интернет-магазин**")
        lines.append(f"📞 {online.get('phone', '')}")
        lines.append(f"📱 Kaspi заказы: {online.get('kaspi_orders', '')}")
        lines.append(f"✉️ {online.get('email', '')}")
        lines.append(f"📷 Instagram: {online.get('instagram', '')}")
        lines.append("")

        # Delivery
        lines.append("**Доставка**")
        lines.append(f"• {delivery.get('pickup', '')}")
        lines.append(f"• {delivery.get('courier', '')}")
        lines.append(f"• {delivery.get('express', '')}")
        lines.append(f"• Транспортные компании: {delivery.get('transport', '')}")
        lines.append("")

        # Payment
        lines.append("**Оплата**")
        for method in payment.get("methods", []):
            lines.append(f"• {method}")

        return "\n".join(lines)

    def _format_manager_response(self, data: dict) -> str:
        """Format manager callback response."""
        status = data.get("status")
        if status == "awaiting_confirmation":
            return ("Вы хотите связаться с менеджером?\n\n"
                    "📞 Интернет-магазин: +7 771 013-00-20\n"
                    "📱 Kaspi заказы: +7 775 894-93-84\n\n"
                    "Напишите 'Да' и менеджер свяжется с вами, или позвоните сами.")
        elif status == "confirmed":
            return ("✅ Заявка принята! Менеджер свяжется с вами в ближайшее время.\n\n"
                    "📞 Можете позвонить сами: +7 771 013-00-20\n"
                    "🕐 Время работы: Пн-Пт 9:00-19:00")
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

    def _format_pc_build(self, data: dict) -> str:
        """Format PC build response - DETERMINISTIC, no LLM."""
        build = data.get("build", {})
        if not build:
            return "Не удалось собрать ПК. Попробуйте изменить бюджет."

        lines = ["**Ваша сборка ПК:**\n"]
        total = 0

        # Component display order and names
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
                display_name = component_names.get(comp_type, comp_type.upper())
                lines.append(f"• **{display_name}**: {name}")
                lines.append(f"  Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")

        # Add peripherals if present
        peripherals = data.get("peripherals", {})
        if peripherals:
            lines.append("\n**Периферия:**")
            for ptype, p in peripherals.items():
                if p:
                    name = p.get("name", "")[:50]
                    price = p.get("price", 0)
                    discount = p.get("discount_price", 0)
                    total += discount or price
                    lines.append(f"• {ptype}: {name}")
                    lines.append(f"  Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸")

        lines.append(f"\n**Итого картой: {total:,.0f} ₸**")

        # Show warnings
        warnings = data.get("warnings", [])
        if warnings:
            lines.append("\n⚠️ **Предупреждения:**")
            for w in warnings:
                lines.append(f"• {w}")

        # Show compatibility notes
        compat_notes = data.get("compatibility_notes", [])
        if compat_notes:
            lines.append("\n❌ **Проблемы совместимости:**")
            for note in compat_notes:
                lines.append(f"• {note}")

        return "\n".join(lines)

    def _format_alternatives(self, data: dict) -> str:
        """Format component alternatives - DETERMINISTIC."""
        alts = data.get("alternatives", [])
        comp_type = data.get("component_type", "компонента")
        warning = data.get("warning")

        component_names = {
            "cpu": "процессора",
            "motherboard": "материнской платы",
            "ram": "оперативной памяти",
            "gpu": "видеокарты",
            "storage": "накопителя",
            "psu": "блока питания",
            "case": "корпуса",
        }
        display_name = component_names.get(comp_type, comp_type)

        # Show warning first if present (e.g., platform change warning)
        lines = []
        if warning:
            lines.append(warning)
            lines.append("")

        if not alts:
            if warning:
                return "\n".join(lines)
            return f"Альтернативы для {display_name} не найдены."

        lines.append(f"**Альтернативы для {display_name}:**\n")
        for i, p in enumerate(alts[:5], 1):
            name = p.get("name", "")[:65]
            price = p.get("price", 0)
            discount = p.get("discount_price", 0)
            lines.append(f"{i}. {name}")
            lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")

        lines.append("Выберите номер для замены.")
        return "\n".join(lines)

    def _format_selection(self, data: dict) -> str:
        """Format item selection response - DETERMINISTIC."""
        action = data.get("action", "")
        selected = data.get("selected", {})

        if action == "replaced_component":
            # Show the updated build
            current_build = data.get("current_build", {})
            if current_build:
                return self._format_pc_build(current_build)
            name = selected.get("name", "")
            return f"✅ Компонент заменён: {name}"

        if action == "added_peripheral":
            current_build = data.get("current_build", {})
            if current_build:
                return self._format_pc_build(current_build)
            name = selected.get("name", "")
            return f"✅ Добавлено: {name}"

        # General selection
        name = selected.get("name", "товар")
        price = selected.get("price", 0)
        discount = selected.get("discount_price", 0)
        return f"✅ Выбран: {name}\nРассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸"

    def _format_peripherals(self, data: dict) -> str:
        """Format peripherals list - DETERMINISTIC."""
        perips = data.get("peripherals", [])
        ptype = data.get("peripheral_type", "периферия")

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
            lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸\n")

        lines.append("Выберите номер для добавления к сборке.")
        return "\n".join(lines)

    def _format_search_results(self, data: dict) -> str:
        """Format search results - DETERMINISTIC."""
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
            lines.append(f"   Рассрочка: {price:,.0f} ₸ | Картой: {discount:,.0f} ₸")
            if stock > 0:
                lines.append(f"   В наличии: {stock} шт.\n")
            else:
                lines.append(f"   Нет в наличии\n")

        return "\n".join(lines)

    def _format_clarification(self, data: dict) -> str:
        """Format clarification question - DETERMINISTIC."""
        question = data.get("question", "Уточните ваш запрос")
        options = data.get("options", [])

        lines = [question, ""]
        for opt in options:
            text = opt.get("text", "")
            lines.append(f"• {text}")

        return "\n".join(lines)

    def _format_context(self, context: dict) -> str:
        """Format context for the prompt."""
        if not context:
            return "Контекст пуст (новая сессия)"

        lines = []
        if context.get("current_build"):
            build = context["current_build"]
            total = build.get("total_price", 0)
            original_budget = context.get("original_budget", total)
            components = list(build.get("build", {}).keys())
            lines.append(f"- Есть сборка ПК на сумму {total:,.0f} ₸ (запрошенный бюджет был {original_budget:,.0f} ₸)")
            lines.append(f"  Компоненты: {', '.join(components)}")

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
            "clarify_intent": Intent.GENERAL,  # Use general for clarifications
            "general_response": Intent.GENERAL,
        }
        return mapping.get(tool_name, Intent.GENERAL)
