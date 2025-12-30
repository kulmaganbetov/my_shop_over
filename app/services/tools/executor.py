"""Tool executor - standardized tool responses.

Response format:
{
    "success": bool,
    "data": dict,  # Tool-specific data
    "metadata": dict  # Optional metadata
}
"""

import logging
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import ChatRepository, ProductRepository
from app.schemas.common import ProductSchema
from app.schemas.llm import PCBuildParams, ProductSearchParams
from app.services.pc_build import PCBuildService
from app.services.product_search import ProductSearchService

logger = logging.getLogger(__name__)

# Peripheral name mapping
PERIPHERAL_NAME_MAPPING = {
    "монитор": "monitor", "дисплей": "monitor", "экран": "monitor",
    "мышь": "mouse", "мышка": "mouse", "мышку": "mouse",
    "клавиатура": "keyboard", "клавиатуру": "keyboard", "клава": "keyboard",
    "наушники": "headset", "гарнитура": "headset", "гарнитуру": "headset",
    "коврик": "mousepad", "коврик для мыши": "mousepad",
    "веб-камера": "webcam", "вебкамера": "webcam", "камера": "webcam",
}


def normalize_peripheral_type(peripheral_type: str) -> str:
    """Convert Russian peripheral name to English key."""
    if not peripheral_type:
        return "mouse"
    normalized = peripheral_type.lower().strip()
    return PERIPHERAL_NAME_MAPPING.get(normalized, normalized)


class ToolExecutor:
    """Executes backend tools with standardized responses."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.product_repo = ProductRepository(session)
        self.chat_repo = ChatRepository(session)

    async def execute(
        self,
        tool_name: str,
        params: dict,
        session_id: str,
    ) -> dict[str, Any]:
        """Execute tool and return standardized result.

        Returns:
            {"success": bool, "data": dict, "metadata": dict}
        """
        try:
            method = getattr(self, f"_tool_{tool_name}", None)
            if not method:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "data": {}
                }

            result = await method(params, session_id)
            return {"success": True, "data": result, "metadata": {"tool": tool_name}}

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e), "data": {}}

    async def _get_context(self, session_id: str) -> dict:
        """Get session context."""
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        if chat_session and chat_session.context:
            return chat_session.context
        return {}

    async def _update_context(self, session_id: str, updates: dict):
        """Update session context."""
        await self.chat_repo.update_session_context(session_id, updates)

    # ===== Product Search =====

    async def _tool_search_products(self, params: dict, session_id: str) -> dict:
        """Search for products."""
        from app.llm.service import LLMService

        query = params.get("query", "").strip()
        query_lower = query.lower()

        # Try exact SKU/code search first
        sku_product = await self.product_repo.get_by_sku(query)
        if sku_product:
            products_data = [ProductSchema.model_validate(sku_product).model_dump()]
            await self._update_context(session_id, {
                "last_search_results": products_data,
                "last_search_query": query,
            })
            return {"products": products_data, "total": 1, "query": query}

        # Try kaspi_code search
        kaspi_product = await self.product_repo.get_by_kaspi_code(query)
        if kaspi_product:
            products_data = [ProductSchema.model_validate(kaspi_product).model_dump()]
            await self._update_context(session_id, {
                "last_search_results": products_data,
                "last_search_query": query,
            })
            return {"products": products_data, "total": 1, "query": query}

        # Auto-detect filters
        manufacturer = params.get("manufacturer") or self._detect_manufacturer(query_lower)
        category = params.get("category") or self._detect_category(query_lower)

        search_params = ProductSearchParams(
            query=query,
            category=category,
            min_price=params.get("min_price"),
            max_price=params.get("max_price"),
            manufacturer=manufacturer,
            in_stock_only=True,
            limit=5,
        )

        search_service = ProductSearchService(self.session, LLMService())
        result = await search_service.search(search_params)

        products_data = [p.model_dump() for p in result.products]

        await self._update_context(session_id, {
            "last_search_results": products_data,
            "last_search_query": query,
        })

        return {"products": products_data, "total": result.total_count, "query": query}

    def _detect_manufacturer(self, query: str) -> Optional[str]:
        """Detect manufacturer from query."""
        nvidia_kw = ["nvidia", "geforce", "rtx", "gtx", "quadro"]
        amd_kw = ["amd", "radeon", "ryzen", "rx ", "rx5", "rx6", "rx7", "athlon"]
        intel_kw = ["intel", "core i", "xeon", "celeron", "pentium", "arc "]

        for kw in nvidia_kw:
            if kw in query:
                return "NVIDIA"
        for kw in amd_kw:
            if kw in query:
                return "AMD"
        for kw in intel_kw:
            if kw in query:
                return "Intel"
        return None

    def _detect_category(self, query: str) -> Optional[str]:
        """Detect category from query."""
        # GPU model names first
        gpu_kw = ["rtx", "gtx", "rx ", "rx5", "rx6", "rx7", "radeon", "geforce", "gt 7"]
        for kw in gpu_kw:
            if kw in query:
                return "Видеокарты"

        category_map = {
            "Видеокарты": ["видеокарт", "gpu", "graphics"],
            "Процессоры": ["процессор", "cpu", "проц", "ryzen", "core i"],
            "Материнские платы": ["материнск", "motherboard", "мат плат"],
            "Оперативная память": ["оперативн", "ram", "память ddr", "ddr4", "ddr5"],
            "SSD накопители": ["ssd", "ссд", "твердотельн"],
            "Жесткие диски": ["hdd", "жестк", "hard drive"],
            "Блоки питания": ["блок питан", "psu", "бп "],
            "Корпуса": ["корпус", "case", "кейс"],
            "Кулеры и охлаждение": ["кулер", "охлажден", "cooler"],
            "Мониторы": ["монитор", "дисплей", "экран"],
            "Мыши": ["мышь", "мышка", "mouse"],
            "Клавиатуры": ["клавиатур", "keyboard"],
            "Ноутбуки": ["ноутбук", "laptop", "лэптоп"],
        }

        for category, keywords in category_map.items():
            for kw in keywords:
                if kw in query:
                    return category
        return None

    # ===== PC Build =====

    async def _tool_build_pc(self, params: dict, session_id: str) -> dict:
        """Build a PC configuration."""
        from app.llm.service import LLMService

        budget = params.get("budget")
        purpose = params.get("purpose", "gaming")

        # Default budgets
        if not budget:
            defaults = {"gaming": 500000, "office": 200000, "work": 700000}
            budget = defaults.get(purpose, 450000)

        build_params = PCBuildParams(budget=budget, purpose=purpose)

        build_service = PCBuildService(self.session, LLMService())
        result = await build_service.recommend_build(build_params)

        build_data = result.model_dump()

        await self._update_context(session_id, {
            "current_build": build_data,
            "original_budget": budget,
        })

        return build_data

    async def _tool_modify_build(self, params: dict, session_id: str) -> dict:
        """Modify current build budget."""
        from app.llm.service import LLMService

        modifier = params.get("modifier", "higher")

        context = await self._get_context(session_id)
        current_build = context.get("current_build", {})

        if not current_build:
            return {"error": "Сначала нужно собрать ПК"}

        prev_price = current_build.get("total_price", 450000)

        if modifier == "higher":
            new_budget = int(prev_price * 1.5)
        else:
            new_budget = int(prev_price * 0.6)

        build_params = PCBuildParams(budget=new_budget, purpose="gaming")

        build_service = PCBuildService(self.session, LLMService())
        result = await build_service.recommend_build(build_params)

        build_data = result.model_dump()
        await self._update_context(session_id, {"current_build": build_data})

        return build_data

    # ===== Component Alternatives =====

    async def _tool_get_alternatives(self, params: dict, session_id: str) -> dict:
        """Get alternative components."""
        from app.llm.service import LLMService
        from app.services.pc_build import normalize_component_type

        component_type_raw = params.get("component_type", "gpu")
        component_type = normalize_component_type(component_type_raw)
        preference = params.get("preference")

        context = await self._get_context(session_id)
        current_build = context.get("current_build", {})

        if not current_build or not current_build.get("build"):
            return {"error": "Сначала нужно собрать ПК"}

        build_service = PCBuildService(self.session, LLMService())
        alternatives, warning = await build_service.get_component_alternatives(
            component_type=component_type,
            current_build=current_build.get("build", {}),
            preference=preference,
            limit=5,
        )

        alternatives_data = [alt.model_dump() for alt in alternatives]

        await self._update_context(session_id, {
            "last_alternatives": alternatives_data,
            "last_component_type": component_type,
        })

        result = {
            "alternatives": alternatives_data,
            "component_type": component_type,
            "preference": preference,
        }

        if warning:
            result["warning"] = warning

        return result

    # ===== Item Selection =====

    async def _tool_select_item(self, params: dict, session_id: str) -> dict:
        """Select an item from a list by number."""
        number = int(params.get("number", 1)) - 1  # 0-indexed

        context = await self._get_context(session_id)

        # Check available lists
        alternatives = context.get("last_alternatives", [])
        peripherals = context.get("last_peripherals", [])
        search_results = context.get("last_search_results", [])

        items_list = alternatives or peripherals or search_results

        if not items_list:
            return {"error": "Нет вариантов для выбора"}

        if number < 0 or number >= len(items_list):
            return {"error": f"Выберите номер от 1 до {len(items_list)}"}

        selected = items_list[number]

        # Component replacement
        if alternatives:
            component_type = context.get("last_component_type", "")
            current_build = context.get("current_build", {})

            if current_build.get("build") and component_type:
                current_build["build"][component_type] = selected

                # Recalculate total
                total = sum(
                    (c.get("discount_price") or c.get("price") or 0)
                    for c in current_build["build"].values()
                    if c
                )
                for p in current_build.get("peripherals", {}).values():
                    if p:
                        total += p.get("discount_price") or p.get("price") or 0
                current_build["total_price"] = total

                await self._update_context(session_id, {
                    "current_build": current_build,
                    "last_alternatives": [],
                })

                return {
                    "selected": selected,
                    "action": "replaced_component",
                    "component_type": component_type,
                    "current_build": current_build,
                }

        # Peripheral selection
        if peripherals:
            peripheral_type = context.get("last_peripheral_type", "mouse")
            current_build = context.get("current_build", {})

            if current_build:
                if "peripherals" not in current_build:
                    current_build["peripherals"] = {}
                current_build["peripherals"][peripheral_type] = selected

                # Recalculate total
                total = sum(
                    (c.get("discount_price") or c.get("price") or 0)
                    for c in current_build.get("build", {}).values()
                    if c
                )
                for p in current_build.get("peripherals", {}).values():
                    if p:
                        total += p.get("discount_price") or p.get("price") or 0
                current_build["total_price"] = total

                await self._update_context(session_id, {
                    "current_build": current_build,
                    "last_peripherals": [],
                })

                return {
                    "selected": selected,
                    "action": "added_peripheral",
                    "peripheral_type": peripheral_type,
                    "current_build": current_build,
                }

        # Just product selection
        return {"selected": selected, "action": "selected_product"}

    # ===== Peripherals =====

    async def _tool_add_peripheral(self, params: dict, session_id: str) -> dict:
        """Add peripheral to build."""
        peripheral_type_raw = params.get("peripheral_type", "mouse")
        peripheral_type = normalize_peripheral_type(peripheral_type_raw)
        budget = params.get("budget")

        logger.info(f"[PERIPHERAL] '{peripheral_type_raw}' -> '{peripheral_type}'")

        categories = {
            "monitor": ("Мониторы", ["Мониторы"]),
            "mouse": ("Мыши", ["Мыши"]),
            "keyboard": ("Клавиатуры", ["Клавиатуры"]),
            "headset": ("Гарнитуры", ["Гарнитуры", "Наушники"]),
            "mousepad": ("Коврики для мыши", ["Коврики"]),
            "webcam": ("Веб-камеры", ["Веб-камеры"]),
        }

        category_name, keywords = categories.get(peripheral_type, ("Мыши", ["Мыши"]))

        products = await self.product_repo.get_by_component_type(
            component_type=category_name,
            max_price=budget,
            in_stock_only=True,
            limit=5,
            search_keywords=keywords,
        )

        if not products:
            return {"error": f"{category_name} не найдены в наличии"}

        products_data = [ProductSchema.model_validate(p).model_dump() for p in products]

        await self._update_context(session_id, {
            "last_peripherals": products_data,
            "last_peripheral_type": peripheral_type,
        })

        return {"peripherals": products_data, "peripheral_type": peripheral_type}

    # ===== Build Display =====

    async def _tool_show_current_build(self, params: dict, session_id: str) -> dict:
        """Show current PC build."""
        context = await self._get_context(session_id)
        current_build = context.get("current_build", {})

        if not current_build or not current_build.get("build"):
            return {"error": "У вас пока нет сборки"}

        return {"current_build": current_build}

    async def _tool_show_specs(self, params: dict, session_id: str) -> dict:
        """Show specifications."""
        with_analysis = params.get("with_analysis", False)

        context = await self._get_context(session_id)
        current_build = context.get("current_build", {})
        search_results = context.get("last_search_results", [])

        if current_build.get("build"):
            return {
                "type": "build_specs",
                "build": current_build,
                "with_analysis": with_analysis,
            }
        elif search_results:
            return {
                "type": "product_specs",
                "products": search_results[:3],
                "with_analysis": with_analysis,
            }
        else:
            return {"error": "Нет данных для показа характеристик"}

    # ===== Delivery Info =====

    async def _tool_get_delivery_info(self, params: dict, session_id: str) -> dict:
        """Get delivery and store information."""
        topic = params.get("topic", "all")
        city = params.get("city", "").lower()

        # Detect city from topic
        if topic in ["almaty", "astana", "pavlodar"]:
            city = topic
            topic = "city"

        all_data = {
            "stores": {
                "almaty": {
                    "name": "Алматы",
                    "address": "г. Алматы, проспект Абылай хана, 7 (вход со стороны ул.Тузова)",
                    "hours": "Пн.-Пт.: 09:00-19:00, Сб.-Вс.: 09:00-17:00",
                    "phones": ["+7 747 601-03-25", "+7 7273 51-28-51"],
                },
                "astana": {
                    "name": "Астана",
                    "address": "г. Астана, проспект Республики, 72",
                    "hours": "Пн.-Пт.: 10:00-19:00, Сб.-Вс.: выходной",
                    "phones": ["+7 707 956-50-26", "+7 7172 39-52-80"],
                },
                "pavlodar": {
                    "name": "Павлодар",
                    "address": "г. Павлодар, ул. Желтоксан, 7",
                    "hours": "Пн.-Пт.: 10:00-19:00, Сб.-Вс.: 10:00-17:00",
                    "phones": ["+7 7182 77-70-55"],
                    "service_center": "+7 7182 39-36-89",
                },
            },
            "online": {
                "phone": "+7 771 013-00-20",
                "kaspi_orders": "+7 775 894-93-84",
                "email": "sales@overclockers.kz",
                "instagram": "https://www.instagram.com/over.kz/",
            },
            "delivery": {
                "pickup": "Самовывоз из магазинов",
                "courier": "Доставка курьером по городу",
                "express": "Экспресс-доставка",
                "transport": "JetLogistic, Exline, DPD, ABT",
            },
            "payment": {
                "methods": ["Картой онлайн", "Рассрочка 0-0-12 от Kaspi", "Наличными"],
            },
        }

        # Filtered responses
        if topic == "phone":
            return {
                "topic": "phone",
                "online_phone": all_data["online"]["phone"],
                "kaspi_orders": all_data["online"]["kaspi_orders"],
                "stores": {k: {"name": v["name"], "phones": v["phones"]} for k, v in all_data["stores"].items()},
            }

        if topic == "hours":
            return {
                "topic": "hours",
                "stores": {k: {"name": v["name"], "hours": v["hours"]} for k, v in all_data["stores"].items()},
            }

        if topic == "address":
            return {
                "topic": "address",
                "stores": {k: {"name": v["name"], "address": v["address"]} for k, v in all_data["stores"].items()},
            }

        if topic == "delivery":
            return {"topic": "delivery", "delivery": all_data["delivery"]}

        if topic == "payment":
            return {"topic": "payment", "payment": all_data["payment"]}

        if topic == "city" and city:
            city_key = city.replace("алматы", "almaty").replace("астана", "astana").replace("павлодар", "pavlodar")
            if city_key in all_data["stores"]:
                return {"topic": "city", "city": city_key, "store": all_data["stores"][city_key]}

        # Full data
        return {
            "topic": "all",
            "stores": all_data["stores"],
            "online": all_data["online"],
            "delivery": all_data["delivery"],
            "payment": all_data["payment"],
        }

    # ===== Manager =====

    async def _tool_call_manager(self, params: dict, session_id: str) -> dict:
        """Escalate to manager."""
        reason = params.get("reason", "запрос клиента")

        await self.chat_repo.escalate_to_manager(session_id, reason)
        logger.warning(f"[MANAGER] Session {session_id} escalated: {reason}")

        return {
            "status": "escalated",
            "reason": reason,
            "message": "Диалог передан менеджеру",
        }

    # ===== FAQ =====

    async def _tool_answer_faq(self, params: dict, session_id: str) -> dict:
        """Answer FAQ question."""
        topic = params.get("topic", "")

        faq = {
            "гарантия": "Гарантия на все товары - от 12 до 36 месяцев.",
            "возврат": "Возврат товара возможен в течение 14 дней.",
            "оплата": "Принимаем картой, наличными и в рассрочку через Kaspi.",
            "доставка": "Бесплатная доставка по Алматы от 50,000₸. В другие города - 3-7 дней.",
        }

        topic_lower = topic.lower()
        for key, answer in faq.items():
            if key in topic_lower:
                return {"answer": answer, "topic": key}

        return {"answer": None, "topic": topic}

    # ===== General =====

    async def _tool_general_response(self, params: dict, session_id: str) -> dict:
        """Handle general conversation."""
        return {"type": "general"}

    async def _tool_clarify_intent(self, params: dict, session_id: str) -> dict:
        """Ask user to clarify their intent."""
        component = params.get("component", "товар")

        component_names = {
            "cpu": "процессор",
            "gpu": "видеокарту",
            "motherboard": "материнскую плату",
            "ram": "оперативную память",
            "storage": "накопитель",
            "psu": "блок питания",
            "case": "корпус",
        }
        display_name = component_names.get(component, component)

        return {
            "type": "clarification",
            "component": display_name,
            "question": f"Уточните: заменить {display_name} в сборке или купить отдельно?",
            "options": [
                {"key": "replace", "text": f"Заменить {display_name} в сборке"},
                {"key": "buy_separate", "text": f"Купить {display_name} отдельно"},
            ],
        }
