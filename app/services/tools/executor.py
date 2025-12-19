"""Tool executor - executes tools called by the orchestrator."""

import logging
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.db.repositories import ChatRepository, ProductRepository
from app.schemas.common import ProductSchema
from app.schemas.llm import PCBuildParams, ProductSearchParams
from app.services.pc_build import PCBuildService
from app.services.product_search import ProductSearchService

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes backend tools and returns structured results."""

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
        """Execute a tool and return the result.

        Returns:
            dict with 'success', 'data', and optionally 'error' keys
        """
        try:
            method = getattr(self, f"_tool_{tool_name}", None)
            if not method:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}

            result = await method(params, session_id)
            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_context(self, session_id: str) -> dict:
        """Get session context."""
        chat_session = await self.chat_repo.get_session_by_id(session_id)
        if chat_session and chat_session.context:
            return chat_session.context
        return {}

    async def _update_context(self, session_id: str, updates: dict):
        """Update session context."""
        await self.chat_repo.update_session_context(session_id, updates)

    # ===== Tool implementations =====

    async def _tool_search_products(self, params: dict, session_id: str) -> dict:
        """Search for products."""
        from app.llm.service import LLMService

        search_params = ProductSearchParams(
            query=params.get("query", ""),
            category=params.get("category"),
            min_price=params.get("min_price"),
            max_price=params.get("max_price"),
            manufacturer=params.get("manufacturer"),
            in_stock_only=True,
            limit=5,
        )

        search_service = ProductSearchService(self.session, LLMService())
        result = await search_service.search(search_params)

        products_data = [p.model_dump() for p in result.products]

        # Save to context for selection
        await self._update_context(session_id, {
            "last_search_results": products_data,
            "last_search_query": params.get("query", ""),
        })

        return {
            "products": products_data,
            "total": result.total,
            "query": params.get("query", ""),
        }

    async def _tool_build_pc(self, params: dict, session_id: str) -> dict:
        """Build a PC configuration."""
        from app.llm.service import LLMService

        budget = params.get("budget")
        purpose = params.get("purpose", "gaming")

        # Default budgets
        if not budget:
            budgets = {"gaming": 500000, "office": 200000, "work": 700000}
            budget = budgets.get(purpose, 450000)

        build_params = PCBuildParams(
            budget=budget,
            purpose=purpose,
        )

        build_service = PCBuildService(self.session, LLMService())
        result = await build_service.recommend_build(build_params)

        build_data = result.model_dump()

        # Save to context
        await self._update_context(session_id, {"current_build": build_data})

        return build_data

    async def _tool_modify_build(self, params: dict, session_id: str) -> dict:
        """Modify current build budget (higher/lower)."""
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

        # Rebuild with new budget
        build_params = PCBuildParams(
            budget=new_budget,
            purpose="gaming",
        )

        build_service = PCBuildService(self.session, LLMService())
        result = await build_service.recommend_build(build_params)

        build_data = result.model_dump()
        await self._update_context(session_id, {"current_build": build_data})

        return build_data

    async def _tool_get_alternatives(self, params: dict, session_id: str) -> dict:
        """Get alternative components."""
        from app.llm.service import LLMService

        component_type = params.get("component_type", "gpu")
        preference = params.get("preference")

        context = await self._get_context(session_id)
        current_build = context.get("current_build", {})

        if not current_build or not current_build.get("build"):
            return {"error": "Сначала нужно собрать ПК"}

        build_service = PCBuildService(self.session, LLMService())
        alternatives = await build_service.get_component_alternatives(
            component_type=component_type,
            current_build=current_build.get("build", {}),
            preference=preference,
            limit=5,
        )

        alternatives_data = [alt.model_dump() for alt in alternatives]

        # Save for selection
        await self._update_context(session_id, {
            "last_alternatives": alternatives_data,
            "last_component_type": component_type,
        })

        return {
            "alternatives": alternatives_data,
            "component_type": component_type,
            "preference": preference,
        }

    async def _tool_select_item(self, params: dict, session_id: str) -> dict:
        """Select an item from a list by number."""
        number = int(params.get("number", 1)) - 1  # 0-indexed

        context = await self._get_context(session_id)

        # Check what lists are available
        alternatives = context.get("last_alternatives", [])
        peripherals = context.get("last_peripherals", [])
        search_results = context.get("last_search_results", [])

        items_list = alternatives or peripherals or search_results

        if not items_list:
            return {"error": "Нет вариантов для выбора"}

        if number < 0 or number >= len(items_list):
            return {"error": f"Выберите номер от 1 до {len(items_list)}"}

        selected = items_list[number]

        # If selecting alternative component - update build
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

        # If selecting peripheral - add to build
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

        # Just a product selection
        return {
            "selected": selected,
            "action": "selected_product",
        }

    async def _tool_add_peripheral(self, params: dict, session_id: str) -> dict:
        """Add peripheral to build."""
        peripheral_type = params.get("peripheral_type", "mouse")
        budget = params.get("budget")

        # Map types to categories
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

        return {
            "peripherals": products_data,
            "peripheral_type": peripheral_type,
        }

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

    async def _tool_get_delivery_info(self, params: dict, session_id: str) -> dict:
        """Get delivery information."""
        return {
            "delivery_info": {
                "almaty": {
                    "free_from": 50000,
                    "days": "1-2 рабочих дня",
                    "pickup": True,
                },
                "other_cities": {
                    "method": "Казпочта или курьерские службы",
                    "days": "3-7 рабочих дней",
                },
                "payment": ["Картой онлайн", "Рассрочка 0-0-12 от Kaspi", "Наличными"],
            }
        }

    async def _tool_call_manager(self, params: dict, session_id: str) -> dict:
        """Request manager callback."""
        reason = params.get("reason", "запрос клиента")

        context = await self._get_context(session_id)
        awaiting = context.get("awaiting_manager_confirmation", False)

        if not awaiting:
            # First request - ask for confirmation
            await self._update_context(session_id, {
                "awaiting_manager_confirmation": True,
                "manager_reason": reason,
            })
            return {"status": "awaiting_confirmation"}
        else:
            # Confirmed
            await self._update_context(session_id, {
                "awaiting_manager_confirmation": False,
                "manager_requested": True,
            })
            logger.warning(f"[MANAGER_REQUEST] session={session_id} reason={reason}")
            return {"status": "confirmed", "reason": reason}

    async def _tool_answer_faq(self, params: dict, session_id: str) -> dict:
        """Answer FAQ question."""
        topic = params.get("topic", "")

        # Static FAQ answers
        faq_answers = {
            "гарантия": "Гарантия на все товары - от 12 до 36 месяцев в зависимости от производителя.",
            "возврат": "Возврат товара возможен в течение 14 дней при сохранении товарного вида.",
            "оплата": "Принимаем оплату картой, наличными и в рассрочку через Kaspi.",
            "доставка": "Бесплатная доставка по Алматы от 50,000₸. В другие города - 3-7 дней.",
        }

        # Find matching FAQ
        topic_lower = topic.lower()
        for key, answer in faq_answers.items():
            if key in topic_lower:
                return {"answer": answer, "topic": key}

        return {"answer": None, "topic": topic}

    async def _tool_general_response(self, params: dict, session_id: str) -> dict:
        """Handle general conversation."""
        return {"type": "general"}
