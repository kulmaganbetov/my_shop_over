"""Tool definitions for LLM orchestrator.

This module defines all available tools that the LLM can call.
Each tool has a name, description, and parameter schema.
"""

from typing import Any

# Tool definitions for the LLM orchestrator
TOOLS = [
    {
        "name": "search_products",
        "description": "Поиск товаров в каталоге по запросу, категории или фильтрам цены. "
                       "Используй для любых запросов о товарах: 'покажи видеокарты', 'найди мышку', 'ноутбуки до 500000'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос (название товара, категория, бренд)"
                },
                "category": {
                    "type": "string",
                    "description": "Категория товара (Видеокарты, Процессоры, Мониторы, и т.д.)"
                },
                "min_price": {
                    "type": "number",
                    "description": "Минимальная цена в тенге"
                },
                "max_price": {
                    "type": "number",
                    "description": "Максимальная цена в тенге"
                },
                "manufacturer": {
                    "type": "string",
                    "description": "Производитель (AMD, Intel, NVIDIA, и т.д.)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "build_pc",
        "description": "Собрать конфигурацию ПК под указанный бюджет и назначение. "
                       "Используй для запросов: 'собери пк', 'хочу сборку', 'нужен компьютер для игр'.",
        "parameters": {
            "type": "object",
            "properties": {
                "budget": {
                    "type": "number",
                    "description": "Бюджет в тенге. Если не указан - 450000 для обычного, 500000 для игрового."
                },
                "purpose": {
                    "type": "string",
                    "enum": ["gaming", "office", "work", "streaming", "general"],
                    "description": "Назначение ПК: gaming (игры), office (офис), work (работа/дизайн), streaming, general"
                }
            },
            "required": []
        }
    },
    {
        "name": "modify_build",
        "description": "Изменить бюджет текущей сборки (дороже/дешевле). "
                       "Используй когда пользователь говорит: 'дороже', 'дешевле', 'более дорогую сборку'.",
        "parameters": {
            "type": "object",
            "properties": {
                "modifier": {
                    "type": "string",
                    "enum": ["higher", "lower"],
                    "description": "higher = дороже, lower = дешевле"
                }
            },
            "required": ["modifier"]
        }
    },
    {
        "name": "get_alternatives",
        "description": "Получить альтернативные варианты для компонента сборки. "
                       "Используй для: 'замени видеокарту', 'другой процессор', 'подешевле процессор', "
                       "'нужен Core i5', 'хочу Ryzen 7'.",
        "parameters": {
            "type": "object",
            "properties": {
                "component_type": {
                    "type": "string",
                    "enum": ["cpu", "gpu", "motherboard", "ram", "storage", "psu", "case", "cooler",
                             "процессор", "видеокарта", "материнка", "память", "накопитель", "бп", "корпус", "кулер"],
                    "description": "Тип компонента для замены"
                },
                "preference": {
                    "type": "string",
                    "description": "Предпочтение: дешевле, дороже, Intel, AMD, Core i5, Core i7, Ryzen 5, Ryzen 7 и т.д. ПЕРЕДАВАЙ ПОЛНЫЙ ТЕКСТ ЗАПРОСА!"
                }
            },
            "required": ["component_type"]
        }
    },
    {
        "name": "select_item",
        "description": "Выбрать товар из предложенного списка по номеру. "
                       "Используй когда пользователь говорит: '1', 'первый', 'выбираю второй'.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {
                    "type": "number",
                    "description": "Номер выбранного товара (1, 2, 3...)"
                }
            },
            "required": ["number"]
        }
    },
    {
        "name": "add_peripheral",
        "description": "Добавить периферию к сборке (мышь, клавиатура, монитор). "
                       "Используй для: 'добавь мышку', 'нужен монитор', 'покажи клавиатуры'. "
                       "ВАЖНО: если показана периферия и пользователь говорит 'дешевле'/'дороже' - "
                       "используй этот инструмент снова с другим бюджетом!",
        "parameters": {
            "type": "object",
            "properties": {
                "peripheral_type": {
                    "type": "string",
                    "enum": ["mouse", "keyboard", "monitor", "headset", "mousepad", "webcam", "монитор", "мышь", "клавиатура", "наушники"],
                    "description": "Тип периферии"
                },
                "budget": {
                    "type": "number",
                    "description": "Максимальный бюджет на периферию в тенге. Для 'дешевле' - меньше текущих цен."
                }
            },
            "required": ["peripheral_type"]
        }
    },
    {
        "name": "show_current_build",
        "description": "Показать текущую сборку ПК. "
                       "Используй для: 'покажи сборку', 'что выбрано', 'моя сборка'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "show_specs",
        "description": "Показать характеристики товаров или сборки с анализом плюсов/минусов. "
                       "Используй для: 'характеристики', 'подробнее', 'плюсы и минусы'.",
        "parameters": {
            "type": "object",
            "properties": {
                "with_analysis": {
                    "type": "boolean",
                    "description": "Добавить анализ плюсов/минусов"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_delivery_info",
        "description": "Получить информацию о магазинах, доставке, оплате, контактах. "
                       "Используй для: 'доставка', 'оплата', 'адрес', 'телефон', 'режим работы', 'открыто ли'.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["phone", "address", "hours", "delivery", "payment", "almaty", "astana", "pavlodar", "all"],
                    "description": "Конкретная тема: phone (телефон), address (адрес), hours (режим работы), delivery (доставка), payment (оплата), almaty/astana/pavlodar (конкретный город), all (всё)"
                },
                "city": {
                    "type": "string",
                    "description": "Город для которого нужна информация (Алматы, Астана, Павлодар)"
                }
            },
            "required": []
        }
    },
    {
        "name": "call_manager",
        "description": "Запросить звонок менеджера. "
                       "Используй для: 'позвоните мне', 'нужен менеджер', 'хочу поговорить с человеком'.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Причина обращения"
                }
            },
            "required": []
        }
    },
    {
        "name": "answer_faq",
        "description": "Ответить на частый вопрос о магазине, гарантии, возврате. "
                       "Используй для: 'гарантия', 'возврат', 'как оформить заказ'.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Тема вопроса"
                }
            },
            "required": []
        }
    },
    {
        "name": "general_response",
        "description": "Общий ответ на приветствия и разговор. "
                       "Используй для: 'привет', 'спасибо', 'пока' и общих вопросов.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def get_tools_for_prompt() -> str:
    """Format tools as a string for the LLM prompt."""
    lines = ["Доступные функции:\n"]
    for tool in TOOLS:
        lines.append(f"### {tool['name']}")
        lines.append(f"Описание: {tool['description']}")
        params = tool["parameters"]["properties"]
        if params:
            lines.append("Параметры:")
            for name, info in params.items():
                required = name in tool["parameters"].get("required", [])
                req_mark = " (обязательный)" if required else ""
                lines.append(f"  - {name}: {info.get('description', '')}{req_mark}")
        lines.append("")
    return "\n".join(lines)


def get_tool_by_name(name: str) -> dict | None:
    """Get tool definition by name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None
