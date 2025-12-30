"""Conversation state management.

Tracks conversation flow and enables context-aware responses.
State machine: idle -> searching/building -> reviewing -> finalizing
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any


class ConversationState(str, Enum):
    """Conversation states for the state machine."""

    IDLE = "idle"  # No active task
    SEARCHING = "searching"  # User is searching for products
    BUILDING = "building"  # User is building a PC
    REVIEWING = "reviewing"  # Reviewing alternatives/options
    FINALIZING = "finalizing"  # Finalizing build/purchase
    MANAGER = "manager"  # Handed off to manager


class LastAction(str, Enum):
    """Last action type for continuation logic."""

    NONE = "none"
    SEARCH = "search"
    BUILD_PC = "build_pc"
    SHOW_ALTERNATIVES = "show_alternatives"
    SHOW_PERIPHERALS = "show_peripherals"
    SHOW_BUILD = "show_build"


class Intent(str, Enum):
    """Simplified core intents."""

    SEARCH_PRODUCT = "search_product"
    BUILD_PC = "build_pc"
    MODIFY_BUILD = "modify_build"  # дороже/дешевле for current build/search
    REPLACE_COMPONENT = "replace_component"
    SELECT_ITEM = "select_item"
    ADD_PERIPHERAL = "add_peripheral"
    SHOW_BUILD = "show_build"
    ASK_QUESTION = "ask_question"
    DELIVERY_INFO = "delivery_info"
    CALL_MANAGER = "call_manager"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ConversationContext:
    """Tracks conversation state and history."""

    state: ConversationState = ConversationState.IDLE
    last_action: LastAction = LastAction.NONE

    # Current build state
    current_build: Optional[dict] = None
    original_budget: int = 0

    # Last shown items for continuation
    last_shown_products: list = field(default_factory=list)
    last_search_query: str = ""
    last_alternatives: list = field(default_factory=list)
    last_component_type: str = ""
    last_peripherals: list = field(default_factory=list)
    last_peripheral_type: str = ""

    # Intent stack for context
    intent_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "state": self.state.value,
            "last_action": self.last_action.value,
            "current_build": self.current_build,
            "original_budget": self.original_budget,
            "last_shown_products": self.last_shown_products,
            "last_search_query": self.last_search_query,
            "last_alternatives": self.last_alternatives,
            "last_component_type": self.last_component_type,
            "last_peripherals": self.last_peripherals,
            "last_peripheral_type": self.last_peripheral_type,
            "intent_history": self.intent_history[-5:],  # Keep last 5
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Create from dictionary."""
        if not data:
            return cls()
        return cls(
            state=ConversationState(data.get("state", "idle")),
            last_action=LastAction(data.get("last_action", "none")),
            current_build=data.get("current_build"),
            original_budget=data.get("original_budget", 0),
            last_shown_products=data.get("last_shown_products", []),
            last_search_query=data.get("last_search_query", ""),
            last_alternatives=data.get("last_alternatives", []),
            last_component_type=data.get("last_component_type", ""),
            last_peripherals=data.get("last_peripherals", []),
            last_peripheral_type=data.get("last_peripheral_type", ""),
            intent_history=data.get("intent_history", []),
        )

    def has_build(self) -> bool:
        """Check if there's an active build."""
        return bool(self.current_build and self.current_build.get("build"))

    def has_shown_items(self) -> bool:
        """Check if there are items shown that can be modified."""
        return bool(
            self.last_shown_products or
            self.last_alternatives or
            self.last_peripherals
        )

    def get_last_shown_items(self) -> list:
        """Get the most recent shown items."""
        if self.last_alternatives:
            return self.last_alternatives
        if self.last_peripherals:
            return self.last_peripherals
        return self.last_shown_products


# Keyword patterns for intent detection
INTENT_KEYWORDS = {
    Intent.BUILD_PC: [
        "собери", "сборка", "сборку", "пк", "компьютер",
        "хочу пк", "нужен пк", "build", "gaming pc",
        "игровой компьютер", "рабочий пк", "офисный пк"
    ],
    Intent.SEARCH_PRODUCT: [
        "покажи", "найди", "поиск", "купить", "в наличии",
        "есть ли", "цена", "сколько стоит", "хочу купить"
    ],
    Intent.MODIFY_BUILD: [
        "дороже", "дешевле", "подороже", "подешевле",
        "более дорогую", "более дешевую", "бюджетнее"
    ],
    Intent.REPLACE_COMPONENT: [
        "замени", "поменяй", "смени", "другой", "другую",
        "альтернатив", "вместо"
    ],
    Intent.ADD_PERIPHERAL: [
        "добавь", "нужен монитор", "нужна мышь", "нужна клавиатура",
        "добавить периферию"
    ],
    Intent.SHOW_BUILD: [
        "покажи сборку", "моя сборка", "текущая сборка",
        "конфигурация", "что выбрано"
    ],
    Intent.DELIVERY_INFO: [
        "доставка", "адрес", "телефон", "магазин", "оплата",
        "режим работы", "как добраться", "где находится"
    ],
    Intent.CALL_MANAGER: [
        "менеджер", "позвоните", "оператор", "человек",
        "живой человек", "с человеком"
    ],
    Intent.GREETING: [
        "привет", "здравствуй", "добрый день", "добрый вечер",
        "салем", "hello", "hi", "хай"
    ],
    Intent.SELECT_ITEM: [
        "первый", "второй", "третий", "четвертый", "пятый",
        "выбираю", "беру", "этот", "выбрал"
    ],
}

# Product category keywords
CATEGORY_KEYWORDS = {
    "Видеокарты": ["видеокарт", "gpu", "rtx", "gtx", "geforce", "radeon", "rx ", "rx5", "rx6", "rx7"],
    "Процессоры": ["процессор", "cpu", "ryzen", "core i", "intel", "amd"],
    "Материнские платы": ["материнск", "motherboard", "мат плат"],
    "Оперативная память": ["оперативн", "ram", "память", "ddr4", "ddr5"],
    "SSD накопители": ["ssd", "ссд", "накопитель", "nvme"],
    "Блоки питания": ["блок питан", "psu", "бп "],
    "Корпуса": ["корпус", "case", "кейс"],
    "Мониторы": ["монитор", "дисплей", "экран"],
    "Мыши": ["мышь", "мышк", "mouse"],
    "Клавиатуры": ["клавиатур", "keyboard", "клав"],
    "Наушники": ["наушник", "гарнитур", "headset"],
    "Ноутбуки": ["ноутбук", "laptop", "лэптоп"],
}

# Component type keywords for replacement
COMPONENT_KEYWORDS = {
    "cpu": ["процессор", "проц", "cpu", "цп"],
    "gpu": ["видеокарт", "видюх", "gpu", "график"],
    "motherboard": ["материнск", "motherboard", "мат плат", "мп"],
    "ram": ["оперативн", "ram", "озу", "память"],
    "storage": ["накопитель", "ssd", "диск", "storage"],
    "psu": ["блок питан", "psu", "бп"],
    "case": ["корпус", "case", "кейс"],
    "cooler": ["кулер", "охлаждени", "cooler"],
}

# Peripheral keywords
PERIPHERAL_KEYWORDS = {
    "monitor": ["монитор", "дисплей", "экран"],
    "mouse": ["мышь", "мышк"],
    "keyboard": ["клавиатур", "клав"],
    "headset": ["наушник", "гарнитур"],
    "mousepad": ["коврик"],
    "webcam": ["веб-камер", "камер"],
}


def detect_intent_from_keywords(message: str, context: ConversationContext) -> tuple[Intent, dict]:
    """Detect intent using keyword matching.

    Returns:
        tuple: (intent, params dict)
    """
    msg_lower = message.lower().strip()
    params = {}

    # 1. Check for number selection (1-9)
    if msg_lower.isdigit() or msg_lower in ["первый", "второй", "третий", "четвертый", "пятый"]:
        number_map = {"первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5}
        num = int(msg_lower) if msg_lower.isdigit() else number_map.get(msg_lower, 1)
        if context.has_shown_items():
            return Intent.SELECT_ITEM, {"number": num}

    # 2. Check for continuation (дороже/дешевле without specifics)
    for kw in INTENT_KEYWORDS[Intent.MODIFY_BUILD]:
        if kw in msg_lower:
            modifier = "lower" if any(w in msg_lower for w in ["дешевле", "подешевле", "бюджетнее"]) else "higher"

            # Apply to last action
            if context.last_action == LastAction.BUILD_PC and context.has_build():
                return Intent.MODIFY_BUILD, {"modifier": modifier}
            elif context.last_action == LastAction.SHOW_ALTERNATIVES:
                return Intent.REPLACE_COMPONENT, {
                    "component_type": context.last_component_type,
                    "preference": "дешевле" if modifier == "lower" else "дороже"
                }
            elif context.last_action == LastAction.SHOW_PERIPHERALS:
                return Intent.ADD_PERIPHERAL, {
                    "peripheral_type": context.last_peripheral_type,
                    "preference": "дешевле" if modifier == "lower" else "дороже"
                }
            elif context.last_action == LastAction.SEARCH:
                # Re-search with price modifier
                return Intent.SEARCH_PRODUCT, {
                    "query": context.last_search_query,
                    "price_modifier": modifier
                }

    # 3. Check for greeting (short messages)
    if len(msg_lower) < 20:
        for kw in INTENT_KEYWORDS[Intent.GREETING]:
            if kw in msg_lower:
                return Intent.GREETING, {}

    # 4. Check for manager call
    for kw in INTENT_KEYWORDS[Intent.CALL_MANAGER]:
        if kw in msg_lower:
            return Intent.CALL_MANAGER, {}

    # 5. Check for delivery info
    for kw in INTENT_KEYWORDS[Intent.DELIVERY_INFO]:
        if kw in msg_lower:
            topic = "all"
            if "телефон" in msg_lower:
                topic = "phone"
            elif "адрес" in msg_lower:
                topic = "address"
            elif "режим" in msg_lower or "работ" in msg_lower:
                topic = "hours"
            elif "оплат" in msg_lower:
                topic = "payment"
            elif "доставк" in msg_lower:
                topic = "delivery"
            return Intent.DELIVERY_INFO, {"topic": topic}

    # 6. Check for show build
    for kw in INTENT_KEYWORDS[Intent.SHOW_BUILD]:
        if kw in msg_lower:
            return Intent.SHOW_BUILD, {}

    # 7. Check for PC build request
    build_score = 0
    for kw in INTENT_KEYWORDS[Intent.BUILD_PC]:
        if kw in msg_lower:
            build_score += 1

    if build_score >= 1:
        # Extract budget if mentioned
        budget = extract_budget(msg_lower)
        purpose = extract_purpose(msg_lower)
        return Intent.BUILD_PC, {"budget": budget, "purpose": purpose}

    # 8. Check for component replacement (need existing build)
    if context.has_build():
        for kw in INTENT_KEYWORDS[Intent.REPLACE_COMPONENT]:
            if kw in msg_lower:
                comp_type = detect_component_type(msg_lower)
                if comp_type:
                    return Intent.REPLACE_COMPONENT, {
                        "component_type": comp_type,
                        "preference": extract_preference(msg_lower)
                    }

    # 9. Check for add peripheral
    peripheral = detect_peripheral_type(msg_lower)
    if peripheral:
        for kw in INTENT_KEYWORDS[Intent.ADD_PERIPHERAL]:
            if kw in msg_lower:
                return Intent.ADD_PERIPHERAL, {"peripheral_type": peripheral}
        # Also if just mentions peripheral by name and has build
        if context.has_build():
            return Intent.ADD_PERIPHERAL, {"peripheral_type": peripheral}

    # 10. Check for product search
    category = detect_category(msg_lower)
    if category or any(kw in msg_lower for kw in INTENT_KEYWORDS[Intent.SEARCH_PRODUCT]):
        return Intent.SEARCH_PRODUCT, {
            "query": message,
            "category": category
        }

    # 11. Check if it's a question
    if "?" in message or any(w in msg_lower for w in ["как", "что", "почему", "зачем", "можно ли", "нужно ли"]):
        return Intent.ASK_QUESTION, {"question": message}

    # Default: unknown (will use LLM)
    return Intent.UNKNOWN, {}


def extract_budget(text: str) -> Optional[int]:
    """Extract budget from text."""
    import re

    # Look for patterns like "500000", "500к", "500 тыс"
    patterns = [
        r'(\d+)\s*(?:тысяч|тыс|к\b)',  # 500 тысяч, 500к
        r'(\d{6,})',  # 500000
        r'(\d+)\s*₸',  # 500000₸
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num = int(match.group(1))
            # If small number, assume thousands
            if num < 1000:
                return num * 1000
            return num

    return None


def extract_purpose(text: str) -> str:
    """Extract PC purpose from text."""
    if any(w in text for w in ["игр", "gaming", "гейм"]):
        return "gaming"
    elif any(w in text for w in ["работ", "work", "дизайн", "програм"]):
        return "work"
    elif any(w in text for w in ["офис", "office", "документ"]):
        return "office"
    elif any(w in text for w in ["стрим", "stream"]):
        return "streaming"
    return "gaming"  # Default for PC builds


def detect_category(text: str) -> Optional[str]:
    """Detect product category from text."""
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return category
    return None


def detect_component_type(text: str) -> Optional[str]:
    """Detect component type from text."""
    for comp_type, keywords in COMPONENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return comp_type
    return None


def detect_peripheral_type(text: str) -> Optional[str]:
    """Detect peripheral type from text."""
    for ptype, keywords in PERIPHERAL_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return ptype
    return None


def extract_preference(text: str) -> Optional[str]:
    """Extract user preference for component replacement."""
    # Brand preferences
    brands = ["intel", "amd", "nvidia", "msi", "asus", "gigabyte", "palit", "zotac", "sapphire"]
    for brand in brands:
        if brand in text.lower():
            return brand

    # Price preferences
    if any(w in text for w in ["дешев", "бюджет", "недорог"]):
        return "дешевле"
    if any(w in text for w in ["дорог", "лучш", "топов", "мощн"]):
        return "дороже"

    return None
