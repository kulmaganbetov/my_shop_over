"""FAQ service with RAG (Retrieval Augmented Generation)."""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories import FAQRepository
from app.llm.service import LLMService
from app.schemas.chat import FAQResponse

logger = logging.getLogger(__name__)


# Default FAQ content for over-shop.kz
DEFAULT_FAQ_CONTENT = {
    "delivery": {
        "title": "Доставка",
        "content": """Доставка по Казахстану:
- Доставка по Алматы: 1-2 рабочих дня
- Доставка по другим городам Казахстана: 3-7 рабочих дней
- Бесплатная доставка при заказе от 50 000 тенге
- Стоимость доставки рассчитывается индивидуально
- Самовывоз из магазина доступен ежедневно с 10:00 до 20:00
- Курьерская доставка по Алматы: 2000 тенге""",
    },
    "warranty": {
        "title": "Гарантия",
        "content": """Гарантийные условия:
- Гарантия на всю технику от 12 до 36 месяцев
- Гарантийный ремонт осуществляется в авторизованных сервисных центрах
- Для гарантийного обслуживания необходим чек и гарантийный талон
- Гарантия не распространяется на механические повреждения
- Расширенная гарантия доступна для некоторых категорий товаров""",
    },
    "payment": {
        "title": "Оплата",
        "content": """Способы оплаты:
- Наличными при получении
- Банковской картой онлайн (Visa, MasterCard)
- Kaspi Pay и Kaspi QR
- Рассрочка через Kaspi Bank (0-0-12, 0-0-24)
- Безналичный расчет для юридических лиц
- Кредит через банки-партнеры""",
    },
    "return": {
        "title": "Возврат товара",
        "content": """Условия возврата:
- Возврат товара надлежащего качества в течение 14 дней
- Товар должен сохранить товарный вид и упаковку
- Возврат денег в течение 3-5 рабочих дней
- Для возврата необходим чек и паспорт
- Некоторые категории товаров не подлежат возврату (ПО, расходные материалы)""",
    },
    "order": {
        "title": "Оформление заказа",
        "content": """Как оформить заказ:
- Выберите товар и добавьте в корзину
- Укажите контактные данные и адрес доставки
- Выберите способ оплаты и доставки
- Подтвердите заказ
- Менеджер свяжется с вами для подтверждения
- Отслеживайте статус заказа в личном кабинете""",
    },
    "contacts": {
        "title": "Контакты",
        "content": """Контактная информация over-shop.kz:
- Телефон: 8 (727) 123-45-67
- WhatsApp: +7 (777) 123-45-67
- Email: info@over-shop.kz
- Адрес: г. Алматы, ул. Примерная, 123
- Режим работы: Пн-Вс 10:00-20:00
- Онлайн-консультант на сайте""",
    },
}


class FAQService:
    """Service for answering FAQ questions using RAG."""

    def __init__(self, session: AsyncSession, llm_service: LLMService):
        self.session = session
        self.llm_service = llm_service
        self.faq_repo = FAQRepository(session)

    async def answer_question(
        self,
        question: str,
        topic: Optional[str] = None,
    ) -> FAQResponse:
        """Answer a FAQ question using RAG."""
        # Try vector search first
        context_docs = []
        sources = []

        try:
            embedding = await self.llm_service.get_embedding(question)
            results = await self.faq_repo.vector_search(
                embedding=embedding,
                limit=3,
                category=topic,
                similarity_threshold=0.4,
            )

            for doc, score in results:
                context_docs.append(f"## {doc.title}\n{doc.content}")
                sources.append(doc.title)

        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # If no results from vector search, try text search
        if not context_docs:
            try:
                docs = await self.faq_repo.search_by_content(
                    query=question,
                    category=topic,
                    limit=3,
                )
                for doc in docs:
                    context_docs.append(f"## {doc.title}\n{doc.content}")
                    sources.append(doc.title)
            except Exception as e:
                logger.warning(f"Text search failed: {e}")

        # If still no results, use default FAQ content
        if not context_docs:
            context_docs, sources = self._get_default_context(question, topic)

        # Generate answer using LLM
        context = "\n\n".join(context_docs) if context_docs else "Информация не найдена."

        answer = await self.llm_service.generate_faq_response(
            context=context,
            question=question,
        )

        # Calculate confidence based on whether we found relevant docs
        confidence = 0.9 if context_docs else 0.5

        return FAQResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
        )

    def _get_default_context(
        self,
        question: str,
        topic: Optional[str] = None,
    ) -> tuple[list[str], list[str]]:
        """Get context from default FAQ content."""
        context_docs = []
        sources = []

        question_lower = question.lower()

        # Topic-specific lookup
        if topic and topic in DEFAULT_FAQ_CONTENT:
            faq = DEFAULT_FAQ_CONTENT[topic]
            context_docs.append(f"## {faq['title']}\n{faq['content']}")
            sources.append(faq['title'])
            return context_docs, sources

        # Keyword-based lookup
        keyword_mapping = {
            "delivery": ["доставка", "доставить", "привезти", "курьер", "самовывоз"],
            "warranty": ["гарантия", "ремонт", "сервис", "сломался"],
            "payment": ["оплата", "платить", "карта", "рассрочка", "кредит", "kaspi"],
            "return": ["возврат", "вернуть", "обмен", "обменять"],
            "order": ["заказ", "заказать", "оформить", "купить", "корзина"],
            "contacts": ["контакт", "телефон", "адрес", "где находитесь", "позвонить"],
        }

        matched_topics = set()
        for faq_topic, keywords in keyword_mapping.items():
            for keyword in keywords:
                if keyword in question_lower:
                    matched_topics.add(faq_topic)
                    break

        if matched_topics:
            for faq_topic in matched_topics:
                faq = DEFAULT_FAQ_CONTENT[faq_topic]
                context_docs.append(f"## {faq['title']}\n{faq['content']}")
                sources.append(faq['title'])
        else:
            # Return all FAQ content if no match
            for faq in DEFAULT_FAQ_CONTENT.values():
                context_docs.append(f"## {faq['title']}\n{faq['content']}")
                sources.append(faq['title'])

        return context_docs, sources

    async def seed_default_faq(self) -> None:
        """Seed the database with default FAQ content."""
        for category, faq in DEFAULT_FAQ_CONTENT.items():
            try:
                # Get embedding for the content
                embedding_text = f"{faq['title']}: {faq['content']}"
                embedding = await self.llm_service.get_embedding(embedding_text)

                await self.faq_repo.upsert_faq(
                    title=faq['title'],
                    content=faq['content'],
                    category=category,
                    embedding=embedding,
                )
                logger.info(f"Seeded FAQ: {faq['title']}")
            except Exception as e:
                logger.error(f"Failed to seed FAQ {faq['title']}: {e}")
