-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Products table
CREATE TABLE IF NOT EXISTS products (
    id BIGSERIAL PRIMARY KEY,
    sku VARCHAR(100) UNIQUE NOT NULL,
    kaspi_code VARCHAR(100),
    name VARCHAR(500) NOT NULL,
    supplier VARCHAR(255),
    stock INTEGER DEFAULT 0,
    manufacturer VARCHAR(255),
    price FLOAT,
    discount_price FLOAT,
    category VARCHAR(255),
    component_type VARCHAR(50),
    specifications JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Product embeddings table
CREATE TABLE IF NOT EXISTS product_embeddings (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT UNIQUE REFERENCES products(id) ON DELETE CASCADE,
    embedding vector(1536),
    embedding_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- FAQ documents table
CREATE TABLE IF NOT EXISTS faq_documents (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- PC presets table
CREATE TABLE IF NOT EXISTS pc_presets (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    purpose VARCHAR(100) NOT NULL,
    budget_min INTEGER,
    budget_max INTEGER,
    description TEXT,
    components JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) UNIQUE NOT NULL,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Chat messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(100) REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    intent VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);
CREATE INDEX IF NOT EXISTS idx_products_kaspi_code ON products(kaspi_code);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_manufacturer ON products(manufacturer);
CREATE INDEX IF NOT EXISTS idx_products_component_type ON products(component_type);
CREATE INDEX IF NOT EXISTS idx_products_name_search ON products(name);
CREATE INDEX IF NOT EXISTS idx_products_category_stock ON products(category, stock);

CREATE INDEX IF NOT EXISTS idx_faq_category ON faq_documents(category);

CREATE INDEX IF NOT EXISTS idx_pc_presets_purpose ON pc_presets(purpose);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);

-- Vector indexes (using IVFFlat for better performance)
CREATE INDEX IF NOT EXISTS idx_product_embeddings_vector ON product_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_faq_embeddings_vector ON faq_documents
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);

-- Insert sample PC presets
INSERT INTO pc_presets (name, purpose, budget_min, budget_max, description, components) VALUES
('Gaming Entry', 'gaming', 300000, 450000, 'Бюджетный игровой ПК для 1080p',
 '{"cpu": "sample-cpu-sku", "gpu": "sample-gpu-sku", "motherboard": "sample-mb-sku", "ram": "sample-ram-sku", "storage": "sample-ssd-sku", "psu": "sample-psu-sku", "case": "sample-case-sku"}'),
('Gaming Mid', 'gaming', 450000, 700000, 'Игровой ПК среднего уровня для 1440p',
 '{"cpu": "sample-cpu-sku", "gpu": "sample-gpu-sku", "motherboard": "sample-mb-sku", "ram": "sample-ram-sku", "storage": "sample-ssd-sku", "psu": "sample-psu-sku", "case": "sample-case-sku"}'),
('Gaming High', 'gaming', 700000, 1200000, 'Топовый игровой ПК для 4K',
 '{"cpu": "sample-cpu-sku", "gpu": "sample-gpu-sku", "motherboard": "sample-mb-sku", "ram": "sample-ram-sku", "storage": "sample-ssd-sku", "psu": "sample-psu-sku", "case": "sample-case-sku"}'),
('Office Basic', 'office', 150000, 250000, 'Офисный ПК для работы с документами',
 '{"cpu": "sample-cpu-sku", "gpu": "sample-gpu-sku", "motherboard": "sample-mb-sku", "ram": "sample-ram-sku", "storage": "sample-ssd-sku", "psu": "sample-psu-sku", "case": "sample-case-sku"}'),
('Workstation', 'work', 500000, 900000, 'Рабочая станция для профессиональных задач',
 '{"cpu": "sample-cpu-sku", "gpu": "sample-gpu-sku", "motherboard": "sample-mb-sku", "ram": "sample-ram-sku", "storage": "sample-ssd-sku", "psu": "sample-psu-sku", "case": "sample-case-sku"}')
ON CONFLICT DO NOTHING;

-- Insert default FAQ content
INSERT INTO faq_documents (title, content, category) VALUES
('Доставка', 'Доставка по Казахстану:
- Доставка по Алматы: 1-2 рабочих дня
- Доставка по другим городам Казахстана: 3-7 рабочих дней
- Бесплатная доставка при заказе от 50 000 тенге
- Стоимость доставки рассчитывается индивидуально
- Самовывоз из магазина доступен ежедневно с 10:00 до 20:00
- Курьерская доставка по Алматы: 2000 тенге', 'delivery'),

('Гарантия', 'Гарантийные условия:
- Гарантия на всю технику от 12 до 36 месяцев
- Гарантийный ремонт осуществляется в авторизованных сервисных центрах
- Для гарантийного обслуживания необходим чек и гарантийный талон
- Гарантия не распространяется на механические повреждения
- Расширенная гарантия доступна для некоторых категорий товаров', 'warranty'),

('Оплата', 'Способы оплаты:
- Наличными при получении
- Банковской картой онлайн (Visa, MasterCard)
- Kaspi Pay и Kaspi QR
- Рассрочка через Kaspi Bank (0-0-12, 0-0-24)
- Безналичный расчет для юридических лиц
- Кредит через банки-партнеры', 'payment'),

('Возврат товара', 'Условия возврата:
- Возврат товара надлежащего качества в течение 14 дней
- Товар должен сохранить товарный вид и упаковку
- Возврат денег в течение 3-5 рабочих дней
- Для возврата необходим чек и паспорт
- Некоторые категории товаров не подлежат возврату (ПО, расходные материалы)', 'return'),

('Оформление заказа', 'Как оформить заказ:
- Выберите товар и добавьте в корзину
- Укажите контактные данные и адрес доставки
- Выберите способ оплаты и доставки
- Подтвердите заказ
- Менеджер свяжется с вами для подтверждения
- Отслеживайте статус заказа в личном кабинете', 'order'),

('Контакты', 'Контактная информация over-shop.kz:
- Телефон: 8 (727) 123-45-67
- WhatsApp: +7 (777) 123-45-67
- Email: info@over-shop.kz
- Адрес: г. Алматы, ул. Примерная, 123
- Режим работы: Пн-Вс 10:00-20:00
- Онлайн-консультант на сайте', 'contacts')
ON CONFLICT DO NOTHING;
