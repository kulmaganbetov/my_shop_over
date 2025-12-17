# Data Pipeline Documentation

## Обзор (Overview)

Этот документ объясняет как данные перемещаются через систему - от FTP сервера до векторного поиска.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐                                                  │
│   │  FTP Server  │  over-shop.kz                                    │
│   │  Dealer.csv  │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          │ 1. Download (Celery Task)                                │
│          ▼                                                          │
│   ┌──────────────┐                                                  │
│   │  CSV Parser  │  pandas + normalization                          │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          │ 2. Save to Database                                      │
│          ▼                                                          │
│   ┌──────────────┐     ┌──────────────┐                            │
│   │  PostgreSQL  │────▶│   products   │  таблица с товарами        │
│   └──────┬───────┘     └──────────────┘                            │
│          │                                                          │
│          │ 3. Generate Embeddings (Celery Task)                     │
│          ▼                                                          │
│   ┌──────────────┐                                                  │
│   │  OpenAI API  │  text-embedding-3-small                         │
│   └──────┬───────┘                                                  │
│          │                                                          │
│          │ 4. Save Vectors                                          │
│          ▼                                                          │
│   ┌──────────────┐     ┌──────────────────────┐                    │
│   │   pgvector   │────▶│  product_embeddings  │  векторы 1536 dim  │
│   └──────────────┘     └──────────────────────┘                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Скачивание CSV с FTP

### Что происходит:
1. Celery task подключается к FTP серверу `over-shop.kz`
2. Скачивает файл `Dealer.csv` в память
3. Декодирует как UTF-8

### Код:
```python
# app/tasks/ingestion.py

def download_csv_from_ftp():
    ftp = ftplib.FTP()
    ftp.connect("over-shop.kz", 21)
    ftp.login("zoomos1", "password")

    buffer = io.BytesIO()
    ftp.retrbinary("RETR Dealer.csv", buffer.write)

    return buffer.read().decode("utf-8-sig")
```

### Когда запускается:
- **Автоматически**: Celery Beat каждый час
- **Вручную**: `POST /api/v1/admin/sync/products`

### Логи:
```
STEP 1: DOWNLOADING CSV FROM FTP
  FTP Host: over-shop.kz
  FTP Port: 21
  Connecting to FTP server...
  ✓ Connected to FTP server
  ✓ Authentication successful
  Downloading Dealer.csv...
  ✓ Downloaded successfully!
  File size: 2,345,678 bytes
```

---

## Step 2: Парсинг и нормализация

### Формат CSV:
```csv
SKU;КодКаспи;Номенклатура;Поставщик;Остаток;Производитель;КредитРассрочка;БонуснаяЦена;Категория
12345;K001;Intel Core i7-13700K;Supplier1;10;Intel;150000;145000;Процессоры
```

### Что происходит:
1. Pandas читает CSV (разделитель `;`)
2. Колонки переименовываются на английский
3. Каждая строка нормализуется:
   - Числа очищаются от пробелов и запятых
   - Определяется `component_type` (CPU, GPU, RAM...)
   - Из названия извлекаются характеристики (socket, DDR type...)

### Код:
```python
# app/tasks/ingestion.py

def normalize_product(row):
    return {
        "sku": row["sku"],
        "name": row["name"],
        "price": float(row["price"]),
        "category": row["category"],
        "component_type": detect_component_type(row["category"]),
        "specifications": extract_specs(row["name"]),
    }
```

### Логи:
```
STEP 2: PARSING CSV
  ✓ Read 15,234 rows
  Original columns: ['SKU', 'КодКаспи', 'Номенклатура'...]
  Renamed columns: ['sku', 'kaspi_code', 'name'...]

  Statistics:
    Total products: 15,234
    Unique categories: 127
    Products with price: 14,892
```

---

## Step 3: Сохранение в PostgreSQL

### Таблица `products`:
```sql
CREATE TABLE products (
    id BIGSERIAL PRIMARY KEY,
    sku VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(500) NOT NULL,
    price FLOAT,
    discount_price FLOAT,
    category VARCHAR(255),
    component_type VARCHAR(50),  -- для PC сборок
    specifications JSONB,        -- технические характеристики
    stock INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Что происходит:
1. Для каждого продукта проверяется SKU
2. Если существует → UPDATE
3. Если новый → INSERT

### Код:
```python
# app/tasks/ingestion.py

with Session(engine) as session:
    for row in df.iterrows():
        product_data = normalize_product(row)

        existing = session.query(Product).filter(
            Product.sku == product_data["sku"]
        ).first()

        if existing:
            # Update
            for key, value in product_data.items():
                setattr(existing, key, value)
        else:
            # Insert
            session.add(Product(**product_data))

    session.commit()
```

### Логи:
```
STEP 3 & 4: SAVING TO POSTGRESQL
  Processing 15,234 products...
    Created: SKU=12345, Name=Intel Core i7-13700K...
    Created: SKU=12346, Name=AMD Ryzen 9 7950X...
    Updated: SKU=12347, Name=NVIDIA RTX 4090...
    Progress: 100/15234 processed...
    Progress: 200/15234 processed...
  ✓ Transaction committed

SYNC COMPLETED
  ✓ Created: 234 new products
  ✓ Updated: 15,000 existing products
  ✗ Errors: 0
```

---

## Step 4: Генерация Embeddings

### Что такое Embedding?
Embedding - это векторное представление текста. Текст преобразуется в массив из 1536 чисел, который позволяет находить семантически похожие товары.

### Как это работает:
```
"Intel Core i7-13700K процессор"  →  OpenAI API  →  [0.023, -0.145, 0.078, ... 1536 чисел]
"AMD Ryzen 7 7800X CPU"           →  OpenAI API  →  [0.021, -0.142, 0.081, ... 1536 чисел]

Эти векторы похожи! Cosine similarity = 0.92
```

### Таблица `product_embeddings`:
```sql
CREATE TABLE product_embeddings (
    id BIGSERIAL PRIMARY KEY,
    product_id BIGINT REFERENCES products(id),
    embedding vector(1536),       -- pgvector тип
    embedding_text TEXT           -- текст который был преобразован
);
```

### Код:
```python
# app/tasks/embeddings.py

def generate_product_embedding(product_id):
    # 1. Загрузить продукт
    product = session.query(Product).get(product_id)

    # 2. Создать текст для embedding
    text = f"""
    Product: {product.name}
    Category: {product.category}
    Manufacturer: {product.manufacturer}
    Price: {product.price}
    """

    # 3. Вызвать OpenAI API
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding  # [0.023, -0.145, ...]

    # 4. Сохранить в pgvector
    session.add(ProductEmbedding(
        product_id=product_id,
        embedding=embedding,
        embedding_text=text
    ))
    session.commit()
```

### Когда запускается:
- **Автоматически**: Celery Beat ежедневно
- **Вручную**: `POST /api/v1/admin/sync/embeddings`

### Логи:
```
[Embedding] Generating embedding for product_id=12345
[Embedding] Product: Intel Core i7-13700K...
[Embedding] Calling OpenAI API...
[Embedding] ✓ Got embedding vector (dim=1536)
[Embedding] ✓ Saved to PostgreSQL (pgvector)
```

---

## Step 5: Векторный поиск (pgvector)

### Как работает поиск:
1. Пользователь вводит: "видеокарта для игр"
2. Текст преобразуется в вектор через OpenAI
3. pgvector ищет похожие векторы в базе
4. Возвращает топ-N самых похожих товаров

### SQL запрос:
```sql
-- Поиск по косинусной близости
SELECT
    p.name,
    p.price,
    1 - (pe.embedding <=> query_embedding) as similarity
FROM products p
JOIN product_embeddings pe ON p.id = pe.product_id
ORDER BY pe.embedding <=> query_embedding
LIMIT 10;
```

### Код:
```python
# app/db/repositories/product.py

async def vector_search(self, query_embedding, limit=10):
    distance = ProductEmbedding.embedding.cosine_distance(query_embedding)

    results = await session.execute(
        select(Product, (1 - distance).label("similarity"))
        .join(ProductEmbedding)
        .order_by(distance)
        .limit(limit)
    )

    return results
```

---

## Как запустить пайплайн вручную

### 1. Запустить Docker контейнеры:
```bash
docker-compose -f docker-compose.infra.yml up -d
```

### 2. Запустить синхронизацию продуктов:
```bash
# Через API
curl -X POST http://localhost:8000/api/v1/admin/sync/products

# Результат
{"status": "queued", "task_id": "abc-123"}
```

### 3. Запустить генерацию embeddings:
```bash
curl -X POST "http://localhost:8000/api/v1/admin/sync/embeddings?batch_size=100"
```

### 4. Проверить результат:
```bash
# Посмотреть логи Celery worker
docker logs overshop-celery-worker -f

# Подключиться к PostgreSQL
docker exec -it overshop-db psql -U postgres -d overshop

# Проверить количество продуктов
SELECT COUNT(*) FROM products;

# Проверить количество embeddings
SELECT COUNT(*) FROM product_embeddings;
```

### 5. Запустить тестовый скрипт:
```bash
python scripts/test_pipeline.py
```

---

## Troubleshooting

### FTP не подключается
```
✗ Connection refused - FTP server may be down
```
**Решение**: Проверьте доступность FTP сервера, VPN, firewall.

### OpenAI API ошибка
```
✗ OpenAI API error: Invalid API key
```
**Решение**: Проверьте OPENAI_API_KEY в .env файле.

### Нет embeddings
```
Products missing embeddings: 15000
```
**Решение**: Запустите `POST /api/v1/admin/sync/embeddings`

### pgvector не установлен
```
ERROR: type "vector" does not exist
```
**Решение**:
```sql
CREATE EXTENSION vector;
```
