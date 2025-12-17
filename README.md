# OverShop AI Assistant Backend

AI-powered assistant backend for over-shop.kz e-commerce store.

## Architecture

This is a **modular monolith** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
├─────────────────────────────────────────────────────────────────┤
│                        Intent Router                             │
├─────────────────────────────────────────────────────────────────┤
│                         LLM Layer                                │
│  (Intent Detection, Parameter Extraction, Response Generation)   │
├──────────────────┬──────────────────┬───────────────────────────┤
│ ProductSearch    │   PCBuild        │         FAQ               │
│   Service        │   Service        │       Service             │
├──────────────────┴──────────────────┴───────────────────────────┤
│                      Data Layer                                  │
│              (PostgreSQL + pgvector + Redis)                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **LLM as Interface Layer Only**
   - LLM is used ONLY for:
     - Intent detection
     - Parameter extraction
     - Natural language response generation
   - LLM NEVER:
     - Accesses the database directly
     - Makes business decisions
     - Determines compatibility

2. **Deterministic Business Logic**
   - PC compatibility rules are rule-based
   - Product search uses vector similarity + structured filters
   - All business decisions are predictable and testable

## Features

### 1. PC Build Assistant
- Help users build compatible PCs
- Support different budgets and purposes (gaming, work, office)
- Guarantee component compatibility using rule engine
- Provide explanations

### 2. Product Search
- Semantic search using embeddings (pgvector)
- Structured filters (category, price, manufacturer)
- Returns real products from database

### 3. Store FAQ
- RAG-based answers for store questions
- Topics: delivery, warranty, payment, returns, orders

## Tech Stack

- **Python 3.11+**
- **FastAPI** - Web framework
- **SQLAlchemy** - ORM
- **PostgreSQL + pgvector** - Database with vector search
- **Celery + Redis** - Task queue
- **OpenAI / Anthropic** - LLM providers

## Quick Start

> **macOS Users**: See the detailed setup guide at [docs/LOCAL_SETUP_MAC.md](docs/LOCAL_SETUP_MAC.md)

### Using Docker Compose

```bash
# Copy environment file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_KEY=your-key-here

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/overshop
export REDIS_URL=redis://localhost:6379/0
export OPENAI_API_KEY=your-key-here

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload

# Start Celery worker (in another terminal)
celery -A app.tasks.celery_app worker --loglevel=info
```

## API Endpoints

### Chat
```
POST /api/v1/chat
```
Main chat endpoint. Automatically detects intent and routes to appropriate service.

**Request:**
```json
{
  "message": "Хочу собрать игровой ПК за 500000 тенге",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "message": "Отличный выбор! Для игрового ПК за 500000 тенге рекомендую...",
  "intent": "pc_build",
  "session_id": "abc-123",
  "data": {
    "build": {...},
    "total_price": 485000,
    "compatibility": "ok"
  },
  "suggestions": ["Показать альтернативы", "Изменить бюджет"]
}
```

### Product Search
```
GET /api/v1/products/search?query=RTX 4070&category=Видеокарты
```

### Admin Endpoints
```
POST /api/v1/admin/sync/products  # Trigger FTP sync
POST /api/v1/admin/sync/embeddings  # Update embeddings
POST /api/v1/admin/faq/seed  # Seed FAQ content
```

## Data Ingestion

Products are synchronized from FTP server periodically:

1. Celery beat triggers `sync_products_from_ftp` task hourly
2. Downloads `Dealer.csv` from FTP
3. Parses and normalizes data
4. Upserts to PostgreSQL
5. Generates embeddings for new products

## Project Structure

```
app/
├── api/              # FastAPI routes
│   ├── routes.py     # Main API endpoints
│   └── admin.py      # Admin endpoints
├── core/             # Configuration
│   └── config.py
├── db/               # Database
│   ├── base.py       # SQLAlchemy setup
│   ├── models.py     # Database models
│   └── repositories/ # Data access layer
├── llm/              # LLM integration
│   ├── client.py     # LLM client abstraction
│   ├── prompts.py    # Prompt templates
│   └── service.py    # LLM service
├── services/         # Business logic
│   ├── router.py     # Intent router
│   ├── product_search.py
│   ├── pc_build.py
│   ├── compatibility.py  # Rule engine
│   └── faq.py
├── tasks/            # Celery tasks
│   ├── celery_app.py
│   ├── ingestion.py
│   └── embeddings.py
├── schemas/          # Pydantic models
└── main.py           # FastAPI app
```

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DATABASE_URL | PostgreSQL async URL | - |
| DATABASE_URL_SYNC | PostgreSQL sync URL | - |
| REDIS_URL | Redis URL | redis://localhost:6379/0 |
| OPENAI_API_KEY | OpenAI API key | - |
| LLM_PROVIDER | openai or anthropic | openai |
| FTP_HOST | FTP server host | over-shop.kz |
| FTP_USER | FTP username | - |
| FTP_PASSWORD | FTP password | - |

## License

Private - over-shop.kz
