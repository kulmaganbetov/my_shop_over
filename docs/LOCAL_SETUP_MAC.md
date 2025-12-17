# Local Development Setup on macOS

This guide will help you set up the OverShop AI Assistant backend on your Mac for local development.

## Prerequisites

### 1. Install Homebrew (if not installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Python 3.11+

```bash
brew install python@3.11
```

Verify installation:
```bash
python3.11 --version
```

### 3. Install PostgreSQL with pgvector

```bash
# Install PostgreSQL 16
brew install postgresql@16

# Start PostgreSQL service
brew services start postgresql@16

# Install pgvector extension
brew install pgvector
```

### 4. Install Redis

```bash
brew install redis

# Start Redis service
brew services start redis
```

### 5. Verify Services are Running

```bash
# Check PostgreSQL
brew services list | grep postgresql

# Check Redis
brew services list | grep redis

# Test Redis connection
redis-cli ping
# Should return: PONG
```

---

## Project Setup

### 1. Clone the Repository

```bash
cd ~/projects  # or your preferred directory
git clone <repository-url> my_shop_over
cd my_shop_over
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit the .env file
nano .env  # or use your preferred editor (vim, code, etc.)
```

Update these values in `.env`:

```env
# Database - use your local PostgreSQL
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/overshop
DATABASE_URL_SYNC=postgresql://postgres:postgres@localhost:5432/overshop

# Redis
REDIS_URL=redis://localhost:6379/0

# IMPORTANT: Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# LLM Provider (openai or anthropic)
LLM_PROVIDER=openai

# FTP Configuration (already set)
FTP_HOST=over-shop.kz
FTP_PORT=21
FTP_USER=zoomos1
FTP_PASSWORD=FJsV6cFv
FTP_CSV_FILENAME=Dealer.csv

# Application
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
```

---

## Database Setup

### 1. Create Database and User

```bash
# Connect to PostgreSQL
psql postgres

# In PostgreSQL shell, run:
CREATE USER postgres WITH PASSWORD 'postgres' SUPERUSER;
CREATE DATABASE overshop OWNER postgres;
\c overshop
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

Or as a single command:
```bash
createdb overshop
psql overshop -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 2. Initialize Database Schema

```bash
# Make sure you're in the project directory with venv activated
cd ~/projects/my_shop_over
source venv/bin/activate

# Run the initialization script
psql overshop < scripts/init-db.sql
```

### 3. Verify Database Setup

```bash
psql overshop -c "\dt"
```

You should see tables like:
- products
- product_embeddings
- faq_documents
- pc_presets
- chat_sessions
- chat_messages

---

## Running the Application

### Option A: Run Everything Separately (Recommended for Development)

Open **3 terminal windows/tabs**:

#### Terminal 1: FastAPI Server

```bash
cd ~/projects/my_shop_over
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2: Celery Worker

```bash
cd ~/projects/my_shop_over
source venv/bin/activate
celery -A app.tasks.celery_app worker --loglevel=info
```

#### Terminal 3: Celery Beat (Scheduler) - Optional

```bash
cd ~/projects/my_shop_over
source venv/bin/activate
celery -A app.tasks.celery_app beat --loglevel=info
```

### Option B: Use Docker Compose (Easier)

If you prefer Docker:

```bash
# Install Docker Desktop for Mac from https://www.docker.com/products/docker-desktop

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
```

---

## Verify Installation

### 1. Check API Health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{"status": "healthy", "service": "overshop-assistant"}
```

### 2. Check Database Connection

```bash
curl http://localhost:8000/api/v1/health/db
```

Expected response:
```json
{"status": "healthy", "database": "connected"}
```

### 3. Access API Documentation

Open in your browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Initial Data Setup

### 1. Seed FAQ Content

```bash
curl -X POST http://localhost:8000/api/v1/admin/faq/seed
```

### 2. Sync Products from FTP

```bash
curl -X POST http://localhost:8000/api/v1/admin/sync/products
```

### 3. Generate Embeddings

```bash
curl -X POST "http://localhost:8000/api/v1/admin/sync/embeddings?batch_size=50"
```

---

## Testing the Chat API

### Example 1: PC Build Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Хочу собрать игровой ПК за 500000 тенге для игр в 1440p"
  }'
```

### Example 2: Product Search

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Покажи видеокарты RTX 4070"
  }'
```

### Example 3: FAQ Question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Как оформить доставку?"
  }'
```

### Example 4: Direct Product Search

```bash
curl "http://localhost:8000/api/v1/products/search?query=RTX%204070&limit=5"
```

---

## Running Tests

```bash
# Make sure venv is activated
source venv/bin/activate

# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_compatibility.py -v
```

---

## Development Workflow

### Making Code Changes

The FastAPI server runs with `--reload`, so it will automatically restart when you save changes.

### Database Migrations

If you need to create new migrations:

```bash
# Create a migration
alembic revision --autogenerate -m "description of changes"

# Apply migrations
alembic upgrade head
```

### Viewing Logs

```bash
# API logs - visible in Terminal 1
# Celery worker logs - visible in Terminal 2

# To see Redis activity
redis-cli MONITOR

# To see PostgreSQL queries (enable in postgresql.conf)
tail -f /usr/local/var/log/postgresql@16.log
```

---

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check if PostgreSQL is running
brew services list | grep postgresql

# Restart PostgreSQL
brew services restart postgresql@16

# Check PostgreSQL logs
tail -f /usr/local/var/log/postgresql@16.log
```

### Redis Connection Issues

```bash
# Check if Redis is running
brew services list | grep redis

# Restart Redis
brew services restart redis

# Test connection
redis-cli ping
```

### pgvector Extension Not Found

```bash
# Make sure pgvector is installed
brew install pgvector

# Then create the extension in your database
psql overshop -c "CREATE EXTENSION vector;"
```

### OpenAI API Key Issues

If you see authentication errors:
1. Verify your API key is correct in `.env`
2. Check you have credits in your OpenAI account
3. Restart the FastAPI server after changing `.env`

### Port Already in Use

```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>
```

---

## Stopping Services

### If Running Manually

- Press `Ctrl+C` in each terminal window

### If Using Docker

```bash
docker-compose down
```

### Stop Background Services

```bash
brew services stop postgresql@16
brew services stop redis
```

---

## Quick Reference Commands

```bash
# Start everything
brew services start postgresql@16
brew services start redis
source venv/bin/activate
uvicorn app.main:app --reload  # Terminal 1
celery -A app.tasks.celery_app worker --loglevel=info  # Terminal 2

# Stop everything
Ctrl+C (in each terminal)
brew services stop postgresql@16
brew services stop redis

# Check status
brew services list
curl http://localhost:8000/api/v1/health
```

---

## Next Steps

1. **Customize PC Presets**: Edit `scripts/init-db.sql` to add real product SKUs
2. **Add More FAQ Content**: Use the admin API to add FAQ documents
3. **Configure Webhooks**: Set up notifications for new orders
4. **Deploy**: See `docker-compose.yml` for production deployment
