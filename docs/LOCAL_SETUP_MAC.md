# Local Development Setup on macOS (Hybrid Docker)

This guide shows how to run **PostgreSQL, Redis, and Celery in Docker**, while running the **FastAPI app locally** on your Mac.

## Prerequisites

### 1. Install Docker Desktop

Download and install Docker Desktop for Mac:
- https://www.docker.com/products/docker-desktop/

After installation, make sure Docker is running (whale icon in menu bar).

### 2. Install Python 3.11+

```bash
# Using Homebrew
brew install python@3.11

# Or download from python.org
# https://www.python.org/downloads/
```

Verify installation:
```bash
python3 --version
```

---

## Project Setup

### 1. Clone/Navigate to the Repository

```bash
cd ~/projects/my_shop_over  # or wherever your project is
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` file:
```bash
nano .env  # or: open -e .env (TextEdit) or: code .env (VS Code)
```

Update with these values:
```env
# Database - connects to Docker container
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/overshop
DATABASE_URL_SYNC=postgresql://postgres:postgres@localhost:5432/overshop

# Redis - connects to Docker container
REDIS_URL=redis://localhost:6379/0

# IMPORTANT: Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# LLM Provider
LLM_PROVIDER=openai

# FTP Configuration
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

## Step 1: Start Infrastructure in Docker

### Create a Docker Compose file for infrastructure only

Create a new file `docker-compose.infra.yml`:

```bash
cat > docker-compose.infra.yml << 'EOF'
version: '3.8'

services:
  # PostgreSQL with pgvector
  db:
    image: pgvector/pgvector:pg16
    container_name: overshop-db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: overshop
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    container_name: overshop-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Celery Worker
  celery-worker:
    image: python:3.11-slim
    container_name: overshop-celery-worker
    working_dir: /app
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/overshop
      - DATABASE_URL_SYNC=postgresql://postgres:postgres@db:5432/overshop
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
    volumes:
      - .:/app
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: >
      bash -c "pip install -r requirements.txt &&
               celery -A app.tasks.celery_app worker --loglevel=info"

  # Celery Beat (Scheduler)
  celery-beat:
    image: python:3.11-slim
    container_name: overshop-celery-beat
    working_dir: /app
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/overshop
      - DATABASE_URL_SYNC=postgresql://postgres:postgres@db:5432/overshop
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: >
      bash -c "pip install -r requirements.txt &&
               celery -A app.tasks.celery_app beat --loglevel=info"

volumes:
  postgres_data:
  redis_data:
EOF
```

### Start the Docker containers

```bash
# Start PostgreSQL, Redis, and Celery
docker-compose -f docker-compose.infra.yml up -d

# Check status
docker-compose -f docker-compose.infra.yml ps

# View logs
docker-compose -f docker-compose.infra.yml logs -f
```

### Wait for services to be ready

```bash
# Check PostgreSQL
docker exec overshop-db pg_isready -U postgres

# Check Redis
docker exec overshop-redis redis-cli ping
```

---

## Step 2: Verify Database Setup

The database should be automatically initialized. Verify:

```bash
# Connect to PostgreSQL in Docker
docker exec -it overshop-db psql -U postgres -d overshop -c "\dt"
```

You should see tables:
- products
- product_embeddings
- faq_documents
- pc_presets
- chat_sessions
- chat_messages

If tables are missing, run:
```bash
docker exec -i overshop-db psql -U postgres -d overshop < scripts/init-db.sql
```

---

## Step 3: Run FastAPI Locally

Open a new terminal:

```bash
cd ~/projects/my_shop_over
source venv/bin/activate

# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Verify Everything is Working

### 1. Check API Health

```bash
curl http://localhost:8000/api/v1/health
```
Expected: `{"status": "healthy", "service": "overshop-assistant"}`

### 2. Check Database Connection

```bash
curl http://localhost:8000/api/v1/health/db
```
Expected: `{"status": "healthy", "database": "connected"}`

### 3. Access API Documentation

Open in browser: http://localhost:8000/docs

---

## Initial Data Setup

### Seed FAQ Content

```bash
curl -X POST http://localhost:8000/api/v1/admin/faq/seed
```

### Sync Products from FTP

```bash
curl -X POST http://localhost:8000/api/v1/admin/sync/products
```

### Generate Embeddings

```bash
curl -X POST "http://localhost:8000/api/v1/admin/sync/embeddings?batch_size=50"
```

---

## Test the Chat API

### PC Build Request

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Хочу собрать игровой ПК за 500000 тенге"}'
```

### Product Search

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Покажи видеокарты RTX 4070"}'
```

### FAQ Question

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Как оформить доставку?"}'
```

---

## Daily Workflow

### Starting Development

```bash
# 1. Start Docker containers
cd ~/projects/my_shop_over
docker-compose -f docker-compose.infra.yml up -d

# 2. Wait a few seconds for services to start

# 3. Start FastAPI locally
source venv/bin/activate
uvicorn app.main:app --reload
```

### Stopping Development

```bash
# Stop FastAPI: Ctrl+C in terminal

# Stop Docker containers
docker-compose -f docker-compose.infra.yml down

# Or keep data and just stop:
docker-compose -f docker-compose.infra.yml stop
```

---

## Viewing Logs

### Docker Container Logs

```bash
# All containers
docker-compose -f docker-compose.infra.yml logs -f

# Specific container
docker-compose -f docker-compose.infra.yml logs -f db
docker-compose -f docker-compose.infra.yml logs -f redis
docker-compose -f docker-compose.infra.yml logs -f celery-worker
```

### Connect to PostgreSQL

```bash
docker exec -it overshop-db psql -U postgres -d overshop
```

### Connect to Redis

```bash
docker exec -it overshop-redis redis-cli
```

---

## Troubleshooting

### Docker containers won't start

```bash
# Check Docker Desktop is running
docker info

# Remove old containers and start fresh
docker-compose -f docker-compose.infra.yml down -v
docker-compose -f docker-compose.infra.yml up -d
```

### Database connection refused

```bash
# Check if PostgreSQL container is running
docker ps | grep overshop-db

# Check PostgreSQL logs
docker logs overshop-db

# Restart the container
docker-compose -f docker-compose.infra.yml restart db
```

### Redis connection refused

```bash
# Check if Redis container is running
docker ps | grep overshop-redis

# Test Redis
docker exec overshop-redis redis-cli ping
```

### Celery worker not processing tasks

```bash
# Check Celery worker logs
docker logs overshop-celery-worker -f

# Restart Celery worker
docker-compose -f docker-compose.infra.yml restart celery-worker
```

### Port already in use

```bash
# Find what's using the port
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :8000  # FastAPI

# Kill the process or change ports in docker-compose.infra.yml
```

### Reset everything

```bash
# Stop and remove all containers and volumes
docker-compose -f docker-compose.infra.yml down -v

# Start fresh
docker-compose -f docker-compose.infra.yml up -d
```

---

## Quick Reference

```bash
# Start infrastructure
docker-compose -f docker-compose.infra.yml up -d

# Start FastAPI
source venv/bin/activate && uvicorn app.main:app --reload

# Stop everything
docker-compose -f docker-compose.infra.yml down

# View all logs
docker-compose -f docker-compose.infra.yml logs -f

# Check status
docker-compose -f docker-compose.infra.yml ps
curl http://localhost:8000/api/v1/health
```
