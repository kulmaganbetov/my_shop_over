"""Celery application configuration."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "overshop",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.tasks.ingestion",
        "app.tasks.embeddings",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Almaty",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    "sync-products-every-hour": {
        "task": "app.tasks.ingestion.sync_products_from_ftp",
        "schedule": 3600.0,  # every hour
    },
    "update-embeddings-daily": {
        "task": "app.tasks.embeddings.update_missing_embeddings",
        "schedule": 86400.0,  # every 24 hours
    },
}
