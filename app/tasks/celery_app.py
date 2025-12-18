"""Celery application configuration."""

import logging
from celery import Celery
from celery.signals import worker_ready, task_prerun, task_postrun, task_failure

from app.core.config import settings

logger = logging.getLogger(__name__)

# Log configuration on module load
logger.info("=" * 60)
logger.info("CELERY APP INITIALIZATION")
logger.info("=" * 60)
logger.info(f"  Broker URL: {settings.redis_url}")
logger.info(f"  Backend URL: {settings.redis_url}")

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
    # Ensure tasks are discovered
    task_routes={
        "app.tasks.ingestion.*": {"queue": "celery"},
        "app.tasks.embeddings.*": {"queue": "celery"},
    },
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


# Celery signals for logging
@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    """Log when worker is ready and show registered tasks."""
    logger.info("=" * 60)
    logger.info("CELERY WORKER READY")
    logger.info("=" * 60)
    logger.info(f"  Worker: {sender}")
    logger.info(f"  Broker: {celery_app.conf.broker_url}")
    logger.info("  Registered tasks:")
    for task_name in sorted(celery_app.tasks.keys()):
        if not task_name.startswith("celery."):
            logger.info(f"    - {task_name}")
    logger.info("=" * 60)


@task_prerun.connect
def on_task_prerun(sender, task_id, task, args, kwargs, **kw):
    """Log when a task starts."""
    logger.info(f"[TASK START] {task.name} (ID: {task_id})")


@task_postrun.connect
def on_task_postrun(sender, task_id, task, args, kwargs, retval, state, **kw):
    """Log when a task completes."""
    logger.info(f"[TASK END] {task.name} (ID: {task_id}) -> {state}")


@task_failure.connect
def on_task_failure(sender, task_id, exception, args, kwargs, traceback, einfo, **kw):
    """Log when a task fails."""
    logger.error(f"[TASK FAILED] {sender.name} (ID: {task_id}): {exception}")
