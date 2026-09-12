from celery import Celery
from app.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "ingestion",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_BACKEND_URL,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT},
    task_always_eager=settings.CELERY_ALWAYS_EAGER,  # Lightweight mode: synchronous execution
    task_eager_propagates=True,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    result_expires=86400,
)

