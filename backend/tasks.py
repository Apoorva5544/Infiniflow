import os

from dotenv import load_dotenv
from celery import Celery

load_dotenv(override=True)

# Prefer REDIS_URL (Render/Upstash/Redis Cloud) over the host/port pair.
REDIS_URL = os.getenv("REDIS_URL")
if REDIS_URL:
    broker_url = REDIS_URL
else:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = os.getenv("REDIS_PORT", "6379")
    broker_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

celery_app = Celery(
    "infiniflow",
    broker=broker_url,
    backend=broker_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="tasks.ping")
def ping():
    return "pong"
