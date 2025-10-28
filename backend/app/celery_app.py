import os
from celery import Celery


BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URL = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("neorose_vfr", broker=BROKER_URL, backend=BACKEND_URL)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"])

