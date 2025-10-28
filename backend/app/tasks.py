from __future__ import annotations

from .celery_app import celery_app
from .pipeline_runner import run_tryon_job


@celery_app.task(name="neorose_vfr.run_tryon")
def run_tryon_task(user_image_path: str, garment_front_path: str, garment_side_path: str | None = None):
    result = run_tryon_job(
        user_image_path=user_image_path,
        garment_front_path=garment_front_path,
        garment_side_path=garment_side_path,
    )
    return result

