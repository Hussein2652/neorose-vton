from __future__ import annotations

from .celery_app import celery_app
from .pipeline_runner import run_tryon_job
from .db import update_job
from .cache import cache_set_job


@celery_app.task(name="neorose_vfr.run_tryon")
def run_tryon_task(job_id: str, user_image_path: str, garment_front_path: str, garment_side_path: str | None = None):
    try:
        update_job(job_id, status="running")
        cache_set_job(job_id, status="running")
    except Exception:
        pass
    try:
        result = run_tryon_job(
            user_image_path=user_image_path,
            garment_front_path=garment_front_path,
            garment_side_path=garment_side_path,
        )
        try:
            update_job(job_id, status="completed", result_path=result.get("result_path"))
            cache_set_job(job_id, status="completed", result_path=result.get("result_path"))
        except Exception:
            pass
        return result
    except Exception as e:  # noqa: BLE001
        try:
            update_job(job_id, status="failed", error=str(e))
            cache_set_job(job_id, status="failed", error=str(e))
        except Exception:
            pass
        raise
