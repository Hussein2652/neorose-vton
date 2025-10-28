from __future__ import annotations

from .celery_app import celery_app
from .pipeline_runner import run_tryon_job
from .db import update_job, get_job, record_usage
from .metrics import jobs_completed, jobs_failed, jobs_in_queue
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
            job_id=job_id,
        )
        try:
            update_job(job_id, status="completed", result_path=result.get("result_path"), result_url=result.get("result_url"), cost_estimate=result.get("cost_estimate"))
            cache_set_job(job_id, status="completed", result_path=result.get("result_path"), result_url=result.get("result_url"))
            # Usage accounting
            job = get_job(job_id)
            if job and job.user_id:
                record_usage(job.user_id, job_id, units=1.0, cost=float(result.get("cost_estimate") or 0.0))
            try:
                jobs_completed.inc(); jobs_in_queue.dec()
            except Exception:
                pass
        except Exception:
            pass
        return result
    except Exception as e:  # noqa: BLE001
        try:
            update_job(job_id, status="failed", error=str(e))
            cache_set_job(job_id, status="failed", error=str(e))
            try:
                jobs_failed.inc(); jobs_in_queue.dec()
            except Exception:
                pass
        except Exception:
            pass
        raise
