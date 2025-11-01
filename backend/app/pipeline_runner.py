from typing import Optional
import os
from pipeline.pipeline import VFRPipeline
from backend.app.config import settings
from .storage import Storage
from .db import get_job, get_plan
from .db import update_job

# Reuse a single in-process pipeline to avoid reloading heavy models per task
_PIPELINE: Optional[VFRPipeline] = None


def run_tryon_job(
    user_image_path: str,
    garment_front_path: str,
    garment_side_path: Optional[str] = None,
    job_id: Optional[str] = None,
):
    # Determine pipeline overrides from job/plan if available
    overrides = {}
    try:
        if job_id:
            j = get_job(job_id)
            if j and j.plan:
                p = get_plan(j.plan)
                if p and p.default_backend:
                    overrides["finisher_backend"] = p.default_backend
    except Exception:
        pass
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = VFRPipeline.from_settings(settings, overrides=overrides)
    pipe = _PIPELINE
    result = pipe.run(
        user_image_path=user_image_path,
        garment_front_path=garment_front_path,
        garment_side_path=garment_side_path,
    )
    # Optionally publish to S3/CDN
    result_url = Storage.publish_result(result.output_path)
    # Estimate cost from plan if available; fallback to backend pricing env
    backend = os.environ.get("FINISHER_BACKEND", str(settings.get("finisher.backend", "local")))
    plan_cost = None
    try:
        if job_id:
            j = get_job(job_id)
            if j and j.plan:
                p = get_plan(j.plan)
                if p and p.per_image_cost is not None:
                    plan_cost = float(p.per_image_cost)
    except Exception:
        plan_cost = None
    cost_local = float(os.environ.get("COST_LOCAL", "0.0"))
    cost_kling = float(os.environ.get("COST_KLING", "0.0"))
    backend_cost = cost_kling if backend == "kling" else cost_local
    cost = plan_cost if plan_cost is not None else backend_cost
    # Note: update_job called by queue/tasks too; this call only sets cost if present
    try:
        if job_id:
            update_job(job_id, cost_estimate=cost)
    except Exception:
        pass
    return {"result_path": result.output_path, "result_url": result_url, "cost_estimate": cost}
