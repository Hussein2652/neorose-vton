import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .models import JobCreateResponse, JobStatusResponse
from .queue import Jobs
from .storage import Storage
from .pipeline_runner import run_tryon_job
from .db import init_db, create_job, get_job, update_job

USE_CELERY = os.environ.get("USE_CELERY", "0") == "1"
if USE_CELERY:
    from .tasks import run_tryon_task
    from .celery_app import celery_app

app = FastAPI(title="Neorose VFR API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/jobs/tryon", response_model=JobCreateResponse)
async def create_tryon_job(
    user_image: UploadFile = File(...),
    garment_front: UploadFile = File(...),
    garment_side: UploadFile | None = File(None),
):
    try:
        # Save uploads
        user_path = Storage.save_upload(user_image, prefix="user")
        g_front_path = Storage.save_upload(garment_front, prefix="garment_front")
        g_side_path = (
            Storage.save_upload(garment_side, prefix="garment_side") if garment_side else None
        )

        # Create DB job and schedule execution
        import uuid

        job_id = str(uuid.uuid4())
        if USE_CELERY:
            async_result = run_tryon_task.delay(
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
            )
            create_job(
                job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                provider=os.environ.get("FINISHER_BACKEND", "local"),
                task_id=async_result.id,
            )
        else:
            create_job(
                job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                provider=os.environ.get("FINISHER_BACKEND", "local"),
            )
            Jobs.enqueue(
                run_tryon_job,
                job_id=job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
            )
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    db_job = get_job(job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if USE_CELERY and db_job.status in {"queued", "running"} and db_job.task_id:
        try:
            res = celery_app.AsyncResult(db_job.task_id)
            if res.successful() and (not db_job.result_path):
                payload = res.get(timeout=0.1)
                result_path = payload.get("result_path") if isinstance(payload, dict) else None
                if result_path:
                    update_job(job_id, status="completed", result_path=result_path)
                    db_job = get_job(job_id) or db_job
            elif res.failed() and not db_job.error:
                update_job(job_id, status="failed", error=str(res.result))
                db_job = get_job(job_id) or db_job
            elif res.state == "STARTED" and db_job.status != "running":
                update_job(job_id, status="running")
                db_job = get_job(job_id) or db_job
        except Exception:
            pass
    return JobStatusResponse(
        job_id=job_id,
        status=db_job.status,
        error=db_job.error,
        result_path=db_job.result_path,
    )


@app.get("/v1/jobs/{job_id}/result")
def get_job_result(job_id: str):
    db_job = get_job(job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if db_job.status != "completed" or not db_job.result_path:
        raise HTTPException(status_code=409, detail="Job not completed yet")
    return FileResponse(db_job.result_path, filename="vfr_result.jpg")


# Start the worker on app start
@app.on_event("startup")
def _startup():
    Storage.ensure_dirs()
    init_db()
    Jobs.ensure_worker()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    try:
        app.mount("/web", StaticFiles(directory="frontend/web", html=True), name="web")
    except Exception:
        pass
