import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .models import JobCreateResponse, JobStatusResponse
from .queue import Jobs
from .storage import Storage
from .pipeline_runner import run_tryon_job
from .db import init_db, create_job, get_job, update_job
from .cache import cache_get_job, cache_set_job
from .auth import require_auth, check_basic_auth
from .ratelimit import rate_limit
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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
    _user=Depends(require_auth),
    _rl=Depends(rate_limit),
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
                job_id=job_id,
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
            cache_set_job(job_id, status="queued")
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
            cache_set_job(job_id, status="queued")
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, _rl: None = Depends(rate_limit)):
    # Try cache first
    c = cache_get_job(job_id)
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
    status = c.get("status") if c else db_job.status
    result_path = c.get("result_path") if c else db_job.result_path
    result_url = c.get("result_url") if c else db_job.result_url
    error = c.get("error") if c else db_job.error
    return JobStatusResponse(job_id=job_id, status=status, error=error, result_path=result_path, result_url=result_url)


from fastapi.responses import RedirectResponse


@app.get("/v1/jobs/{job_id}/result")
def get_job_result(job_id: str, _rl: None = Depends(rate_limit)):
    db_job = get_job(job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if db_job.status != "completed" or not (db_job.result_path or db_job.result_url):
        raise HTTPException(status_code=409, detail="Job not completed yet")
    # If S3/CDN URL exists and redirect is enabled, redirect
    if os.environ.get("RESULT_REDIRECT", "1") == "1" and db_job.result_url:
        return RedirectResponse(url=db_job.result_url, status_code=307)
    # Else serve local file path
    if not db_job.result_path:
        raise HTTPException(status_code=404, detail="Local result not available")
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


@app.middleware("http")
async def _protect_web(request, call_next):
    if os.environ.get("PROTECT_WEB", "0") == "1" and request.url.path.startswith("/web"):
        try:
            # Allow Basic auth or the regular API auth
            basic = check_basic_auth(request)
            if not basic:
                await require_auth(request)
        except HTTPException as e:
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
    return await call_next(request)
