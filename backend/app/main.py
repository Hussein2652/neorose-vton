import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .models import JobCreateResponse, JobStatusResponse, JobCreateFromUrlsRequest
from .queue import Jobs
from .storage import Storage
from .pipeline_runner import run_tryon_job
from .db import init_db, create_job, get_job, update_job
from .cache import cache_get_job, cache_set_job
from .auth import require_auth, check_basic_auth
from .ratelimit import rate_limit
from .logging_config import setup_logging
from .metrics import jobs_created, jobs_completed, jobs_failed, jobs_in_queue
from prometheus_client import make_asgi_app as make_prom_app
from .graphql_schema import graphql_app
from .billing import router as billing_router
from .validators import enforce_max_upload_size
from .artifacts import ensure_artifact, ArtifactSpec, ingest_manual_assets
import yaml
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


# Configure CORS and mount auxiliary apps/routes at import time
try:
    origins = os.environ.get("CORS_ORIGINS", "*")
    origin_list = [o.strip() for o in origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    # Safe-guard: if middleware cannot be added (should not happen before startup), continue
    pass

# Optional mounts (best-effort): static web, Prometheus, GraphQL, billing routes
try:
    app.mount("/web", StaticFiles(directory="frontend/web", html=True), name="web")
except Exception:
    pass
try:
    app.mount("/metrics", make_prom_app())
except Exception:
    pass
try:
    app.mount("/graphql", graphql_app)
except Exception:
    pass
# Billing routes
try:
    app.include_router(billing_router, prefix="/v1")
except Exception:
    pass


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
    _lim=Depends(enforce_max_upload_size),
):
    try:
        user_id = None
        try:
            # _user may be None when auth is disabled
            user_id = getattr(_user, 'uid', None)
        except Exception:
            user_id = None
        # Save uploads
        user_path = Storage.save_upload(user_image, prefix="user")
        g_front_path = Storage.save_upload(garment_front, prefix="garment_front")
        g_side_path = (
            Storage.save_upload(garment_side, prefix="garment_side") if garment_side else None
        )

        # Create DB job and schedule execution
        import uuid

        job_id = str(uuid.uuid4())
        # Optional plan quota enforcement
        try:
            if user_id:
                from .db import ensure_user, get_plan, get_month_usage_units
                u = ensure_user(user_id)
                p = get_plan(u.plan) if u else None
                if p and p.monthly_limit is not None:
                    used = get_month_usage_units(user_id)
                    if used >= p.monthly_limit:
                        raise HTTPException(status_code=402, detail="Quota exceeded; upgrade plan")
        except HTTPException:
            raise
        except Exception:
            pass
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
                user_id=user_id,
                plan=os.environ.get("DEFAULT_PLAN", "free"),
            )
            cache_set_job(job_id, status="queued")
        else:
            create_job(
                job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                provider=os.environ.get("FINISHER_BACKEND", "local"),
                user_id=user_id,
                plan=os.environ.get("DEFAULT_PLAN", "free"),
            )
            Jobs.enqueue(
                run_tryon_job,
                queue_job_id=job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                job_id=job_id,
            )
            cache_set_job(job_id, status="queued")
        jobs_created.inc()
        jobs_in_queue.inc()
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/jobs/tryon-from-urls", response_model=JobCreateResponse)
async def create_tryon_job_from_urls(body: JobCreateFromUrlsRequest, _user=Depends(require_auth), _rl=Depends(rate_limit)):
    try:
        user_id = None
        try:
            user_id = getattr(_user, 'uid', None)
        except Exception:
            user_id = None
        # Fetch URLs to local storage
        user_path = Storage.save_from_url(body.user_image_url, prefix="user")
        g_front_path = Storage.save_from_url(body.garment_front_url, prefix="garment_front")
        g_side_path = Storage.save_from_url(body.garment_side_url, prefix="garment_side") if body.garment_side_url else None

        import uuid
        job_id = str(uuid.uuid4())
        # Optional plan quota enforcement
        try:
            if user_id:
                from .db import ensure_user, get_plan, get_month_usage_units
                u = ensure_user(user_id)
                p = get_plan(u.plan) if u else None
                if p and p.monthly_limit is not None:
                    used = get_month_usage_units(user_id)
                    if used >= p.monthly_limit:
                        raise HTTPException(status_code=402, detail="Quota exceeded; upgrade plan")
        except HTTPException:
            raise
        except Exception:
            pass

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
                user_id=user_id,
                plan=os.environ.get("DEFAULT_PLAN", "free"),
            )
        else:
            create_job(
                job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                provider=os.environ.get("FINISHER_BACKEND", "local"),
                user_id=user_id,
                plan=os.environ.get("DEFAULT_PLAN", "free"),
            )
            Jobs.enqueue(
                run_tryon_job,
                queue_job_id=job_id,
                user_image_path=user_path,
                garment_front_path=g_front_path,
                garment_side_path=g_side_path,
                job_id=job_id,
            )
        jobs_created.inc(); jobs_in_queue.inc(); cache_set_job(job_id, status="queued")
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/config")
def get_config():
    from .config import settings as s
    import yaml
    cfg_path = os.environ.get("VFR_CONFIG", "configs/pipeline.yaml")
    raw = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
    return {"config": raw}


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


# Admin endpoints
def _require_admin(request) -> None:
    key = os.environ.get("ADMIN_API_KEY")
    if not key or request.headers.get("x-admin-key") != key:
        raise HTTPException(status_code=401, detail="Admin key required")


@app.get("/v1/admin/jobs")
def admin_list_jobs(limit: int = 50, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import list_jobs as db_list_jobs

    rows = db_list_jobs(limit)
    return [
        {
            "id": j.id,
            "status": j.status,
            "error": j.error,
            "provider": j.provider,
            "user_id": j.user_id,
            "plan": j.plan,
            "quality": j.quality,
            "cost_estimate": j.cost_estimate,
            "result_url": j.result_url,
            "created_at": str(j.created_at),
        }
        for j in rows
    ]


@app.get("/v1/admin/feature-flags")
def admin_list_flags(request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import FeatureFlagORM, engine
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        rows = list(s.query(FeatureFlagORM).all())  # type: ignore[attr-defined]
        return [{"key": r.key, "value": r.value, "description": r.description} for r in rows]


@app.post("/v1/admin/feature-flags")
def admin_set_flag(key: str, value: str, description: str | None = None, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import set_feature_flag

    set_feature_flag(key, value, description)
    return {"ok": True}


@app.get("/v1/admin/models")
def admin_list_models(request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import ModelArtifactORM, engine
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        rows = list(s.query(ModelArtifactORM).all())  # type: ignore[attr-defined]
        return [
            {"name": r.name, "version": r.version, "sha256": r.sha256, "s3_path": r.s3_path, "created_at": str(r.created_at)}
            for r in rows
        ]


@app.post("/v1/admin/models")
def admin_upsert_model(name: str, version: str, sha256: str, s3_path: str | None = None, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import ModelArtifactORM, engine
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        obj = s.get(ModelArtifactORM, name)
        if not obj:
            obj = ModelArtifactORM(name=name, version=version, sha256=sha256, s3_path=s3_path)
        else:
            obj.version = version
            obj.sha256 = sha256
            obj.s3_path = s3_path
        s.add(obj)
        s.commit()
    return {"ok": True}


@app.get("/v1/admin/artifacts")
def admin_list_artifacts(request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import list_artifacts
    rows = list_artifacts()
    def present(p):
        try:
            return bool(p and os.path.exists(p))
        except Exception:
            return False
    return [
        {
            "name": r.name,
            "version": r.version,
            "sha256": r.sha256,
            "s3_path": r.s3_path,
            "local_path": r.local_path,
            "present": present(r.local_path),
            "size_bytes": r.size_bytes,
            "downloaded_at": str(r.downloaded_at) if r.downloaded_at else None,
        }
        for r in rows
    ]


@app.post("/v1/admin/artifacts/ensure")
def admin_ensure_artifact(name: str, version: str, sha256: str | None = None, url: str | None = None, s3_uri: str | None = None, request=Depends(lambda r: r)):
    _require_admin(request)
    from .artifacts import ensure_artifact, ArtifactSpec
    from .db import set_artifact_local
    local, _ = ensure_artifact(ArtifactSpec(name=name, version=version, sha256=sha256, url=url, s3_uri=s3_uri, unpack=True))
    try:
        size = 0
        if os.path.isdir(local):
            for root, _dirs, files in os.walk(local):
                for fn in files:
                    size += os.path.getsize(os.path.join(root, fn))
        else:
            size = os.path.getsize(local)
        set_artifact_local(name, version, local, size)
    except Exception:
        pass
    return {"ok": True, "local": local}


@app.get("/v1/admin/plans")
def admin_list_plans(request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import list_plans
    rows = list_plans()
    return [
        {
            "name": r.name,
            "default_backend": r.default_backend,
            "max_res_long": r.max_res_long,
            "monthly_limit": r.monthly_limit,
            "per_image_cost": r.per_image_cost,
        }
        for r in rows
    ]


@app.post("/v1/admin/plans")
def admin_upsert_plan(name: str, default_backend: str | None = None, max_res_long: str | None = None, monthly_limit: int | None = None, per_image_cost: float | None = None, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import upsert_plan
    upsert_plan(name, default_backend, max_res_long, monthly_limit, per_image_cost)
    return {"ok": True}


@app.post("/v1/admin/users/{user_id}/plan")
def admin_set_user_plan(user_id: str, plan: str, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import set_user_plan
    set_user_plan(user_id, plan)
    return {"ok": True}


@app.get("/v1/admin/usage")
def admin_get_usage(user_id: str, request=Depends(lambda r: r)):
    _require_admin(request)
    from .db import get_usage_summary
    return get_usage_summary(user_id)


@app.get("/v1/me")
def me(_user=Depends(require_auth)):
    if not _user:
        return {"anon": True}
    return {"uid": _user.uid, "email": getattr(_user, "email", None)}


# Start the worker on app start
@app.on_event("startup")
def _startup():
    setup_logging()
    Storage.ensure_dirs()
    init_db()
    Jobs.ensure_worker()

    # Prefetch models if configured
    try:
        if os.environ.get("PREFETCH_MODELS", "0") == "1":
            cfg_path = os.environ.get("MODELS_CONFIG", "configs/models.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    mcfg = yaml.safe_load(f) or {}
                for a in mcfg.get('artifacts', []) or []:
                    spec = ArtifactSpec(
                        name=a.get('name'),
                        version=a.get('version'),
                        sha256=a.get('sha256'),
                        url=a.get('url'),
                        s3_uri=a.get('s3_uri'),
                        unpack=bool(a.get('unpack', False)),
                    )
                    local, _ = ensure_artifact(spec)
                    # update registry info
                    try:
                        from .db import set_artifact_local
                        size = 0
                        if os.path.isdir(local):
                            for root, _dirs, files in os.walk(local):
                                for fn in files:
                                    size += os.path.getsize(os.path.join(root, fn))
                        else:
                            size = os.path.getsize(local)
                        set_artifact_local(spec.name, spec.version, local, size)
                    except Exception:
                        pass
            # Ingest manual licensed assets (SMPL-X/PIXIE)
            manual_dir = os.environ.get("MANUAL_MODELS_DIR", "manual_downloads")
            if os.path.isdir(manual_dir):
                ingest_manual_assets(manual_dir)
            # Also prefetch HF models if configured
            if os.environ.get("PREFETCH_HF_MODELS", "0") == "1":
                try:
                    from huggingface_hub import snapshot_download  # type: ignore
                    # SDXL base/refiner + common controlnets + SDXL turbo + Flux
                    ids = []
                    for mid in (mcfg.get('hf_models') or []):
                        if isinstance(mid, str) and mid:
                            ids.append(mid)
                    import re
                    def san(s: str) -> str:
                        return re.sub(r'[^a-zA-Z0-9_.-]+', '-', s)
                    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')
                    for mid in ids:
                        try:
                            local_dir = os.path.join(os.environ.get('MODELS_DIR', 'storage/models'), 'snapshots', san(mid))
                            os.makedirs(local_dir, exist_ok=True)
                            snapshot_download(repo_id=mid, token=token, local_files_only=False, local_dir=local_dir, local_dir_use_symlinks=False)
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception as e:
        # Log but don't fail startup
        import logging
        logging.getLogger(__name__).warning(f"Prefetch models failed: {e}")


@app.get("/v1/health/models")
def health_models() -> dict:
    required = []
    # Critical local paths for strictly-offline finisher & preprocessors
    base = os.environ.get("SDXL_BASE_PATH", "/app/storage/models/sdxl_base/1.0/sd_xl_base_1.0.safetensors")
    refiner = os.environ.get("SDXL_REFINER_PATH", "/app/storage/models/sdxl_refiner/1.0/sd_xl_refiner_1.0.safetensors")
    union = os.environ.get("CONTROLNET_UNION_SDXL_DIR", "/app/storage/models/snapshots/xinsir-controlnet-union-sdxl-1.0")
    annot = os.environ.get("ANNOTATOR_DIR", "/app/storage/models/snapshots/lllyasviel-Annotators")

    # Perception models (pose, depth, matting, upscaler)
    yolo = os.environ.get("YOLOV8_POSE_MODEL", "/app/storage/models/yolo_v8_pose/v8x/yolov8x-pose.pt")
    zoe = os.path.join(os.environ.get("MODELS_DIR", "/app/storage/models"), "zoedepth_m12_nk", "v1", "ZoeD_M12_NK.pt")
    rvm = os.path.join(os.environ.get("MODELS_DIR", "/app/storage/models"), "rvm_mobilenetv3", "v1.0.0", "rvm_mobilenetv3.pth")
    esr = os.environ.get("REALESRGAN_WEIGHTS", "/app/storage/models/realesrgan/x4plus/RealESRGAN_x4plus.pth")
    models_root = os.environ.get("MODELS_DIR", "/app/storage/models")
    lama_root = os.path.join(models_root, "big_lama", "v1")
    lama_candidates = [
        os.path.join(lama_root, "big-lama.pt"),
        os.path.join(lama_root, "big-lama.ckpt"),
        os.path.join(lama_root, "unpacked", "big-lama", "big-lama.pt"),
        os.path.join(lama_root, "unpacked", "big-lama", "best.ckpt"),
    ]

    # Licensed geometry models (manual ingestion)
    smplx_home = os.environ.get("SMPLX_HOME", "/app/storage/models/smplx")
    pixie_home = os.environ.get("PIXIE_HOME", "/app/storage/models/pixie")
    smplx_npz = os.path.join(smplx_home, "SMPLX_NEUTRAL.npz")

    checks: list[tuple[str, str]] = [
        (base, 'file'),
        (refiner, 'file'),
        (union, 'dir'),
        (annot, 'dir'),
        (yolo, 'file'),
        (zoe, 'file'),
        (rvm, 'file'),
        (esr, 'file'),
        (smplx_home, 'dir'),
        (smplx_npz, 'file'),
        (pixie_home, 'dir'),
    ]

    # Evaluate existence
    exists_map: dict[str, bool] = {}
    for p, t in checks:
        exists = os.path.isfile(p) if t == 'file' else os.path.isdir(p)
        exists_map[p] = exists
        required.append({"path": p, "type": t, "exists": exists})

    # LaMa can be one of several weight file names/locations
    lama_exists_flags = []
    for p in lama_candidates:
        ex = os.path.isfile(p)
        lama_exists_flags.append(ex)
        required.append({"path": p, "type": 'file', "exists": ex})

    mandatory_ok = all(exists_map.values())
    lama_ok = any(lama_exists_flags)
    ok = mandatory_ok and lama_ok
    return {"ok": ok, "items": required}

@app.on_event("shutdown")
def _shutdown():
    # best-effort: reset queue gauge
    try:
        jobs_in_queue.set(0)
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
