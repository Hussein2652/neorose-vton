from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from .models import JobCreateResponse, JobStatusResponse
from .queue import Jobs
from .storage import Storage
from .pipeline_runner import run_tryon_job

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

        # Enqueue background job
        job_id = Jobs.enqueue(
            run_tryon_job,
            user_image_path=user_path,
            garment_front_path=g_front_path,
            garment_side_path=g_side_path,
        )
        return JobCreateResponse(job_id=job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = Jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        error=job.error,
        result_path=job.result_path,
    )


@app.get("/v1/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = Jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result_path:
        raise HTTPException(status_code=409, detail="Job not completed yet")
    return FileResponse(job.result_path, filename="vfr_result.jpg")


# Start the worker on app start
@app.on_event("startup")
def _startup():
    Storage.ensure_dirs()
    Jobs.ensure_worker()

