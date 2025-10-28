Neorose VFR (Virtual Fitting Room)

Overview
- Implements the project skeleton described in “Virtual Fitting Room Requirements.pdf”.
- Provides a FastAPI backend, an end-to-end pipeline skeleton with clear I/O contracts, local storage, baseline configs, and a simple CLI/demo.
- Heavy ML components are stubbed with lightweight placeholders so the flow runs locally without large model downloads.

What’s Included
- FastAPI backend with endpoints to submit try-on jobs, check status, and fetch results.
- In-process background job runner (no Redis required) to keep things simple.
- Optional Celery/Redis integration for distributed background execution.
- SQLite by default, with Postgres support via `DATABASE_URL`.
- Alembic migrations for DB schema.
- Redis-backed job status cache for fast polling.
- Two UIs: a simple static demo and a React app scaffold.
- Pipeline skeleton modules matching the PDF stages: person parsing, pose extraction, garment warping, geometry fitting, finisher, and post-processing.
- Configs in YAML per the PDF (baseline hyperparameters included and easily overridden via env).
- Simple Python SDK client and a local CLI/demo script.

Quickstart
1) Setup
   - Python 3.10+
   - Create a venv and install dependencies:
     python -m venv .venv
     . .venv/bin/activate
     pip install -r requirements.txt

2) Run the API
   - Start the server:
     uvicorn backend.app.main:app --reload
   - Docs at:
     http://127.0.0.1:8000/docs
   - Optional web UI (static):
     http://127.0.0.1:8000/web

3) Submit a Try-On Job (multipart form)
   - From docs UI or via curl:
     curl -X POST \
       -F "user_image=@sample_data/user.jpg" \
       -F "garment_front=@sample_data/garment_front.jpg" \
       -F "garment_side=@sample_data/garment_side.jpg" \
       http://127.0.0.1:8000/v1/jobs/tryon

   - Response contains a job_id. Check status:
     curl http://127.0.0.1:8000/v1/jobs/<job_id>

   - When done, download result:
     curl -OJ http://127.0.0.1:8000/v1/jobs/<job_id>/result

Use Celery + Redis (Optional)
- Start infra via docker-compose (Redis + Postgres + Celery worker):
  - docker compose up -d redis postgres
  - export DATABASE_URL=postgresql+psycopg://vfr:vfr@localhost:5432/vfr
  - docker compose build celery_worker && docker compose up -d celery_worker
- Start API with Celery enabled:
  - export USE_CELERY=1
  - export CELERY_BROKER_URL=redis://localhost:6379/0
  - export CELERY_RESULT_BACKEND=redis://localhost:6379/1
  - uvicorn backend.app.main:app --reload
  - Jobs will be enqueued to the Celery worker; status is synced via task id.

Database Migrations (Alembic)
- Ensure dependencies installed, then run:
  - chmod +x scripts/db_upgrade.sh
  - ./scripts/db_upgrade.sh
  - Set `DATABASE_URL` to target Postgres before running if needed.

Job Status Cache (Redis)
- Optional: export `REDIS_URL=redis://localhost:6379/2` (default). Cache TTL via `CACHE_TTL`.

React Frontend
- A React app scaffold is in `frontend/react-app` (Vite + TS):
  - cd frontend/react-app
  - npm install
  - npm run dev
  - Opens on http://127.0.0.1:5173 (proxy to API is configured)

4) Run Local Demo (no API)
   - Provide a user and garment image; outputs a composite result to storage:
     python scripts/run_local_demo.py \
       --user sample_data/user.jpg \
       --garment sample_data/garment_front.jpg \
       --out storage/demo_result.jpg

Project Layout
- backend/
  - app/main.py              FastAPI app and routes
  - app/models.py            Pydantic request/response models
  - app/config.py            Settings loader (env + YAML)
  - app/queue.py             In-process job queue (worker thread)
  - app/storage.py           Local file storage helpers
  - app/pipeline_runner.py   Bridges API jobs to pipeline
- pipeline/
  - io_types.py              Shared I/O contracts for stages
  - pipeline.py              Orchestrates all stages
  - person_parsing.py        Stub segmentation/masks
  - pose_extraction.py       Stub pose/keypoints
  - garment_warping.py       Stub garment alignment/warp
  - geometry_fitting.py      Stub draping placeholder
  - finisher.py              Stub photoreal finisher placeholder
  - post_processing.py       Stub post-process filters
- configs/
  - pipeline.yaml            Baseline hyperparameters from PDF
- sdk/python/neorose_vfr_client.py  Minimal Python client SDK
- scripts/run_local_demo.py         CLI to run pipeline locally
- requirements.txt           Minimal runtime deps
- sample_data/               Placeholder images directory (add your own)

Notes
- Advanced features (SMPL-X, real segmentation/pose, ReclothVITON/StableVITON, SDXL/Flux, CodeFormer, Kling API integration) are stubbed. The code isolates these behind interfaces so they can be swapped with real implementations.
- Storage uses the local filesystem under `storage/`. Swap with S3/MinIO later via `backend/app/storage.py`.
- Configs live in `configs/pipeline.yaml` and are overridable by env vars.
 - DB: defaults to SQLite at `storage/vfr.sqlite3`. Override with `DATABASE_URL` to use Postgres.
 - Celery: set `USE_CELERY=1` and provide `CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND`.
 - Finisher backend: set `FINISHER_BACKEND=local` (default) or `FINISHER_BACKEND=kling` to use the Kling stub. For real Kling integration, implement providers/kling_api.py HTTP call.
  - Kling API: set `KLING_API_KEY`, `KLING_API_ENDPOINT`, and optional `KLING_TIMEOUT` to call a real endpoint. Falls back to local enhancement if not configured.

Next Steps (Suggested)
- Wire Redis+Celery and Postgres for robust job orchestration and persistence.
- Replace stubs with real models (segmentation, pose, VTON, img2img, QA metrics).
- Add authentication/authorization, rate limiting, and observability (Prometheus/Grafana).
- Add front-end (React/Flutter) per PDF, using the provided API.
 - Swap local storage for S3/MinIO and add CDN.
