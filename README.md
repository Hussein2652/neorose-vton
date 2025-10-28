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
- GraphQL endpoint at `/graphql` (Query: job, health; Mutation: create_tryon_job_from_paths).
- Prometheus metrics at `/metrics` (jobs created/completed/failed, queue size).
- Basic billing/tiers scaffolding (plans/users/usage tables) and model registry/feature flags.
 - Optional pose-aware garment placement (MediaPipe Pose if installed) and GrabCut segmentation (OpenCV).
 - Control maps generation (edge, depth, normal, pose, seg) to mirror ControlNets inputs.
 - VTON placeholder stage (seamless clone or alpha composite) before finisher.

Quickstart
1) Setup
   - Python 3.10+
   - Create a venv and install dependencies:
     python -m venv .venv
     . .venv/bin/activate
     pip install -r requirements.txt
   - Seed placeholder images (optional):
     make seed

2) Run the API
   - Start the server:
     uvicorn backend.app.main:app --reload
   - Docs at:
     http://127.0.0.1:8000/docs
   - Optional web UI (static):
     http://127.0.0.1:8000/web
   - Or run via Makefile:
     make api

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

Alternative JSON Endpoint (URLs)
- Submit images by URL instead of uploading files:
  curl -X POST http://127.0.0.1:8000/v1/jobs/tryon-from-urls \
    -H 'Content-Type: application/json' \
    -d '{"user_image_url":"https://.../user.jpg","garment_front_url":"https://.../garment.jpg"}'

Inspect Config
- Fetch the current YAML config used by the pipeline:
  curl http://127.0.0.1:8000/v1/config

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

Full Docker Setup (API + Celery + Redis + Postgres)
- Run all services:
  - docker compose up -d --build redis postgres celery_worker api
  - API available at http://127.0.0.1:8000
  - Static UI at http://127.0.0.1:8000/web
  - Run Alembic migrations automatically via api entrypoint.
  - Healthchecks are configured for Redis, Postgres, and API.
  - Persistent storage: a named volume `app_storage` is mounted at `/app/storage` for the API and Celery worker so artifacts/results persist across container restarts.

Migrations Service (recommended)
- To run explicit migrations on Postgres before API starts, use the dedicated service:
  - docker compose up -d --build migrate
  - Then bring up API:
    - docker compose up -d --build api

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

GraphQL
- Visit http://127.0.0.1:8000/graphql for GraphiQL
- Sample mutation:
  mutation { createTryonJobFromPaths(userImagePath:"sample_data/user.jpg", garmentFrontPath:"sample_data/garment_front.jpg") }

Prometheus Metrics
- Scrape http://127.0.0.1:8000/metrics
- Exposed metrics: vfr_jobs_created_total, vfr_jobs_completed_total, vfr_jobs_failed_total, vfr_jobs_in_queue

SDKs
- Python: `sdk/python/neorose_vfr_client.py`
- Node: `sdk/node/neoroseVfrClient.js` (requires node-fetch and form-data)

Environment Configuration
- Copy `.env.example` to `.env` and adjust values for your setup.
- When running locally (without Docker), we auto-load `.env` via python-dotenv.
- Docker Compose also uses `.env` for variable substitution.

S3 Profile for Compose
- Use the MinIO-backed S3 preset by layering the override file:
  - docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d --build minio minio_setup
  - docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d --build api celery_worker
  - This sets STORAGE_BACKEND=s3 and points to the MinIO service.

Makefile Shortcuts
- `make install` – create venv and install deps
- `make api` – run Uvicorn API locally
- `make demo` – run local pipeline demo with sample images
- `make seed` – generate sample images in `sample_data/`
- `make migrate` – run Alembic migrations (uses `DATABASE_URL`)
- `make compose-up` – bring up Redis, Postgres, Celery worker, API
- `make compose-up-s3` – additionally bring up MinIO and configure S3 backend
- `make compose-down` – stop and remove containers/volumes

Local MinIO Service
- docker compose up -d minio minio_setup
- Access console at http://127.0.0.1:9001 (minioadmin/minioadmin)
- Use with API by setting:
  - export STORAGE_BACKEND=s3
  - export S3_BUCKET=vfr
  - export S3_REGION=us-east-1
  - export S3_ENDPOINT_URL=http://127.0.0.1:9000
  - export S3_ACCESS_KEY_ID=minioadmin
  - export S3_SECRET_ACCESS_KEY=minioadmin

Authentication (Optional)
- Enable auth enforcement: `export AUTH_REQUIRED=1`
- Firebase: `export FIREBASE_ENABLED=1` and ensure default credentials are set via `GOOGLE_APPLICATION_CREDENTIALS`.
- API key fallback: set `API_KEY=...` and send `x-api-key` header.

Rate Limiting (Optional)
- Enable: `export RATE_LIMIT_ENABLED=1`
- Configure window/limit: `RL_WINDOW` (seconds), `RL_LIMIT` (requests/window). Uses Redis.
 - Per-route limits: set `RL_LIMITS_JSON`, e.g.
   RL_LIMITS_JSON='{"POST:/v1/jobs/tryon": {"limit": 20, "window": 60}, "GET:/v1/jobs/*": {"limit": 200, "window": 60}}'

Protect Web UI
- Require auth for `/web` routes: `export PROTECT_WEB=1` (uses the same auth mechanism configured for the API).

S3/MinIO Storage + CDN (Optional)
- Keep local files for processing, but publish results to S3/CDN:
  - `export STORAGE_BACKEND=s3`
  - `export S3_BUCKET=...`
  - `export S3_REGION=...`
  - Optional: `S3_ENDPOINT_URL` (for MinIO), `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_PREFIX`.
  - Optional: `CDN_BASE_URL` to build public URLs; otherwise uses a presigned URL.
  - `RESULT_REDIRECT=1` makes `/v1/jobs/{id}/result` redirect to the S3/CDN URL when available.

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
 - Optional pose extraction: if `mediapipe` is installed, the system extracts pose landmarks for improved garment placement. Without it, center placement is used.
 - Optional segmentation: uses OpenCV GrabCut when available (opencv-python); otherwise a full mask.
 - Control maps: generated under `storage/results/work/controls/` to approximate ControlNet inputs.
 - VTON stage: enabled by default (`vton.enabled: true` in configs); creates a blended soft render for the finisher.
 - Optional ML QA extras: install via `pip install -r requirements-ml.txt` to enable LPIPS and CLIP-like similarity; otherwise lightweight QA is used.
 - Caching: person canonicalization and keypoints are cached under `storage/cache/users/<sha256>/` keyed by image hash; garment asset cache directories are reserved under `storage/cache/garments/<sha256>/`.

Plans & Quotas
- Plans table supports `monthly_limit` and `per_image_cost` (see Alembic 0004).
- If `AUTH_REQUIRED=1` and a user is identified, the API enforces `monthly_limit` at job submission (returns 402 if exceeded).

Admin & Billing
- Admin list jobs: `GET /v1/admin/jobs` with header `x-admin-key: $ADMIN_API_KEY`
- Feature flags: `GET/POST /v1/admin/feature-flags`
- Model registry: `GET/POST /v1/admin/models`
- Stripe webhook (stub): `POST /v1/billing/stripe/webhook` (set `STRIPE_WEBHOOK_SECRET` for signature verification)

Next Steps (Suggested)
- Wire Redis+Celery and Postgres for robust job orchestration and persistence.
- Replace stubs with real models (segmentation, pose, VTON, img2img, QA metrics).
- Add authentication/authorization, rate limiting, and observability (Prometheus/Grafana).
- Add front-end (React/Flutter) per PDF, using the provided API.
 - Swap local storage for S3/MinIO and add CDN.
Model Artifacts (Download Once)
- Configure artifacts in `configs/models.yaml` and set `PREFETCH_MODELS=1` to prefetch at API startup.
- Artifacts download to `MODELS_DIR` (default `storage/models/`), with file locking to avoid duplicates and a `.complete` marker to skip re-downloads.
- Supports `url:` or `s3_uri:` sources and optional `sha256:` verification, with optional `unpack: true` for archives.
- Admin endpoints:
  - List: `GET /v1/admin/artifacts`
  - Ensure: `POST /v1/admin/artifacts/ensure?name=...&version=...&sha256=...&url=...` (or `s3_uri=...`)
- StableVITON stub can use env vars to ensure weights once: `STABLEVITON_URL` or `STABLEVITON_S3_URI`, plus `STABLEVITON_SHA256` and `STABLEVITON_VERSION`.
Monitoring (Optional)
- Start Prometheus + Grafana overlay:
  - docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d --build prometheus grafana
- Prometheus: http://127.0.0.1:9090 (scrapes API /metrics)
- Grafana: http://127.0.0.1:3000 (admin/admin by default)
GPU Profile (Optional)
- Use the GPU Dockerfile and compose override to enable CUDA for API/worker:
  - docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build api celery_worker
  - Requires a host with NVIDIA drivers and Docker GPU support.

Remote Worker (Optional)
- A simple remote worker is included for testing `FINISHER_BACKEND=remote` and `vton.provider: remote`:
  - docker compose up -d --build remote_worker
  - The API is preconfigured to use http://remote_worker:9000 for remote calls.

Provider Selection
- Configure providers in `configs/pipeline.yaml`:
  - finisher.backend: local | kling | remote | sdxl | flux
  - vton.provider: local | stableviton | remote
  - providers.segmentation: local | torchvision | remote
  - providers.pose: mediapipe | yolov8 | remote
  - providers.geometry: stub | smplx | remote
  - post.upscaler: none | realesrgan; post.face_restore: true|false

ML Extras Installation
- For advanced QA/finisher/providers, install ML extras (bare metal or in GPU image):
  - pip install -r requirements-ml.txt
  - Diffusers (SDXL/Flux) models download into persistent cache at `storage/models/hf` and are reused.
  - TorchVision/Ultralytics download weights once into `storage/models/torch` (set by env automatically).
