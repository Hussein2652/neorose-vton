SHELL := /bin/bash

.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Targets: install, api, demo, seed, compose-up, compose-up-s3, compose-down, compose-down-volumes, migrate, react-dev, compose-up-monitoring, prefetch, verify-models, ingest-lama"
	@echo "Extra (optional): rclone-config, fetch-stableviton, verify-stableviton"

install:
	python -m venv $(VENV)
	. $(VENV)/bin/activate; $(PIP) install --upgrade pip
	. $(VENV)/bin/activate; $(PIP) install -r requirements.txt

api:
	. $(VENV)/bin/activate; uvicorn backend.app.main:app --reload

demo:
	. $(VENV)/bin/activate; python scripts/run_local_demo.py --user sample_data/user.jpg --garment sample_data/garment_front.jpg --out storage/demo_result.jpg

seed:
	. $(VENV)/bin/activate; python scripts/seed_sample_data.py

migrate:
	chmod +x scripts/db_upgrade.sh; ./scripts/db_upgrade.sh

compose-up:
	docker compose up -d --build redis postgres celery_worker api

compose-up-s3:
	docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d --build minio minio_setup
	docker compose -f docker-compose.yml -f docker-compose.s3.yml up -d --build redis postgres celery_worker api

compose-down:
	docker compose down

# Explicitly remove volumes only when you really mean it
compose-down-volumes:
	docker compose down -v

react-dev:
	cd frontend/react-app && npm install && npm run dev

compose-up-monitoring:
	docker compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d --build prometheus grafana

prefetch:
	. $(VENV)/bin/activate || true; python scripts/prefetch_models.py --config configs/models.yaml --models-dir storage/models

verify-models:
	@echo "GET /v1/health/models; expecting ok=true"; \
	curl -sf http://127.0.0.1:8000/v1/health/models | python -c 'import sys,json; d=json.load(sys.stdin); print("models_ok=", d.get("ok")); sys.exit(0 if d.get("ok") else 1)'

ingest-lama:
	. $(VENV)/bin/activate || true; python scripts/ingest_lama.py --models-dir storage/models --manual-dir manual_downloads || true

# ---------- Optional helpers for large OneDrive downloads (run on host) ----------
RCLONE_REMOTE ?= onedrive
STABLEVITON_DIR ?= StableVITON

rclone-config:
	rclone config

fetch-stableviton:
	@mkdir -p storage/models/stableviton/weights
	@echo "Copying StableVITON weights from $(RCLONE_REMOTE):$(STABLEVITON_DIR) -> storage/models/stableviton/weights";
	rclone copy "$(RCLONE_REMOTE):$(STABLEVITON_DIR)/VITONHD.ckpt" storage/models/stableviton/weights --progress --checkers 8 --transfers 4 --retries 999 --low-level-retries 999 --retries-sleep 10s || true
	rclone copy "$(RCLONE_REMOTE):$(STABLEVITON_DIR)/VITONHD_VAE_finetuning.ckpt" storage/models/stableviton/weights --progress --checkers 8 --transfers 4 --retries 999 --low-level-retries 999 --retries-sleep 10s || true
	rclone copy "$(RCLONE_REMOTE):$(STABLEVITON_DIR)/VITONHD_PBE_pose.ckpt" storage/models/stableviton/weights --progress --checkers 8 --transfers 4 --retries 999 --low-level-retries 999 --retries-sleep 10s || true
	@echo "Done. Files in storage/models/stableviton/weights:"; ls -lh storage/models/stableviton/weights || true

verify-stableviton:
	@python - <<'PY'
import hashlib,sys
from pathlib import Path
p=Path('storage/models/stableviton/weights')
missing=False
for name in ['VITONHD.ckpt','VITONHD_VAE_finetuning.ckpt','VITONHD_PBE_pose.ckpt']:
    f=p/name
    if not f.exists():
        print('MISSING', f)
        missing=True
        continue
    h=hashlib.sha256(f.read_bytes()).hexdigest()
    print(f'{f.name} sha256={h}')
sys.exit(1 if missing else 0)
PY
