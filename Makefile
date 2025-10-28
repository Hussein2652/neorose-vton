SHELL := /bin/bash

.DEFAULT_GOAL := help

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Targets: install, api, demo, seed, compose-up, compose-up-s3, compose-down, migrate, react-dev, compose-up-monitoring, prefetch, verify-models"

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
