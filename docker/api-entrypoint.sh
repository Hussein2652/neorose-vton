#!/usr/bin/env bash
set -euo pipefail

echo "[api] Running DB migrations..."
alembic upgrade head || true

echo "[api] Starting Uvicorn..."
exec uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

