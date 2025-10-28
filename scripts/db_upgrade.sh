#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL="${DATABASE_URL:-sqlite:///storage/vfr.sqlite3}"

echo "Running migrations on ${DATABASE_URL}"
alembic upgrade head

