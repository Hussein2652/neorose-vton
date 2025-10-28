from __future__ import annotations

import os
from typing import Optional

import redis


REDIS_URL = os.environ.get("REDIS_URL", os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"))
CACHE_TTL = int(os.environ.get("CACHE_TTL", "120"))


def _client() -> Optional[redis.Redis]:
    try:
        return redis.from_url(REDIS_URL)
    except Exception:
        return None


def cache_set_job(job_id: str, status: str, result_path: str | None = None, error: str | None = None, result_url: str | None = None) -> None:
    r = _client()
    if not r:
        return
    payload = {"status": status}
    if result_path:
        payload["result_path"] = result_path
    if error:
        payload["error"] = error
    if result_url:
        payload["result_url"] = result_url
    key = f"job:{job_id}"
    r.setex(key, CACHE_TTL, str(payload))


def cache_get_job(job_id: str) -> Optional[dict]:
    r = _client()
    if not r:
        return None
    key = f"job:{job_id}"
    v = r.get(key)
    if not v:
        return None
    try:
        # payload was stored as str(dict); eval safely is non-trivial; prefer literal_eval.
        import ast

        return ast.literal_eval(v.decode("utf-8"))
    except Exception:
        return None
