from __future__ import annotations

import os
import time
import json
from typing import Optional, Tuple

from fastapi import HTTPException, Request
import redis


REDIS_URL = os.environ.get("REDIS_URL", os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"))
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "0") == "1"
RL_LIMIT = int(os.environ.get("RL_LIMIT", "60"))  # default requests per window
RL_WINDOW = int(os.environ.get("RL_WINDOW", "60"))  # default window (seconds)
RL_LIMITS_JSON = os.environ.get("RL_LIMITS_JSON", "")

_limits_map = None
if RL_LIMITS_JSON:
    try:
        _limits_map = json.loads(RL_LIMITS_JSON)
    except Exception:
        _limits_map = None

_redis: Optional[redis.Redis] = None
if RATE_LIMIT_ENABLED:
    try:
        _redis = redis.from_url(REDIS_URL)
    except Exception:
        _redis = None


def _match_limit(method: str, path: str) -> Tuple[int, int]:
    """
    Return (limit, window) for a given method+path using RL_LIMITS_JSON.
    Supports wildcard suffixes like '/v1/jobs/*'. Keys should be 'METHOD:/path'.
    """
    if not _limits_map:
        return RL_LIMIT, RL_WINDOW
    key = f"{method.upper()}:{path}"
    if key in _limits_map:
        cfg = _limits_map[key]
        return int(cfg.get("limit", RL_LIMIT)), int(cfg.get("window", RL_WINDOW))
    # simple wildcard match
    for k, cfg in _limits_map.items():
        if not isinstance(k, str):
            continue
        if not k.upper().startswith(method.upper() + ":"):
            continue
        p = k.split(":", 1)[1]
        if p.endswith("*") and path.startswith(p[:-1]):
            return int(cfg.get("limit", RL_LIMIT)), int(cfg.get("window", RL_WINDOW))
    return RL_LIMIT, RL_WINDOW


async def rate_limit(request: Request) -> None:
    if not RATE_LIMIT_ENABLED or _redis is None:
        return
    ip = request.client.host if request.client else "unknown"
    route = request.url.path
    method = request.method
    limit, window = _match_limit(method, route)
    key = f"rl:{method}:{route}:{ip}"
    now = int(time.time())
    pipe = _redis.pipeline()
    # Use a simple counter with TTL
    pipe.incr(key, 1)
    pipe.expire(key, window)
    count, _ = pipe.execute()
    if int(count) > limit:
        raise HTTPException(status_code=429, detail="Too Many Requests")
