from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import HTTPException, Request
import redis


REDIS_URL = os.environ.get("REDIS_URL", os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"))
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "0") == "1"
RL_LIMIT = int(os.environ.get("RL_LIMIT", "60"))  # requests per window
RL_WINDOW = int(os.environ.get("RL_WINDOW", "60"))  # seconds

_redis: Optional[redis.Redis] = None
if RATE_LIMIT_ENABLED:
    try:
        _redis = redis.from_url(REDIS_URL)
    except Exception:
        _redis = None


async def rate_limit(request: Request) -> None:
    if not RATE_LIMIT_ENABLED or _redis is None:
        return
    ip = request.client.host if request.client else "unknown"
    route = request.url.path
    key = f"rl:{route}:{ip}"
    now = int(time.time())
    pipe = _redis.pipeline()
    # Use a simple counter with TTL
    pipe.incr(key, 1)
    pipe.expire(key, RL_WINDOW)
    count, _ = pipe.execute()
    if int(count) > RL_LIMIT:
        raise HTTPException(status_code=429, detail="Too Many Requests")

