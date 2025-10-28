import os
from fastapi import HTTPException, Request


MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))


async def enforce_max_upload_size(request: Request) -> None:
    cl = request.headers.get("content-length")
    if not cl:
        return
    try:
        size = int(cl)
    except Exception:
        return
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Upload too large")

