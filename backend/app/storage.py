import os
import shutil
import time
from typing import Optional
from fastapi import UploadFile


STORAGE_ROOT = os.environ.get("VFR_STORAGE", "storage")
UPLOADS_DIR = os.path.join(STORAGE_ROOT, "uploads")
RESULTS_DIR = os.path.join(STORAGE_ROOT, "results")


class Storage:
    @staticmethod
    def ensure_dirs() -> None:
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    @staticmethod
    def save_upload(upload: UploadFile, prefix: Optional[str] = None) -> str:
        Storage.ensure_dirs()
        ts = int(time.time() * 1000)
        name = upload.filename or "upload.bin"
        base = f"{prefix}_{ts}_" if prefix else ""
        safe_name = name.replace("/", "_").replace("\\", "_")
        path = os.path.join(UPLOADS_DIR, base + safe_name)
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        return path

    @staticmethod
    def save_result_bytes(data: bytes, filename: str) -> str:
        Storage.ensure_dirs()
        path = os.path.join(RESULTS_DIR, filename)
        with open(path, "wb") as f:
            f.write(data)
        return path

