import os
import shutil
import time
from typing import Optional
from fastapi import UploadFile

try:
    import boto3  # type: ignore
    from botocore.client import Config as BotoConfig  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    boto3 = None
    BotoConfig = None


STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")  # local | s3
STORAGE_ROOT = os.environ.get("VFR_STORAGE", "storage")
UPLOADS_DIR = os.path.join(STORAGE_ROOT, "uploads")
RESULTS_DIR = os.path.join(STORAGE_ROOT, "results")

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_REGION = os.environ.get("S3_REGION")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
S3_PREFIX = os.environ.get("S3_PREFIX", "vfr/")
CDN_BASE_URL = os.environ.get("CDN_BASE_URL")


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

    @staticmethod
    def _s3_client():
        if STORAGE_BACKEND != "s3" or boto3 is None:
            return None
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=S3_REGION,
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            config=BotoConfig(signature_version="s3v4") if BotoConfig else None,
        )
        return client

    @staticmethod
    def publish_result(local_path: str) -> Optional[str]:
        """
        Upload the result to S3 (if configured) and return a CDN or presigned URL.
        Always keeps local files for compatibility.
        """
        client = Storage._s3_client()
        if not client or not S3_BUCKET:
            return None
        key = S3_PREFIX.rstrip("/") + "/results/" + os.path.basename(local_path)
        client.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ACL": "public-read", "ContentType": "image/jpeg"})
        if CDN_BASE_URL:
            return f"{CDN_BASE_URL.rstrip('/')}/{key}"
        # fallback: presigned URL
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=3600,
        )
        return url
