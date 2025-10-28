from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join("storage", "models"))
LOCKS_DIR = os.environ.get("LOCKS_DIR", os.path.join("storage", "locks"))


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _lock_path(key: str) -> str:
    _safe_makedirs(LOCKS_DIR)
    return os.path.join(LOCKS_DIR, f"{key}.lock")


class FileLock:
    def __init__(self, key: str, timeout: int = 300) -> None:
        self.path = _lock_path(key)
        self.timeout = timeout

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL ensures exclusive creation
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                return self
            except FileExistsError:
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Timeout acquiring lock {self.path}")
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


def _download_http(url: str, dest_path: str) -> None:
    import requests

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _download_s3(s3_uri: str, dest_path: str) -> None:
    # s3_uri: s3://bucket/key
    import boto3  # type: ignore

    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")
    bucket_key = s3_uri[len("s3://") :]
    bucket, key = bucket_key.split("/", 1)
    s3 = boto3.client(
        "s3",
        region_name=os.environ.get("S3_REGION"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    )
    s3.download_file(bucket, key, dest_path)


def _maybe_unpack(src_path: str, target_dir: str) -> Optional[str]:
    # Returns directory path if unpacked, else None
    if src_path.endswith((".zip", ".ZIP")):
        with zipfile.ZipFile(src_path, "r") as zf:
            zf.extractall(target_dir)
        return target_dir
    if any(src_path.endswith(ext) for ext in (".tar.gz", ".tgz", ".tar")):
        with tarfile.open(src_path, "r:*") as tf:
            tf.extractall(target_dir)
        return target_dir
    return None


@dataclass
class ArtifactSpec:
    name: str
    version: str
    sha256: Optional[str] = None
    url: Optional[str] = None
    s3_uri: Optional[str] = None
    unpack: bool = False


def ensure_artifact(spec: ArtifactSpec) -> Tuple[str, bool]:
    """
    Ensure an artifact exists locally under MODELS_DIR/name/version.
    - Returns (local_path, downloaded) where local_path is the file or directory path.
    - If 'unpack' is true and the artifact is an archive, the directory path is returned.
    - Only downloads once per name/version; verifies SHA256 if provided.
    """
    _safe_makedirs(MODELS_DIR)
    base_dir = os.path.join(MODELS_DIR, spec.name, spec.version)
    _safe_makedirs(base_dir)
    marker = os.path.join(base_dir, ".complete")

    # If marker exists and sha is not provided, assume done
    if os.path.exists(marker) and (spec.sha256 is None or any(os.scandir(base_dir))):
        # Prefer unpacked dir if present, else single file in dir
        entries = [e.path for e in os.scandir(base_dir) if not e.name.startswith(".")]
        if entries:
            # If a single file, return it; else the directory
            if len(entries) == 1 and os.path.isfile(entries[0]):
                return entries[0], False
            return base_dir, False

    lock_key = f"artifact_{spec.name}_{spec.version}"
    with FileLock(lock_key):
        # Check again inside lock
        if os.path.exists(marker) and (spec.sha256 is None or any(os.scandir(base_dir))):
            entries = [e.path for e in os.scandir(base_dir) if not e.name.startswith(".")]
            if entries:
                if len(entries) == 1 and os.path.isfile(entries[0]):
                    return entries[0], False
                return base_dir, False

        # Download to temp
        tmp_dir = tempfile.mkdtemp(prefix="dl_", dir=base_dir)
        try:
            tmp_file = os.path.join(tmp_dir, "artifact.bin")
            if spec.url:
                _download_http(spec.url, tmp_file)
            elif spec.s3_uri:
                _download_s3(spec.s3_uri, tmp_file)
            else:
                raise ValueError("Artifact spec must have url or s3_uri")

            if spec.sha256:
                digest = _sha256_file(tmp_file)
                if digest.lower() != spec.sha256.lower():
                    raise ValueError("SHA256 mismatch for artifact")

            # Move to final and optionally unpack
            # Derive final filename from URL/S3 key when possible
            fname = "artifact.bin"
            if spec.url:
                try:
                    fname = os.path.basename(spec.url.split("?")[0]) or fname
                except Exception:
                    pass
            elif spec.s3_uri:
                try:
                    key = spec.s3_uri[len("s3://"):].split("/", 1)[1]
                    fname = os.path.basename(key) or fname
                except Exception:
                    pass
            final_path = os.path.join(base_dir, fname)
            shutil.move(tmp_file, final_path)
            if spec.unpack:
                unpack_dir = os.path.join(base_dir, "unpacked")
                _safe_makedirs(unpack_dir)
                _maybe_unpack(final_path, unpack_dir)
                # Keep the archive too; marker points to unpacked dir
                with open(marker, "w", encoding="utf-8") as f:
                    f.write("unpacked\n")
                return unpack_dir, True
            else:
                with open(marker, "w", encoding="utf-8") as f:
                    f.write(os.path.basename(final_path) + "\n")
                return final_path, True
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
