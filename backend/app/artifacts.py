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


def _download_http(url: str, dest_path: str, max_retries: int = 6) -> None:
    """Robust HTTP downloader with resume support and retries.
    - Resumes using HTTP Range when server supports it.
    - Uses exponential backoff on transient errors.
    """
    import requests
    import time as _time
    from requests.exceptions import ChunkedEncodingError, ConnectionError

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    attempt = 0
    chunk_size = 4 * 1024 * 1024  # 4 MB
    session = requests.Session()
    headers_base = {"Accept-Encoding": "identity", "User-Agent": "neorose-prefetch/1.0"}
    # Hugging Face token support for gated/large files
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if "huggingface.co" in url and hf_token:
        headers_base["Authorization"] = f"Bearer {hf_token}"
        headers_base["Referer"] = "https://huggingface.co"

    while True:
        attempt += 1
        try:
            start_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            headers = dict(headers_base)
            if start_size > 0:
                headers["Range"] = f"bytes={start_size}-"
            r = session.get(url, stream=True, timeout=(20, 600), headers=headers, allow_redirects=True)
            # If server doesn't support Range and we had a partial file, restart from scratch
            if r.status_code == 200 and start_size > 0:
                r.close()
                # restart from scratch
                try:
                    os.remove(dest_path)
                except FileNotFoundError:
                    pass
                start_size = 0
                r = session.get(url, stream=True, timeout=(20, 600), headers=headers_base, allow_redirects=True)
            r.raise_for_status()

            mode = "ab" if start_size > 0 else "wb"
            with open(dest_path, mode) as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            # Download completed
            return
        except (ChunkedEncodingError, ConnectionError, IOError) as e:
            if attempt >= max_retries:
                raise
            # backoff (1,2,4,8,16,32)
            _time.sleep(min(32, 2 ** (attempt - 1)))
            continue


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
    # Coerce to str to handle YAML numeric versions (e.g., 20190826, 1.0)
    name_str = str(spec.name)
    version_str = str(spec.version)
    base_dir = os.path.join(MODELS_DIR, name_str, version_str)
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

    lock_key = f"artifact_{name_str}_{version_str}"
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


def _copytree(src: str, dst: str) -> None:
    if os.path.isdir(src):
        if os.path.exists(dst):
            # Merge copy
            for root, dirs, files in os.walk(src):
                rel = os.path.relpath(root, src)
                troot = os.path.join(dst, rel) if rel != "." else dst
                os.makedirs(troot, exist_ok=True)
                for f in files:
                    shutil.copy2(os.path.join(root, f), os.path.join(troot, f))
        else:
            shutil.copytree(src, dst)


def ingest_manual_assets(manual_dir: str) -> dict:
    """
    Ingest manually downloaded, licensed assets into the persistent models directory.
    - Expects optional subdirectories under `manual_dir`, e.g., `smplx/`, `pixie/`.
    - Recognizes common archive filenames and extracts them into target locations.
    Returns a dict report of actions taken.
    """
    report = {"manual_dir": manual_dir, "actions": []}
    if not manual_dir or not os.path.isdir(manual_dir):
        return report
    try:
        # Resolve typical folder names
        smplx_src = None
        for name in ("smplx_downloads", "smplx"):
            cand = os.path.join(manual_dir, name)
            if os.path.isdir(cand):
                smplx_src = cand
                break
        pixie_src = None
        for name in ("pixie_downloads", "pixie"):
            cand = os.path.join(manual_dir, name)
            if os.path.isdir(cand):
                pixie_src = cand
                break
        # LaMa (big-lama) optional manual drop
        lama_src = None
        for name in ("lama_downloads", "lama"):
            cand = os.path.join(manual_dir, name)
            if os.path.isdir(cand):
                lama_src = cand
                break

        # SMPL-X assets
        if smplx_src:
            smplx_dst = os.path.join(MODELS_DIR, "smplx")
            os.makedirs(smplx_dst, exist_ok=True)
            # Copy core files
            for fname in os.listdir(smplx_src):
                srcp = os.path.join(smplx_src, fname)
                if os.path.isfile(srcp):
                    # Rename SMPLX_NEUTRAL_*.npz to SMPLX_NEUTRAL.npz (compat)
                    if fname.startswith("SMPLX_NEUTRAL") and fname.endswith(".npz"):
                        shutil.copy2(srcp, os.path.join(smplx_dst, fname))
                        shutil.copy2(srcp, os.path.join(smplx_dst, "SMPLX_NEUTRAL.npz"))
                        report["actions"].append({"copied": [srcp, os.path.join(smplx_dst, "SMPLX_NEUTRAL.npz")]})
                    elif fname.endswith(".pkl"):
                        shutil.copy2(srcp, os.path.join(smplx_dst, fname))
                        report["actions"].append({"copied": [srcp, os.path.join(smplx_dst, fname)]})
                    elif fname.lower().endswith(".zip"):
                        # Extract known zips
                        target = "vposer" if "vposer" in fname.lower() else ("uv" if "uv" in fname.lower() else None)
                        with zipfile.ZipFile(srcp, "r") as zf:
                            outdir = os.path.join(smplx_dst, target) if target else smplx_dst
                            os.makedirs(outdir, exist_ok=True)
                            zf.extractall(outdir)
                            report["actions"].append({"unzipped": [srcp, outdir]})

        # PIXIE assets
        if pixie_src:
            pixie_dst = os.path.join(MODELS_DIR, "pixie")
            os.makedirs(pixie_dst, exist_ok=True)
            for fname in os.listdir(pixie_src):
                srcp = os.path.join(pixie_src, fname)
                if os.path.isfile(srcp):
                    if fname.lower().endswith(".tar"):
                        # Use auto-detect mode to handle various tar formats
                        with tarfile.open(srcp, "r:*") as tf:
                            tf.extractall(pixie_dst)
                        report["actions"].append({"untarred": [srcp, pixie_dst]})
                    elif fname.lower().endswith(".zip"):
                        with zipfile.ZipFile(srcp, "r") as zf:
                            zf.extractall(pixie_dst)
                        report["actions"].append({"unzipped": [srcp, pixie_dst]})
                    else:
                        shutil.copy2(srcp, os.path.join(pixie_dst, fname))
                        report["actions"].append({"copied": [srcp, os.path.join(pixie_dst, fname)]})
        # LaMa assets (look for big-lama.pt or big-lama.zip)
        if lama_src:
            lama_dst = os.path.join(MODELS_DIR, "big_lama", "v1")
            os.makedirs(lama_dst, exist_ok=True)
            for fname in os.listdir(lama_src):
                srcp = os.path.join(lama_src, fname)
                if os.path.isfile(srcp):
                    if fname.lower() == "big-lama.pt":
                        shutil.copy2(srcp, os.path.join(lama_dst, fname))
                        report["actions"].append({"copied": [srcp, os.path.join(lama_dst, fname)]})
                    elif fname.lower() == "big-lama.zip":
                        with zipfile.ZipFile(srcp, "r") as zf:
                            zf.extractall(lama_dst)
                        # Normalize common layouts: ensure a convenience copy at v1/big-lama.pt if present inside a folder
                        for root, _dirs, files in os.walk(lama_dst):
                            if "big-lama.pt" in files:
                                try:
                                    shutil.copy2(os.path.join(root, "big-lama.pt"), os.path.join(lama_dst, "big-lama.pt"))
                                except Exception:
                                    pass
                                break
                        report["actions"].append({"unzipped": [srcp, lama_dst]})

        # SCHP code/weights (optional)
        schp_src = None
        for name in ("schp", "SCHP", "schp_downloads"):
            cand = os.path.join(manual_dir, name)
            if os.path.isdir(cand):
                schp_src = cand
                break
        if schp_src:
            tp_dst = os.path.join("third_party", "schp")
            os.makedirs(tp_dst, exist_ok=True)
            # Copy all files and unzip any repo zips into third_party/schp
            _copytree(schp_src, tp_dst)
            for fn in os.listdir(schp_src):
                srcp = os.path.join(schp_src, fn)
                if os.path.isfile(srcp) and fn.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(srcp, 'r') as zf:
                            zf.extractall(tp_dst)
                        report["actions"].append({"unzipped": [srcp, tp_dst]})
                    except Exception:
                        pass
            # Also place the LIP weights into the expected models dir
            lip_dst_dir = os.path.join(MODELS_DIR, 'schp_lip', '20190826')
            os.makedirs(lip_dst_dir, exist_ok=True)
            for fn in os.listdir(schp_src):
                if fn.endswith('.pth') and 'exp-schp-201908261155-lip' in fn:
                    try:
                        shutil.copy2(os.path.join(schp_src, fn), os.path.join(lip_dst_dir, fn))
                        # Ensure canonical name exists
                        dst_canon = os.path.join(lip_dst_dir, 'exp-schp-201908261155-lip.pth')
                        shutil.copy2(os.path.join(schp_src, fn), dst_canon)
                        report["actions"].append({"weights": [os.path.join(schp_src, fn), dst_canon]})
                    except Exception:
                        pass
            report["actions"].append({"third_party": [schp_src, tp_dst]})

        # StableVITON code/weights (optional)
        stv_src = None
        for name in ("stableviton", "StableVITON", "stable_viton", "stableviton_downloads"):
            cand = os.path.join(manual_dir, name)
            if os.path.isdir(cand):
                stv_src = cand
                break
        if stv_src:
            stv_tp = os.path.join("third_party", "stableviton")
            os.makedirs(stv_tp, exist_ok=True)
            _copytree(stv_src, stv_tp)
            # Unzip repo archives into third_party/stableviton if present
            for fn in os.listdir(stv_src):
                srcp = os.path.join(stv_src, fn)
                if os.path.isfile(srcp) and fn.lower().endswith('.zip'):
                    try:
                        with zipfile.ZipFile(srcp, 'r') as zf:
                            zf.extractall(stv_tp)
                        report["actions"].append({"unzipped": [srcp, stv_tp]})
                    except Exception:
                        pass
            report["actions"].append({"third_party": [stv_src, stv_tp]})
            # Copy any .ckpt weights found anywhere under the source to models/stableviton/weights
            stv_w_dst = os.path.join(MODELS_DIR, "stableviton", "weights")
            os.makedirs(stv_w_dst, exist_ok=True)
            copied = []
            for root, _dirs, files in os.walk(stv_src):
                for fn in files:
                    if fn.lower().endswith('.ckpt'):
                        srcp = os.path.join(root, fn)
                        dstp = os.path.join(stv_w_dst, fn)
                        try:
                            shutil.copy2(srcp, dstp)
                            copied.append((srcp, dstp))
                        except Exception:
                            pass
            if copied:
                report["actions"].append({"weights": copied})
    except Exception as e:  # pragma: no cover
        report["error"] = str(e)
    return report
