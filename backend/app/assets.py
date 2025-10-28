from __future__ import annotations

import hashlib
import os
from typing import Tuple

from .storage import CACHE_DIR


def file_sha256(path: str, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def user_cache_dir(user_image_path: str) -> Tuple[str, str]:
    h = file_sha256(user_image_path)
    d = os.path.join(CACHE_DIR, 'users', h[:2], h)
    os.makedirs(d, exist_ok=True)
    return d, h


def garment_cache_dir(garment_image_path: str) -> Tuple[str, str]:
    h = file_sha256(garment_image_path)
    d = os.path.join(CACHE_DIR, 'garments', h[:2], h)
    os.makedirs(d, exist_ok=True)
    return d, h

