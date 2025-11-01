from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager


LOCKS_DIR = os.environ.get("LOCKS_DIR", "/app/storage/locks")
os.makedirs(LOCKS_DIR, exist_ok=True)
GPU_LOCK_PATH = os.path.join(LOCKS_DIR, "gpu.lock")


def _write_lock(owner: str) -> None:
    os.makedirs(os.path.dirname(GPU_LOCK_PATH), exist_ok=True)
    data = {"owner": owner, "pid": os.getpid(), "ts": time.time()}
    with open(GPU_LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _try_acquire(owner: str) -> bool:
    try:
        # O_EXCL ensures exclusive create
        fd = os.open(GPU_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        _write_lock(owner)
        return True
    except FileExistsError:
        return False


def _release_if_owner(owner: str) -> None:
    try:
        with open(GPU_LOCK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("owner") == owner:
            os.remove(GPU_LOCK_PATH)
    except FileNotFoundError:
        pass
    except Exception:
        # Best effort: try unlink anyway
        try:
            os.remove(GPU_LOCK_PATH)
        except Exception:
            pass


def is_reserved() -> bool:
    return os.path.exists(GPU_LOCK_PATH)


@contextmanager
def gpu_reservation(owner: str, wait_timeout: float = 600.0, poll: float = 0.25):
    """Cooperative GPU reservation across containers via a shared lock file.
    Blocks until lock can be acquired or timeout elapses. Always releases on exit.
    """
    start = time.time()
    acquired = _try_acquire(owner)
    while not acquired and (time.time() - start) < wait_timeout:
        time.sleep(poll)
        acquired = _try_acquire(owner)
    try:
        yield acquired
    finally:
        if acquired:
            _release_if_owner(owner)

