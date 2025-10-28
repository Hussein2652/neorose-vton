import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import Any, Callable, Optional


@dataclass
class Job:
    id: str
    func: Callable[..., Any]
    kwargs: dict[str, Any]
    status: str = "queued"  # queued | running | completed | failed
    error: Optional[str] = None
    result_path: Optional[str] = None


class Jobs:
    _jobs: dict[str, Job] = {}
    _queue: "Queue[Job]" = Queue()
    _worker: Optional[threading.Thread] = None
    _stop: bool = False

    @classmethod
    def enqueue(cls, func: Callable[..., Any], **kwargs: Any) -> str:
        # Allow caller to supply a stable job_id (e.g., to match DB row)
        provided_id = kwargs.pop("job_id", None)
        job_id = provided_id or str(uuid.uuid4())
        job = Job(id=job_id, func=func, kwargs=kwargs)
        cls._jobs[job_id] = job
        cls._queue.put(job)
        return job_id

    @classmethod
    def get(cls, job_id: str) -> Optional[Job]:
        return cls._jobs.get(job_id)

    @classmethod
    def _worker_loop(cls) -> None:
        while not cls._stop:
            try:
                job = cls._queue.get(timeout=0.5)
            except Empty:
                continue
            job.status = "running"
            # Persist status transition
            try:
                from .db import update_job
                from .cache import cache_set_job

                update_job(job.id, status="running")
                cache_set_job(job.id, status="running")
            except Exception:
                pass
            try:
                result = job.func(**job.kwargs)
                if isinstance(result, dict) and "result_path" in result:
                    job.result_path = result["result_path"]
                    job_result_url = result.get("result_url") if isinstance(result, dict) else None
                job.status = "completed"
                try:
                    from .db import update_job
                    from .cache import cache_set_job

                    update_job(job.id, status="completed", result_path=job.result_path, result_url=job_result_url)
                    cache_set_job(job.id, status="completed", result_path=job.result_path)
                except Exception:
                    pass
            except Exception as e:  # noqa: BLE001
                job.error = str(e)
                job.status = "failed"
                try:
                    from .db import update_job
                    from .cache import cache_set_job

                    update_job(job.id, status="failed", error=job.error)
                    cache_set_job(job.id, status="failed", error=job.error)
                except Exception:
                    pass
            finally:
                self_ack = getattr(job.func, "ack", None)
                if callable(self_ack):
                    try:
                        self_ack()
                    except Exception:
                        pass
                self = None  # GC hint

    @classmethod
    def ensure_worker(cls) -> None:
        if cls._worker and cls._worker.is_alive():
            return
        cls._stop = False
        cls._worker = threading.Thread(target=cls._worker_loop, daemon=True)
        cls._worker.start()

    @classmethod
    def shutdown(cls) -> None:
        cls._stop = True
        time.sleep(0.1)
