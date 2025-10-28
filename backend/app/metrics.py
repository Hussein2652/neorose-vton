from __future__ import annotations

from prometheus_client import Counter, Gauge

jobs_created = Counter("vfr_jobs_created_total", "Total try-on jobs submitted")
jobs_completed = Counter("vfr_jobs_completed_total", "Total try-on jobs completed")
jobs_failed = Counter("vfr_jobs_failed_total", "Total try-on jobs failed")
jobs_in_queue = Gauge("vfr_jobs_in_queue", "Jobs currently queued or running")

