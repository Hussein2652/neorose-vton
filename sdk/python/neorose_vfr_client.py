import io
import os
import requests
from typing import Optional


class VFRClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def try_on(
        self,
        user_image_path: str,
        garment_front_path: str,
        garment_side_path: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url}/v1/jobs/tryon"
        files = {
            "user_image": (os.path.basename(user_image_path), open(user_image_path, "rb"), "image/jpeg"),
            "garment_front": (os.path.basename(garment_front_path), open(garment_front_path, "rb"), "image/jpeg"),
        }
        if garment_side_path:
            files["garment_side"] = (
                os.path.basename(garment_side_path),
                open(garment_side_path, "rb"),
                "image/jpeg",
            )
        r = requests.post(url, files=files, timeout=60)
        r.raise_for_status()
        return r.json()["job_id"]

    def job_status(self, job_id: str) -> dict:
        r = requests.get(f"{self.base_url}/v1/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        return r.json()

    def download_result(self, job_id: str) -> bytes:
        r = requests.get(f"{self.base_url}/v1/jobs/{job_id}/result", timeout=60)
        r.raise_for_status()
        return r.content

    def wait_for_result(self, job_id: str, timeout_s: int = 120, interval_s: float = 1.0) -> dict:
        import time
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            last = self.job_status(job_id)
            if last.get("status") in ("completed", "failed"):
                return last
            time.sleep(interval_s)
        return last or {"status": "timeout", "job_id": job_id}

    def download_result_to(self, job_id: str, out_path: str) -> str:
        data = self.download_result(job_id)
        import os
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        return out_path
