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

