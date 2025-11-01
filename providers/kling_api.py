from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image, ImageEnhance, ImageFilter


class KlingAPIError(Exception):
    pass


class KlingFinisher:
    """
    Production-oriented finisher using Kling API.
    - Uses multipart/form-data to send the soft render image and JSON params.
    - Accepts either binary image response or JSON with base64 image.
    - Falls back to a local enhancement pipeline if API env is missing or the call fails.
    """

    def __init__(self) -> None:
        self.api_key = os.environ.get("KLING_API_KEY")
        self.endpoint = os.environ.get("KLING_API_ENDPOINT", "")
        self.timeout = float(os.environ.get("KLING_TIMEOUT", "60"))

    def _fallback_local(self, soft_render_path: str, out_dir: str, denoise: float) -> str:
        os.makedirs(out_dir, exist_ok=True)
        im = Image.open(soft_render_path).convert("RGB")
        # Simulated premium pass: detail, bloom, contrast, color
        # PIL uses uppercase constants for builtin filters
        im = im.filter(ImageFilter.DETAIL)
        im = im.filter(ImageFilter.GaussianBlur(radius=max(1, int(denoise * 6))))
        im = ImageEnhance.Contrast(im).enhance(1.10)
        im = ImageEnhance.Color(im).enhance(1.05)
        out_path = os.path.join(out_dir, "kling_polished.png")
        im.save(out_path)
        return out_path

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.RequestException, KlingAPIError)),
    )
    def _call_api(self, soft_render_path: str, params: dict) -> tuple[bytes, str]:
        if not self.api_key or not self.endpoint:
            raise KlingAPIError("Kling API not configured")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {
            "image": (
                os.path.basename(soft_render_path),
                open(soft_render_path, "rb"),
                mimetypes.guess_type(soft_render_path)[0] or "image/png",
            ),
            "params": (None, json.dumps(params), "application/json"),
        }
        resp = requests.post(self.endpoint, headers=headers, files=files, timeout=self.timeout)
        if resp.status_code >= 400:
            raise KlingAPIError(f"Kling API error: {resp.status_code} {resp.text[:200]}")
        ctype = resp.headers.get("content-type", "")
        if ctype.startswith("image/"):
            return resp.content, ctype
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise KlingAPIError("Unexpected response type from Kling API") from e
        # Expected JSON may include base64 image or URL; handle base64 for offline safety
        if "image_base64" in data:
            return base64.b64decode(data["image_base64"]), "image/png"
        raise KlingAPIError("No image in Kling API response")

    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.18, controls: Optional[dict] = None, adapters: Optional[dict] = None) -> str:
        # Params aligned loosely with the PDF config
        params = {
            "mode": "img2img",
            "denoise": denoise,
            "steps": 38,
            "cfg": 6.2,
            "res_long": 1344,
        }
        os.makedirs(out_dir, exist_ok=True)
        try:
            data, ctype = self._call_api(soft_render_path, params)
            ext = ".png" if "png" in ctype else ".jpg"
            out_path = os.path.join(out_dir, f"kling_polished{ext}")
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        except Exception:
            # Fallback locally if API not configured or failed
            return self._fallback_local(soft_render_path, out_dir, denoise)
