from __future__ import annotations

import os
import json
from typing import Optional

import requests


class RemoteError(Exception):
    pass


class RemoteFinisher:
    def __init__(self, url: Optional[str] = None) -> None:
        self.url = (url or os.environ.get("REMOTE_FINISHER_URL", "")).rstrip("/")
        if not self.url:
            raise RemoteError("REMOTE_FINISHER_URL not configured")

    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.18, controls: Optional[dict] = None, adapters: Optional[dict] = None) -> str:
        files = {
            "soft": (os.path.basename(soft_render_path), open(soft_render_path, "rb"), "image/png"),
        }
        data = {"denoise": denoise}
        if controls:
            # attach control files if present
            for k, v in controls.items():
                if v and os.path.exists(v):
                    files[f"control_{k}"] = (os.path.basename(v), open(v, "rb"), "image/png")
        r = requests.post(self.url + "/img2img", files=files, data=data, timeout=180)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        ext = ".png" if "png" in ctype else ".jpg"
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "remote_polished" + ext)
        with open(out, "wb") as f:
            f.write(r.content)
        return out


class RemoteVTON:
    def __init__(self, url: Optional[str] = None) -> None:
        self.url = (url or os.environ.get("REMOTE_VTON_URL", "")).rstrip("/")
        if not self.url:
            raise RemoteError("REMOTE_VTON_URL not configured")

    def process(self, user_image_path: str, warped_garment_path: str, mask_path: Optional[str], out_dir: str) -> str:
        files = {
            "user": (os.path.basename(user_image_path), open(user_image_path, "rb"), "image/jpeg"),
            "garment": (os.path.basename(warped_garment_path), open(warped_garment_path, "rb"), "image/png"),
        }
        if mask_path and os.path.exists(mask_path):
            files["mask"] = (os.path.basename(mask_path), open(mask_path, "rb"), "image/png")
        r = requests.post(self.url + "/vton", files=files, timeout=180)
        r.raise_for_status()
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "vton.png")
        with open(out, "wb") as f:
            f.write(r.content)
        return out

