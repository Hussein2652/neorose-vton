from __future__ import annotations

import os
import requests
from typing import Optional


class StableVITONClient:
    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.environ.get("VTON_EXPERT_URL", "http://vton_expert:9010")

    def process(self, user_image_path: str, garment_image_path: str, out_dir: str) -> Optional[str]:
        os.makedirs(out_dir, exist_ok=True)
        try:
            with open(user_image_path, 'rb') as fu, open(garment_image_path, 'rb') as fg:
                resp = requests.post(
                    f"{self.url}/infer",
                    files={
                        'user': (os.path.basename(user_image_path), fu, 'image/jpeg'),
                        'garment': (os.path.basename(garment_image_path), fg, 'image/png'),
                    },
                    timeout=120,
                )
            if resp.status_code != 200:
                return None
            out_path = os.path.join(out_dir, 'vton.png')
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            return out_path
        except Exception:
            return None

