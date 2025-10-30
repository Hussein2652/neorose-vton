from __future__ import annotations

from typing import Optional
from pipeline.vton_expert import apply_vton
from .vton_stable import StableVITONClient


class LocalVTON:
    def process(self, user_image_path: str, warped_garment_path: str, mask_path: Optional[str], out_dir: str) -> str:
        # Try expert StableVITON if available
        try:
            client = StableVITONClient()
            res = client.process(user_image_path, warped_garment_path, out_dir, user_mask_path=mask_path)
            if res:
                return res
        except Exception:
            pass
        # Fallback to local basic compositor
        return apply_vton(user_image_path, warped_garment_path, mask_path, out_dir)
