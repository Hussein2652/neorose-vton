from __future__ import annotations

from typing import Optional
import os
from pipeline.vton_expert import apply_vton
from .vton_stable import StableVITONClient


class LocalVTON:
    def process(self, user_image_path: str, garment_front_path: str, mask_path: Optional[str], out_dir: str) -> str:
        """Call the StableVITON expert with the ORIGINAL garment image.
        Do not pass a pre-warped composite as garment.
        """
        try:
            client = StableVITONClient()
            res = client.process(user_image_path, garment_front_path, out_dir, user_mask_path=mask_path)
            if res:
                return res
        except Exception:
            if os.environ.get("VTON_EXPERT_REQUIRED", "0") == "1":
                raise
        # Fallback: if expert is unavailable, try to use a soft render if present; else composite the garment
        try:
            # Typical warp output path in our pipeline work dir
            cand = os.path.join(os.path.dirname(out_dir), "warp", "soft_render.png")
            g = cand if os.path.exists(cand) else garment_front_path
        except Exception:
            g = garment_front_path
        return apply_vton(user_image_path, g, mask_path, out_dir)
