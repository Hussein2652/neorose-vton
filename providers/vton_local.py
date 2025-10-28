from __future__ import annotations

from typing import Optional
from pipeline.vton_expert import apply_vton


class LocalVTON:
    def process(self, user_image_path: str, warped_garment_path: str, mask_path: Optional[str], out_dir: str) -> str:
        return apply_vton(user_image_path, warped_garment_path, mask_path, out_dir)

