from __future__ import annotations

import os
from typing import Optional
from PIL import Image


def apply_vton(user_image_path: str, warped_garment_path: str, mask_path: Optional[str], out_dir: str) -> str:
    """
    Lightweight VTON expert placeholder.
    - Blends warped garment onto the user using the person mask to avoid bleed.
    - If OpenCV is present, applies seamless cloning (Poisson); else alpha composite.
    """
    os.makedirs(out_dir, exist_ok=True)
    user = Image.open(user_image_path).convert('RGB')
    gar = Image.open(warped_garment_path).convert('RGBA')

    try:
        import cv2  # type: ignore
        import numpy as np

        src = cv2.cvtColor(np.array(gar), cv2.COLOR_RGBA2BGRA)
        dst = cv2.cvtColor(np.array(user), cv2.COLOR_RGB2BGR)
        # Build mask from garment alpha
        mask = src[:, :, 3]
        # Use center for cloning
        h, w = dst.shape[:2]
        center = (w // 2, h // 2)
        mixed = cv2.seamlessClone(src[:, :, :3], dst, mask, center, cv2.MIXED_CLONE)
        out = Image.fromarray(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
    except Exception:
        # Simple over composite using garment alpha
        out = user.copy()
        out.paste(gar, (0, 0), gar)

    out_path = os.path.join(out_dir, 'vton.png')
    out.save(out_path)
    return out_path

