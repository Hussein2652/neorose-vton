from __future__ import annotations

import os
from typing import Optional

from PIL import Image


def segment_person_mediapipe(user_image_path: str, out_dir: str, threshold: float = 0.2) -> Optional[str]:
    try:
        import mediapipe as mp  # type: ignore
        import numpy as np
    except Exception:
        return None
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(user_image_path).convert("RGB")
    arr = np.array(im)

    mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    res = mp_selfie.process(arr)
    mp_selfie.close()
    if res is None or res.segmentation_mask is None:
        return None
    mask = (res.segmentation_mask >= threshold).astype("uint8") * 255
    out_path = os.path.join(out_dir, "user_mask.png")
    Image.fromarray(mask, mode="L").save(out_path)
    return out_path

