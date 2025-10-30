from PIL import Image
import os
from typing import Optional
from .person_parsing_mediapipe import segment_person_mediapipe
from providers.segmentation_schp import SCHPSegmenter

def _grabcut_mask(user_image_path: str) -> Image.Image | None:
    try:
        import cv2  # type: ignore
        import numpy as np
    except Exception:
        return None
    img = cv2.imread(user_image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    # Initialize mask and rect covering central tall region
    mask = np.zeros((h, w), np.uint8)
    rect_w, rect_h = int(w * 0.6), int(h * 0.8)
    x = (w - rect_w) // 2
    y = int(h * 0.1)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x, y, rect_w, rect_h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        return Image.fromarray(mask2, mode='L')
    except Exception:
        return None


def segment_person(user_image_path: str, out_dir: str) -> str:
    """
    Person segmentation (Tier-S path prefers accurate masks):
    1) Try MediaPipe selfie segmentation (fast, good for portrait)
    2) Fallback to OpenCV GrabCut
    3) Fallback to full-white mask
    """
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(user_image_path).convert("RGB")
    # Try SCHP first if repo/weights are present
    try:
        if os.path.isdir(os.path.join("third_party", "schp")) and os.path.exists(
            os.environ.get(
                "SCHP_MODEL_PATH",
                os.path.join(
                    os.environ.get("MODELS_DIR", "storage/models"),
                    "schp_lip",
                    "20190826",
                    "exp-schp-201908261155-lip.pth",
                ),
            )
        ):
            schp = SCHPSegmenter()
            m = schp.process(user_image_path, out_dir)
            if m and os.path.exists(m):
                return m
    except Exception:
        pass
    mask_path: Optional[str] = segment_person_mediapipe(user_image_path, out_dir)
    if not mask_path:
        mask_img = _grabcut_mask(user_image_path) or Image.new("L", im.size, color=255)
        mask_path = os.path.join(out_dir, "user_mask.png")
        mask_img.save(mask_path)
    return mask_path
