from PIL import Image
import os

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
    Person segmentation using OpenCV GrabCut when available; otherwise full-white mask.
    """
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(user_image_path).convert("RGB")
    mask = _grabcut_mask(user_image_path) or Image.new("L", im.size, color=255)
    mask_path = os.path.join(out_dir, "user_mask.png")
    mask.save(mask_path)
    return mask_path
