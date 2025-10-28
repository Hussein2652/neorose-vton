from PIL import Image
import os


def segment_person(user_image_path: str, out_dir: str) -> str:
    """
    Stub person segmentation.
    - Produces a trivial full-white mask (same size) as a placeholder.
    """
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(user_image_path).convert("RGB")
    mask = Image.new("L", im.size, color=255)
    mask_path = os.path.join(out_dir, "user_mask.png")
    mask.save(mask_path)
    return mask_path

