from PIL import Image, ImageFilter, ImageOps
import os


def post_process(polished_path: str, out_dir: str) -> str:
    """
    Stub post-processing.
    - Applies a subtle sharpen and auto-contrast to emulate a finisher stack.
    """
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(polished_path).convert("RGB")
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    out = os.path.join(out_dir, "final.jpg")
    im.save(out, quality=92)
    return out

