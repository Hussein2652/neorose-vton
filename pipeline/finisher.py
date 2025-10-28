from PIL import Image, ImageFilter
import os


def img2img_polish(soft_render_path: str, out_dir: str, denoise: float = 0.18) -> str:
    """
    Stub finisher (img2img).
    - Applies a mild smoothing filter to emulate denoise.
    """
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(soft_render_path).convert("RGB")
    strength = max(1, int(denoise * 10))
    im = im.filter(ImageFilter.GaussianBlur(radius=strength))
    out = os.path.join(out_dir, "polished.png")
    im.save(out)
    return out

