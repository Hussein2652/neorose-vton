from __future__ import annotations

import os
from typing import Optional
from PIL import Image


class RealESRGANUpscaler:
    """Upscale using realesrgan if available; otherwise no-op."""
    def __init__(self, model: str = "x4plus") -> None:
        self._model_name = model
        self._ready = False
        try:
            import realesrgan  # type: ignore
            import torch  # type: ignore
            self._ready = True
        except Exception:
            self._ready = False

    def process(self, image_path: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        if not self._ready:
            # no-op
            out = os.path.join(out_dir, os.path.basename(image_path))
            Image.open(image_path).save(out)
            return out
        try:
            import torch
            from realesrgan import RealESRGAN  # type: ignore
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = RealESRGAN(device, scale=4)
            model.load_weights("RealESRGAN_x4plus.pth", download=True)  # downloads to default cache
            img = Image.open(image_path).convert('RGB')
            sr_img = model.predict(img)
            out = os.path.join(out_dir, "upscaled.png")
            sr_img.save(out)
            return out
        except Exception:
            out = os.path.join(out_dir, os.path.basename(image_path))
            Image.open(image_path).save(out)
            return out

