from __future__ import annotations

import os
from typing import Optional
from PIL import Image


class SDXLDiffusersFinisher:
    """
    Simple SDXL img2img finisher using diffusers. Requires diffusers + torch + transformers.
    Honors HF cache directories to persist models in storage/models.
    """

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or os.environ.get("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")
        # Persist cache under storage/models
        try:
            from backend.app.artifacts import MODELS_DIR
            os.environ.setdefault("HF_HOME", os.path.join(MODELS_DIR, "hf"))
            os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(MODELS_DIR, "hf", "hub"))
        except Exception:
            pass
        self._pipe = None
        try:
            from diffusers import AutoPipelineForImage2Image  # type: ignore
            import torch  # type: ignore
            self._pipe = AutoPipelineForImage2Image.from_pretrained(self.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
        except Exception:
            self._pipe = None

    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.18, controls: Optional[dict] = None, adapters: Optional[dict] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        im = Image.open(soft_render_path).convert("RGB")
        if not self._pipe:
            # Fallback: gentle sharpen
            im = im.filter(Image.Filter.SHARPEN)  # type: ignore[attr-defined]
            out = os.path.join(out_dir, "sdxl_fallback.png")
            im.save(out)
            return out
        try:
            from diffusers.utils import load_image  # type: ignore
            image = load_image(soft_render_path)
            prompt = os.environ.get("SDXL_PROMPT", "photo-realistic garment try-on, detailed fabric, natural lighting")
            negative = os.environ.get("SDXL_NEGATIVE", "low quality, blur, artifacts")
            strength = max(0.05, min(0.95, float(denoise)))
            out_im = self._pipe(prompt=prompt, negative_prompt=negative, image=image, strength=strength).images[0]
            out = os.path.join(out_dir, "sdxl_out.png")
            out_im.save(out)
            return out
        except Exception:
            out = os.path.join(out_dir, "sdxl_fallback.png")
            im.save(out)
            return out


class FluxDiffusersFinisher(SDXLDiffusersFinisher):
    def __init__(self, model_id: Optional[str] = None) -> None:
        super().__init__(model_id or os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev"))

