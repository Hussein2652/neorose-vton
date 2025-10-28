from __future__ import annotations

import os
from typing import Optional, List
from PIL import Image


class SDXLControlNetFinisher:
    """
    SDXL + Multi-ControlNet img2img finisher with optional Refiner pass.
    Uses diffusers. Control images are expected to be precomputed and passed in via `controls` dict.
    Downloads models once into HF cache (persisted under storage/models/hf via env set by other providers).
    """

    def __init__(self) -> None:
        # Model IDs (override via env)
        self.base_id = os.environ.get("SDXL_BASE_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        # SDXL ControlNet model IDs (canny/pose/depth/normal/seg)
        self.cn_ids: List[str] = []
        # Allow comma-separated list in env, else use useful defaults
        cn_env = os.environ.get("SDXL_CN_IDS")
        if cn_env:
            self.cn_ids = [s.strip() for s in cn_env.split(",") if s.strip()]
        else:
            # Defaults may require availability; adjust as needed
            self.cn_ids = [
                os.environ.get("SDXL_CN_CANNY", "diffusers/controlnet-canny-sdxl-1.0"),
                os.environ.get("SDXL_CN_POSE", "diffusers/controlnet-openpose-sdxl-1.0"),
                os.environ.get("SDXL_CN_DEPTH", "diffusers/controlnet-depth-sdxl-1.0"),
                os.environ.get("SDXL_CN_NORMAL", "diffusers/controlnet-normal-sdxl-1.0"),
                os.environ.get("SDXL_CN_SEG", "diffusers/controlnet-seg-sdxl-1.0"),
            ]
        self.refiner_id = os.environ.get("SDXL_REFINER_ID", "stabilityai/stable-diffusion-xl-refiner-1.0")
        self._pipe = None
        self._refiner = None
        self._ready = False
        try:
            import torch  # type: ignore
            from diffusers import (
                AutoencoderKL,
                ControlNetModel,
                StableDiffusionXLControlNetImg2ImgPipeline,
                StableDiffusionXLImg2ImgPipeline,
                DDIMScheduler,
            )  # type: ignore
            # Load multiple controlnets
            cns = []
            for mid in self.cn_ids:
                try:
                    cns.append(ControlNetModel.from_pretrained(mid, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32))
                except Exception:
                    pass
            if not cns:
                # If none loaded, fallback to base SDXL img2img pipeline
                self._pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.base_id)
            else:
                self._pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    self.base_id, controlnet=cns
                )
            if torch.cuda.is_available():
                self._pipe = self._pipe.to("cuda")
            try:
                self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.refiner_id)
                if torch.cuda.is_available():
                    self._refiner = self._refiner.to("cuda")
            except Exception:
                self._refiner = None
            self._ready = True
        except Exception:
            self._pipe = None
            self._refiner = None
            self._ready = False

    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.20, controls: Optional[dict] = None, adapters: Optional[dict] = None) -> str:
        os.makedirs(out_dir, exist_ok=True)
        image = Image.open(soft_render_path).convert("RGB")
        prompt = os.environ.get("SDXL_PROMPT", "photo-realistic garment try-on, detailed fabric, natural lighting")
        negative = os.environ.get("SDXL_NEGATIVE", "low quality, blur, artifacts, watermark")
        if not self._ready:
            out = os.path.join(out_dir, "cnx_fallback.png")
            image.save(out)
            return out
        try:
            # Build list of control images aligned with loaded ControlNets order
            control_images: List[Image.Image] = []
            cn_map_keys = [
                ("canny", ["edge", "canny"]),
                ("openpose", ["pose"]),
                ("depth", ["depth"]),
                ("normal", ["normal"]),
                ("seg", ["seg", "mask"]),
            ]
            if controls is None:
                controls = {}
            # Attempt to map control images to cns by basic name matching
            for cn_id in self.cn_ids:
                sel: Optional[Image.Image] = None
                lid = cn_id.lower()
                for cname, keys in cn_map_keys:
                    if cname in lid:
                        for k in keys:
                            p = controls.get(k)
                            if p and os.path.exists(p):
                                sel = Image.open(p).convert("RGB")
                                break
                    if sel:
                        break
                # fallback: just feed the base image if no mapped control
                control_images.append(sel if sel is not None else image)

            # Run controlnet img2img
            strength = max(0.05, min(0.95, float(denoise)))
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=image,
                control_image=control_images if control_images else None,
                strength=strength,
                num_inference_steps=int(os.environ.get("SDXL_STEPS", "30")),
                guidance_scale=float(os.environ.get("SDXL_GUIDANCE", "6.0")),
            ).images[0]
            # Optional refiner pass
            if self._refiner is not None:
                ref_strength = float(os.environ.get("SDXL_REFINER_STRENGTH", "0.15"))
                result = self._refiner(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=result,
                    strength=ref_strength,
                    num_inference_steps=int(os.environ.get("SDXL_REFINER_STEPS", "20")),
                    guidance_scale=float(os.environ.get("SDXL_REFINER_GUIDANCE", "5.0")),
                ).images[0]
            out = os.path.join(out_dir, "sdxl_cnx.png")
            result.save(out)
            return out
        except Exception:
            out = os.path.join(out_dir, "cnx_fallback.png")
            image.save(out)
            return out

