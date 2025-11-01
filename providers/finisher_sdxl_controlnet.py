from __future__ import annotations

import os
from typing import Optional, List
import time
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
        self.base_path = os.environ.get("SDXL_BASE_PATH")
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
        self.refiner_path = os.environ.get("SDXL_REFINER_PATH")
        self.cn_union_dir = os.environ.get("CONTROLNET_UNION_SDXL_DIR")
        self._pipe = None
        self._base = None
        self._refiner = None
        self._ready = False
        try:
            import torch  # type: ignore
            from diffusers import (
                ControlNetModel,
                StableDiffusionXLControlNetImg2ImgPipeline,
                StableDiffusionXLImg2ImgPipeline,
            )  # type: ignore
            from typing import Any

            # Cooperatively yield GPU to the VTON expert when reserved
            locks_dir = os.environ.get("LOCKS_DIR", os.path.join("storage", "locks"))
            gpu_lock_path = os.path.join(locks_dir, "gpu.lock")
            use_cuda = torch.cuda.is_available()
            if os.path.exists(gpu_lock_path):
                try:
                    max_age = int(os.environ.get("GPU_LOCK_STALE_SEC", "1800"))
                    st = os.stat(gpu_lock_path)
                    if (time.time() - st.st_mtime) <= max_age:
                        use_cuda = False
                except Exception:
                    use_cuda = False
            dtype = torch.float16 if use_cuda else torch.float32
            device = "cuda" if use_cuda else "cpu"

            # Prefer from_pretrained (snapshot) for full components; fall back to single-file if requested only
            prefer_pretrained = os.environ.get("FORCE_SDXL_PRETRAINED", "1") == "1"
            base_pipe = None
            if prefer_pretrained:
                try:
                    base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.base_id, torch_dtype=dtype, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE"))
                    )
                except Exception:
                    base_pipe = None
            if base_pipe is None:
                if self.base_path and os.path.exists(self.base_path):
                    base_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(self.base_path, torch_dtype=dtype)
                else:
                    base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.base_id, torch_dtype=dtype, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE"))
                    )

            # Load ControlNet: prefer offline union snapshot dir; else try configured list; else none
            controlnet = None
            # Prefer explicit CN IDs when provided; only use union snapshot when explicitly enabled
            use_union = os.environ.get("USE_CONTROLNET_UNION", "0") in ("1", "true", "yes")
            if use_union and self.cn_union_dir and os.path.isdir(self.cn_union_dir):
                try:
                    controlnet = ControlNetModel.from_pretrained(self.cn_union_dir, torch_dtype=dtype, local_files_only=True)
                except Exception:
                    controlnet = None
            if (controlnet is None) and self.cn_ids:
                cns = []
                for mid in self.cn_ids:
                    try:
                        cns.append(ControlNetModel.from_pretrained(mid, torch_dtype=dtype, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE"))))
                    except Exception:
                        pass
                if cns:
                    controlnet = cns if len(cns) > 1 else cns[0]

            # Build final pipeline
            if controlnet is not None:
                self._pipe = StableDiffusionXLControlNetImg2ImgPipeline(
                    vae=base_pipe.vae,
                    text_encoder=base_pipe.text_encoder,
                    text_encoder_2=base_pipe.text_encoder_2,
                    tokenizer=base_pipe.tokenizer,
                    tokenizer_2=base_pipe.tokenizer_2,
                    unet=base_pipe.unet,
                    scheduler=base_pipe.scheduler,
                    image_encoder=getattr(base_pipe, "image_encoder", None),
                    controlnet=controlnet,
                    feature_extractor=getattr(base_pipe, "feature_extractor", None),
                )
            else:
                self._pipe = base_pipe

            if use_cuda:
                self._pipe = self._pipe.to(device)
                try:
                    self._pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            try:
                self._pipe.enable_vae_slicing()
                self._pipe.enable_vae_tiling()
            except Exception:
                pass
            # Keep a base-only pipeline for fallback render if ControlNet path fails at runtime
            self._base = base_pipe.to(device) if use_cuda else base_pipe
            try:
                self._base.enable_vae_slicing(); self._base.enable_vae_tiling()
            except Exception:
                pass

            # Optionally load SDXL refiner
            try:
                if self.refiner_path and os.path.exists(self.refiner_path):
                    self._refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(self.refiner_path, torch_dtype=dtype)
                elif self.refiner_id:
                    # Use local snapshot if offline
                    self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                        self.refiner_id, torch_dtype=dtype, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE"))
                    )
                if self._refiner is not None and torch.cuda.is_available():
                    self._refiner = self._refiner.to(device)
            except Exception:
                self._refiner = None

            # Optional: Load IP-Adapter (image) for garment/detail conditioning
            # Prefer local snapshot dir for h94/IP-Adapter SDXL weights
            try:
                ip_dir = os.environ.get("IP_ADAPTER_SDXL_DIR", os.path.join(os.environ.get('MODELS_DIR', 'storage/models'), 'snapshots', 'h94-IP-Adapter', 'sdxl_models'))
                ip_weights = os.environ.get("IP_ADAPTER_SDXL_WEIGHTS_NAME", "ip-adapter_sdxl.safetensors")
                weight_path = os.path.join(ip_dir, ip_weights)
                if os.path.exists(weight_path) and hasattr(self._pipe, 'load_ip_adapter'):
                    # Newer diffusers API
                    self._pipe.load_ip_adapter(ip_dir, weight_name=ip_weights)
                    # Safety: set default scale from env later during call
                    self._has_ip_adapter = True  # type: ignore[attr-defined]
                else:
                    self._has_ip_adapter = False  # type: ignore[attr-defined]
            except Exception:
                self._has_ip_adapter = False  # type: ignore[attr-defined]

            # Optional: Load FaceID PlusV2 SDXL + LoRA (offline)
            self._has_faceid_adapter = False  # type: ignore[attr-defined]
            try:
                face_dir = os.environ.get("IP_ADAPTER_FACEID_DIR", os.path.join(os.environ.get('MODELS_DIR', 'storage/models'), 'snapshots', 'h94-IP-Adapter-FaceID'))
                face_bin = os.environ.get("IP_ADAPTER_FACEID_BIN", "ip-adapter-faceid-plusv2_sdxl.bin")
                face_lora = os.environ.get("IP_ADAPTER_FACEID_LORA", "ip-adapter-faceid-plusv2_sdxl_lora.safetensors")
                bin_path = os.path.join(face_dir, face_bin)
                lora_path = os.path.join(face_dir, face_lora)
                if hasattr(self._pipe, 'load_ip_adapter') and os.path.exists(bin_path):
                    try:
                        self._pipe.load_ip_adapter(face_dir, weight_name=face_bin)
                        self._has_faceid_adapter = True
                    except Exception:
                        pass
                if hasattr(self._pipe, 'load_lora_weights') and os.path.exists(lora_path):
                    try:
                        self._pipe.load_lora_weights(face_dir, weight_name=face_lora)
                    except Exception:
                        pass
            except Exception:
                self._has_faceid_adapter = False  # type: ignore[attr-defined]

            # Load refiner from single safetensors if provided
            try:
                if self.refiner_path and os.path.exists(self.refiner_path):
                    self._refiner = StableDiffusionXLImg2ImgPipeline.from_single_file(self.refiner_path, torch_dtype=dtype)
                    if use_cuda:
                        self._refiner = self._refiner.to(device)
                else:
                    self._refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.refiner_id, torch_dtype=dtype, local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")))
                    if use_cuda:
                        self._refiner = self._refiner.to(device)
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
        # Optional pre-upscale to target long side for more details
        try:
            tgt_long = int(os.environ.get("FINISHER_RESIZE_LONG", "1280"))
        except Exception:
            tgt_long = 1280
        if max(image.size) < tgt_long:
            w, h = image.size
            if w >= h:
                nw, nh = tgt_long, int(h * (tgt_long / float(w)))
            else:
                nh, nw = tgt_long, int(w * (tgt_long / float(h)))
            image = image.resize((nw, nh), Image.LANCZOS)

        prompt = os.environ.get(
            "SDXL_PROMPT",
            "ultra-detailed, photorealistic apparel try-on, crisp fabric texture, natural lighting, sharp focus, high resolution",
        )
        negative = os.environ.get(
            "SDXL_NEGATIVE",
            "blurry, low-resolution, out of focus, artifacts, deformed body, watermark, oversmooth, noise",
        )
        if not self._ready:
            out = os.path.join(out_dir, "cnx_fallback.png")
            image.save(out)
            return out
        try:
            ip_adapter_images: List[Image.Image] = []
            if adapters:
                # Prefer garment image for detail conditioning
                g_path = adapters.get('garment_image') or adapters.get('garment')
                if g_path and os.path.exists(g_path):
                    try:
                        ip_adapter_images.append(Image.open(g_path).convert('RGB'))
                    except Exception:
                        pass
                # Face image for identity (FaceID)
                f_path = adapters.get('face_image') or adapters.get('user')
                if f_path and os.path.exists(f_path):
                    try:
                        ip_adapter_images.append(Image.open(f_path).convert('RGB'))
                    except Exception:
                        pass
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
            control_scales: List[float] = []
            for cn_id in self.cn_ids:
                sel: Optional[Image.Image] = None
                lid = cn_id.lower()
                scale_val = float(os.environ.get("CTRL_DEFAULT", "0.6"))
                for cname, keys in cn_map_keys:
                    if cname in lid:
                        # Map per-control default scales
                        if cname == 'openpose':
                            scale_val = float(os.environ.get('CTRL_POSE', '0.9'))
                        elif cname == 'depth':
                            scale_val = float(os.environ.get('CTRL_DEPTH', '0.6'))
                        elif cname == 'normal':
                            scale_val = float(os.environ.get('CTRL_NORMAL', '0.5'))
                        elif cname == 'seg':
                            scale_val = float(os.environ.get('CTRL_SEG', '0.35'))
                        elif cname == 'canny':
                            scale_val = float(os.environ.get('CTRL_EDGE', '0.4'))
                        for k in keys:
                            p = controls.get(k)
                            if p and os.path.exists(p):
                                sel = Image.open(p).convert("RGB")
                                break
                    if sel:
                        break
                # fallback: just feed the base image if no mapped control
                control_images.append(sel if sel is not None else image)
                control_scales.append(scale_val)

            # Run controlnet img2img
            strength = max(0.05, min(0.95, float(denoise)))
            call_kwargs: dict = dict(
                prompt=prompt,
                negative_prompt=negative,
                image=image,
                control_image=control_images if control_images else None,
                strength=strength,
                num_inference_steps=int(os.environ.get("SDXL_STEPS", "30")),
                guidance_scale=float(os.environ.get("SDXL_GUIDANCE", "6.0")),
            )
            if control_images:
                call_kwargs["controlnet_conditioning_scale"] = control_scales
            if os.environ.get("SDXL_DEBUG", "0") == "1":
                try:
                    import sys
                    print(f"[SDXL finisher] controls={len(control_images)} scales={control_scales} strength={strength}", file=sys.stderr)
                except Exception:
                    pass
            # If IP-Adapter(s) loaded and we have images, pass them (list supports multi-adapter in newer APIs)
            try:
                if (getattr(self, '_has_ip_adapter', False) or getattr(self, '_has_faceid_adapter', False)) and ip_adapter_images:
                    call_kwargs['ip_adapter_image'] = ip_adapter_images if len(ip_adapter_images) > 1 else ip_adapter_images[0]
                    g_scale = float(os.environ.get('IP_ADAPTER_GARMENT_SCALE', '1.0'))
                    f_scale = float(os.environ.get('IP_ADAPTER_FACE_SCALE', '0.85'))
                    scales: List[float] = []
                    if getattr(self, '_has_ip_adapter', False) and ip_adapter_images:
                        scales.append(g_scale)
                    if getattr(self, '_has_faceid_adapter', False) and len(ip_adapter_images) > 1:
                        scales.append(f_scale)
                    if hasattr(self._pipe, 'set_ip_adapter_scale') and scales:
                        self._pipe.set_ip_adapter_scale(scales if len(scales) > 1 else scales[0])
            except Exception:
                pass
            try:
                result = self._pipe(**call_kwargs).images[0]
            except Exception as e:
                # Fallback: try base-only img2img without ControlNet
                try:
                    print(f"[SDXL finisher] controlnet path failed: {e}", file=sys.stderr)
                except Exception:
                    pass
                result = self._base(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=image,
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

            # Optional: Preserve face/outside regions from the original to keep identity sharp
            try:
                preserve = os.environ.get("SDXL_BLEND_PRESERVE_OUTSIDE", "1") == "1"
                mask_path = (controls or {}).get("seg") if controls else None
                user_path = (adapters or {}).get("user") if adapters else None
                if preserve and mask_path and user_path and os.path.exists(mask_path) and os.path.exists(user_path):
                    base_im = Image.open(user_path).convert("RGB").resize(result.size, Image.LANCZOS)
                    m = Image.open(mask_path).convert("L").resize(result.size, Image.BILINEAR)
                    # Compute head preserve region: top portion of person mask
                    import numpy as np
                    arr = np.array(m, dtype=np.uint8)
                    h = arr.shape[0]
                    head = np.zeros_like(arr)
                    head[: int(h * 0.32), :] = 255
                    preserve_mask = Image.fromarray(((arr > 127) & (head > 0)).astype("uint8") * 255, mode="L")
                    ksz = int(os.environ.get("SDXL_BLEND_PRESERVE_KERNEL", "21"))
                    preserve_mask = preserve_mask.filter(ImageFilter.GaussianBlur(radius=max(3, ksz // 2)))
                    result = Image.composite(base_im, result, preserve_mask)
            except Exception:
                pass
            out = os.path.join(out_dir, "sdxl_cnx.png")
            result.save(out)
            return out
        except Exception as e:
            # Last-resort fallback: return input and log error
            try:
                import sys
                print(f"[SDXL finisher] fatal error: {e}", file=sys.stderr)
            except Exception:
                pass
            out = os.path.join(out_dir, "cnx_fallback.png")
            image.save(out)
            return out
