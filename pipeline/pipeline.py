import os
from dataclasses import dataclass
from typing import Optional

from backend.app.storage import RESULTS_DIR
from backend.app.storage import CACHE_DIR
from backend.app.assets import user_cache_dir, garment_cache_dir
from backend.app.db import get_asset_cache, set_asset_cache
from .io_types import UserCanonical, GarmentAssets, DrapedOutput
from providers.local_stub import (
    LocalPersonParser,
    LocalPoseExtractor,
    LocalGarmentWarper,
    LocalGeometryFitter,
    LocalFinisher,
    LocalPostProcessor,
    LocalQA,
)
from providers.kling_api import KlingFinisher
from .controls import build_controls
from .vton_expert import apply_vton
from providers.vton_local import LocalVTON
from providers.stable_vton_stub import StableVTON
from providers.remote import RemoteFinisher, RemoteVTON, RemoteError
from providers.finisher_diffusers import SDXLDiffusersFinisher, FluxDiffusersFinisher
from providers.finisher_sdxl_controlnet import SDXLControlNetFinisher
from providers.upscale_esrgan import RealESRGANUpscaler
from providers.face_codeformer import CodeFormerRestorer
from providers.segmentation_torchvision import TorchVisionPersonSegmenter
from providers.pose_ultralytics import YOLOv8PoseExtractor
from providers.geometry_smplx_stub import SMPLXGeometry


@dataclass
class VFRRunResult:
    output_path: str


class VFRPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Build providers
        # Segmentation provider
        seg = str(self.cfg.get("providers.segmentation", "local")).lower()
        if seg == "torchvision":
            self.person = TorchVisionPersonSegmenter()
        else:
            self.person = LocalPersonParser()

        # Pose provider
        posep = str(self.cfg.get("providers.pose", "mediapipe")).lower()
        if posep == "yolov8":
            self.pose = YOLOv8PoseExtractor()
        else:
            self.pose = LocalPoseExtractor()

        self.warper = LocalGarmentWarper()

        # Geometry provider
        geom = str(self.cfg.get("providers.geometry", "stub")).lower()
        if geom == "smplx":
            self.geometry = SMPLXGeometry()
        else:
            self.geometry = LocalGeometryFitter()
        backend = self.cfg.get("finisher_backend", "local")
        if backend == "kling":
            self.finisher = KlingFinisher()
        elif backend == "remote":
            try:
                self.finisher = RemoteFinisher()
            except Exception:
                self.finisher = LocalFinisher()
        elif backend == "sdxl":
            self.finisher = SDXLDiffusersFinisher()
        elif backend == "sdxl_controlnet":
            self.finisher = SDXLControlNetFinisher()
        elif backend == "flux":
            self.finisher = FluxDiffusersFinisher()
        else:
            self.finisher = LocalFinisher()
        self.post = LocalPostProcessor()
        self.qa = LocalQA()
        vton_provider = self.cfg.get("vton_provider", "local")
        if vton_provider == "stableviton":
            self.vton = StableVTON()
        elif vton_provider == "remote":
            try:
                self.vton = RemoteVTON()
            except Exception:
                self.vton = LocalVTON()
        else:
            self.vton = LocalVTON()

        # Optional post providers
        self.upscaler = RealESRGANUpscaler() if str(self.cfg.get("post.upscaler", "")).lower() == "realesrgan" else None
        self.face_restorer = CodeFormerRestorer() if bool(self.cfg.get("post.face_restore", False)) else None

    @classmethod
    def from_settings(cls, settings, overrides: Optional[dict] = None) -> "VFRPipeline":
        # Settings is the backend.app.config.settings instance
        cfg = {
            "img2img_denoise": float(settings.get("finisher.denoise", 0.18)),
            "finisher_backend": str(settings.get("finisher.backend", "local")),
            "qa_thresholds": {
                "clip_min": float(settings.get("qa_thresholds.garment_fidelity_clip_min", 0.28)),
                "lpips_max": float(settings.get("qa_thresholds.garment_fidelity_lpips_max", 0.32)),
                "aesthetic_min": float(settings.get("qa_thresholds.aesthetics_min", 0.58)),
                "identity_min": float(settings.get("qa_thresholds.identity_min", 0.75)),
                "iou_min": float(settings.get("qa_thresholds.iou_garment_skin_min", 0.90)),
                "max_retries": int(settings.get("qa_thresholds.retry.max_retries", 1)),
                "denoise_delta": float(settings.get("qa_thresholds.retry.denoise_delta", -0.02)),
            },
            "escalate_on_fail": bool(settings.get("finisher.escalate_on_fail", False)),
            "vton_enabled": bool(settings.get("vton.enabled", True)),
            "vton_provider": str(settings.get("vton.provider", "local")),
        }
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    cfg[k] = v
        # Feature flag overrides (best-effort)
        try:
            from backend.app.db import get_feature_flag  # type: ignore

            fb = get_feature_flag("finisher_backend")
            if fb:
                cfg["finisher_backend"] = fb
            vton_flag = get_feature_flag("vton_enabled")
            if vton_flag is not None:
                cfg["vton_enabled"] = vton_flag.lower() in ("1", "true", "yes")
            esc_flag = get_feature_flag("escalate_on_fail")
            if esc_flag is not None:
                cfg["escalate_on_fail"] = esc_flag.lower() in ("1", "true", "yes")
        except Exception:
            pass
        return cls(cfg)

    def run(
        self,
        user_image_path: str,
        garment_front_path: str,
        garment_side_path: Optional[str] = None,
    ) -> VFRRunResult:
        work_root = os.path.join(RESULTS_DIR, "work")
        os.makedirs(work_root, exist_ok=True)

        # Stage: Person Canonicalization (stub)
        # Cacheable person canonicalization
        person_cache_dir, user_hash = user_cache_dir(user_image_path)
        person_dir = person_cache_dir
        mask_path = os.path.join(person_dir, "user_mask.png")
        keypoints_path = os.path.join(person_dir, "keypoints.json")
        if not os.path.exists(mask_path):
            mask_path = self.person.process(user_image_path, person_dir)
        if not os.path.exists(keypoints_path):
            keypoints_path = self.pose.process(user_image_path, person_dir)
        user_can = UserCanonical(
            user_image_path=user_image_path,
            mask_path=mask_path,
            keypoints_path=keypoints_path,
        )

        # Stage: Garment Assetization (stub)
        garment_cache, garment_hash = garment_cache_dir(garment_front_path)
        garment = GarmentAssets(
            garment_front_path=garment_front_path,
            garment_side_path=garment_side_path,
        )

        # Stage: Warping (soft render placeholder)
        warp_dir = os.path.join(work_root, "warp")
        warped_garment_path = self.warper.process(
            garment.garment_front_path,
            user_can.user_image_path,
            warp_dir,
            keypoints_path=user_can.keypoints_path,
        )
        # Optional VTON expert blend
        if self.cfg.get("vton_enabled", True):
            vton_dir = os.path.join(work_root, "vton")
            vton_path = self.vton.process(user_can.user_image_path, warped_garment_path, user_can.mask_path, vton_dir)
            soft_input = vton_path
        else:
            soft_input = warped_garment_path
        draped = DrapedOutput(soft_render_path=soft_input)

        # Stage: Geometry (stub)
        drape_dir = os.path.join(work_root, "drape")
        _marker = self.geometry.process(None, None, drape_dir)
        _ = _marker  # unused placeholder

        # Stage: Controls for finisher
        controls_dir = os.path.join(work_root, "controls")
        controls = build_controls(user_can.user_image_path, user_can.mask_path, user_can.keypoints_path, controls_dir)

        # Stage: Finisher (img2img polish placeholder)
        attempts = 0
        max_retries = int(self.cfg.get("qa_thresholds", {}).get("max_retries", 1))
        denoise = float(self.cfg.get("img2img_denoise", 0.18))
        final_path = None
        last_scores = None
        while attempts <= max_retries:
            fin_dir = os.path.join(work_root, f"finisher_try_{attempts}")
            polished_path = self.finisher.process(
                draped.soft_render_path,
                fin_dir,
                denoise=denoise,
                controls=controls,
                adapters={
                    'garment_image': garment.garment_front_path,
                    'face_image': user_can.user_image_path,
                },
            )
            post_dir = os.path.join(work_root, f"post_try_{attempts}")
            final_path = self.post.process(polished_path, post_dir)
            # Optional post chain
            if self.upscaler:
                up_dir = os.path.join(work_root, f"upscale_try_{attempts}")
                final_path = self.upscaler.process(final_path, up_dir)
            if self.face_restorer:
                fr_dir = os.path.join(work_root, f"face_try_{attempts}")
                final_path = self.face_restorer.process(final_path, fr_dir)
            # QA
            scores = self.qa.evaluate(
                final_path,
                refs={
                    "user": user_can.user_image_path,
                    "garment": garment.garment_front_path,
                    "work_root": work_root,
                },
            )
            last_scores = scores
            if scores.get("passed"):
                break
            attempts += 1
            # Adjust finisher params for next try
            denoise = max(0.0, denoise + float(self.cfg["qa_thresholds"]["denoise_delta"]))
            # Increase steps by configured increment for retries
            try:
                inc = int(self.cfg.get("qa_thresholds", {}).get("step_increase", 5))
                cur_steps = int(os.environ.get("SDXL_STEPS", str(int(self.cfg.get("finisher_steps", 38)))))
                os.environ["SDXL_STEPS"] = str(cur_steps + inc)
            except Exception:
                pass

        # Optional escalation to Kling after retries
        if (not last_scores or not last_scores.get("passed")) and self.cfg.get("escalate_on_fail"):
            fin_dir = os.path.join(work_root, "finisher_kling")
            kling = KlingFinisher()
            polished_path = kling.process(draped.soft_render_path, fin_dir, denoise=denoise)
            post_dir = os.path.join(work_root, "post_kling")
            final_path = self.post.process(polished_path, post_dir)

        return VFRRunResult(output_path=final_path)
