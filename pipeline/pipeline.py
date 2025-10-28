import os
from dataclasses import dataclass
from typing import Optional

from backend.app.storage import RESULTS_DIR
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


@dataclass
class VFRRunResult:
    output_path: str


class VFRPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Build providers
        self.person = LocalPersonParser()
        self.pose = LocalPoseExtractor()
        self.warper = LocalGarmentWarper()
        self.geometry = LocalGeometryFitter()
        backend = self.cfg.get("finisher_backend", "local")
        self.finisher = KlingFinisher() if backend == "kling" else LocalFinisher()
        self.post = LocalPostProcessor()
        self.qa = LocalQA()

    @classmethod
    def from_settings(cls, settings) -> "VFRPipeline":
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
        }
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
        person_dir = os.path.join(work_root, "person")
        mask_path = self.person.process(user_image_path, person_dir)
        keypoints_path = self.pose.process(user_image_path, person_dir)
        user_can = UserCanonical(
            user_image_path=user_image_path,
            mask_path=mask_path,
            keypoints_path=keypoints_path,
        )

        # Stage: Garment Assetization (stub)
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
            vton_path = apply_vton(user_can.user_image_path, warped_garment_path, user_can.mask_path, vton_dir)
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
            polished_path = self.finisher.process(draped.soft_render_path, fin_dir, denoise=denoise, controls=controls)
            post_dir = os.path.join(work_root, f"post_try_{attempts}")
            final_path = self.post.process(polished_path, post_dir)
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
            denoise = max(0.0, denoise + float(self.cfg["qa_thresholds"]["denoise_delta"]))

        # Optional escalation to Kling after retries
        if (not last_scores or not last_scores.get("passed")) and self.cfg.get("escalate_on_fail"):
            fin_dir = os.path.join(work_root, "finisher_kling")
            kling = KlingFinisher()
            polished_path = kling.process(draped.soft_render_path, fin_dir, denoise=denoise)
            post_dir = os.path.join(work_root, "post_kling")
            final_path = self.post.process(polished_path, post_dir)

        return VFRRunResult(output_path=final_path)
