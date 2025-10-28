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
        }
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
        soft_render_path = self.warper.process(garment.garment_front_path, user_can.user_image_path, warp_dir)
        draped = DrapedOutput(
            soft_render_path=soft_render_path,
        )

        # Stage: Geometry (stub)
        drape_dir = os.path.join(work_root, "drape")
        _marker = self.geometry.process(None, None, drape_dir)
        _ = _marker  # unused placeholder

        # Stage: Finisher (img2img polish placeholder)
        attempts = 0
        max_retries = int(self.cfg.get("qa_thresholds", {}).get("max_retries", 1))
        denoise = float(self.cfg.get("img2img_denoise", 0.18))
        final_path = None
        last_scores = None
        while attempts <= max_retries:
            fin_dir = os.path.join(work_root, f"finisher_try_{attempts}")
            polished_path = self.finisher.process(draped.soft_render_path, fin_dir, denoise=denoise)
            post_dir = os.path.join(work_root, f"post_try_{attempts}")
            final_path = self.post.process(polished_path, post_dir)
            # QA
            scores = self.qa.evaluate(final_path, refs={"user": user_can.user_image_path, "garment": garment.garment_front_path})
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
