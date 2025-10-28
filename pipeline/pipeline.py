import os
from dataclasses import dataclass
from typing import Optional

from backend.app.storage import RESULTS_DIR
from .io_types import UserCanonical, GarmentAssets, DrapedOutput, FinalOutput
from .person_parsing import segment_person
from .pose_extraction import extract_pose
from .garment_warping import warp_garment_to_user
from .geometry_fitting import drape_garment
from .finisher import img2img_polish
from .post_processing import post_process


@dataclass
class VFRRunResult:
    output_path: str


class VFRPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    @classmethod
    def from_settings(cls, settings) -> "VFRPipeline":
        # Settings is the backend.app.config.settings instance
        cfg = {
            "img2img_denoise": float(settings.get("finisher.denoise", 0.18)),
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
        mask_path = segment_person(user_image_path, person_dir)
        keypoints_path = extract_pose(user_image_path, person_dir)
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
        soft_render_path = warp_garment_to_user(garment.garment_front_path, user_can.user_image_path, warp_dir)
        draped = DrapedOutput(
            soft_render_path=soft_render_path,
        )

        # Stage: Geometry (stub)
        drape_dir = os.path.join(work_root, "drape")
        _marker = drape_garment(None, None, drape_dir)
        _ = _marker  # unused placeholder

        # Stage: Finisher (img2img polish placeholder)
        fin_dir = os.path.join(work_root, "finisher")
        denoise = float(self.cfg.get("img2img_denoise", 0.18))
        polished_path = img2img_polish(draped.soft_render_path, fin_dir, denoise=denoise)

        # Stage: Post Processing
        post_dir = os.path.join(work_root, "post")
        final_path = post_process(polished_path, post_dir)

        return VFRRunResult(output_path=final_path)

