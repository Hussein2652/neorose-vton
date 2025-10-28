from __future__ import annotations

import os
from typing import Optional

from pipeline.person_parsing import segment_person
from pipeline.pose_extraction import extract_pose
from pipeline.garment_warping import warp_garment_to_user
from pipeline.geometry_fitting import drape_garment
from pipeline.finisher import img2img_polish
from pipeline.post_processing import post_process
from pipeline.qa import compute_qa


class LocalPersonParser:
    def process(self, user_image_path: str, out_dir: str) -> str:
        return segment_person(user_image_path, out_dir)


class LocalPoseExtractor:
    def process(self, user_image_path: str, out_dir: str) -> str:
        # Try MediaPipe extractor if installed; fallback to stub
        try:
            from pipeline.pose_extraction_mediapipe import extract_pose as mp_extract

            out = mp_extract(user_image_path, out_dir)
            if out:
                return out
        except Exception:
            pass
        return extract_pose(user_image_path, out_dir)


class LocalGarmentWarper:
    def process(self, garment_front_path: str, user_image_path: str, out_dir: str, keypoints_path: Optional[str] = None) -> str:
        return warp_garment_to_user(garment_front_path, user_image_path, out_dir, keypoints_path=keypoints_path)


class LocalGeometryFitter:
    def process(self, smplx_mesh_path: Optional[str], garment_mesh_path: Optional[str], out_dir: str) -> str:
        return drape_garment(smplx_mesh_path, garment_mesh_path, out_dir)


class LocalFinisher:
    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.18, controls: Optional[dict] = None, adapters: Optional[dict] = None) -> str:
        # controls/adapters accepted for API parity; ignored in stub
        return img2img_polish(soft_render_path, out_dir, denoise)


class LocalPostProcessor:
    def process(self, polished_path: str, out_dir: str) -> str:
        return post_process(polished_path, out_dir)


class LocalQA:
    def evaluate(self, final_path: str, refs: dict) -> dict:
        user = refs.get("user")
        garment = refs.get("garment")
        work_root = refs.get("work_root")
        scores = compute_qa(final_path, user, garment, work_root=work_root)
        res = {
            "garment_fidelity": scores.garment_similarity,
            "identity": scores.identity_score,
            "aesthetic": scores.aesthetics,
        }
        # Simple pass/fail based on config-like thresholds (fallback defaults)
        thr = {
            "garment_min": float(os.environ.get("QA_GARMENT_MIN", "0.28")),
            "identity_min": float(os.environ.get("QA_IDENTITY_MIN", "0.75")),
            "aesthetic_min": float(os.environ.get("QA_AESTH_MIN", "0.58")),
        }
        res["passed"] = (
            res["garment_fidelity"] >= thr["garment_min"]
            and res["identity"] >= thr["identity_min"]
            and res["aesthetic"] >= thr["aesthetic_min"]
        )
        return res
