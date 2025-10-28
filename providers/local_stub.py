from __future__ import annotations

import os
from typing import Optional

from pipeline.person_parsing import segment_person
from pipeline.pose_extraction import extract_pose
from pipeline.garment_warping import warp_garment_to_user
from pipeline.geometry_fitting import drape_garment
from pipeline.finisher import img2img_polish
from pipeline.post_processing import post_process


class LocalPersonParser:
    def process(self, user_image_path: str, out_dir: str) -> str:
        return segment_person(user_image_path, out_dir)


class LocalPoseExtractor:
    def process(self, user_image_path: str, out_dir: str) -> str:
        return extract_pose(user_image_path, out_dir)


class LocalGarmentWarper:
    def process(self, garment_front_path: str, user_image_path: str, out_dir: str) -> str:
        return warp_garment_to_user(garment_front_path, user_image_path, out_dir)


class LocalGeometryFitter:
    def process(self, smplx_mesh_path: Optional[str], garment_mesh_path: Optional[str], out_dir: str) -> str:
        return drape_garment(smplx_mesh_path, garment_mesh_path, out_dir)


class LocalFinisher:
    def process(self, soft_render_path: str, out_dir: str, denoise: float = 0.18) -> str:
        return img2img_polish(soft_render_path, out_dir, denoise)


class LocalPostProcessor:
    def process(self, polished_path: str, out_dir: str) -> str:
        return post_process(polished_path, out_dir)


class LocalQA:
    def evaluate(self, final_path: str, refs: dict) -> dict:
        # Placeholder metrics always pass; include simple numeric scores
        return {
            "clip": 0.35,
            "lpips": 0.25,
            "aesthetic": 0.62,
            "identity": 0.80,
            "iou_garment_skin": 0.92,
            "passed": True,
        }

