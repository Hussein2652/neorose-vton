from dataclasses import dataclass
from typing import Optional


@dataclass
class UserCanonical:
    user_image_path: str
    mask_path: Optional[str] = None
    keypoints_path: Optional[str] = None
    smplx_mesh_path: Optional[str] = None
    uv_texture_path: Optional[str] = None


@dataclass
class GarmentAssets:
    garment_front_path: str
    garment_side_path: Optional[str] = None
    mesh_path: Optional[str] = None
    uv_path: Optional[str] = None
    texture_path: Optional[str] = None


@dataclass
class DrapedOutput:
    soft_render_path: str
    control_pose_path: Optional[str] = None
    control_depth_path: Optional[str] = None
    control_normal_path: Optional[str] = None
    control_seg_path: Optional[str] = None
    control_edge_path: Optional[str] = None


@dataclass
class FinalOutput:
    output_path: str

