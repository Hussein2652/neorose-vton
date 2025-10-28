from __future__ import annotations

from typing import Protocol, Optional


class PersonParser(Protocol):
    def process(self, user_image_path: str, out_dir: str) -> str: ...


class PoseExtractor(Protocol):
    def process(self, user_image_path: str, out_dir: str) -> str: ...


class GarmentWarper(Protocol):
    def process(self, garment_front_path: str, user_image_path: str, out_dir: str, keypoints_path: Optional[str] = None) -> str: ...


class GeometryFitter(Protocol):
    def process(self, smplx_mesh_path: Optional[str], garment_mesh_path: Optional[str], out_dir: str) -> str: ...


class Finisher(Protocol):
    def process(
        self,
        soft_render_path: str,
        out_dir: str,
        denoise: float = 0.18,
        controls: Optional[dict] = None,
        adapters: Optional[dict] = None,
    ) -> str: ...


class PostProcessor(Protocol):
    def process(self, polished_path: str, out_dir: str) -> str: ...


class QA(Protocol):
    def evaluate(self, final_path: str, refs: dict) -> dict: ...
