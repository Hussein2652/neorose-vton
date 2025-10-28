from typing import Optional
from pipeline.pipeline import VFRPipeline
from backend.app.config import settings


def run_tryon_job(
    user_image_path: str,
    garment_front_path: str,
    garment_side_path: Optional[str] = None,
):
    pipe = VFRPipeline.from_settings(settings)
    result = pipe.run(
        user_image_path=user_image_path,
        garment_front_path=garment_front_path,
        garment_side_path=garment_side_path,
    )
    return {"result_path": result.output_path}

