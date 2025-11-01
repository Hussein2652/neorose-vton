from __future__ import annotations

from typing import Optional
from PIL import Image


def compute_densepose_image(user_im: Image.Image) -> Image.Image:
    """
    Optional DensePose via detectron2 if installed in the image.
    Returns an RGB visualization compatible with StableVITON's expected 'image-densepose'.
    If detectron2/densepose are not available, raises ImportError so caller can fallback.
    """
    # Lazy imports to avoid mandatory deps
    try:
        import torch  # type: ignore
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultPredictor  # type: ignore
        from densepose import add_densepose_config  # type: ignore
        from densepose.vis.extractor import DensePoseResultExtractor  # type: ignore
        from densepose.vis.base import CompoundVisualizer  # type: ignore
        from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer  # type: ignore
        from densepose.vis.mesh import DensePoseOutputsVertexBasedVisualizer  # type: ignore
        from densepose.vis.segm import DensePoseResultsFineSegmentationVisualizer  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ImportError("densepose not available") from e

    # Require config & weights from env for deterministic setup
    import os  # type: ignore
    from detectron2 import model_zoo  # type: ignore
    cfg_file = os.environ.get("DENSEPOSE_CFG")
    weights_path = os.environ.get("DENSEPOSE_WEIGHTS")
    # Validate provided paths
    def _valid_file(p: str | None, min_size: int = 1024) -> bool:
        try:
            return bool(p) and os.path.exists(p) and os.path.getsize(p) >= min_size  # type: ignore[arg-type]
        except Exception:
            return False
    # Try provided cfg; else fall back to model_zoo packaged config
    if not _valid_file(cfg_file, min_size=256):
        try:
            cfg_file = model_zoo.get_config_file("projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        except Exception as e:  # noqa: BLE001
            raise ImportError("densepose config not available") from e
    if not _valid_file(weights_path, min_size=1024):
        raise ImportError("densepose weights not provided or too small")
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(cfg_file)  # type: ignore[arg-type]
    cfg.MODEL.WEIGHTS = weights_path  # type: ignore[assignment]
    # Users should bake detectron2+densepose model into the container for this path.
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Run predictor
    import numpy as np  # type: ignore

    img = np.array(user_im.convert("RGB"))
    outputs = predictor(img)

    # Visualize DensePose IUV onto an RGB image
    extractor = DensePoseResultExtractor()
    visualizers = [
        DensePoseOutputsVertexBasedVisualizer(),
        DensePoseResultsFineSegmentationVisualizer(),
        ScoredBoundingBoxVisualizer(),
    ]
    comp_vis = CompoundVisualizer(visualizers)
    dp = extractor(outputs)
    out = comp_vis.visualize(img, dp)
    return Image.fromarray(out[:, :, ::-1])
