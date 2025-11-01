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

    # Minimal DensePose config (use default model if available in cache)
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(cfg.DENSEPOSE_CONFIG_FILE if hasattr(cfg, "DENSEPOSE_CONFIG_FILE") else "")
    # Use a commonly available model id if present; otherwise rely on env/preload
    # Users should bake a detectron2+densepose model into the container for this path.
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

