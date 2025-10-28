from __future__ import annotations

import os
from typing import Optional

from PIL import Image


class TorchVisionPersonSegmenter:
    """
    Person segmenter using torchvision DeepLabV3-ResNet50 pretrained weights.
    Requires torch+torchvision (see requirements-ml.txt). Falls back to full mask if unavailable.
    """

    def __init__(self) -> None:
        self._ready = False
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            from torchvision.models.segmentation import deeplabv3_resnet50
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
            # Ensure HF/Torch caches persist
            from backend.app.artifacts import MODELS_DIR
            os.environ.setdefault("TORCH_HOME", os.path.join(MODELS_DIR, "torch"))
            self._model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1).eval()
            self._weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            self._ready = True
        except Exception:
            self._ready = False

    def process(self, user_image_path: str, out_dir: str) -> str:
        import os
        from PIL import Image
        os.makedirs(out_dir, exist_ok=True)
        im = Image.open(user_image_path).convert("RGB")
        if not self._ready:
            # fallback full mask
            mask = Image.new("L", im.size, color=255)
            out = os.path.join(out_dir, "user_mask.png")
            mask.save(out)
            return out
        try:
            import torch
            import numpy as np
            preprocess = self._weights.transforms()
            batch = preprocess(im).unsqueeze(0)
            with torch.no_grad():
                out = self._model(batch)["out"]  # [1, C, H, W]
                pred = out.argmax(1)[0].cpu().numpy().astype("uint8")
            # VOC labels: person index is 15
            person_idx = 15
            mask = (pred == person_idx).astype("uint8") * 255
            mask_img = Image.fromarray(mask, mode="L").resize(im.size)
            out_path = os.path.join(out_dir, "user_mask.png")
            mask_img.save(out_path)
            return out_path
        except Exception:
            mask = Image.new("L", im.size, color=255)
            out = os.path.join(out_dir, "user_mask.png")
            mask.save(out)
            return out

