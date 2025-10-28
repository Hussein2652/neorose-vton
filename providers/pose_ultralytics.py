from __future__ import annotations

import os
from typing import Optional


class YOLOv8PoseExtractor:
    """
    Pose extractor using ultralytics YOLOv8 (yolov8n-pose by default).
    Requires `ultralytics` (see requirements-ml.txt). Writes keypoints.json.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get("YOLOV8_POSE_MODEL", "yolov8n-pose.pt")
        self._ready = False
        try:
            from ultralytics import YOLO  # type: ignore
            self._YOLO = YOLO
            self._ready = True
        except Exception:
            self._ready = False

    def process(self, user_image_path: str, out_dir: str) -> str:
        import os, json
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "keypoints.json")
        if not self._ready:
            # fallback empty
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"width": 0, "height": 0, "keypoints": []}, f)
            return out_path
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(self.model_name)
            res = model(user_image_path)
            # Take first result
            r0 = res[0]
            w, h = r0.orig_shape[1], r0.orig_shape[0]
            kps = []
            if hasattr(r0, 'keypoints') and r0.keypoints is not None:
                np_kps = r0.keypoints.xy.cpu().numpy() if hasattr(r0.keypoints, 'xy') else None
                if np_kps is not None and len(np_kps) > 0:
                    # choose first person detected
                    arr = np_kps[0]
                    for idx, (x, y) in enumerate(arr):
                        kps.append({"index": int(idx), "x": float(x), "y": float(y), "visibility": 1.0})
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"width": w, "height": h, "keypoints": kps}, f)
            return out_path
        except Exception:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"width": 0, "height": 0, "keypoints": []}, f)
            return out_path

