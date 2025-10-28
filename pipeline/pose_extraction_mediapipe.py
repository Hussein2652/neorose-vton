from __future__ import annotations

import json
import os
from typing import Optional


def extract_pose(user_image_path: str, out_dir: str) -> Optional[str]:
    """
    Extract body keypoints using MediaPipe Pose if available.
    Returns path to keypoints.json or None if MediaPipe is not installed.
    """
    try:
        import mediapipe as mp  # type: ignore
        import cv2  # type: ignore
    except Exception:
        return None

    os.makedirs(out_dir, exist_ok=True)
    image = cv2.imread(user_image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, enable_segmentation=False) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        h, w = image.shape[:2]
        landmarks = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            landmarks.append(
                {
                    "index": idx,
                    "x": float(lm.x * w),
                    "y": float(lm.y * h),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                }
            )
        data = {"width": w, "height": h, "keypoints": landmarks}
        out_path = os.path.join(out_dir, "keypoints.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return out_path

