from __future__ import annotations

import os
from typing import Optional, Dict

from PIL import Image, ImageOps, ImageFilter, ImageDraw


def _save(im: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return path


def control_edges(user_image_path: str, out_dir: str) -> str:
    im = Image.open(user_image_path).convert("L")
    try:
        import cv2  # type: ignore
        import numpy as np

        arr = np.asarray(im)
        edges = cv2.Canny(arr, 80, 150)
        out = Image.fromarray(edges)
    except Exception:
        out = im.filter(ImageFilter.FIND_EDGES)
    return _save(out, os.path.join(out_dir, "edge.png"))


def control_depth(user_image_path: str, out_dir: str) -> str:
    # Fake "depth" via blurred luminance inverse
    im = Image.open(user_image_path).convert("L")
    d = ImageOps.autocontrast(im.filter(ImageFilter.GaussianBlur(radius=6)))
    d = ImageOps.invert(d)
    return _save(d, os.path.join(out_dir, "depth.png"))


def control_normals(user_image_path: str, out_dir: str) -> str:
    # Approximate normals from Sobel gradients
    try:
        import cv2  # type: ignore
        import numpy as np

        g = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        nz = 1.0
        nx = gx / 255.0
        ny = gy / 255.0
        # normalize
        length = (nx ** 2 + ny ** 2 + nz ** 2) ** 0.5
        nx = (nx / length + 1.0) * 127.5
        ny = (ny / length + 1.0) * 127.5
        nz = (nz / length + 1.0) * 127.5
        normals = cv2.merge([nx.astype('uint8'), ny.astype('uint8'), (nz * 255).astype('uint8')])
        out = Image.fromarray(normals, mode='RGB')
    except Exception:
        out = Image.open(user_image_path).convert("RGB").filter(ImageFilter.SMOOTH)
    return _save(out, os.path.join(out_dir, "normal.png"))


def control_pose(user_image_path: str, keypoints_path: Optional[str], out_dir: str) -> Optional[str]:
    if not keypoints_path or not os.path.exists(keypoints_path):
        return None
    try:
        import json
        data = json.load(open(keypoints_path, 'r', encoding='utf-8'))
        im = Image.open(user_image_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        # Render simple skeleton lines between chosen joints (MediaPipe indices)
        pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26)]
        kps = {kp['index']: (kp['x'], kp['y']) for kp in data.get('keypoints', []) if 'index' in kp}
        for a, b in pairs:
            if a in kps and b in kps:
                draw.line([kps[a], kps[b]], fill=(255, 0, 0), width=4)
        return _save(im, os.path.join(out_dir, "pose.png"))
    except Exception:
        return None


def build_controls(user_image_path: str, mask_path: Optional[str], keypoints_path: Optional[str], out_dir: str) -> Dict[str, Optional[str]]:
    os.makedirs(out_dir, exist_ok=True)
    controls = {
        'edge': control_edges(user_image_path, out_dir),
        'depth': control_depth(user_image_path, out_dir),
        'normal': control_normals(user_image_path, out_dir),
        'seg': mask_path,
        'pose': control_pose(user_image_path, keypoints_path, out_dir),
    }
    return controls

