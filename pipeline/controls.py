from __future__ import annotations

import os
from typing import Optional, Dict

from PIL import Image
import torch


def _save(im: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path)
    return path


def control_edges(user_image_path: str, out_dir: str) -> str:
    # Prefer HED from controlnet-aux; fallback to OpenCV canny
    try:
        from controlnet_aux import HEDdetector  # type: ignore
        annot_dir = os.environ.get('ANNOTATOR_DIR')
        if annot_dir and os.path.isdir(annot_dir):
            hed = HEDdetector.from_pretrained(annotator_path=annot_dir, cache_dir=annot_dir, local_files_only=True)
        else:
            hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
        # Prefer GPU for annotators if available
        try:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if hasattr(hed, 'to'):
                hed.to(dev)
        except Exception:
            pass
        im = Image.open(user_image_path).convert('RGB')
        out_im = hed(im)
        return _save(out_im, os.path.join(out_dir, 'edge.png'))
    except Exception:
        try:
            import cv2  # type: ignore
            import numpy as np
            im = Image.open(user_image_path).convert('L')
            arr = np.asarray(im)
            edges = cv2.Canny(arr, 80, 150)
            out = Image.fromarray(edges)
            return _save(out, os.path.join(out_dir, 'edge.png'))
        except Exception:
            return _save(Image.open(user_image_path).convert('L'), os.path.join(out_dir, 'edge.png'))


def control_depth(user_image_path: str, out_dir: str) -> str:
    # Prefer ZoeDepth via controlnet-aux; fallback to MiDaS DPT Large
    try:
        from controlnet_aux import ZoeDetector  # type: ignore
        annot_dir = os.environ.get('ANNOTATOR_DIR')
        if annot_dir and os.path.isdir(annot_dir):
            zoe = ZoeDetector.from_pretrained(annotator_path=annot_dir, cache_dir=annot_dir, local_files_only=True)
        else:
            zoe = ZoeDetector.from_pretrained('lllyasviel/Annotators')
        try:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if hasattr(zoe, 'to'):
                zoe.to(dev)
        except Exception:
            pass
        im = Image.open(user_image_path).convert('RGB')
        out_im = zoe(im)
        return _save(out_im, os.path.join(out_dir, 'depth.png'))
    except Exception:
        try:
            midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            midas.to(device).eval()
            transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            transform = transforms.dpt_transform
            im = Image.open(user_image_path).convert('RGB')
            import numpy as np
            inp = transform(np.array(im)).to(device)
            with torch.no_grad():
                pred = midas(inp)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=im.size[::-1], mode='bicubic', align_corners=False
                ).squeeze()
            d = pred.cpu().numpy()
            d = (d - d.min()) / (d.max() - d.min() + 1e-8)
            d_img = Image.fromarray((255 * (1.0 - d)).astype('uint8'))
            return _save(d_img, os.path.join(out_dir, 'depth.png'))
        except Exception:
            im = Image.open(user_image_path).convert('L')
            return _save(im, os.path.join(out_dir, 'depth.png'))


def control_normals(user_image_path: str, out_dir: str) -> str:
    # Prefer NormalBae from controlnet-aux; fallback to a smoothed RGB proxy
    try:
        from controlnet_aux import NormalBaeDetector  # type: ignore
        annot_dir = os.environ.get('ANNOTATOR_DIR')
        if annot_dir and os.path.isdir(annot_dir):
            nb = NormalBaeDetector.from_pretrained(annotator_path=annot_dir, cache_dir=annot_dir, local_files_only=True)
        else:
            nb = NormalBaeDetector.from_pretrained('lllyasviel/Annotators')
        try:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if hasattr(nb, 'to'):
                nb.to(dev)
        except Exception:
            pass
        im = Image.open(user_image_path).convert('RGB')
        out_im = nb(im)
        return _save(out_im, os.path.join(out_dir, 'normal.png'))
    except Exception:
        return _save(Image.open(user_image_path).convert('RGB'), os.path.join(out_dir, 'normal.png'))


def control_pose(user_image_path: str, keypoints_path: Optional[str], out_dir: str) -> Optional[str]:
    # Prefer OpenPose rendering via controlnet-aux; fallback to MediaPipe keypoints path render
    try:
        from controlnet_aux import OpenposeDetector  # type: ignore
        annot_dir = os.environ.get('ANNOTATOR_DIR')
        if annot_dir and os.path.isdir(annot_dir):
            op = OpenposeDetector.from_pretrained(annotator_path=annot_dir, cache_dir=annot_dir, local_files_only=True)
        else:
            op = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
        try:
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            if hasattr(op, 'to'):
                op.to(dev)
        except Exception:
            pass
        im = Image.open(user_image_path).convert('RGB')
        out_im = op(im)
        return _save(out_im, os.path.join(out_dir, 'pose.png'))
    except Exception:
        if not keypoints_path or not os.path.exists(keypoints_path):
            return None
        try:
            import json
            from PIL import ImageDraw
            data = json.load(open(keypoints_path, 'r', encoding='utf-8'))
            im = Image.open(user_image_path).convert('RGB')
            draw = ImageDraw.Draw(im)
            pairs = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26)]
            kps = {kp['index']: (kp['x'], kp['y']) for kp in data.get('keypoints', []) if 'index' in kp}
            for a, b in pairs:
                if a in kps and b in kps:
                    draw.line([kps[a], kps[b]], fill=(255, 0, 0), width=4)
            return _save(im, os.path.join(out_dir, 'pose.png'))
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
