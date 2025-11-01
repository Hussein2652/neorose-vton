from __future__ import annotations

import os
import sys
from typing import Optional
from PIL import Image


def refine_mask(user_image_path: str, mask_path: str, out_dir: str) -> str:
    """
    Edge-aware refinement of a binary person mask to a soft alpha matte.
    Preference order:
      1) Vendored Robust Video Matting (drop repo under third_party/rvm) with
         weights at $RVM_WEIGHTS → single-frame matting.
      2) Soft edge via distance transform around binary mask (no deps beyond cv2).
    Returns path to refined PNG (L mode). Falls back to the input mask on error.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Try vendored Robust Video Matting
    weights = os.environ.get(
        "RVM_WEIGHTS",
        os.path.join(os.environ.get("MODELS_DIR", "storage/models"), "rvm_mobilenetv3", "v1.0.0", "rvm_mobilenetv3.pth"),
    )
    rvm_root = os.path.join("third_party", "rvm")
    if os.path.isdir(rvm_root) and weights and os.path.exists(weights):
        try:
            import torch  # type: ignore
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Ensure third_party/rvm is importable
            if rvm_root not in sys.path:
                sys.path.insert(0, rvm_root)
            # Official repo defines MattingNetwork in model.py at project root
            from model import MattingNetwork  # type: ignore

            net = MattingNetwork("mobilenetv3").to(device).eval()
            sd = torch.load(weights, map_location=device)
            if isinstance(sd, dict):
                # Handle state dict nested formats transparently
                try:
                    net.load_state_dict(sd)
                except Exception:
                    sd = sd.get("state_dict", sd)
                    net.load_state_dict(sd)
            # Prepare single-frame input
            import numpy as np
            img = Image.open(user_image_path).convert("RGB")
            im = np.asarray(img).astype("float32") / 255.0
            src = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(device)
            # RVM expects recurrent states; initialize zeros
            r1 = r2 = r3 = r4 = [None] * 4
            downsample_ratio = 0.25
            with torch.no_grad():
                pha, fgr, *_ = net(src, *([None] * 4), downsample_ratio)
            alpha = (pha[0, 0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            out_img = Image.fromarray(alpha, mode="L")
            out_path = os.path.join(out_dir, "user_mask.png")
            out_img.save(out_path)
            return out_path
        except Exception:
            # Fall through to soft-edge refinement
            pass

    # Soft-edge refinement from binary mask using distance transform
    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore
        base = Image.open(user_image_path).convert("RGB")
        m = Image.open(mask_path).convert("L").resize(base.size, Image.NEAREST)
        m_np = np.asarray(m)
        fg = (m_np > 200).astype("uint8")
        dist_out = cv2.distanceTransform((1 - fg) * 255, cv2.DIST_L2, 3)
        dist_in = cv2.distanceTransform(fg * 255, cv2.DIST_L2, 3)
        # Soft edge ~15px radius
        edge = dist_out - dist_in
        edge = np.clip((edge + 15.0) / 30.0, 0.0, 1.0)
        alpha = (1.0 - edge) * fg + (1 - fg) * (1.0 - edge)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_img = Image.fromarray((alpha * 255).astype("uint8"), mode="L")
        out_path = os.path.join(out_dir, "user_mask.png")
        alpha_img.save(out_path)
        return out_path
    except Exception:
        return mask_path
