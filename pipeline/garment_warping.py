from PIL import Image
import os
import json
from typing import Optional


def warp_garment_to_user(garment_front_path: str, user_image_path: str, out_dir: str, keypoints_path: Optional[str] = None) -> str:
    """
    Stub garment warping.
    - Resizes garment to 60% of user image long side and centers it.
    - Outputs a simple composite as a soft render placeholder.
    """
    os.makedirs(out_dir, exist_ok=True)
    user = Image.open(user_image_path).convert("RGB")
    garment = Image.open(garment_front_path).convert("RGBA")

    # Try to place garment using pose keypoints (shoulders/hips)
    def _pose_box(img_w: int, img_h: int) -> Optional[tuple[int, int, int, int]]:
        if not keypoints_path or not os.path.exists(keypoints_path):
            return None
        try:
            data = json.load(open(keypoints_path, 'r', encoding='utf-8'))
            kps = {kp.get('index'): kp for kp in data.get('keypoints', [])}
            # MediaPipe indices
            L_SHO, R_SHO, L_HIP, R_HIP = 11, 12, 23, 24
            ls = kps.get(L_SHO); rs = kps.get(R_SHO); lh = kps.get(L_HIP); rh = kps.get(R_HIP)
            if not (ls and rs and lh and rh):
                return None
            shoulder_w = abs(rs['x'] - ls['x'])
            hip_w = abs(rh['x'] - lh['x'])
            mid_sh_y = (ls['y'] + rs['y']) / 2.0
            mid_hip_y = (lh['y'] + rh['y']) / 2.0
            box_w = int(max(shoulder_w, hip_w) * 1.35)
            box_h = int(abs(mid_hip_y - mid_sh_y) * 1.55)
            cx = int((ls['x'] + rs['x']) / 2.0)
            top = int(mid_sh_y - box_h * 0.10)  # slightly above shoulders
            x = max(0, int(cx - box_w // 2)); y = max(0, int(top))
            box_w = min(box_w, img_w)
            box_h = min(box_h, img_h - y)
            return x, y, box_w, box_h
        except Exception:
            return None

    g_w, g_h = garment.size
    pose_box = _pose_box(user.width, user.height)
    if pose_box:
        x, y, target_w, target_h = pose_box
        # Fit garment into target box preserving aspect
        scale = min(target_w / g_w, target_h / g_h)
        new_w = max(1, int(g_w * scale))
        new_h = max(1, int(g_h * scale))
    else:
        # Fallback: resize relative to long side
        long_side = max(user.size)
        target_long = int(long_side * 0.6)
        if g_w >= g_h:
            new_w = target_long
            new_h = int(g_h * (target_long / g_w))
        else:
            new_h = target_long
            new_w = int(g_w * (target_long / g_h))
    garment = garment.resize((new_w, new_h), Image.LANCZOS)

    # Position garment
    if pose_box:
        # Center within pose box
        x = x + (target_w - new_w) // 2
        y = y + (target_h - new_h) // 2
    else:
        x = (user.width - new_w) // 2
        y = (user.height - new_h) // 2
    composite = user.copy()
    composite.paste(garment, (x, y), garment)

    out_path = os.path.join(out_dir, "soft_render.png")
    composite.save(out_path)
    # Save simple metadata for QA (bounding box)
    meta = {"x": int(x), "y": int(y), "w": int(new_w), "h": int(new_h), "pose_used": bool(pose_box)}
    with open(os.path.join(out_dir, "warp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return out_path
