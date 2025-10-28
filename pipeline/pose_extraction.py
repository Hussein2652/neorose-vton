import json
import os


def extract_pose(user_image_path: str, out_dir: str) -> str:
    """
    Stub pose/keypoints extractor.
    - Emits an empty keypoints JSON with image size.
    """
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image

    im = Image.open(user_image_path)
    data = {"width": im.width, "height": im.height, "keypoints": []}
    pose_path = os.path.join(out_dir, "keypoints.json")
    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return pose_path

