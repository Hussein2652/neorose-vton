from __future__ import annotations

import os
import subprocess
from typing import Optional
from PIL import Image


class SCHPSegmenter:
    """
    Wrapper around a local SCHP inference script.
    Expects:
      - code at third_party/schp (or SCHP_INFER_CMD env to point to a CLI)
      - weights at $SCHP_MODEL_PATH (defaults to storage/models/schp_lip/20190826/exp-schp-201908261155-lip.pth)
    Produces a binary person mask at out_dir/user_mask.png and returns its path.
    """

    def __init__(self) -> None:
        self.repo_dir = os.environ.get("SCHP_REPO_DIR", os.path.join("third_party", "schp"))
        self.model_path = os.environ.get(
            "SCHP_MODEL_PATH",
            os.path.join(
                os.environ.get("MODELS_DIR", "storage/models"),
                "schp_lip",
                "20190826",
                "exp-schp-201908261155-lip.pth",
            ),
        )
        self.cmd = os.environ.get("SCHP_INFER_CMD")

    def _default_cmd(self, image_path: str, out_dir: str) -> Optional[list[str]]:
        # Prefer our portable wrapper that calls SCHP simple_extractor
        wrapper = os.path.join("scripts", "schp_infer.py")
        if os.path.exists(wrapper):
            return [
                "python",
                wrapper,
                "--weights",
                self.model_path,
                "--image",
                image_path,
                "--out",
                out_dir,
            ]
        # Try common inference entrypoints in the vendored repo (if someone provided their own)
        candidates = [
            os.path.join(self.repo_dir, "tools", "inference.py"),
            os.path.join(self.repo_dir, "inference.py"),
            os.path.join(self.repo_dir, "demo_inference.py"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return ["python", c, "--weights", self.model_path, "--image", image_path, "--out", out_dir]
        return None

    def process(self, image_path: str, out_dir: str) -> Optional[str]:
        os.makedirs(out_dir, exist_ok=True)
        cmd: Optional[list[str]] = None
        if self.cmd:
            cmd = [p for p in self.cmd.split(" ") if p]
            # append std args if placeholders present
            cmd = [
                s.replace("{WEIGHTS}", self.model_path).replace("{IMAGE}", image_path).replace("{OUT}", out_dir)
                for s in cmd
            ]
        else:
            cmd = self._default_cmd(image_path, out_dir)
        if not cmd:
            return None
        try:
            subprocess.run(cmd, check=True)
            # Expect output mask at out_dir/user_mask.png; if not present, try common names
            mask_path = os.path.join(out_dir, "user_mask.png")
            if os.path.exists(mask_path):
                return mask_path
            # search for first PNG
            for fn in os.listdir(out_dir):
                if fn.lower().endswith(".png"):
                    # ensure binary mask
                    p = os.path.join(out_dir, fn)
                    try:
                        im = Image.open(p).convert("L")
                        im.save(mask_path)
                        return mask_path
                    except Exception:
                        pass
            return None
        except Exception:
            return None
