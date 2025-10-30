from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from PIL import Image


def _to_binary_mask(palette_png: str, out_mask: str) -> None:
    im = Image.open(palette_png)
    # SCHP outputs a palette PNG with class indices in pixel values
    # Background is class 0; everything else becomes person (255)
    if im.mode != "P":
        im = im.convert("P")
    mask = im.point(lambda v: 0 if v == 0 else 255).convert("L")
    mask.save(out_mask)


def run(weights: str, image_path: str, out_dir: str, gpu: Optional[str] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    repo_dir = os.environ.get("SCHP_REPO_DIR", os.path.join("third_party", "schp", "Self-Correction-Human-Parsing-master"))
    entry = os.path.join(repo_dir, "simple_extractor.py")
    if not os.path.exists(entry):
        raise FileNotFoundError("SCHP simple_extractor.py not found; ensure the repo is unzipped under third_party/schp")
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "in"); out_seg = os.path.join(td, "out")
        os.makedirs(inp, exist_ok=True); os.makedirs(out_seg, exist_ok=True)
        base = os.path.basename(image_path)
        in_path = os.path.join(inp, base)
        shutil.copy2(image_path, in_path)

        cmd = [
            sys.executable,
            entry,
            "--dataset", "lip",
            "--model-restore", weights,
            "--input-dir", inp,
            "--output-dir", out_seg,
        ]
        if gpu is None:
            gpu = os.environ.get("SCHP_GPU", "0")
        if gpu:
            cmd += ["--gpu", gpu]
        subprocess.run(cmd, check=True, cwd=repo_dir)
        stem = os.path.splitext(base)[0]
        out_png = os.path.join(out_seg, f"{stem}.png")
        if not os.path.exists(out_png):
            cands = [f for f in os.listdir(out_seg) if f.lower().endswith(".png")]
            if not cands:
                raise RuntimeError("SCHP produced no segmentation PNGs")
            out_png = os.path.join(out_seg, cands[0])
        user_mask = os.path.join(out_dir, "user_mask.png")
        _to_binary_mask(out_png, user_mask)
        return user_mask


def main() -> None:
    ap = argparse.ArgumentParser(description="SCHP wrapper: --weights --image --out")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    m = run(args.weights, args.image, args.out, gpu=args.gpu)
    print(m)


if __name__ == "__main__":
    main()

