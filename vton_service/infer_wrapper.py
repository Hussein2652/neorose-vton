from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import List

import torch


def _clean_ckpt(src_path: str) -> str:
    ckpt = torch.load(src_path, map_location="cpu")
    has_wrapper = isinstance(ckpt, dict) and "state_dict" in ckpt
    state = ckpt["state_dict"] if has_wrapper else ckpt

    # Drop keys that are known to be incompatible across CLIP/HF versions
    drop_keys: List[str] = [
        "cond_stage_model.transformer.vision_model.embeddings.position_ids",
    ]
    removed = 0
    for k in list(state.keys()):
        if k in drop_keys:
            state.pop(k, None)
            removed += 1

    # Write to a temporary file
    fd, tmp_path = tempfile.mkstemp(suffix="_stv_clean.ckpt")
    os.close(fd)
    if has_wrapper:
        torch.save({"state_dict": state}, tmp_path)
    else:
        torch.save(state, tmp_path)
    print(f"[infer_wrapper] Cleaned ckpt: removed {removed} keys -> {tmp_path}", file=sys.stderr)
    return tmp_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--model_load_path", required=True)
    ap.add_argument("--data_root_dir", required=True)
    ap.add_argument("--batch_size", default="1")
    ap.add_argument("--img_H", default="512")
    ap.add_argument("--img_W", default="384")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--unpair", action="store_true")
    args, unknown = ap.parse_known_args()

    # Clean checkpoint then call upstream inference.py
    cleaned = _clean_ckpt(args.model_load_path)
    cmd = [
        sys.executable,
        "third_party/stableviton/StableVITON-master/inference.py",
        "--config_path",
        args.config_path,
        "--model_load_path",
        cleaned,
        "--data_root_dir",
        args.data_root_dir,
        "--batch_size",
        str(args.batch_size),
        "--img_H",
        str(args.img_H),
        "--img_W",
        str(args.img_W),
        "--save_dir",
        args.save_dir,
    ]
    if args.unpair:
        cmd.append("--unpair")
    cmd.extend(unknown)
    verbose = os.environ.get("STABLEVITON_VERBOSE") == "1"
    try:
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if verbose and r.stdout:
            print(r.stdout)
        if verbose and r.stderr:
            print(r.stderr, file=sys.stderr)
    except subprocess.CalledProcessError as e:
        # Always surface upstream CLI output to help diagnose
        print("[infer_wrapper] Upstream inference.py failed", file=sys.stderr)
        if e.stdout:
            print("[infer_wrapper stdout]", e.stdout, file=sys.stderr)
        if e.stderr:
            print("[infer_wrapper stderr]", e.stderr, file=sys.stderr)
        raise
    finally:
        try:
            os.remove(cleaned)
        except Exception:
            pass


if __name__ == "__main__":
    main()
