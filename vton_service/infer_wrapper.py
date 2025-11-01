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

def _patch_infer_script(src_path: str, enable_fp16: bool) -> str:
    """Create a temporary patched copy of inference.py to enable fp16 and autocast.
    The original third_party script is read-only in the container, so we patch at runtime.
    """
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return src_path

    out: List[str] = []
    inserted_import = False
    added_arg = False
    replaced_cuda = False
    inserted_with = False
    pending_indent_for_xsamples = False

    for i, line in enumerate(lines):
        # After DataLoader import, add autocast import
        if (not inserted_import) and "from torch.utils.data import DataLoader" in line:
            out.append(line)
            out.append("from torch.cuda.amp import autocast\n")
            inserted_import = True
            continue

        # After --eta arg, add --fp16
        if (not added_arg) and "parser.add_argument(\"--eta\"" in line:
            out.append(line)
            out.append("    parser.add_argument(\"--fp16\", action=\"store_true\", help=\"Run inference in float16 (mixed precision)\")\n")
            added_arg = True
            continue

        # Replace model.cuda() with conditional half().cuda()
        if (not replaced_cuda) and "model = model.cuda()" in line:
            # Keep original dtype; rely on autocast to avoid input/weight dtype mismatch
            out.append(line)
            replaced_cuda = True
            continue

        # Wrap sampling + decode in autocast
        if not inserted_with and "samples, _, _ = sampler.sample(" in line:
            # Insert with autocast at the same indentation
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}with autocast(enabled=args.fp16, dtype=torch.float16 if args.fp16 else torch.float32):\n")
            out.append(f"{indent}    {line.lstrip()}")
            inserted_with = True
            pending_indent_for_xsamples = True
            continue

        if pending_indent_for_xsamples and "x_samples = model.decode_first_stage(samples)" in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}    {line.lstrip()}")
            pending_indent_for_xsamples = False
            continue

        # Reduce DataLoader worker count to limit memory footprint
        if "DataLoader(dataset, num_workers=4" in line:
            out.append(line.replace("num_workers=4", "num_workers=0"))
            continue

        out.append(line)

    # Write patched file
    fd, tmp_path = tempfile.mkstemp(suffix="_stv_infer_fp16.py")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(out)
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
    # Patch upstream inference script for fp16 if requested via env
    upstream_path = "third_party/stableviton/StableVITON-master/inference.py"
    use_fp16 = os.environ.get("STABLEVITON_FORCE_FP16", "0") == "1"
    infer_entry = _patch_infer_script(upstream_path, enable_fp16=use_fp16) if use_fp16 else upstream_path
    cmd = [
        sys.executable,
        infer_entry,
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
    if use_fp16:
        cmd.append("--fp16")
    cmd.extend(unknown)
    verbose = os.environ.get("STABLEVITON_VERBOSE") == "1"
    try:
        env = os.environ.copy()
        # Ensure third_party modules (cldm, ldm, utils, dataset, etc.) are importable
        tp_root = os.path.abspath("third_party/stableviton/StableVITON-master")
        env["PYTHONPATH"] = (env.get("PYTHONPATH", "") + (os.pathsep if env.get("PYTHONPATH") else "") + tp_root)
        r = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
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
