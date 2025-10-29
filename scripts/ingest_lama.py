#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Optional


def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def try_hf_download(models_dir: str, token: Optional[str]) -> Optional[str]:
    """Attempt to fetch big-lama.pt via Hugging Face with several known repo IDs.
    Returns local path if successful, else None.
    """
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        return None

    repo_ids = [
        # primary model card (often gated)
        "advimman/lama",
        # common variants seen in mirrors
        "saic-mdal/lama",
        "saic-mdal/LaMa",
        "saic-mdal/Big-Lama",
    ]
    for rid in repo_ids:
        try:
            fp = hf_hub_download(
                repo_id=rid,
                filename="big-lama.pt",
                token=token,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
            return fp
        except Exception:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest LaMa big-lama.pt into the models directory.")
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "storage/models"))
    ap.add_argument("--manual-dir", default=os.environ.get("MANUAL_MODELS_DIR", "manual_downloads"))
    args = ap.parse_args()

    target_dir = os.path.join(args.models_dir, "big_lama", "v1")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "big-lama.pt")

    # 1) If already present, done
    if os.path.isfile(target_path):
        print(f"Found existing LaMa at {target_path}")
        print(f"sha256={sha256_file(target_path)}")
        return

    # 2) Try manual ingestion
    manual_candidates = []
    for sub in ("lama", "lama_downloads"):
        cand = os.path.join(args.manual_dir, sub, "big-lama.pt")
        manual_candidates.append(cand)
    for cand in manual_candidates:
        if os.path.isfile(cand):
            shutil.copy2(cand, target_path)
            print(f"Copied manual LaMa from {cand} -> {target_path}")
            print(f"sha256={sha256_file(target_path)}")
            return

    # 3) Try via HF programmatic download (requires acceptance + token)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    fp = try_hf_download(target_dir, token)
    if fp and os.path.isfile(fp):
        final_path = target_path
        if os.path.abspath(fp) != os.path.abspath(final_path):
            shutil.copy2(fp, final_path)
        print(f"Downloaded LaMa via HF: {final_path}")
        print(f"sha256={sha256_file(final_path)}")
        return

    print("ERROR: Could not ingest LaMa big-lama.pt. Place the file at manual_downloads/lama/big-lama.pt and re-run.", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()

