from __future__ import annotations

import os
import sys
from pathlib import Path


def _download(url: str, to_path: Path, timeout: float = 30.0) -> bool:
    try:
        import requests  # type: ignore
    except Exception:
        return False
    try:
        to_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as r:  # type: ignore
            r.raise_for_status()
            with open(to_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):  # type: ignore
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def main() -> None:
    cfg_path = os.environ.get("DENSEPOSE_CFG", "/app/storage/models/densepose/densepose_rcnn_R_50_FPN_s1x.yaml")
    weights_path = os.environ.get("DENSEPOSE_WEIGHTS", "/app/storage/models/densepose/densepose_rcnn_R_50_FPN_s1x.pkl")
    cfg_url = os.environ.get(
        "DENSEPOSE_CFG_URL",
        "https://raw.githubusercontent.com/facebookresearch/DensePose/main/configs/densepose_rcnn_R_50_FPN_s1x.yaml",
    )
    weights_url = os.environ.get(
        "DENSEPOSE_WEIGHTS_URL",
        "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl",
    )

    ok_any = True
    if cfg_path and not os.path.exists(cfg_path):
        ok = _download(cfg_url, Path(cfg_path))
        print(f"[fetch_densepose] cfg {'downloaded' if ok else 'skip/failed'} -> {cfg_path}")
        ok_any = ok_any and ok
    if weights_path and not os.path.exists(weights_path):
        ok = _download(weights_url, Path(weights_path))
        print(f"[fetch_densepose] weights {'downloaded' if ok else 'skip/failed'} -> {weights_path}")
        ok_any = ok_any and ok

    # Exit code 0 even if downloads failed (expert will fallback to pseudo-DensePose)
    sys.exit(0)


if __name__ == "__main__":
    main()

