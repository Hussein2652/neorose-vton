from __future__ import annotations

import os
from typing import Optional


def compute_face_embedding(image_path: str) -> Optional[object]:
    """
    Best-effort face embedding via insightface if available.
    Returns a numpy array embedding or None.
    Requires: `insightface` and `onnxruntime` installed and an AntelopeV2 model bundle present under MODELS_DIR.
    """
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        return None

    models_dir = os.environ.get("MODELS_DIR", "storage/models")
    # Probe common locations
    cand_roots = [
        os.path.join(models_dir, "instantid_antelopev2_bundle"),
        os.path.join(models_dir, "antelopev2"),
    ]
    root = None
    for r in cand_roots:
        if os.path.isdir(r):
            root = r
            break
    if root is None:
        return None
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name="antelopev2", root=root, providers=providers, allowed_modules=["detection", "recognition"])  # type: ignore
        app.prepare(ctx_id=0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else -1)
        img = Image.open(image_path).convert("RGB")
        import numpy as np  # type: ignore
        arr = np.array(img)[:, :, ::-1]  # BGR for insightface
        faces = app.get(arr)
        if not faces:
            return None
        # Use the largest face
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding  # L2-normalized
        return emb.astype("float32")
    except Exception:
        return None

