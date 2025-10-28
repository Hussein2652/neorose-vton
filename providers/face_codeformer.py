from __future__ import annotations

import os
from typing import Optional
from PIL import Image


class CodeFormerRestorer:
    """Restore faces using CodeFormer if available; otherwise pass-through."""
    def __init__(self) -> None:
        try:
            import facexlib  # type: ignore
            import basicsr  # type: ignore
            self._ready = True
        except Exception:
            self._ready = False

    def process(self, image_path: str, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        if not self._ready:
            out = os.path.join(out_dir, os.path.basename(image_path))
            Image.open(image_path).save(out)
            return out
        # Placeholder; integrate real CodeFormer pipeline if libs present
        out = os.path.join(out_dir, os.path.basename(image_path))
        Image.open(image_path).save(out)
        return out

