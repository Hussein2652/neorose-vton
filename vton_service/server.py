from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image

app = FastAPI(title="StableVITON Expert Service")


def _infer_third_party(user_im: Image.Image, garment_im: Image.Image) -> Optional[Image.Image]:
    """
    Placeholder hook to call third_party.stableviton code if present.
    For now, do a simple alpha paste if code is not present.
    """
    try:
        import importlib
        mod = importlib.import_module('third_party.stableviton.infer')
        if hasattr(mod, 'run_infer'):
            out = mod.run_infer(user_im, garment_im)  # type: ignore
            return out
    except Exception:
        pass
    try:
        # Fallback: alpha paste centered (until runtime code is present)
        user = user_im.convert('RGB')
        g = garment_im.convert('RGBA')
        w, h = user.size
        gw, gh = g.size
        x = (w - gw) // 2
        y = (h - gh) // 2
        comp = user.copy()
        comp.paste(g, (x, y), g)
        return comp
    except Exception:
        return None


@app.post('/infer')
async def infer(user: UploadFile = File(...), garment: UploadFile = File(...)):
    try:
        ui = Image.open(user.file).convert('RGB')
        gi = Image.open(garment.file).convert('RGBA')
        out = _infer_third_party(ui, gi)
        if out is None:
            return JSONResponse({"error": "inference unavailable"}, status_code=503)
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        return Response(buf.getvalue(), media_type='image/png')
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

