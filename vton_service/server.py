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
    # Preferred: import a Python hook if present
    try:
        import importlib
        mod = importlib.import_module('third_party.stableviton.infer')
        if hasattr(mod, 'run_infer'):
            out = mod.run_infer(user_im, garment_im)  # type: ignore
            return out
    except Exception:
        pass
    # Fallback: run a CLI if defined via env STABLEVITON_INFER_CMD
    try:
        import subprocess, tempfile
        cmd_tpl = os.environ.get('STABLEVITON_INFER_CMD')
        if cmd_tpl:
            with tempfile.TemporaryDirectory() as td:
                u = os.path.join(td, 'user.png'); g = os.path.join(td, 'garment.png'); o = os.path.join(td, 'out.png')
                user_im.save(u); garment_im.save(g)
                weights_dir = os.environ.get('STABLEVITON_WEIGHTS_DIR', os.path.join('storage','models','stableviton','weights'))
                cmd = [s.replace('{USER}', u).replace('{GARMENT}', g).replace('{OUT}', o).replace('{WEIGHTS_DIR}', weights_dir) for s in cmd_tpl.split(' ') if s]
                subprocess.run(cmd, check=True)
                if os.path.exists(o):
                    return Image.open(o).convert('RGB')
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
