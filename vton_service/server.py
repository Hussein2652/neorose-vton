from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image

app = FastAPI(title="StableVITON Expert Service")


def _save_image(img: Image.Image, path: str, mode: str = 'RGB') -> None:
    img.convert(mode).save(path)


def _build_onepair_dataset(user_im: Image.Image, garment_im: Image.Image, mask_im: Optional[Image.Image]) -> tuple[str, str, str]:
    """
    Build a minimal StableVITON dataset structure for a single pair under a temp dir.
    Returns (data_root_dir, im_name, cloth_name).
    """
    import tempfile
    import numpy as np
    import cv2  # type: ignore

    td = tempfile.mkdtemp(prefix="stv_", dir=None)
    root = os.path.join(td)
    test = os.path.join(root, 'test')
    os.makedirs(test, exist_ok=True)
    # Needed dirs
    d_agn = os.path.join(test, 'agnostic-v3.2')
    d_agnm = os.path.join(test, 'agnostic-mask')
    d_img = os.path.join(test, 'image')
    d_dp = os.path.join(test, 'image-densepose')
    d_cloth = os.path.join(test, 'cloth')
    d_cmask = os.path.join(test, 'cloth-mask')
    d_gtcm = os.path.join(test, 'gt_cloth_warped_mask')
    for d in (d_agn, d_agnm, d_img, d_dp, d_cloth, d_cmask, d_gtcm):
        os.makedirs(d, exist_ok=True)

    im_name = 'user.jpg'
    cloth_name = 'garment.png'
    # Save base user/cloth
    _save_image(user_im, os.path.join(d_img, im_name))
    # Cloth image saved as PNG
    _save_image(garment_im, os.path.join(d_cloth, cloth_name), mode='RGBA')

    # Cloth mask from alpha if present, else Otsu on luminance
    gar_np = np.array(garment_im.convert('RGBA'))
    alpha = gar_np[:, :, 3]
    if alpha.max() == 0 or np.mean(alpha) < 1:
        gray = cv2.cvtColor(gar_np[:, :, :3], cv2.COLOR_RGB2GRAY)
        _, cmask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        cmask = alpha
    cv2.imwrite(os.path.join(d_cmask, cloth_name), cmask)

    # Person mask: use provided mask if available; else full white
    if mask_im is None:
        mask_im = Image.new('L', user_im.size, color=255)
    _save_image(mask_im, os.path.join(d_agnm, im_name.replace('.jpg', '_mask.png')), mode='L')

    # Build agnostic-v3.2: blur masked user as a coarse canonical input
    u_np = np.array(user_im.convert('RGB'))
    m_np = np.array(mask_im.convert('L'))
    m01 = (m_np > 127).astype(np.uint8)
    fg = u_np * m01[:, :, None] + 128 * (1 - m01)[:, :, None]
    fg_blur = cv2.GaussianBlur(fg, (21, 21), 0)
    cv2.imwrite(os.path.join(d_agn, im_name), cv2.cvtColor(fg_blur, cv2.COLOR_RGB2BGR))

    # DensePose placeholder: zeros image (StableVITON uses it as a hint)
    dp = np.zeros_like(u_np, dtype=np.uint8)
    cv2.imwrite(os.path.join(d_dp, im_name), cv2.cvtColor(dp, cv2.COLOR_RGB2BGR))

    # gt_cloth_warped_mask: zeros for test
    z = np.zeros_like(m_np, dtype=np.uint8)
    cv2.imwrite(os.path.join(d_gtcm, im_name), z)

    # Write pairs file
    pairs = os.path.join(root, 'test_pairs.txt')
    with open(pairs, 'w', encoding='utf-8') as f:
        f.write(f"{im_name} {cloth_name}\n")
    return root, im_name, cloth_name


def _infer_third_party(user_im: Image.Image, garment_im: Image.Image, mask_im: Optional[Image.Image]) -> Optional[Image.Image]:
    """
    Placeholder hook to call third_party.stableviton code if present.
    For now, do a simple alpha paste if code is not present.
    """
    # Preferred: import a Python hook if present
    try:
        import importlib
        mod = importlib.import_module('third_party.stableviton.infer')
        if hasattr(mod, 'run_infer'):
            out = mod.run_infer(user_im, garment_im, mask_im)  # type: ignore
            return out
    except Exception:
        pass
    # Fallback: run a CLI if defined via env STABLEVITON_INFER_CMD
    try:
        import subprocess, tempfile, sys, traceback
        cmd_tpl = os.environ.get('STABLEVITON_INFER_CMD')
        if cmd_tpl:
            # Build one-pair dataset
            root, _im, _cl = _build_onepair_dataset(user_im, garment_im, mask_im)
            with tempfile.TemporaryDirectory() as td:
                o = os.path.join(td, 'out')
                os.makedirs(o, exist_ok=True)
                weights_dir = os.environ.get('STABLEVITON_WEIGHTS_DIR', os.path.join('storage','models','stableviton','weights'))
                cmd = [s.replace('{DATA_ROOT}', root).replace('{OUT}', o).replace('{WEIGHTS_DIR}', weights_dir) for s in cmd_tpl.split(' ') if s]
                try:
                    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    if os.environ.get('STABLEVITON_VERBOSE') == '1':
                        print('[StableVITON CLI stdout]', r.stdout)
                        print('[StableVITON CLI stderr]', r.stderr)
                except subprocess.CalledProcessError as e:
                    print('[StableVITON CLI failed]', e, file=sys.stderr)
                    if e.stdout:
                        print('[stdout]', e.stdout, file=sys.stderr)
                    if e.stderr:
                        print('[stderr]', e.stderr, file=sys.stderr)
                    if os.environ.get('STABLEVITON_STRICT', '0') == '1':
                        return None
                    # else continue to fallback below
                # Find result image in save_dir (pair or unpair)
                cand = None
                for sub in ('pair', 'unpair', ''):
                    p = os.path.join(o, sub)
                    if os.path.isdir(p):
                        for fn in os.listdir(p):
                            if fn.lower().endswith(('.jpg', '.png')):
                                cand = os.path.join(p, fn)
                                break
                        if cand:
                            break
                if cand and os.path.exists(cand):
                    return Image.open(cand).convert('RGB')
    except Exception:
        if os.environ.get('STABLEVITON_STRICT', '0') == '1':
            return None
        traceback.print_exc()
    try:
        # Fallback: alpha paste centered (until runtime code is present)
        if os.environ.get('STABLEVITON_FALLBACK', '1') == '1':
            user = user_im.convert('RGB')
            g = garment_im.convert('RGBA')
            w, h = user.size
            gw, gh = g.size
            x = (w - gw) // 2
            y = (h - gh) // 2
            comp = user.copy()
            comp.paste(g, (x, y), g)
            return comp
        return None
    except Exception:
        return None


@app.post('/infer')
async def infer(user: UploadFile = File(...), garment: UploadFile = File(...), mask: UploadFile | None = File(None)):
    try:
        ui = Image.open(user.file).convert('RGB')
        gi = Image.open(garment.file).convert('RGBA')
        mi = Image.open(mask.file).convert('L') if mask is not None else None
        out = _infer_third_party(ui, gi, mi)
        if out is None:
            return JSONResponse({"error": "inference unavailable"}, status_code=503)
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        return Response(buf.getvalue(), media_type='image/png')
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
