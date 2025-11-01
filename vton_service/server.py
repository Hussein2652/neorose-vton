from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image, ImageFilter
import numpy as np
import math
from typing import Tuple

try:
    # Optional DensePose helper (detectron2). Falls back if unavailable.
    from .densepose_util import compute_densepose_image  # type: ignore
except Exception:  # noqa: BLE001
    compute_densepose_image = None  # type: ignore

app = FastAPI(title="StableVITON Expert Service")


def _save_image(img: Image.Image, path: str, mode: str = 'RGB') -> None:
    img.convert(mode).save(path)


def _best_size_from_input(size: Tuple[int, int], max_h: int = 640, max_w: int = 512) -> Tuple[int, int]:
    """Compute StableVITON target (H,W) from input, preserving aspect and snapping to /64.
    Caps by max_h/max_w and ensures a reasonable minimum to avoid degenerate latents.
    """
    w0, h0 = size
    if h0 <= 0 or w0 <= 0:
        return (512, 384)
    # Scale to fit caps while preserving aspect (note PIL size is (W,H))
    scale = min(max_h / h0, max_w / w0, 1.0)
    ht = int(round((h0 * scale) / 64.0) * 64)
    wt = int(round((w0 * scale) / 64.0) * 64)
    ht = max(128, min(max_h, ht))
    wt = max(128, min(max_w, wt))
    # Avoid too small latents like 8x6; bump to >= 192 each when possible
    if ht < 192 or wt < 192:
        ht = max(ht, 192)
        wt = max(wt, 192)
    return (ht, wt)


def _run_schp_mask(tmp_dir: str, user_path: str) -> tuple[Optional[str], Optional[str]]:
    try:
        import subprocess
        weights = os.environ.get('SCHP_MODEL_PATH', '')
        cmd_tpl = os.environ.get('SCHP_INFER_CMD')
        if not cmd_tpl or not weights or not os.path.exists(weights):
            # Log why SCHP is skipped
            try:
                import sys
                print(f"[SCHP] skipped: cmd_tpl set={bool(cmd_tpl)}, weights_exists={os.path.exists(weights) if weights else False}", file=sys.stderr)
            except Exception:
                pass
            return None, None
        out_dir = os.path.join(tmp_dir, 'schp')
        os.makedirs(out_dir, exist_ok=True)
        cmd = [s.replace('{WEIGHTS}', weights).replace('{IMAGE}', user_path).replace('{OUT}', out_dir) for s in cmd_tpl.split(' ') if s]
        r = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # scripts/schp_infer.py prints mask path; segmentation PNG is inside out_dir
        mask_path = None
        last = (r.stdout or '').strip().splitlines()
        if last:
            p = last[-1].strip()
            if os.path.exists(p):
                mask_path = p
        seg_path = None
        for fn in os.listdir(out_dir):
            if fn.lower().endswith('.png'):
                seg_path = os.path.join(out_dir, fn)
                break
        return mask_path, seg_path
    except Exception:
        return None, None


def _build_onepair_dataset(user_im: Image.Image, garment_im: Image.Image, mask_im: Optional[Image.Image]) -> tuple[str, str, str]:
    """
    Build a minimal StableVITON dataset structure for a single pair under a temp dir.
    Returns (data_root_dir, im_name, cloth_name).
    """
    import tempfile

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

    # Cloth mask from alpha if present, else Otsu on luminance; refine morphologically
    gar_rgba = garment_im.convert('RGBA')
    gar_np = np.array(gar_rgba)
    alpha = gar_np[:, :, 3].astype(np.uint8)
    if alpha.max() <= 0 or float(alpha.mean()) < 1.0:
        gray = np.array(garment_im.convert('L'))
        # Otsu threshold
        hist = np.bincount(gray.flatten(), minlength=256).astype(np.float64)
        total = gray.size
        cum_hist = np.cumsum(hist)
        cum_mean = np.cumsum(hist * np.arange(256))
        global_mean = cum_mean[-1] / (total + 1e-8)
        numerator = (global_mean * cum_hist - cum_mean) ** 2
        denominator = cum_hist * (total - cum_hist) + 1e-8
        sigma_b2 = numerator / denominator
        t = int(np.nanargmax(sigma_b2))
        cmask = (gray >= t).astype(np.uint8) * 255
    else:
        cmask = alpha
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cm = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, k, iterations=1)
        cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, k, iterations=1)
        cmask = cm
    except Exception:
        pass
    Image.fromarray(cmask).save(os.path.join(d_cmask, cloth_name))

    # Person mask: use provided mask, else SCHP if available, else full white
    if mask_im is None:
        # Save temp user image to disk for SCHP
        user_tmp = os.path.join(test, 'user_tmp.jpg')
        _save_image(user_im, user_tmp)
        mpath, seg_path = _run_schp_mask(root, user_tmp)
        if mpath and os.path.exists(mpath):
            try:
                mask_im = Image.open(mpath).convert('L')
            except Exception:
                mask_im = None
        if mask_im is None:
            mask_im = Image.new('L', user_im.size, color=255)
    _save_image(mask_im, os.path.join(d_agnm, im_name.replace('.jpg', '_mask.png')), mode='L')

    # Build agnostic-v3.2 using SCHP full segmentation if available: remove garment regions
    u_np = np.array(user_im.convert('RGB'))
    m_np = np.array(mask_im.convert('L'))
    m01 = (m_np > 127).astype(np.uint8)
    try:
        import cv2
        # If seg_path exists from SCHP, remove upper-body garment classes
        seg_map = None
        schp_dir = os.path.join(root, 'schp')
        # find any png in schp dir
        cand = None
        if os.path.isdir(schp_dir):
            for fn in os.listdir(schp_dir):
                if fn.lower().endswith('.png'):
                    cand = os.path.join(schp_dir, fn)
                    break
        if cand and os.path.exists(cand):
            seg_map = np.array(Image.open(cand).convert('P'))
        if seg_map is not None:
            # LIP indices for garments to erase
            GARMENT_IDS = {5, 6, 7, 10, 11, 12}
            garment_mask = np.isin(seg_map, list(GARMENT_IDS)).astype(np.uint8)
            # restrict to person area
            garment_mask = garment_mask * m01
            if garment_mask.sum() > 0:
                # Dilate for safety
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                garment_mask = cv2.dilate(garment_mask, k, iterations=2)
                # Inpaint removed garment area for agnostic
                inpaint_mask = (garment_mask * 255).astype(np.uint8)
                agn_bgr = cv2.cvtColor(u_np, cv2.COLOR_RGB2BGR)
                agn_bgr = cv2.inpaint(agn_bgr, inpaint_mask, 3, cv2.INPAINT_TELEA)
                agn_rgb = cv2.cvtColor(agn_bgr, cv2.COLOR_BGR2RGB)
            else:
                agn_rgb = u_np
        else:
            agn_rgb = u_np
        # Blur background only
        bg = 128 * (1 - m01)[:, :, None]
        fg = agn_rgb * m01[:, :, None] + bg
        fg_img = Image.fromarray(fg.astype(np.uint8), mode='RGB').filter(ImageFilter.GaussianBlur(radius=3))
    except Exception:
        fg = u_np * m01[:, :, None] + 128 * (1 - m01)[:, :, None]
        fg_img = Image.fromarray(fg.astype(np.uint8), mode='RGB').filter(ImageFilter.GaussianBlur(radius=6))
    fg_img.save(os.path.join(d_agn, im_name))

    # DensePose map: prefer detectron2 if available, else pseudo (u,v,dist)
    try:
        dp_path = os.path.join(d_dp, im_name)
        if compute_densepose_image is not None:
            dpi = compute_densepose_image(user_im)
            dpi.save(dp_path)
        else:
            import cv2  # available in the expert image
            h, w = m01.shape
            ys, xs = np.mgrid[0:h, 0:w]
            ys_nonzero, xs_nonzero = np.nonzero(m01)
            if len(xs_nonzero) > 0:
                x0, x1 = xs_nonzero.min(), xs_nonzero.max()
                y0, y1 = ys_nonzero.min(), ys_nonzero.max()
            else:
                x0, x1, y0, y1 = 0, w - 1, 0, h - 1
            u = (xs - x0) / max(1, (x1 - x0))
            v = (ys - y0) / max(1, (y1 - y0))
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            dist = cv2.distanceTransform((m01 * 255).astype(np.uint8), cv2.DIST_L2, 3)
            if dist.max() > 0:
                dist = dist / dist.max()
            dp_np = np.stack([u, v, dist], axis=2)
            dp_rgb = (dp_np * 255.0).astype(np.uint8)
            Image.fromarray(dp_rgb, mode='RGB').save(dp_path)
    except Exception:
        Image.new('RGB', user_im.size, color=(0, 0, 0)).save(os.path.join(d_dp, im_name))

    # gt_cloth_warped_mask: zeros for test
    z = Image.fromarray(np.zeros_like(m_np, dtype=np.uint8), mode='L')
    z.save(os.path.join(d_gtcm, im_name))

    # Write pairs file
    pairs = os.path.join(root, 'test_pairs.txt')
    with open(pairs, 'w', encoding='utf-8') as f:
        f.write(f"{im_name} {cloth_name}\n")
    # Optional: copy debug dataset to storage for inspection
    if os.environ.get('STABLEVITON_DEBUG') == '1':
        try:
            dbg_root = os.path.join('/app/storage', 'debug', os.path.basename(root))
            os.makedirs(dbg_root, exist_ok=True)
            import shutil
            for d in (test, os.path.dirname(pairs)):
                dd = os.path.join(dbg_root, os.path.basename(d))
                if os.path.isdir(d):
                    shutil.copytree(d, dd, dirs_exist_ok=True)
                else:
                    shutil.copy2(d, dbg_root)
        except Exception:
            pass
    return root, im_name, cloth_name


def _infer_third_party(user_im: Image.Image, garment_im: Image.Image, mask_im: Optional[Image.Image]) -> Optional[Image.Image]:
    """
    Placeholder hook to call third_party.stableviton code if present.
    For now, do a simple alpha paste if code is not present.
    """
    # Preferred: import a Python hook if present; if it returns None, treat as failure and continue
    try:
        import importlib
        mod = importlib.import_module('third_party.stableviton.infer')
        if hasattr(mod, 'run_infer'):
            out = mod.run_infer(user_im, garment_im, mask_im)  # type: ignore
            if out is not None:
                return out
            # Upstream returned None; fall through to CLI path
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
                # Adapt resolution to input while staying in safe caps (640x512) and /64 grid
                Ht, Wt = _best_size_from_input(user_im.size, max_h=int(os.environ.get('STV_MAX_H', '640')), max_w=int(os.environ.get('STV_MAX_W', '512')))
                for i, tok in enumerate(cmd):
                    if tok == '--img_H' and i + 1 < len(cmd):
                        cmd[i+1] = str(Ht)
                    if tok == '--img_W' and i + 1 < len(cmd):
                        cmd[i+1] = str(Wt)
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
                def _find_out(save_root: str) -> Optional[str]:
                    for sub in ('pair', 'unpair', ''):
                        pth = os.path.join(save_root, sub)
                        if os.path.isdir(pth):
                            for fn in os.listdir(pth):
                                if fn.lower().endswith(('.jpg', '.png')):
                                    return os.path.join(pth, fn)
                    return None
                cand = _find_out(o)
                if not cand:
                    # Retry with smaller safe sizes if nothing produced
                    for (hh, ww) in [(512, 384), (320, 256)]:
                        cmd2 = list(cmd)
                        for i, tok in enumerate(cmd2):
                            if tok == '--img_H' and i + 1 < len(cmd2):
                                cmd2[i+1] = str(hh)
                            if tok == '--img_W' and i + 1 < len(cmd2):
                                cmd2[i+1] = str(ww)
                        try:
                            r2 = subprocess.run(cmd2, check=True, capture_output=True, text=True)
                            if os.environ.get('STABLEVITON_VERBOSE') == '1':
                                print('[StableVITON CLI stdout retry]', r2.stdout)
                                print('[StableVITON CLI stderr retry]', r2.stderr)
                        except subprocess.CalledProcessError as e2:
                            if os.environ.get('STABLEVITON_VERBOSE') == '1':
                                print('[StableVITON CLI retry failed]', e2, file=sys.stderr)
                            continue
                        cand = _find_out(o)
                        if cand:
                            break
                if cand and os.path.exists(cand):
                    return Image.open(cand).convert('RGB')
        else:
            # No CLI template configured
            print('[StableVITON] STABLEVITON_INFER_CMD not set; skipping CLI', file=sys.stderr)
    except Exception as ex:
        # Always print exception for debugging
        try:
            print('[StableVITON] exception in CLI path:', ex, file=sys.stderr)
            import traceback as _tb
            _tb.print_exc()
        except Exception:
            pass
        if os.environ.get('STABLEVITON_STRICT', '0') == '1':
            return None
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


@app.get('/health')
def health():
    return {"status": "ok"}
