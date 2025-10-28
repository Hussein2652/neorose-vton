from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from skimage.metrics import structural_similarity as ssim

_torch_ready = False
_lpips_ready = False
try:
    import torch
    import torchvision
    from torchvision import transforms
    _torch_ready = True
    try:
        import lpips  # type: ignore
        _lpips_ready = True
    except Exception:
        _lpips_ready = False
except Exception:
    _torch_ready = False


def _load_img_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _laplacian_var(im: Image.Image) -> float:
    # Approximate sharpness using Laplacian variance via PIL (no cv2 dependency here)
    # Convert to grayscale and apply edge-like filter
    g = ImageOps.grayscale(im)
    arr = np.asarray(g, dtype=np.float32)
    # Simple 3x3 Laplacian kernel
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d  # scikit-image installs scipy

    edges = convolve2d(arr, k, mode="same", boundary="symm")
    return float(edges.var())


def _hist_corr(a: Image.Image, b: Image.Image) -> float:
    # 3D RGB histogram correlation (normalized)
    ha = np.histogramdd(np.asarray(a).reshape(-1, 3), bins=8, range=[(0, 255)] * 3)[0]
    hb = np.histogramdd(np.asarray(b).reshape(-1, 3), bins=8, range=[(0, 255)] * 3)[0]
    ha = ha / (ha.sum() + 1e-8)
    hb = hb / (hb.sum() + 1e-8)
    num = (ha * hb).sum()
    den = np.linalg.norm(ha) * np.linalg.norm(hb) + 1e-8
    return float(num / den)


def _ssim(a: Image.Image, b: Image.Image) -> float:
    a_g = ImageOps.grayscale(a)
    b_g = ImageOps.grayscale(b)
    return float(ssim(np.asarray(a_g, dtype=np.float32), np.asarray(b_g, dtype=np.float32)))


def _lpips_distance(a: Image.Image, b: Image.Image) -> Optional[float]:
    if not _lpips_ready:
        return None
    import torch
    import lpips  # type: ignore
    # lpips expects [-1,1] normalized tensors (NCHW)
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    a_t = t(a).unsqueeze(0)
    b_t = t(b).unsqueeze(0)
    net = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        net = net.cuda(); a_t = a_t.cuda(); b_t = b_t.cuda()
    with torch.no_grad():
        d = net(a_t, b_t)
    return float(d.item())


def _clip_similarity(a: Image.Image, b: Image.Image) -> Optional[float]:
    # Try open-clip; fallback to ResNet50 embeddings cosine similarity
    try:
        import open_clip  # type: ignore
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        def emb(img):
            with torch.no_grad():
                t = preprocess(img).unsqueeze(0).to(device)
                feats = model.encode_image(t)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                return feats.squeeze(0).cpu().numpy()
        va = emb(a); vb = emb(b)
        num = float((va * vb).sum()); den = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return num / den
    except Exception:
        if not _torch_ready:
            return None
        import torch
        import torchvision.models as models
        from torchvision import transforms
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
        resnet.eval()
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        def emb(img):
            with torch.no_grad():
                t = preprocess(img).unsqueeze(0).to(device)
                feats = resnet.forward(t)
                feats = feats / torch.norm(feats, dim=-1, keepdim=True)
                return feats.squeeze(0).cpu().numpy()
        va = emb(a); vb = emb(b)
        num = float((va * vb).sum()); den = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        return num / den


@dataclass
class QAScores:
    garment_similarity: float
    identity_score: float
    aesthetics: float
    lpips: Optional[float] = None
    clip_sim: Optional[float] = None


def compute_qa(final_path: str, user_path: str, garment_path: str, work_root: Optional[str] = None) -> QAScores:
    final_im = _load_img_rgb(final_path)
    user_im = _load_img_rgb(user_path)
    garment_im = _load_img_rgb(garment_path)

    # Determine garment ROI from warp metadata if available
    roi_im: Optional[Image.Image] = None
    if work_root:
        meta_path = os.path.join(work_root, "warp", "warp_meta.json")
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                x, y, w, h = [int(meta[k]) for k in ("x", "y", "w", "h")]
                roi_im = final_im.crop((x, y, x + w, y + h))
            except Exception:
                roi_im = None
    if roi_im is None:
        # Fallback ROI: central crop
        W, H = final_im.size
        w = int(W * 0.6)
        h = int(H * 0.6)
        x = (W - w) // 2
        y = (H - h) // 2
        roi_im = final_im.crop((x, y, x + w, y + h))

    # Garment fidelity: histogram correlation; attempt CLIP similarity as well
    garment_similarity = _hist_corr(garment_im.resize(roi_im.size), roi_im)
    clip_sim = _clip_similarity(garment_im.resize(roi_im.size), roi_im)

    # Identity retention: SSIM between upper region of user and final image
    W, H = final_im.size
    box = (0, 0, W, int(H * 0.35))  # top 35%
    user_head = user_im.crop(box).resize((256, 256))
    final_head = final_im.crop(box).resize((256, 256))
    identity_score = _ssim(user_head, final_head)

    # Aesthetics: combine sharpness and contrast (simple proxy)
    sharp = _laplacian_var(final_im)
    contrast = float(np.std(np.asarray(ImageOps.grayscale(final_im), dtype=np.float32)) / 64.0)
    aesthetics = float(np.tanh((sharp / 200.0) + contrast))  # map to 0..1

    # Optional LPIPS distance (lower is better)
    lp = _lpips_distance(garment_im.resize(roi_im.size), roi_im)

    return QAScores(
        garment_similarity=garment_similarity,
        identity_score=identity_score,
        aesthetics=aesthetics,
        lpips=lp,
        clip_sim=clip_sim,
    )
