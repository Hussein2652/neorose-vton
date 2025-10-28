Awesome—here’s a **no-placeholders, production list** of the **exact repos / model IDs / weight names** you asked for, plus **finisher constants**, and **licensing notes**. Where a project does **not publish an official SHA256**, I call that out explicitly (you can compute and pin it in your artifact registry at download time).

---

# 1) Segmentation (SCHP / CIHP / LIP)

**Preferred model:** SCHP (Self-Correction for Human Parsing) — use the **CIHP** label space for apparel (19 classes).

* **Repo (mirrors):**

  * GoGoDuck912/Self-Correction-Human-Parsing (includes trained LIP/CIHP weights). ([GitHub][1])
  * andrewjong/Self-Correction-Human-Parsing-Mirrored-Dirs (duplicate hosting of weights). ([GitHub][2])
* **Weights (file names you’ll fetch):**

  * `exp-schp-201908261155-lip.pth` (LIP)
  * `exp-schp-201908301523-atr.pth` (if you ever use ATR)
  * `exp-schp-201908261155-cihp.pth` (CIHP)
* **Class mapping (CIHP 19 classes):** face, hair, torso_skin, upperclothes, right_arm, left_arm, pants, coat, left_shoe, right_shoe, right_leg, left_leg, hat, dress, socks, sunglasses, skirt, scarf, glove. (Use these to build UV-aligned masks post SMPL-X.) ([Dataset Ninja][3])
* **SHA256:** **Not officially published** by authors; **compute on download** and store in your registry.

> Tip: export masks as PNG + JSON label map; then warp to UV after PIXIE fit.

---

# 2) Depth (ZoeDepth)

**Chosen variant:** ZoeDepth **ZoeD-M12-NK** (robust indoor/outdoor).

* **Repo:** `isl-org/ZoeDepth` (MIT). Note: repo is now archived (read-only) but weights remain. ([GitHub][4])
* **Where to fetch:** Use the **Releases** or torch.hub pull from `isl-org/ZoeDepth`. ([GitHub][5])
* **SHA256:** Not published; **compute on download**.

---

# 3) Normals + Edges (annotators)

* **Normals:** ControlNet “normalbae” preprocessor via **controlnet-aux** (PyPI). ([PyPI][6])

  * You’ll call the annotator to **generate a Normal map** (no separate weights to fetch beyond the package’s internal pull).
* **Edges (Seams):** **HED** detector (via controlnet-aux) for strong seam edges. ([PyPI][6])
* **(Optional) LineArt:** Available in ComfyUI ControlNet aux packs (useful for crisp hems/collars). ([GitHub][7])
* **SHA256:** Not published for annotator bundles; **compute on first model cache**.

---

# 4) Identity adapters

* **InstantID** (face identity guidance; tuning-free):

  * **Repo:** `instantX-research/InstantID` (includes model + usage). ([GitHub][8])
  * **SHA256:** Not published; compute & pin.

* **IP-Adapter FaceID-PlusV2** (higher-fidelity face lock):

  * **Model card / files:** `h94/IP-Adapter-FaceID` (look for **FaceID-Plus** and **FaceID-Plus-V2** safetensors). ([Hugging Face][9])
  * **SHA256:** Not published; compute & pin.

> Use **both** (InstantID + FaceID-PlusV2) and enforce a **face similarity threshold** (≥ 0.80) in QA.

---

# 5) SMPL-X fitting

* **PIXIE (preferred):**

  * **Repo:** `yfeng95/PIXIE` (single-image SMPL-X with face & hands; returns textures). ([GitHub][10])
  * **Project site:** pixie.is.tue.mpg.de (docs & downloads). ([pixie.is.tue.mpg.de][11])
* **SMPLify-X (fallback / refinement):** `vchoutas/smplify-x`. ([GitHub][12])
* **SMPL-X body model:** get from SMPL-X official site (license-gated; you will host internally). ([pixie.is.tue.mpg.de][11])
* **SHA256:** Not published; compute & pin for **SMPL-X models** and **PIXIE checkpoints**.

---

# 6) Garment reconstruction + UV inpainting + fabric classifier

* **Silhouette carving + depth prior:** Use your own pipeline with **ZoeDepth** + **SCHP** masks + **HED** seams; no canonical “single repo” exists for 2–3 photo garment recon—this is an **algorithmic module** using the above annotators (no extra weights).
* **UV unwrapping:** tool-side (Blender or your own unwrap; no weights).
* **UV inpainting:** **LaMa**

  * **Repo:** `advimman/lama` (official). ([GitHub][13])
  * **Project page:** (paper & details). ([advimman.github.io][14])
  * **SHA256:** Not published; compute & pin.
* **Fabric classifier:** lightweight CNN you train in-house (e.g., ResNet-18) on your curated fabric crops (denim/knit/satin/leather/printed). **No third-party weights**—ship your own (commercial-safe).

---

# 7) Cloth simulation

* **Coarse sim:** **Taichi/ARCSim-lite** approach (implement internally; no public “weights”).
* **Refinement:** differentiable contact/thickness refinement (internal).
* **NVIDIA Flex** is an alternative library if you go CUDA route; again **no weights**.

  * (These are **engine/runtime builds**, not trained models.)
* **SHA256:** N/A (no checkpoints). Build artifacts can still be hashed for provenance.

---

# 8) Gaussian-splat renderer (2.5D prior)

* **Official repo:** `graphdeco-inria/gaussian-splatting` (authors’ implementation). ([GitHub][15])
* **Project page & materials:** fungraph page (paper, resources). ([repo-sam.inria.fr][16])
* **Weights:** Not needed for your use; you will **train / optimize splats per input** (or render from reconstructed geometry).
* **SHA256:** N/A (no fixed model; you’ll hash output assets if you cache them).

---

# 9) Matting (BGMv2 vs RVM)

* **BGMv2 (Background Matting V2)** — **preferred** if a clean background plate is available.

  * **Repo:** `PeterL1n/BackgroundMattingV2` (official). ([GitHub][17])
  * (Weights are linked from the repo; download and pin.)
* **RVM (Robust Video Matting)** — acceptable fallback when background plate is **not** available; quality is often slightly below BGMv2 with a plate (author confirms BGMv2 > RVM when bg image exists). ([GitHub][18])
* **SHA256:** Not published; compute & pin.

---

# 10) Finisher stack (SDXL / FLUX) + ControlNets

## 10.1 Base models

* **SDXL 1.0 (base + refiner)** — **stabilityai/stable-diffusion-xl-base-1.0** and **…-refiner-1.0**. (Apache-2.0). ([Hugging Face][19])
* **FLUX.1 (dev/schnell)** — **black-forest-labs/FLUX.1-dev** and **FLUX.1-schnell** (check license before commercial use). ([GitHub][20])
* **SHA256:** Not published; compute & pin specific **revision hashes** (Hugging Face commit IDs) in your registry for reproducibility.

## 10.2 ControlNet weights (SD1.5)

* **OpenPose:** `lllyasviel/sd-controlnet-openpose`. ([Hugging Face][21])
* **Depth:** `lllyasviel/sd-controlnet-depth` (MiDaS-conditioned). ([Hugging Face][22])
* **Normals:** `lllyasviel/control_v11p_sd15_normalbae`. ([Hugging Face][23])
* **Edges (HED):** `lllyasviel/sd-controlnet-hed` (listed on model index). ([Hugging Face][24])
* **Segmentation:** (for SD1.5) use the **seg** ControlNet from ControlNet suite. ([Hugging Face][25])
* **SHA256:** Not published; compute & pin.

## 10.3 ControlNet weights (SDXL native)

* **Union SDXL (multi-control):** `xinsir/controlnet-union-sdxl-1.0` (Apache-2.0). ([Hugging Face][26])
* (You can also use xinsir’s **openpose-sdxl** and **depth-sdxl** individually if you prefer separate controls.) ([Hugging Face][27])
* **SHA256:** Not published; compute & pin.

## 10.4 Annotators package

* **controlnet-aux** (HED, normalbae, OpenPose proxy, etc., auto-pulls detector weights). ([PyPI][6])

## 10.5 Post-processing

* **Real-ESRGAN:** `xinntao/Real-ESRGAN` (use `realesrgan-x4plus`). ([GitHub][20])
* **CodeFormer:** `sczhou/CodeFormer`. ([Hugging Face][26])
* **SHA256:** Not published; compute & pin.

---

# 11) Finisher tuning constants (Tier-Ω defaults)

Use the **same constants** for SDXL and FLUX img2img unless noted.

```
# Resolution & sampler
RES_LONG=1408
STEPS=40
CFG=6.2
SAMPLER=DPM++_2M_Karras

# Denoise (main img2img + optional passes)
IMG2IMG_DENOISE=0.20         # 0.16–0.24 by garment
REFINER_DENOISE=0.14
TILEPASS_DENOISE=0.12
TILEPASS_STEPS=16            # enable only for knits/denim/embroidery

# ControlNet scales (when using separate SD1.5 CNs)
CTRL_POSE=0.55
CTRL_DEPTH=0.65
CTRL_NORMAL=0.45
CTRL_SEG=0.78
CTRL_EDGE=0.35

# If using xinsir/controlnet-union-sdxl-1.0
UNION_POSE=0.60
UNION_DEPTH=0.60
UNION_SEG=0.75
UNION_NORMAL=0.40
UNION_EDGE=0.35

# Adapters / identity
ADAPT_GARMENT=1.00           # IP-Adapter (Image-Plus) on garment photo
ADAPT_FACEID=0.85            # IP-Adapter FaceID-PlusV2 (with InstantID)

# Identity guardrail
MIN_FACE_SIM=0.80            # if below: +0.05 FACEID, -0.02 denoise, +5 steps
```

---

# 12) Licensing constraints / notes

* **SDXL base/refiner:** Apache-2.0; commercial-friendly. **Pin the HF revision.** ([Hugging Face][19])
* **FLUX.1**: check **black-forest-labs** license; some variants/dev builds may have restrictions—**review before commercial use**. ([GitHub][20])
* **ControlNet models (lllyasviel)**: typically MIT/Apache-style per model card; respect each card. ([Hugging Face][21])
* **xinsir SDXL Union**: Apache-2.0 (good for commercial). ([Hugging Face][28])
* **ZoeDepth**: MIT (repo archived but license stands). ([GitHub][4])
* **SCHP**: research code; model weights hosted on GitHub mirrors—**verify permissible commercial usage** or train your own parser on CIHP/LIP if needed. ([GitHub][1])
* **PIXIE / SMPL-X**: **license-gated** assets (non-redistributable). You must accept terms and **host internally**. ([pixie.is.tue.mpg.de][11])
* **LaMa / Real-ESRGAN / CodeFormer / BGMv2**: check respective repos (LaMa often under Apache-2.0; Real-ESRGAN under BSD-style; CodeFormer under BSD-3-Clause; BGMv2 under custom but permissive). Verify at integration. ([GitHub][13])

---

## What you’ll get after you pin these:

* I/O-clean **providers** for: segmentation, pose, depth, normals, edges, identity, SMPL-X fit, garment UV+mesh, drape, gaussian-splat render, finisher, matting.
* `configs/models.yaml` populated with **model IDs, file names, and (your) SHA256**.
* A `prefetch_models.py` that downloads and **verifies SHA256** before enabling the provider.
* Metrics (CLIP/LPIPS/face-ID/IoU) + policy retries wired, de-dupe hashing and EXIF/PII guard.

If you want, I’ll draft the **exact `models.yaml`** (with Hugging Face revision hashes for SDXL/ControlNets + placeholders for your computed SHA256 fields) and the **`prefetch_models.py`** that resolves all of the above in one go.

[1]: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing?utm_source=chatgpt.com "GoGoDuck912/Self-Correction-Human-Parsing"
[2]: https://github.com/andrewjong/Self-Correction-Human-Parsing-Mirrored-Dirs?utm_source=chatgpt.com "andrewjong/Self-Correction-Human-Parsing-Mirrored-Dirs"
[3]: https://datasetninja.com/cihp?utm_source=chatgpt.com "CIHP Dataset"
[4]: https://github.com/isl-org/ZoeDepth?utm_source=chatgpt.com "isl-org/ZoeDepth: Metric depth estimation from a single image"
[5]: https://github.com/isl-org/ZoeDepth/releases?utm_source=chatgpt.com "Releases · isl-org/ZoeDepth"
[6]: https://pypi.org/project/controlnet-aux/?utm_source=chatgpt.com "controlnet-aux"
[7]: https://github.com/Fannovel16/comfyui_controlnet_aux?utm_source=chatgpt.com "ComfyUI's ControlNet Auxiliary Preprocessors"
[8]: https://github.com/instantX-research/InstantID?utm_source=chatgpt.com "InstantID: Zero-shot Identity-Preserving Generation in ..."
[9]: https://huggingface.co/h94/IP-Adapter-FaceID?utm_source=chatgpt.com "h94/IP-Adapter-FaceID"
[10]: https://github.com/yfeng95/PIXIE?utm_source=chatgpt.com "yfeng95/PIXIE"
[11]: https://pixie.is.tue.mpg.de/?utm_source=chatgpt.com "PIXIE"
[12]: https://github.com/vchoutas/smplify-x?utm_source=chatgpt.com "vchoutas/smplify-x: Expressive Body Capture: 3D Hands, ..."
[13]: https://github.com/advimman/lama?utm_source=chatgpt.com "LaMa Image Inpainting, Resolution-robust Large Mask ..."
[14]: https://advimman.github.io/lama-project/?utm_source=chatgpt.com "Resolution-robust Large Mask Inpainting with Fourier ..."
[15]: https://github.com/graphdeco-inria/gaussian-splatting?utm_source=chatgpt.com "graphdeco-inria/gaussian-splatting"
[16]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field ..."
[17]: https://github.com/PeterL1n/BackgroundMattingV2?utm_source=chatgpt.com "Real-Time High-Resolution Background Matting"
[18]: https://github.com/PeterL1n/RobustVideoMatting/issues/1?utm_source=chatgpt.com "Any different with BackgroundMattingV2? · Issue #1"
[19]: https://huggingface.co/models?other=controlnet&utm_source=chatgpt.com "Models"
[20]: https://github.com/xinntao/Real-ESRGAN?utm_source=chatgpt.com "xinntao/Real-ESRGAN"
[21]: https://huggingface.co/lllyasviel/sd-controlnet-openpose?utm_source=chatgpt.com "lllyasviel/sd-controlnet-openpose"
[22]: https://huggingface.co/lllyasviel/sd-controlnet-depth?utm_source=chatgpt.com "lllyasviel/sd-controlnet-depth"
[23]: https://huggingface.co/lllyasviel/control_v11p_sd15_normalbae?utm_source=chatgpt.com "lllyasviel/control_v11p_sd15_normalbae"
[24]: https://huggingface.co/lllyasviel/sd-controlnet-canny?utm_source=chatgpt.com "lllyasviel/sd-controlnet-canny"
[25]: https://huggingface.co/lllyasviel/ControlNet?utm_source=chatgpt.com "lllyasviel/ControlNet"
[26]: https://huggingface.co/xinsir/controlnet-union-sdxl-1.0?utm_source=chatgpt.com "xinsir/controlnet-union-sdxl-1.0"
[27]: https://huggingface.co/xinsir?utm_source=chatgpt.com "xinsir (qi)"
[28]: https://huggingface.co/InvokeAI/Xinsir-SDXL_Controlnet_Union?utm_source=chatgpt.com "InvokeAI/Xinsir-SDXL_Controlnet_Union"
