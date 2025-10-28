from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageFilter
import io

app = FastAPI(title="VFR Remote Worker")


@app.post("/img2img")
async def img2img(soft: UploadFile = File(...)):
    im = Image.open(soft.file).convert("RGB")
    out = im.filter(ImageFilter.DETAIL)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/vton")
async def vton(user: UploadFile = File(...), garment: UploadFile = File(...)):
    u = Image.open(user.file).convert("RGB")
    g = Image.open(garment.file).convert("RGBA")
    # Simple center overlay
    w, h = u.size; gw, gh = g.size
    x = (w - gw) // 2; y = (h - gh) // 2
    out = u.copy(); out.paste(g, (x, y), g)
    buf = io.BytesIO(); out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

