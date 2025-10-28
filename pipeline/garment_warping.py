from PIL import Image
import os


def warp_garment_to_user(garment_front_path: str, user_image_path: str, out_dir: str) -> str:
    """
    Stub garment warping.
    - Resizes garment to 60% of user image long side and centers it.
    - Outputs a simple composite as a soft render placeholder.
    """
    os.makedirs(out_dir, exist_ok=True)
    user = Image.open(user_image_path).convert("RGB")
    garment = Image.open(garment_front_path).convert("RGBA")

    # Resize garment maintaining aspect
    long_side = max(user.size)
    target_long = int(long_side * 0.6)
    g_w, g_h = garment.size
    if g_w >= g_h:
        new_w = target_long
        new_h = int(g_h * (target_long / g_w))
    else:
        new_h = target_long
        new_w = int(g_w * (target_long / g_h))
    garment = garment.resize((new_w, new_h), Image.LANCZOS)

    # Center garment
    x = (user.width - new_w) // 2
    y = (user.height - new_h) // 2
    composite = user.copy()
    composite.paste(garment, (x, y), garment)

    out_path = os.path.join(out_dir, "soft_render.png")
    composite.save(out_path)
    return out_path

