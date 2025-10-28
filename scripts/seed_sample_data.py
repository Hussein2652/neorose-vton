from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
import os


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def create_user_image(path: str) -> None:
    w, h = 800, 1000
    im = Image.new("RGB", (w, h), (230, 230, 230))
    d = ImageDraw.Draw(im)
    # Simple stick figure silhouette
    d.ellipse((w//2-80, 80, w//2+80, 240), fill=(200, 200, 200))  # head
    d.rectangle((w//2-60, 240, w//2+60, 700), fill=(180, 180, 180))  # torso
    d.rectangle((w//2-120, 700, w//2-40, 980), fill=(160, 160, 160))  # left leg
    d.rectangle((w//2+40, 700, w//2+120, 980), fill=(160, 160, 160))  # right leg
    d.text((20, 20), "user.jpg (placeholder)", fill=(50, 50, 50))
    im.save(path)


def create_garment_image(path: str) -> None:
    w, h = 600, 600
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # Simple T-shirt shape
    body_color = (120, 160, 255, 230)
    d.rectangle((150, 150, 450, 500), fill=body_color)  # body
    d.polygon([(150, 150), (80, 260), (150, 260)], fill=body_color)  # left sleeve
    d.polygon([(450, 150), (520, 260), (450, 260)], fill=body_color)  # right sleeve
    d.text((10, 10), "garment_front.png (placeholder)", fill=(10, 10, 10, 255))
    im.save(path)


def main() -> None:
    ensure_dir("sample_data")
    user_path = os.path.join("sample_data", "user.jpg")
    garment_path = os.path.join("sample_data", "garment_front.jpg")
    if not os.path.exists(user_path):
        create_user_image(user_path)
        print(f"Created {user_path}")
    else:
        print(f"Exists {user_path}")
    if not os.path.exists(garment_path):
        # Save as JPEG for consistency
        tmp_png = os.path.join("sample_data", "garment_front.png")
        create_garment_image(tmp_png)
        Image.open(tmp_png).convert("RGB").save(garment_path, quality=92)
        os.remove(tmp_png)
        print(f"Created {garment_path}")
    else:
        print(f"Exists {garment_path}")


if __name__ == "__main__":
    main()

