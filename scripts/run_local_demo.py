import argparse
import os
from pipeline.pipeline import VFRPipeline
from backend.app.config import settings


def main():
    parser = argparse.ArgumentParser(description="Run local VFR demo pipeline")
    parser.add_argument("--user", required=True, help="Path to user image")
    parser.add_argument("--garment", required=True, help="Path to garment front image")
    parser.add_argument("--garment_side", default=None, help="Optional garment side image")
    parser.add_argument("--out", required=True, help="Output image path")
    args = parser.parse_args()

    pipe = VFRPipeline.from_settings(settings)
    res = pipe.run(
        user_image_path=args.user,
        garment_front_path=args.garment,
        garment_side_path=args.garment_side,
    )

    # Copy result to desired output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(res.output_path, "rb") as src, open(args.out, "wb") as dst:
        dst.write(src.read())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

