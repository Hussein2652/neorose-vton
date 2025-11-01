from __future__ import annotations

from backend.app.db import upsert_plan


def main() -> None:
    # Default plans: free (limit 30/mo), pro (300/mo), enterprise (unlimited)
    upsert_plan("free", default_backend="local", max_res_long="1344", monthly_limit=30, per_image_cost=0.0)
    upsert_plan("pro", default_backend="local", max_res_long="1536", monthly_limit=300, per_image_cost=0.05)
    upsert_plan("enterprise", default_backend="sdxl_controlnet", max_res_long="2048", monthly_limit=None, per_image_cost=0.10)
    print("Plans initialized.")


if __name__ == "__main__":
    main()
