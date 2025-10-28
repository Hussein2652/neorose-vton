import os


def drape_garment(smplx_mesh_path: str | None, garment_mesh_path: str | None, out_dir: str) -> str:
    """
    Stub draping stage.
    - For now, acts as a pass-through; in a real system, this would run cloth sim.
    - Emits a placeholder NPZ path marker.
    """
    os.makedirs(out_dir, exist_ok=True)
    marker_path = os.path.join(out_dir, "drape_done.npz")
    with open(marker_path, "wb") as f:
        f.write(b"placeholder")
    return marker_path

