from __future__ import annotations

import os


class SMPLXGeometry:
    """Stub for SMPL-X geometry/draping provider.
    Ensures SMPL-X body/garment artifacts if configured, then returns a marker.
    """

    def __init__(self) -> None:
        self._ensured = False
        try:
            smplx_home = os.environ.get("SMPLX_HOME", os.path.join(os.environ.get("MODELS_DIR", os.path.join("storage","models")), "smplx"))
            # Simple readiness check: any SMPLX_NEUTRAL*.npz or any vposer* dir
            has_neutral = any(f.startswith("SMPLX_NEUTRAL") and f.endswith(".npz") for f in os.listdir(smplx_home)) if os.path.isdir(smplx_home) else False
            has_vposer = any(d.lower().startswith("vposer") and os.path.isdir(os.path.join(smplx_home, d)) for d in os.listdir(smplx_home)) if os.path.isdir(smplx_home) else False
            has_core = has_neutral or has_vposer
            if has_core:
                self._ensured = True
            else:
                # Try ensure via artifact env if provided
                from backend.app.artifacts import ensure_artifact, ArtifactSpec  # type: ignore
                url = os.environ.get("SMPLX_URL"); s3 = os.environ.get("SMPLX_S3_URI"); sha = os.environ.get("SMPLX_SHA256"); ver = os.environ.get("SMPLX_VERSION", "v1")
                if (url or s3) and sha:
                    ensure_artifact(ArtifactSpec(name="smplx", version=ver, sha256=sha, url=url, s3_uri=s3, unpack=True))
                    self._ensured = True
        except Exception:
            self._ensured = False

    def process(self, smplx_mesh_path: str | None, garment_mesh_path: str | None, out_dir: str) -> str:
        import os
        os.makedirs(out_dir, exist_ok=True)
        marker_path = os.path.join(out_dir, "drape_done.npz")
        with open(marker_path, "wb") as f:
            f.write(b"smplx_stub")
        return marker_path
