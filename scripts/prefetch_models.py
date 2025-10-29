#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import yaml
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.artifacts import ensure_artifact, ArtifactSpec  # noqa: E402


def sha256_file(path: str, chunk: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def snapshot_hf(repo_id: str, models_dir: str) -> str:
    from huggingface_hub import snapshot_download
    import re

    def san(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_.-]+', '-', s)

    local_dir = os.path.join(models_dir, 'snapshots', san(repo_id))
    os.makedirs(local_dir, exist_ok=True)
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    snapshot_download(repo_id=repo_id, token=token, local_files_only=False, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir


def main() -> None:
    ap = argparse.ArgumentParser(description='Prefetch models and HF snapshots once and record registry')
    ap.add_argument('--config', default=os.environ.get('MODELS_CONFIG', 'configs/models.yaml'))
    ap.add_argument('--models-dir', default=os.environ.get('MODELS_DIR', 'storage/models'))
    ap.add_argument('--registry', default=None, help='Path to write registry.json (default: <models-dir>/registry.json)')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    os.makedirs(args.models_dir, exist_ok=True)
    reg_path = args.registry or os.path.join(args.models_dir, 'registry.json')
    registry: dict = {"artifacts": [], "snapshots": [], "generated_at": datetime.utcnow().isoformat()}

    # Artifacts
    for a in cfg.get('artifacts', []) or []:
        spec = ArtifactSpec(
            name=a.get('name'),
            version=a.get('version'),
            sha256=a.get('sha256'),
            url=a.get('url'),
            s3_uri=a.get('s3_uri'),
            unpack=bool(a.get('unpack', False)),
        )
        local, downloaded = ensure_artifact(spec)
        size = 0
        if os.path.isdir(local):
            for root, _dirs, files in os.walk(local):
                for fn in files:
                    size += os.path.getsize(os.path.join(root, fn))
        else:
            size = os.path.getsize(local)
        entry = {
            "name": spec.name,
            "version": spec.version,
            "local": local,
            "size": size,
            "downloaded": downloaded,
        }
        if spec.sha256 and os.path.isfile(local):
            entry["sha256_match"] = (sha256_file(local).lower() == spec.sha256.lower())
        registry["artifacts"].append(entry)

    # HF snapshots
    for repo_id in cfg.get('hf_models', []) or []:
        local_dir = snapshot_hf(repo_id, args.models_dir)
        registry["snapshots"].append({"repo": repo_id, "local": local_dir})

    with open(reg_path, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)
    print(f'Wrote registry: {reg_path}')


if __name__ == '__main__':
    main()
