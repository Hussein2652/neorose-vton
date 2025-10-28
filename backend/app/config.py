import os
import yaml
from typing import Any


CONFIG_PATH = os.environ.get("VFR_CONFIG", "configs/pipeline.yaml")


class Settings:
    def __init__(self) -> None:
        self._cfg: dict[str, Any] = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self._cfg = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        # Env wins over YAML
        env_key = key.upper().replace(".", "_")
        if env_key in os.environ:
            return os.environ[env_key]

        # Dot path lookup in YAML
        parts = key.split(".")
        cur: Any = self._cfg
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur


settings = Settings()

