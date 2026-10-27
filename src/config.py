import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_config(profile: str | None = None) -> dict[str, Any]:
    """Load config/<profile>.yaml. Profile comes from APP_ENV (default: local)."""
    name = profile or os.getenv("APP_ENV", "local")
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    return cfg
