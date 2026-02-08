from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and require the top-level to be a mapping/object.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object: {path}")

    return data

