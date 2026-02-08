from __future__ import annotations

import json
from pathlib import Path

from .shotlist import ShotList


def load_shotlist_json(path: Path) -> ShotList:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShotList.model_validate(data)

