from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str

    width: int
    height: int
    steps: int
    guidance: float

    seed: int

    # img2img (optional). Use None to disable.
    reference_image: Optional[Path]
    strength: Optional[float]


@dataclass(frozen=True)
class ImageGenerationResult:
    image_path: Path

