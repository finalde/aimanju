from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Shot(BaseModel):
    shot_id: str = Field(..., description="e.g. 01, 02, 03")

    # Keeping these optional to support partial shotlists.
    duration_seconds: Optional[float] = Field(default=None)
    summary: Optional[str] = Field(default=None)
    framing: Optional[str] = Field(default=None)
    camera: Optional[str] = Field(default=None)
    action: Optional[str] = Field(default=None)
    dialogue: Optional[str] = Field(default=None)

    prompt: str = Field(..., description="Positive prompt for image generation.")
    negative_prompt: Optional[str] = Field(default=None)
    seed: Optional[int] = Field(default=None)


class ShotList(BaseModel):
    episode: str
    global_prompt: str
    global_negative_prompt: str
    shots: List[Shot]

