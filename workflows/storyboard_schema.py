from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Shot(BaseModel):
    shot_id: str = Field(..., description="e.g. 01, 02, 03")
    duration_seconds: Optional[float] = Field(
        default=None, description="Optional duration in seconds."
    )
    summary: str = Field(..., description="One-line purpose of the shot.")

    framing: Optional[str] = Field(
        default=None, description="e.g. close-up, medium, wide, low angle"
    )
    camera: Optional[str] = Field(
        default=None, description="e.g. static, slow push-in, handheld"
    )
    action: Optional[str] = Field(
        default=None, description="What happens visually in this shot."
    )
    dialogue: Optional[str] = Field(
        default=None, description="Spoken line(s) for this shot."
    )

    prompt: str = Field(..., description="Positive prompt for image generation.")
    negative_prompt: Optional[str] = Field(
        default=None, description="Negative prompt for image generation."
    )
    seed: Optional[int] = Field(default=None, description="Optional RNG seed.")


class ShotList(BaseModel):
    episode: str
    style: str = Field(
        default="realistic cinematic",
        description="High-level style string for the storyboard.",
    )
    global_prompt: str
    global_negative_prompt: str
    shots: List[Shot]

