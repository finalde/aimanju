from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class ImageGenModelConfig(BaseModel):
    """
    Model selection for image generation.

    All fields are required (no defaults in code).
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description='Currently supported: "sdxl_diffusers"')
    path: str = Field(..., description="Model path or repo id (depends on type).")
    device: str = Field(..., description='"cuda" or "cpu"')


class ImageGenGenerationConfig(BaseModel):
    """
    Single-image generation request.

    All fields are required (no defaults in code).
    Optional behavior is controlled by explicit values (e.g. reference_image: "").
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(...)
    negative_prompt: str = Field(...)
    out_path: str = Field(...)

    width: int = Field(...)
    height: int = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)
    seed: int = Field(...)

    reference_image: str = Field(..., description='"" disables img2img, else a path')
    strength: float = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ImageGenModelConfig = Field(...)
    generation: ImageGenGenerationConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        data = load_yaml_mapping(path)
        return cls.model_validate(data)

