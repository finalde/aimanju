from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class StoryGenModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description='Currently supported: "sdxl_diffusers"')
    path: str = Field(..., description="Model path or repo id (depends on type).")
    device: str = Field(..., description='"cuda" or "cpu"')


class StoryboardRenderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_json: str = Field(...)
    out_dir: str = Field(...)

    width: int = Field(...)
    height: int = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)

    reference_image: str = Field(..., description='"" disables img2img, else a path')
    strength: float = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: StoryGenModelConfig = Field(...)
    storyboard: StoryboardRenderConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        data = load_yaml_mapping(path)
        return cls.model_validate(data)

