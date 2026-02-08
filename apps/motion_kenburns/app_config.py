from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames_dir: str = Field(...)
    pattern: str = Field(...)
    start_number: int = Field(...)


class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seconds_per_image: float = Field(...)
    fps: int = Field(...)
    width: int = Field(...)
    height: int = Field(...)
    zoom_per_frame: float = Field(...)
    max_zoom: float = Field(...)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_path: str = Field(...)


class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: str = Field(...)
    pix_fmt: str = Field(...)
    crf: int = Field(...)
    preset: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: InputConfig = Field(...)
    motion: MotionConfig = Field(...)
    output: OutputConfig = Field(...)
    encoding: EncodingConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

