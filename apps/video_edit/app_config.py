from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frames_dir: str = Field(..., description="Directory containing frames")
    pattern: str = Field(..., description='printf-style pattern, e.g. "%02d.png"')
    start_number: int = Field(..., description="First frame number")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_path: str = Field(..., description="Output MP4 path")
    fps: int = Field(..., description="Output frames per second")


class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    codec: str = Field(..., description='Usually "libx264"')
    pix_fmt: str = Field(..., description='Usually "yuv420p"')
    crf: int = Field(..., description="x264 CRF")
    preset: str = Field(..., description="x264 preset")


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: InputConfig = Field(...)
    output: OutputConfig = Field(...)
    encoding: EncodingConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

