from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_json: str = Field(...)
    frames_dir: str = Field(...)
    frame_ext: str = Field(...)


class TimingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_duration_seconds: float = Field(...)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edl_json: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: InputConfig = Field(...)
    timing: TimingConfig = Field(...)
    output: OutputConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

