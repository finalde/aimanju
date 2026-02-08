from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


SubtitlesMode = Literal["burn_in", "soft", "none"]


class InputsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_mp4: str = Field(...)
    tts_manifest_json: str = Field(...)
    subtitles_srt: str = Field(...)


class SubtitlesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: SubtitlesMode = Field(...)
    force_style: str = Field(..., description='ASS force_style for burn-in; "" disables.')


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: str = Field(...)
    out_mp4: str = Field(...)
    voice_mp3: str = Field(...)


class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vcodec: str = Field(...)
    crf: int = Field(...)
    preset: str = Field(...)
    acodec: str = Field(...)
    abitrate: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: InputsConfig = Field(...)
    subtitles: SubtitlesConfig = Field(...)
    output: OutputConfig = Field(...)
    encoding: EncodingConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

