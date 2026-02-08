from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class ScriptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lines: str = Field(..., description="Multiline script. Each non-empty line becomes one cue.")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: str = Field(...)
    manifest_json: str = Field(...)
    write_text_files: bool = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script: ScriptConfig = Field(...)
    output: OutputConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

