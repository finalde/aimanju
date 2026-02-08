from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class EngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(..., description='Currently supported: "piper"')
    voice_model: str = Field(..., description="Piper .onnx voice model path")
    voice_config: str = Field(..., description='"" to disable')
    speaker_id: int = Field(..., description="-1 to disable")
    length_scale: float = Field(..., description="Piper --length-scale (speech speed). >1 = slower")
    noise_scale: float = Field(..., description="Piper --noise-scale")
    noise_w_scale: float = Field(..., description="Piper --noise-w-scale")
    sentence_silence: float = Field(..., description="Piper --sentence-silence seconds")
    volume: float = Field(..., description="Piper --volume")
    no_normalize: bool = Field(..., description="Piper --no-normalize")
    cuda: bool = Field(..., description="Piper --cuda (if supported)")


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    script_manifest_json: str = Field(..., description="Path to outputs/scripts/*/manifest.json from apps.script_gen")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: str = Field(...)
    combined_wav: str = Field(...)
    manifest_json: str = Field(..., description="JSON manifest mapping lines -> wav/text")
    write_text_files: bool = Field(..., description="If true, write 001.txt next to 001.wav")
    write_mp3_files: bool = Field(..., description="If true, write 001.mp3 next to 001.wav (requires ffmpeg)")


class AudioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 0 means "auto" (do not pass --sample_rate; use model's default)
    sample_rate: int = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    engine: EngineConfig = Field(...)
    input: InputConfig = Field(...)
    output: OutputConfig = Field(...)
    audio: AudioConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

