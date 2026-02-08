from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_model: str = Field(..., description="SDXL base model path or HF repo.")
    device: Literal["cuda", "cpu"] = Field(...)


class FaceIdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ip_adapter_faceid_sdxl: str = Field(..., description="ip-adapter-faceid_sdxl.bin path")
    insightface_model: str = Field(..., description='InsightFace model name, e.g. "buffalo_l" or "antelopev2".')
    insightface_home: str = Field(..., description="Folder containing insightface models (buffalo_l).")
    provider: Literal["CUDAExecutionProvider", "CPUExecutionProvider"] = Field(...)


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(...)
    negative_prompt: str = Field(...)


class IdentityGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(...)
    height: int = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)
    seed: int = Field(...)


class PoseControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    controlnet_path: str = Field(..., description="SDXL ControlNet (openpose) model folder path.")
    control_image: str = Field(..., description="OpenPose/DW pose image.")
    auto_generate: bool = Field(..., description="Auto-generate OpenPose image if missing.")
    auto_from: Literal["subject_image", "identity_image"] = Field(...)
    conditioning_scale: float = Field(...)
    strength: float = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)
    seed: int = Field(...)


class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_image: str = Field(..., description="Reference image for face ID embedding.")


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    identity_png: str = Field(...)
    posed_png: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(...)
    faceid: FaceIdConfig = Field(...)
    inputs: InputConfig = Field(...)
    prompt: PromptConfig = Field(...)
    identity: IdentityGenConfig = Field(...)
    pose: PoseControlConfig = Field(...)
    output: OutputConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))
