from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = Field(..., description='ComfyUI base URL, e.g. "http://127.0.0.1:8188".')


class InputImagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_image: str = Field(..., description="Character reference image.")
    pose_image: str = Field(..., description="Pose control image (OpenPose or DW pose).")
    keypoints_image: str = Field(..., description="Face keypoints image for InstantID.")


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(...)
    negative_prompt: str = Field(...)


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(...)
    height: int = Field(...)
    seed: int = Field(...)
    steps: int = Field(...)
    cfg: float = Field(...)
    denoise: float = Field(...)


class WeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instantid_weight: float = Field(...)
    controlnet_strength: float = Field(...)
    ipadapter_weight: float = Field(...)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_dir: str = Field(...)


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workflow_json: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(...)
    workflow: WorkflowConfig = Field(...)
    inputs: InputImagesConfig = Field(...)
    prompt: PromptConfig = Field(...)
    generation: GenerationConfig = Field(...)
    weights: WeightsConfig = Field(...)
    output: OutputConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))
