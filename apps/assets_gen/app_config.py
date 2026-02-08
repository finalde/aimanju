from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from libs.common.yaml_config import load_yaml_mapping


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sdxl_base: str = Field(..., description="HF repo id or local diffusers folder (recommended).")
    device: str = Field(..., description='"cuda" or "cpu".')


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(...)
    height: int = Field(...)
    steps: int = Field(...)
    guidance: float = Field(...)
    negative_prompt: str = Field(...)
    img2img_strength: float = Field(..., description="For mode=img2img: 0..1 (lower keeps identity more).")


class FaceCropConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(...)
    output_size: int = Field(..., description="Square size (px) for the face reference image.")
    min_face_size: int = Field(..., description="Minimum face size (px) to accept; fallback to center crop.")
    scale: float = Field(..., description="Expand detected face box by this scale factor.")


class IpAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = Field(...)
    path: str = Field(..., description="Folder containing sdxl_models/...")
    weight_name: str = Field(...)
    scale: float = Field(...)
    reference_image: str = Field(..., description='Optional reference image path; "" disables.')


class ControlNetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="Diffusers-format ControlNet folder (contains config.json + weights).")
    conditioning_scale: float = Field(...)


class ViewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(...)
    seed: int = Field(...)
    strength: Optional[float] = Field(
        ...,
        description='For mode=img2img: optional per-view strength override (0..1). Use null to use generation.img2img_strength.',
    )
    prompt_suffix: str = Field(
        ...,
        description='Optional extra prompt text for this view. "" disables.',
    )
    reference_image: str = Field(
        ...,
        description='For mode=img2img: reference image path. Use "generated:front" or "generated:front_face".',
    )
    control_image: str = Field(
        ...,
        description='Path to conditioning image (pose/depth/canny), PNG recommended. For ip_adapter_only/img2img, set "".',
    )


class CharacterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["controlnet", "ip_adapter_only", "img2img"] = Field(...)
    id: str = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    controlnet: Optional[ControlNetConfig] = Field(...)
    views: List[ViewConfig] = Field(...)


class BackgroundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(...)
    name: str = Field(...)
    description: str = Field(...)
    controlnet: ControlNetConfig = Field(...)
    views: List[ViewConfig] = Field(...)


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_dir: str = Field(...)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: BaseConfig = Field(...)
    generation: GenerationConfig = Field(...)
    face_crop: FaceCropConfig = Field(...)
    ip_adapter: IpAdapterConfig = Field(...)
    characters: List[CharacterConfig] = Field(...)
    backgrounds: List[BackgroundConfig] = Field(...)
    output: OutputConfig = Field(...)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        return cls.model_validate(load_yaml_mapping(path))

