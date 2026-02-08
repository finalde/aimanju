from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from .types import ImageGenerationResult


Device = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class SdxlControlNetConfig:
    path: str


@dataclass(frozen=True)
class SdxlIpAdapterConfig:
    path: str
    weight_name: str
    scale: float = 0.75


class SdxlControlNetImageGenerator:
    def __init__(
        self,
        *,
        base_model: str,
        controlnet: SdxlControlNetConfig,
        device: Device,
        ip_adapter: Optional[SdxlIpAdapterConfig] = None,
    ) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        self._torch = torch
        self._ControlNetModel = ControlNetModel
        self._Pipeline = StableDiffusionXLControlNetPipeline

        self._base_model = base_model
        self._controlnet_cfg = controlnet
        self._ip_adapter_cfg = ip_adapter
        self._device: Device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipe = self._load_pipeline()

    def _load_pipeline(self):
        controlnet = self._ControlNetModel.from_pretrained(
            self._controlnet_cfg.path, torch_dtype=self._dtype
        )

        base_path = Path(self._base_model)
        try_single_file = (
            base_path.exists()
            and base_path.is_file()
            and base_path.suffix.lower() in {".safetensors", ".ckpt"}
        )

        if try_single_file:
            # Not all diffusers versions support this.
            pipe = self._Pipeline.from_single_file(
                str(base_path),
                controlnet=controlnet,
                torch_dtype=self._dtype,
            )
        else:
            pipe = self._Pipeline.from_pretrained(
                self._base_model,
                controlnet=controlnet,
                torch_dtype=self._dtype,
                use_safetensors=True,
            )

        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)

        if self._ip_adapter_cfg is not None:
            pipe.load_ip_adapter(
                self._ip_adapter_cfg.path,
                subfolder="sdxl_models",
                weight_name=self._ip_adapter_cfg.weight_name,
            )
            pipe.set_ip_adapter_scale(self._ip_adapter_cfg.scale)

        return pipe

class SdxlControlNetImg2ImgGenerator:
    def __init__(
        self,
        *,
        base_model: str,
        controlnet: SdxlControlNetConfig,
        device: Device,
        ip_adapter: Optional[SdxlIpAdapterConfig] = None,
    ) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        self._torch = torch
        self._ControlNetModel = ControlNetModel
        self._Pipeline = StableDiffusionXLControlNetImg2ImgPipeline

        self._base_model = base_model
        self._controlnet_cfg = controlnet
        self._ip_adapter_cfg = ip_adapter
        self._device: Device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipe = self._load_pipeline()

    def _load_pipeline(self):
        controlnet = self._ControlNetModel.from_pretrained(
            self._controlnet_cfg.path, torch_dtype=self._dtype
        )

        base_path = Path(self._base_model)
        try_single_file = (
            base_path.exists()
            and base_path.is_file()
            and base_path.suffix.lower() in {".safetensors", ".ckpt"}
        )

        if try_single_file:
            pipe = self._Pipeline.from_single_file(
                str(base_path),
                controlnet=controlnet,
                torch_dtype=self._dtype,
            )
        else:
            pipe = self._Pipeline.from_pretrained(
                self._base_model,
                controlnet=controlnet,
                torch_dtype=self._dtype,
                use_safetensors=True,
            )

        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)

        if self._ip_adapter_cfg is not None:
            pipe.load_ip_adapter(
                self._ip_adapter_cfg.path,
                subfolder="sdxl_models",
                weight_name=self._ip_adapter_cfg.weight_name,
            )
            pipe.set_ip_adapter_scale(self._ip_adapter_cfg.scale)

        return pipe

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image: Path,
        init_image: Path,
        out_path: Path,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: float = 1.0,
        ip_adapter_image: Optional[Path] = None,
    ) -> ImageGenerationResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ctrl = Image.open(control_image).convert("RGB").resize((width, height))
        init = Image.open(init_image).convert("RGB").resize((width, height))
        ref_img: Optional[Image.Image] = None
        if ip_adapter_image is not None:
            ref_img = Image.open(ip_adapter_image).convert("RGB")

        g = self._torch.Generator(device=self._device).manual_seed(int(seed))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init,
            control_image=ctrl,
            strength=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=g,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        )
        if ref_img is not None:
            kwargs["ip_adapter_image"] = ref_img

        image = self._pipe(**kwargs).images[0]
        image.save(out_path)
        return ImageGenerationResult(image_path=out_path)

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image: Path,
        out_path: Path,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        ip_adapter_image: Optional[Path] = None,
        controlnet_conditioning_scale: float = 1.0,
    ) -> ImageGenerationResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ctrl = Image.open(control_image).convert("RGB").resize((width, height))
        ref_img: Optional[Image.Image] = None
        if ip_adapter_image is not None:
            ref_img = Image.open(ip_adapter_image).convert("RGB")

        g = self._torch.Generator(device=self._device).manual_seed(int(seed))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=ctrl,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=g,
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        )
        if ref_img is not None:
            kwargs["ip_adapter_image"] = ref_img

        image = self._pipe(**kwargs).images[0]
        image.save(out_path)
        return ImageGenerationResult(image_path=out_path)
