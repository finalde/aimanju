from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from .types import ImageGenerationResult


Device = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class SdxlIpAdapterModel:
    """
    SDXL base model reference for IP-Adapter-only generation.

    - If `base_model` points to a `.safetensors`/`.ckpt`, load via `from_single_file`.
    - Otherwise treat it as a diffusers folder path or HF repo id.
    """

    base_model: str
    ip_adapter_path: str
    weight_name: str
    scale: float = 0.75


class SdxlIpAdapterImageGenerator:
    def __init__(self, model: SdxlIpAdapterModel, device: Device) -> None:
        import torch
        from diffusers import StableDiffusionXLPipeline

        self._torch = torch
        self._Pipeline = StableDiffusionXLPipeline

        self._model = model
        self._device: Device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipe = self._load_pipeline()

    def _load_pipeline(self):
        base_path = Path(self._model.base_model)
        is_single_file = base_path.exists() and base_path.is_file() and base_path.suffix.lower() in {
            ".safetensors",
            ".ckpt",
        }

        if is_single_file:
            pipe = self._Pipeline.from_single_file(str(base_path), torch_dtype=self._dtype)
        else:
            pipe = self._Pipeline.from_pretrained(
                self._model.base_model,
                torch_dtype=self._dtype,
                use_safetensors=True,
            )

        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)

        pipe.load_ip_adapter(
            self._model.ip_adapter_path,
            subfolder="sdxl_models",
            weight_name=self._model.weight_name,
        )
        pipe.set_ip_adapter_scale(float(self._model.scale))

        return pipe

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        ip_adapter_image: Path,
        out_path: Path,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> ImageGenerationResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ref_img = Image.open(ip_adapter_image).convert("RGB")
        g = self._torch.Generator(device=self._device).manual_seed(int(seed))

        image = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=ref_img,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=int(width),
            height=int(height),
            generator=g,
        ).images[0]

        image.save(out_path)
        return ImageGenerationResult(image_path=out_path)

