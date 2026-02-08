from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from .types import ImageGenerationResult

Device = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class SdxlFaceIdModel:
    base_model: str
    ip_adapter_path: str


class SdxlIpAdapterFaceIdImageGenerator:
    def __init__(self, *, model: SdxlFaceIdModel, device: Device) -> None:
        import torch
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

        self._torch = torch
        self._StableDiffusionXLPipeline = StableDiffusionXLPipeline
        self._DDIMScheduler = DDIMScheduler
        self._IPAdapterFaceIDXL = IPAdapterFaceIDXL

        self._model = model
        self._device: Device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32

        self._pipe = self._load_pipeline()
        self._ip = self._IPAdapterFaceIDXL(self._pipe, self._model.ip_adapter_path, device)

    def _load_pipeline(self):
        scheduler = self._DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        model_path = Path(self._model.base_model)
        if model_path.exists() and model_path.is_file() and model_path.suffix.lower() in {
            ".safetensors",
            ".ckpt",
        }:
            pipe = self._StableDiffusionXLPipeline.from_single_file(
                str(model_path),
                torch_dtype=self._dtype,
                scheduler=scheduler,
                add_watermarker=False,
            )
        else:
            pipe = self._StableDiffusionXLPipeline.from_pretrained(
                self._model.base_model,
                torch_dtype=self._dtype,
                scheduler=scheduler,
                add_watermarker=False,
                use_safetensors=True,
            )

        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)
        return pipe

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        faceid_embeds,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        out_path: Path,
    ) -> ImageGenerationResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        images = self._ip.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            width=int(width),
            height=int(height),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            seed=int(seed),
        )

        image: Image.Image = images[0]
        image.save(out_path)
        return ImageGenerationResult(image_path=out_path)
