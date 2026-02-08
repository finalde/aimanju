from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from PIL import Image

from .types import ImageGenerationRequest, ImageGenerationResult


Device = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class SdxlDiffusersModel:
    """
    SDXL model reference.

    - If `path` is a `.safetensors`/`.ckpt`, we load via diffusers `from_single_file`.
    - Otherwise, we treat it as a diffusers folder path or a HF repo id string.
    """

    path: str


class SdxlDiffusersImageGenerator:
    """
    Pure infrastructure: take a prompt and produce an image.

    This class intentionally does NOT know what a storyboard is.
    """

    def __init__(self, model: SdxlDiffusersModel, device: Device) -> None:
        # Lazy imports so importing this module doesn't require torch/diffusers until used.
        import torch
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

        self._torch = torch
        self._StableDiffusionXLPipeline = StableDiffusionXLPipeline
        self._StableDiffusionXLImg2ImgPipeline = StableDiffusionXLImg2ImgPipeline

        self._model = model
        self._device: Device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32

        self._txt2img = self._load_txt2img_pipeline()
        self._img2img = None

    def _load_txt2img_pipeline(self):
        model_path = Path(self._model.path)
        if model_path.exists() and model_path.is_file() and model_path.suffix.lower() in {
            ".safetensors",
            ".ckpt",
        }:
            pipe = self._StableDiffusionXLPipeline.from_single_file(
                str(model_path),
                torch_dtype=self._dtype,
            )
        else:
            pipe = self._StableDiffusionXLPipeline.from_pretrained(
                self._model.path,
                torch_dtype=self._dtype,
                use_safetensors=True,
            )
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)
        return pipe

    def _load_img2img_pipeline(self):
        model_path = Path(self._model.path)
        if model_path.exists() and model_path.is_file() and model_path.suffix.lower() in {
            ".safetensors",
            ".ckpt",
        }:
            pipe = self._StableDiffusionXLImg2ImgPipeline.from_single_file(
                str(model_path),
                torch_dtype=self._dtype,
            )
        else:
            pipe = self._StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self._model.path,
                torch_dtype=self._dtype,
                use_safetensors=True,
            )
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=False)
        return pipe

    def generate(self, request: ImageGenerationRequest, out_path: Path) -> ImageGenerationResult:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        generator = self._torch.Generator(device=self._device).manual_seed(int(request.seed))

        image: Image.Image
        if request.reference_image is None:
            image = self._txt2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                width=request.width,
                height=request.height,
                generator=generator,
            ).images[0]
        else:
            if request.strength is None:
                raise ValueError("img2img requires request.strength to be set")

            if self._img2img is None:
                self._img2img = self._load_img2img_pipeline()

            init_image = (
                Image.open(request.reference_image)
                .convert("RGB")
                .resize((request.width, request.height))
            )
            image = self._img2img(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                image=init_image,
                strength=float(request.strength),
                num_inference_steps=request.steps,
                guidance_scale=request.guidance,
                generator=generator,
            ).images[0]

        image.save(out_path)
        return ImageGenerationResult(image_path=out_path)

