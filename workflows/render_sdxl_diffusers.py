from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .storyboard_schema import ShotList

app = typer.Typer(add_completion=False)


def _load_shotlist(path: Path) -> ShotList:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShotList.model_validate(data)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def main(
    shot_json: Path = typer.Option(..., exists=True, dir_okay=False, help="shotlist.json"),
    out_dir: Path = typer.Option(..., help="Output frames directory."),
    model: str = typer.Option(
        "models/sdxl-base-1.0",
        help=(
            "Either: (1) a local folder in diffusers format (e.g. models/sdxl-base-1.0), "
            "(2) a single checkpoint file (.safetensors/.ckpt) from Civitai, "
            "or (3) a HF repo id like stabilityai/stable-diffusion-xl-base-1.0"
        ),
    ),
    steps: int = typer.Option(30),
    guidance: float = typer.Option(5.5),
    width: int = typer.Option(832, help="9:16-ish width; SDXL likes multiples of 64."),
    height: int = typer.Option(1472, help="9:16-ish height; SDXL likes multiples of 64."),
    reference_image: Optional[Path] = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Optional: reference image for img2img consistency (e.g. first frame).",
    ),
    strength: float = typer.Option(
        0.65,
        help="img2img strength (0-1). Higher = more change, lower = more consistency.",
    ),
    device: Optional[str] = typer.Option(
        None, help="cuda / cpu. Default: auto-detect."
    ),
) -> None:
    """
    Render storyboard frames using SDXL via diffusers.
    """
    # Lazy imports so base workflow doesn't require torch/diffusers.
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline
    from PIL import Image

    shotlist = _load_shotlist(shot_json)
    _ensure_dir(out_dir)

    use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_device == "cuda" else torch.float32

    model_path = Path(model)
    use_img2img = reference_image is not None

    if use_img2img:
        pipe_cls = StableDiffusionXLImg2ImgPipeline
    else:
        pipe_cls = StableDiffusionXLPipeline

    if model_path.exists() and model_path.is_file() and model_path.suffix.lower() in {
        ".safetensors",
        ".ckpt",
    }:
        # Civitai single-file SDXL checkpoint
        pipe = pipe_cls.from_single_file(str(model_path), torch_dtype=dtype)
    else:
        # Diffusers folder or HF repo id
        pipe = pipe_cls.from_pretrained(model, torch_dtype=dtype, use_safetensors=True)

    pipe = pipe.to(use_device)
    pipe.set_progress_bar_config(disable=False)

    ref_img: Optional[Image.Image] = None
    if use_img2img:
        ref_img = Image.open(reference_image).convert("RGB").resize((width, height))

    for shot in shotlist.shots:
        seed = shot.seed or 0
        g = torch.Generator(device=use_device).manual_seed(int(seed))

        prompt = f"{shotlist.global_prompt}, {shot.prompt}"
        neg = shot.negative_prompt or shotlist.global_negative_prompt

        if use_img2img:
            image: Image.Image = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=ref_img,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g,
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=g,
            ).images[0]

        out_path = out_dir / f"{shot.shot_id}.png"
        image.save(out_path)
        typer.echo(f"Saved {out_path}")


if __name__ == "__main__":
    app()

