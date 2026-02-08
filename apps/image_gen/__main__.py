from __future__ import annotations

import argparse
from pathlib import Path

from libs.infrastructure.image_generation.sdxl_diffusers import (
    SdxlDiffusersImageGenerator,
    SdxlDiffusersModel,
)
from libs.infrastructure.image_generation.types import ImageGenerationRequest

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.image_gen",
        description="Generate a single image from a prompt (infra-level app).",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    config = AppConfig.from_yaml(Path(args.config))

    if config.model.type != "sdxl_diffusers":
        raise SystemExit(f"Unsupported model.type: {config.model.type}")

    generator = SdxlDiffusersImageGenerator(
        model=SdxlDiffusersModel(path=config.model.path),
        device=config.model.device,  # type: ignore[arg-type]
    )

    ref = config.generation.reference_image.strip()
    request = ImageGenerationRequest(
        prompt=config.generation.prompt,
        negative_prompt=config.generation.negative_prompt,
        width=config.generation.width,
        height=config.generation.height,
        steps=config.generation.steps,
        guidance=config.generation.guidance,
        seed=config.generation.seed,
        reference_image=Path(ref) if ref else None,
        strength=config.generation.strength if ref else None,
    )

    result = generator.generate(request=request, out_path=Path(config.generation.out_path))
    print(f"Saved {result.image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

