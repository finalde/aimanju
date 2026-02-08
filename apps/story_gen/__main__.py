from __future__ import annotations

import argparse
import json
from pathlib import Path

from libs.domain.storyboard.shotlist import ShotList
from libs.infrastructure.image_generation.sdxl_diffusers import (
    SdxlDiffusersImageGenerator,
    SdxlDiffusersModel,
)
from libs.infrastructure.image_generation.types import ImageGenerationRequest

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _load_shotlist(path: Path) -> ShotList:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShotList.model_validate(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.story_gen",
        description="Storyboard-aware wrapper: takes a shotlist and generates frames via image_gen.",
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

    shotlist = _load_shotlist(Path(config.storyboard.shot_json))

    generator = SdxlDiffusersImageGenerator(
        model=SdxlDiffusersModel(path=config.model.path),
        device=config.model.device,  # type: ignore[arg-type]
    )

    ref = config.storyboard.reference_image.strip()
    ref_path = Path(ref) if ref else None

    out_dir = Path(config.storyboard.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for shot in shotlist.shots:
        seed = int(shot.seed) if shot.seed is not None else 0
        neg = shot.negative_prompt or shotlist.global_negative_prompt
        full_prompt = f"{shotlist.global_prompt}, {shot.prompt}"

        req = ImageGenerationRequest(
            prompt=full_prompt,
            negative_prompt=neg,
            width=config.storyboard.width,
            height=config.storyboard.height,
            steps=config.storyboard.steps,
            guidance=config.storyboard.guidance,
            seed=seed,
            reference_image=ref_path,
            strength=config.storyboard.strength if ref_path else None,
        )

        out_path = out_dir / f"{shot.shot_id}.png"
        generator.generate(request=req, out_path=out_path)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

