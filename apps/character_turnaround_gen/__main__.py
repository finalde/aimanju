from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

import numpy as np
import cv2

from apps.character_turnaround_gen.app_config import AppConfig
from libs.infrastructure.image_generation.sdxl_controlnet_diffusers import (
    SdxlControlNetConfig as InfraControlNetConfig,
    SdxlControlNetImg2ImgGenerator,
)
from libs.infrastructure.image_generation.sdxl_diffusers import (
    SdxlDiffusersImageGenerator,
    SdxlDiffusersModel,
)
from libs.infrastructure.image_generation.types import ImageGenerationRequest


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_repo_root() / pp).resolve()


def _build_turnaround_prompt(desc: str, style: str) -> str:
    # Keep this explicit and short enough; SDXL CLIP has limited effective context.
    return (
        "character turnaround, 4 views in one row with wide spacing, same person, same outfit, full body, neutral pose, "
        "front view; left profile view; BACK view (show back of head and back of clothes); right profile view; "
        "plain gray background, studio, no text, "
        f"{desc}, {style}"
    )

def _build_base_identity_prompt(desc: str, style: str) -> str:
    return (
        "full body character reference, ONE person only, centered, no other people, "
        "neutral standing pose, head to toe in frame, "
        "plain gray background, studio, no text, "
        f"{desc}, {style}"
    )


def _build_view_prompt(desc: str, style: str, suffix: str) -> str:
    suffix = suffix.strip()
    return (
        "same character identity, same outfit, full body, head to toe in frame, neutral standing pose, "
        "plain gray background, studio, no text, "
        f"{suffix}, {desc}, {style}"
    )


def _crop_grid(img: Image.Image, rows: int, cols: int) -> list[Image.Image]:
    w, h = img.size
    cell_w = w // cols
    cell_h = h // rows
    cells: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            x0 = c * cell_w
            y0 = r * cell_h
            x1 = (c + 1) * cell_w if c < cols - 1 else w
            y1 = (r + 1) * cell_h if r < rows - 1 else h
            cells.append(img.crop((x0, y0, x1, y1)))
    return cells


def _make_canny_image(src: Image.Image, low: int, high: int) -> Image.Image:
    arr = np.array(src.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.character_turnaround_gen",
        description="Generate a 2x2 character turnaround sheet and split into 4 views.",
    )
    parser.add_argument("-c", "--config", default=str(_default_config_path()))
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    out_dir = _abs_path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_path = _abs_path(cfg.output.sheet_png)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)

    base_gen = SdxlDiffusersImageGenerator(
        model=SdxlDiffusersModel(path=cfg.model.path),
        device=cfg.model.device,  # type: ignore[arg-type]
    )

    if cfg.strategy.type == "turnaround_sheet":
        prompt = _build_turnaround_prompt(cfg.prompt.character_description, cfg.prompt.style)

        base_gen.generate(
            request=ImageGenerationRequest(
                prompt=prompt,
                negative_prompt=cfg.prompt.negative_prompt,
                width=cfg.generation.width,
                height=cfg.generation.height,
                steps=cfg.generation.steps,
                guidance=cfg.generation.guidance,
                seed=cfg.generation.seed,
                reference_image=None,
                strength=None,
            ),
            out_path=sheet_path,
        )
        print(f"Saved {sheet_path}")

        sheet = Image.open(sheet_path).convert("RGB")
        cells = _crop_grid(sheet, rows=int(cfg.layout.rows), cols=int(cfg.layout.cols))

        def save_cropped_view(key: str, path_str: str) -> None:
            idx = int(cfg.layout.order[key])
            if idx < 0 or idx >= len(cells):
                raise SystemExit(f"Invalid layout.order for {key}: {idx} (cells={len(cells)})")
            outp = _abs_path(path_str)
            outp.parent.mkdir(parents=True, exist_ok=True)
            src = cells[idx].convert("RGB")
            tw, th = int(cfg.output.view_width), int(cfg.output.view_height)
            if tw <= 0 or th <= 0:
                raise SystemExit(f"Invalid output view size: {tw}x{th}")

            # Letterbox: keep aspect ratio, no stretching.
            sw, sh = src.size
            scale = min(tw / sw, th / sh)
            nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
            resized = src.resize((nw, nh), Image.LANCZOS)

            # Background color from top-left pixel (usually plain gray)
            bg_color = src.getpixel((0, 0))
            canvas = Image.new("RGB", (tw, th), bg_color)
            x = (tw - nw) // 2
            y = (th - nh) // 2
            canvas.paste(resized, (x, y))
            canvas.save(outp)
            print(f"Saved {outp}")

        save_cropped_view("front", cfg.output.front_png)
        save_cropped_view("left", cfg.output.left_png)
        save_cropped_view("back", cfg.output.back_png)
        save_cropped_view("right", cfg.output.right_png)
    else:
        # strategy = img2img_views
        base_path = _abs_path(cfg.img2img.base_png)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        base_prompt = _build_base_identity_prompt(cfg.prompt.character_description, cfg.prompt.style)
        base_gen.generate(
            request=ImageGenerationRequest(
                prompt=base_prompt,
                negative_prompt=cfg.prompt.negative_prompt,
                width=cfg.generation.width,
                height=cfg.generation.height,
                steps=cfg.generation.steps,
                guidance=cfg.generation.guidance,
                seed=int(cfg.img2img.base_seed),
                reference_image=None,
                strength=None,
            ),
            out_path=base_path,
        )
        print(f"Saved {base_path}")

        view_path_by_id = {
            "front": cfg.output.front_png,
            "left": cfg.output.left_png,
            "back": cfg.output.back_png,
            "right": cfg.output.right_png,
        }

        control_gen: SdxlControlNetImg2ImgGenerator | None = None
        if cfg.controlnet.enabled:
            control_gen = SdxlControlNetImg2ImgGenerator(
                base_model=cfg.model.path,
                controlnet=InfraControlNetConfig(path=cfg.controlnet.path),
                device=cfg.model.device,  # type: ignore[arg-type]
            )

        control_dir = out_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)

        for v in cfg.views:
            outp = _abs_path(view_path_by_id[v.id])
            prompt = _build_view_prompt(cfg.prompt.character_description, cfg.prompt.style, v.prompt_suffix)
            strength = float(v.strength) if v.strength is not None else float(cfg.img2img.strength_default)
            if not (0.0 <= strength <= 1.0):
                raise SystemExit(f"Invalid strength for {v.id}: {strength}")

            if control_gen is not None:
                pose_path = _abs_path(v.pose_image)
                pose_img = Image.open(pose_path).convert("RGB")
                canny_img = _make_canny_image(
                    pose_img, cfg.controlnet.low_threshold, cfg.controlnet.high_threshold
                )
                control_path = control_dir / f"{v.id}_canny.png"
                canny_img.resize(
                    (int(cfg.generation.width), int(cfg.generation.height)), Image.NEAREST
                ).save(control_path)

                control_gen.generate(
                    prompt=prompt,
                    negative_prompt=cfg.prompt.negative_prompt,
                    control_image=control_path,
                    init_image=base_path,
                    out_path=outp,
                    width=cfg.generation.width,
                    height=cfg.generation.height,
                    steps=cfg.generation.steps,
                    guidance=cfg.generation.guidance,
                    seed=int(v.seed),
                    strength=strength,
                    controlnet_conditioning_scale=cfg.controlnet.conditioning_scale,
                )
            else:
                base_gen.generate(
                    request=ImageGenerationRequest(
                        prompt=prompt,
                        negative_prompt=cfg.prompt.negative_prompt,
                        width=cfg.generation.width,
                        height=cfg.generation.height,
                        steps=cfg.generation.steps,
                        guidance=cfg.generation.guidance,
                        seed=int(v.seed),
                        reference_image=base_path,
                        strength=strength,
                    ),
                    out_path=outp,
                )
            print(f"Saved {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

