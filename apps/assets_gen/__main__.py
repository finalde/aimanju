from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from apps.assets_gen.app_config import AppConfig, ViewConfig
from libs.infrastructure.image_generation.sdxl_diffusers import SdxlDiffusersImageGenerator, SdxlDiffusersModel
from libs.infrastructure.image_generation.sdxl_controlnet_diffusers import (
    SdxlControlNetConfig,
    SdxlControlNetImageGenerator,
    SdxlIpAdapterConfig,
)
from libs.infrastructure.image_generation.sdxl_ip_adapter_diffusers import (
    SdxlIpAdapterImageGenerator,
    SdxlIpAdapterModel,
)


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _repo_root() -> Path:
    # apps/assets_gen/__main__.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _abs_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_repo_root() / pp).resolve()


def _default_view_suffix(view_id: str) -> str:
    v = view_id.strip().lower()
    if v in {"front", "frontal"}:
        return "frontal view, facing camera"
    if v in {"side", "profile", "left"}:
        return "profile view, side view, facing left"
    if v in {"back", "rear"}:
        return "back view, facing away from camera"
    return ""


def _square_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(size, h, w)
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


def _make_face_reference(
    *,
    src_path: Path,
    out_path: Path,
    output_size: int,
    min_face_size: int,
    scale: float,
) -> None:
    img = cv2.imread(str(src_path))
    if img is None:
        raise SystemExit(f"Failed to read image for face crop: {src_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w / 2.0, y + h / 2.0
        size = int(round(max(w, h) * float(scale)))
        x0 = int(round(cx - size / 2.0))
        y0 = int(round(cy - size / 2.0))
        x1 = x0 + size
        y1 = y0 + size
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.shape[1], x1)
        y1 = min(img.shape[0], y1)
        crop = img[y0:y1, x0:x1]
    else:
        crop = _square_center_crop(img, output_size)

    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.assets_gen",
        description="Generate character/background reference assets (ControlNet + optional IP-Adapter).",
    )
    parser.add_argument("-c", "--config", default=str(_default_config_path()))
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    out_dir = _abs_path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # IP-Adapter (optional)
    ip_cfg = None
    ip_ref_img = None
    if cfg.ip_adapter.enabled:
        ip_cfg = SdxlIpAdapterConfig(
            path=str(_abs_path(cfg.ip_adapter.path)),
            weight_name=cfg.ip_adapter.weight_name,
            scale=float(cfg.ip_adapter.scale),
        )
        if cfg.ip_adapter.reference_image.strip():
            ip_ref_img = _abs_path(cfg.ip_adapter.reference_image)
            if not ip_ref_img.exists():
                raise SystemExit(f"Missing ip_adapter.reference_image: {ip_ref_img}")

    manifest: dict = {"characters": [], "backgrounds": []}

    # Characters
    for ch in cfg.characters:
        ch_dir = out_dir / "characters" / ch.id
        ch_dir.mkdir(parents=True, exist_ok=True)

        img2img_gen = None
        img2img_init = None
        img2img_default_ref = None
        if ch.mode == "img2img":
            if not cfg.ip_adapter.reference_image.strip():
                raise SystemExit('mode=img2img requires ip_adapter.reference_image to be set (identity image)')
            img2img_default_ref = _abs_path(cfg.ip_adapter.reference_image)
            if not img2img_default_ref.exists():
                raise SystemExit(f"Missing identity image: {img2img_default_ref}")
            img2img_gen = SdxlDiffusersImageGenerator(
                model=SdxlDiffusersModel(path=cfg.base.sdxl_base),
                device=cfg.base.device,
            )

        def resolve_img2img_ref(
            view: ViewConfig, front_path: Path | None, front_face: Path | None
        ) -> Path:
            raw = view.reference_image.strip()
            if raw == "generated:front":
                if front_path is None or not front_path.exists():
                    raise SystemExit("reference_image=generated:front requires front view to be generated first")
                return front_path
            if raw == "generated:front_face":
                if front_face is None or not front_face.exists():
                    raise SystemExit("reference_image=generated:front_face requires a face ref to be generated first")
                return front_face
            if raw:
                p = _abs_path(raw)
                if not p.exists():
                    raise SystemExit(f"Missing reference image: {p}")
                return p
            if img2img_default_ref is not None:
                return img2img_default_ref
            raise SystemExit("mode=img2img requires a reference_image per view or ip_adapter.reference_image")

        front_out_path = ch_dir / "front.png"
        front_face_path = ch_dir / "front_face.png"
        if ch.mode == "img2img":
            primary_views = [v for v in ch.views if v.reference_image.strip() != "generated:front"]
            dependent_views = [v for v in ch.views if v.reference_image.strip() == "generated:front"]
        else:
            primary_views = list(ch.views)
            dependent_views = []

        for view in primary_views:
            suffix = view.prompt_suffix.strip() if view.prompt_suffix.strip() else _default_view_suffix(view.id)
            prompt = (
                f"character reference, {ch.description}, {view.id} view, "
                f"consistent identity, clean background, high detail, no subtitles, no text"
            )
            if suffix:
                prompt = prompt + f", {suffix}"
            out_path = ch_dir / f"{view.id}.png"

            if ch.mode == "controlnet":
                if ch.controlnet is None:
                    raise SystemExit(f"Character {ch.id} mode=controlnet requires controlnet config")
                if not view.control_image.strip():
                    raise SystemExit(f"Character {ch.id} view {view.id} missing control_image")
                ctrl = _abs_path(view.control_image)
                if not ctrl.exists():
                    raise SystemExit(f"Missing control image: {ctrl}")

                gen = SdxlControlNetImageGenerator(
                    base_model=cfg.base.sdxl_base,
                    controlnet=SdxlControlNetConfig(path=str(_abs_path(ch.controlnet.path))),
                    device=cfg.base.device,  # "cuda"/"cpu"
                    ip_adapter=ip_cfg,
                )
                gen.generate(
                    prompt=prompt,
                    negative_prompt=cfg.generation.negative_prompt,
                    control_image=ctrl,
                    out_path=out_path,
                    width=cfg.generation.width,
                    height=cfg.generation.height,
                    steps=cfg.generation.steps,
                    guidance=cfg.generation.guidance,
                    seed=view.seed,
                    ip_adapter_image=ip_ref_img,
                    controlnet_conditioning_scale=float(ch.controlnet.conditioning_scale),
                )
            else:
                if ch.mode == "ip_adapter_only":
                    # ip_adapter_only
                    if ip_cfg is None or ip_ref_img is None:
                        raise SystemExit(
                            "mode=ip_adapter_only requires ip_adapter.enabled=true and ip_adapter.reference_image to be set"
                        )
                    gen = SdxlIpAdapterImageGenerator(
                        SdxlIpAdapterModel(
                            base_model=cfg.base.sdxl_base,
                            ip_adapter_path=str(_abs_path(cfg.ip_adapter.path)),
                            weight_name=cfg.ip_adapter.weight_name,
                            scale=float(cfg.ip_adapter.scale),
                        ),
                        device=cfg.base.device,
                    )
                    gen.generate(
                        prompt=prompt,
                        negative_prompt=cfg.generation.negative_prompt,
                        ip_adapter_image=ip_ref_img,
                        out_path=out_path,
                        width=cfg.generation.width,
                        height=cfg.generation.height,
                        steps=cfg.generation.steps,
                        guidance=cfg.generation.guidance,
                        seed=view.seed,
                    )
                else:
                    # img2img: use identity image as init image, low strength for consistency
                    from libs.infrastructure.image_generation.types import ImageGenerationRequest

                    assert img2img_gen is not None
                    img2img_init = resolve_img2img_ref(
                        view,
                        front_out_path if front_out_path.exists() else None,
                        front_face_path if front_face_path.exists() else None,
                    )
                    strength = (
                        float(view.strength)
                        if view.strength is not None
                        else float(cfg.generation.img2img_strength)
                    )
                    if not (0.0 <= strength <= 1.0):
                        raise SystemExit(f"Invalid img2img strength for {ch.id}/{view.id}: {strength}")

                    img2img_gen.generate(
                        ImageGenerationRequest(
                            prompt=prompt,
                            negative_prompt=cfg.generation.negative_prompt,
                            width=cfg.generation.width,
                            height=cfg.generation.height,
                            steps=cfg.generation.steps,
                            guidance=cfg.generation.guidance,
                            seed=int(view.seed),
                            reference_image=img2img_init,
                            strength=strength,
                        ),
                        out_path,
                    )
                    if view.id == "front" and cfg.face_crop.enabled:
                        _make_face_reference(
                            src_path=out_path,
                            out_path=front_face_path,
                            output_size=int(cfg.face_crop.output_size),
                            min_face_size=int(cfg.face_crop.min_face_size),
                            scale=float(cfg.face_crop.scale),
                        )

            manifest["characters"].append(
                {"id": ch.id, "name": ch.name, "view": view.id, "path": str(out_path)}
            )
            print(f"Saved {out_path}")

        for view in dependent_views:
            suffix = view.prompt_suffix.strip() if view.prompt_suffix.strip() else _default_view_suffix(view.id)
            prompt = (
                f"character reference, {ch.description}, {view.id} view, "
                f"consistent identity, clean background, high detail, no subtitles, no text"
            )
            if suffix:
                prompt = prompt + f", {suffix}"
            out_path = ch_dir / f"{view.id}.png"

            if ch.mode == "img2img":
                from libs.infrastructure.image_generation.types import ImageGenerationRequest

                assert img2img_gen is not None
                img2img_init = resolve_img2img_ref(
                    view,
                    front_out_path if front_out_path.exists() else None,
                    front_face_path if front_face_path.exists() else None,
                )
                strength = (
                    float(view.strength)
                    if view.strength is not None
                    else float(cfg.generation.img2img_strength)
                )
                if not (0.0 <= strength <= 1.0):
                    raise SystemExit(f"Invalid img2img strength for {ch.id}/{view.id}: {strength}")

                img2img_gen.generate(
                    ImageGenerationRequest(
                        prompt=prompt,
                        negative_prompt=cfg.generation.negative_prompt,
                        width=cfg.generation.width,
                        height=cfg.generation.height,
                        steps=cfg.generation.steps,
                        guidance=cfg.generation.guidance,
                        seed=int(view.seed),
                        reference_image=img2img_init,
                        strength=strength,
                    ),
                    out_path,
                )
                manifest["characters"].append(
                    {"id": ch.id, "name": ch.name, "view": view.id, "path": str(out_path)}
                )
                print(f"Saved {out_path}")

    # Backgrounds
    for bg in cfg.backgrounds:
        gen = SdxlControlNetImageGenerator(
            base_model=cfg.base.sdxl_base,
            controlnet=SdxlControlNetConfig(path=str(_abs_path(bg.controlnet.path))),
            device=cfg.base.device,
            ip_adapter=None,  # backgrounds usually don't need IP-Adapter
        )

        bg_dir = out_dir / "backgrounds" / bg.id
        bg_dir.mkdir(parents=True, exist_ok=True)

        for view in bg.views:
            ctrl = _abs_path(view.control_image)
            if not ctrl.exists():
                raise SystemExit(f"Missing control image: {ctrl}")

            prompt = (
                f"environment reference, {bg.description}, {view.id} view, "
                f"consistent layout, high detail, no text"
            )
            out_path = bg_dir / f"{view.id}.png"

            gen.generate(
                prompt=prompt,
                negative_prompt=cfg.generation.negative_prompt,
                control_image=ctrl,
                out_path=out_path,
                width=cfg.generation.width,
                height=cfg.generation.height,
                steps=cfg.generation.steps,
                guidance=cfg.generation.guidance,
                seed=view.seed,
                ip_adapter_image=None,
                controlnet_conditioning_scale=float(bg.controlnet.conditioning_scale),
            )

            manifest["backgrounds"].append(
                {"id": bg.id, "name": bg.name, "view": view.id, "path": str(out_path)}
            )
            print(f"Saved {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

