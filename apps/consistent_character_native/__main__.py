from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import torch
from PIL import Image
from insightface.app import FaceAnalysis

from apps.consistent_character_native.app_config import AppConfig
from libs.infrastructure.image_generation.sdxl_controlnet_diffusers import (
    SdxlControlNetConfig,
    SdxlControlNetImg2ImgGenerator,
)
from libs.infrastructure.image_generation.sdxl_ip_adapter_faceid import (
    SdxlFaceIdModel,
    SdxlIpAdapterFaceIdImageGenerator,
)


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_repo_root() / pp).resolve()


def _extract_faceid_embedding(image_path: Path, model_name: str, provider: str):
    app = FaceAnalysis(name=model_name, providers=[provider, "CPUExecutionProvider"])
    app.prepare(ctx_id=0 if provider == "CUDAExecutionProvider" else -1, det_size=(640, 640))
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read subject image: {image_path}")
    faces = app.get(img)
    if not faces:
        raise SystemExit(f"No face detected in subject image: {image_path}")
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
    return torch.from_numpy(face.normed_embedding).unsqueeze(0)


def _ensure_openpose_image(src_path: Path, out_path: Path) -> None:
    from controlnet_aux import OpenposeDetector

    img = Image.open(src_path).convert("RGB")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    pose = openpose(img)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pose.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.consistent_character_native",
        description="Native consistent-character pipeline: FaceID -> ControlNet pose.",
    )
    parser.add_argument("-c", "--config", default=str(_default_config_path()))
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    os.environ["INSIGHTFACE_HOME"] = str(_abs_path(cfg.faceid.insightface_home))

    subject_image = _abs_path(cfg.inputs.subject_image)
    if not subject_image.exists():
        raise SystemExit(f"Missing subject_image: {subject_image}")

    faceid_embeds = _extract_faceid_embedding(
        subject_image, cfg.faceid.insightface_model, cfg.faceid.provider
    )

    faceid_gen = SdxlIpAdapterFaceIdImageGenerator(
        model=SdxlFaceIdModel(
            base_model=cfg.model.base_model,
            ip_adapter_path=str(_abs_path(cfg.faceid.ip_adapter_faceid_sdxl)),
        ),
        device=cfg.model.device,
    )

    identity_path = _abs_path(cfg.output.identity_png)
    faceid_gen.generate(
        prompt=cfg.prompt.prompt,
        negative_prompt=cfg.prompt.negative_prompt,
        faceid_embeds=faceid_embeds,
        width=cfg.identity.width,
        height=cfg.identity.height,
        steps=cfg.identity.steps,
        guidance=cfg.identity.guidance,
        seed=cfg.identity.seed,
        out_path=identity_path,
    )
    print(f"Saved {identity_path}")

    pose_ctrl = _abs_path(cfg.pose.control_image)
    if not pose_ctrl.exists():
        if not cfg.pose.auto_generate:
            raise SystemExit(f"Missing pose control image: {pose_ctrl}")
        auto_src = subject_image if cfg.pose.auto_from == "subject_image" else identity_path
        if not auto_src.exists():
            raise SystemExit(f"Auto-generate pose failed; source missing: {auto_src}")
        _ensure_openpose_image(auto_src, pose_ctrl)
        print(f"Saved {pose_ctrl}")

    control_gen = SdxlControlNetImg2ImgGenerator(
        base_model=cfg.model.base_model,
        controlnet=SdxlControlNetConfig(path=str(_abs_path(cfg.pose.controlnet_path))),
        device=cfg.model.device,
    )

    posed_path = _abs_path(cfg.output.posed_png)
    control_gen.generate(
        prompt=cfg.prompt.prompt,
        negative_prompt=cfg.prompt.negative_prompt,
        control_image=pose_ctrl,
        init_image=identity_path,
        out_path=posed_path,
        width=cfg.identity.width,
        height=cfg.identity.height,
        steps=cfg.pose.steps,
        guidance=cfg.pose.guidance,
        seed=cfg.pose.seed,
        strength=cfg.pose.strength,
        controlnet_conditioning_scale=cfg.pose.conditioning_scale,
    )
    print(f"Saved {posed_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
