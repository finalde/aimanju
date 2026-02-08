from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

from apps.consistent_character_gen.app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _abs_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_repo_root() / pp).resolve()


def _load_workflow(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_overrides(workflow: dict, cfg: AppConfig) -> None:
    workflow["7"]["inputs"]["image"] = str(_abs_path(cfg.inputs.subject_image))
    workflow["94"]["inputs"]["image"] = str(_abs_path(cfg.inputs.keypoints_image))
    workflow["95"]["inputs"]["image"] = str(_abs_path(cfg.inputs.pose_image))

    workflow["9"]["inputs"]["text"] = cfg.prompt.prompt
    workflow["10"]["inputs"]["text"] = cfg.prompt.negative_prompt

    workflow["11"]["inputs"]["seed"] = int(cfg.generation.seed)
    workflow["11"]["inputs"]["steps"] = int(cfg.generation.steps)
    workflow["11"]["inputs"]["cfg"] = float(cfg.generation.cfg)
    workflow["11"]["inputs"]["denoise"] = float(cfg.generation.denoise)

    workflow["29"]["inputs"]["width"] = int(cfg.generation.width)
    workflow["29"]["inputs"]["height"] = int(cfg.generation.height)

    workflow["2"]["inputs"]["weight"] = float(cfg.weights.instantid_weight)
    workflow["74"]["inputs"]["strength"] = float(cfg.weights.controlnet_strength)
    workflow["52"]["inputs"]["weight"] = float(cfg.weights.ipadapter_weight)


def _post_prompt(base_url: str, workflow: dict) -> str:
    resp = requests.post(f"{base_url}/prompt", json={"prompt": workflow}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["prompt_id"]


def _wait_for_result(base_url: str, prompt_id: str, timeout_s: int = 300) -> dict:
    start = time.time()
    while True:
        resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if prompt_id in data and data[prompt_id].get("outputs"):
            return data[prompt_id]
        if time.time() - start > timeout_s:
            raise SystemExit("ComfyUI timeout: no output produced.")
        time.sleep(1.5)


def _download_outputs(base_url: str, outputs: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for node_id, node_out in outputs.items():
        images = node_out.get("images") or []
        for img in images:
            filename = img["filename"]
            subfolder = img.get("subfolder", "")
            img_type = img.get("type", "output")
            url = f"{base_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            out_path = out_dir / filename
            out_path.write_bytes(resp.content)
            saved.append(out_path)
    return saved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.consistent_character_gen",
        description="Run ComfyUI InstantID workflow for consistent characters.",
    )
    parser.add_argument("-c", "--config", default=str(_default_config_path()))
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    workflow_path = _abs_path(cfg.workflow.workflow_json)
    if not workflow_path.exists():
        raise SystemExit(f"Missing workflow JSON: {workflow_path}")

    for p in (cfg.inputs.subject_image, cfg.inputs.pose_image, cfg.inputs.keypoints_image):
        pp = _abs_path(p)
        if not pp.exists():
            raise SystemExit(f"Missing input image: {pp}")

    workflow = _load_workflow(workflow_path)
    _apply_overrides(workflow, cfg)

    prompt_id = _post_prompt(cfg.server.base_url, workflow)
    result = _wait_for_result(cfg.server.base_url, prompt_id)

    out_dir = _abs_path(cfg.output.out_dir)
    saved = _download_outputs(cfg.server.base_url, result["outputs"], out_dir)
    for p in saved:
        print(f"Saved {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
