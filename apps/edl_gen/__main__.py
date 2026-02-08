from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from libs.domain.storyboard.io import load_shotlist_json

from .app_config import AppConfig


@dataclass(frozen=True)
class EdlClip:
    clip_id: str
    shot_id: str
    image_path: str
    start_seconds: float
    duration_seconds: float
    end_seconds: float
    text: Optional[str]


@dataclass(frozen=True)
class Edl:
    source_shotlist: str
    clips: List[EdlClip]


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.edl_gen",
        description="Generate an EDL (edit decision list) from a shotlist.json.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))
    shotlist_path = Path(cfg.input.shot_json)
    frames_dir = Path(cfg.input.frames_dir)
    ext = cfg.input.frame_ext

    shotlist = load_shotlist_json(shotlist_path)

    clips: List[EdlClip] = []
    t = 0.0
    for idx, shot in enumerate(shotlist.shots, start=1):
        dur = float(
            shot.duration_seconds
            if shot.duration_seconds is not None
            else cfg.timing.default_duration_seconds
        )
        image_path = str(frames_dir / f"{shot.shot_id}{ext}")
        text = shot.dialogue or shot.summary
        clip = EdlClip(
            clip_id=f"{idx:03d}",
            shot_id=shot.shot_id,
            image_path=image_path,
            start_seconds=t,
            duration_seconds=dur,
            end_seconds=t + dur,
            text=text,
        )
        clips.append(clip)
        t += dur

    edl = Edl(source_shotlist=str(shotlist_path), clips=clips)

    out_path = Path(cfg.output.edl_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(edl), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

