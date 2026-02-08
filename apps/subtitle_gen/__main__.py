from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from libs.common.srt import SrtCue, render_srt
from libs.common.timecode import clamp_positive
from libs.domain.storyboard.io import load_shotlist_json

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.subtitle_gen",
        description="Generate an SRT file from shotlist.json.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))
    shotlist = load_shotlist_json(Path(cfg.input.shot_json))

    cues: List[SrtCue] = []
    t = 0.0

    for i, shot in enumerate(shotlist.shots, start=1):
        text = shot.dialogue
        if not text and cfg.text.include_summary_fallback:
            text = shot.summary
        text = (text or "").strip()

        dur = float(
            shot.duration_seconds
            if shot.duration_seconds is not None
            else cfg.timing.default_duration_seconds
        )
        dur = clamp_positive(dur, min_value=cfg.timing.min_duration_seconds)

        cues.append(SrtCue(index=i, start_seconds=t, end_seconds=t + dur, text=text))
        t += dur

    out_path = Path(cfg.output.srt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_srt(cues), encoding="utf-8")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

