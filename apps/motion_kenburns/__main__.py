from __future__ import annotations

import argparse
from pathlib import Path

from libs.common.subprocess_runner import require_executable, run

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.motion_kenburns",
        description="Turn still frames into a video with simple Ken Burns motion (ffmpeg zoompan).",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)
    cfg = AppConfig.from_yaml(Path(args.config))

    require_executable("ffmpeg")

    frames_dir = Path(cfg.input.frames_dir)
    out_path = Path(cfg.output.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = int(round(float(cfg.motion.seconds_per_image) * int(cfg.motion.fps)))
    if d <= 0:
        raise SystemExit("seconds_per_image too small for given fps")

    # Feed images at 1 fps, then zoompan duplicates each input image for d frames.
    vf = (
        f"zoompan=z='min(zoom+{cfg.motion.zoom_per_frame},{cfg.motion.max_zoom})'"
        f":d={d}:s={cfg.motion.width}x{cfg.motion.height}:fps={cfg.motion.fps}"
    )

    argv_ffmpeg = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        "1",
        "-start_number",
        str(int(cfg.input.start_number)),
        "-i",
        cfg.input.pattern,
        "-vf",
        vf,
        "-c:v",
        cfg.encoding.codec,
        "-pix_fmt",
        cfg.encoding.pix_fmt,
        "-crf",
        str(int(cfg.encoding.crf)),
        "-preset",
        cfg.encoding.preset,
        str(out_path),
    ]

    # Run in frames dir so input pattern is simple (e.g. %02d.png)
    run(argv_ffmpeg, cwd=frames_dir, check=True)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

