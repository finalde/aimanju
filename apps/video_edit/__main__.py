from __future__ import annotations

import argparse
from pathlib import Path

from libs.common.subprocess_runner import require_executable, run

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def build_video_from_frames(
    *,
    frames_dir: Path,
    pattern: str,
    start_number: int,
    fps: int,
    out_path: Path,
    codec: str,
    pix_fmt: str,
    crf: int,
    preset: str,
) -> None:
    require_executable("ffmpeg")
    # We run ffmpeg with cwd=frames_dir so `pattern` can be simple (e.g. %02d.png).
    # Therefore the output MUST be an absolute path, otherwise it would be resolved
    # relative to frames_dir and fail to create parent directories.
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Example input: -framerate 6 -start_number 1 -i %02d.png
    argv = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(int(fps)),
        "-start_number",
        str(int(start_number)),
        "-i",
        pattern,
        "-c:v",
        codec,
        "-pix_fmt",
        pix_fmt,
        "-crf",
        str(int(crf)),
        "-preset",
        preset,
        str(out_path),
    ]
    run(argv, cwd=frames_dir, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.video_edit",
        description="Build an MP4 from a sequence of frames using ffmpeg.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    build_video_from_frames(
        frames_dir=Path(cfg.input.frames_dir),
        pattern=cfg.input.pattern,
        start_number=cfg.input.start_number,
        fps=cfg.output.fps,
        out_path=Path(cfg.output.out_path),
        codec=cfg.encoding.codec,
        pix_fmt=cfg.encoding.pix_fmt,
        crf=cfg.encoding.crf,
        preset=cfg.encoding.preset,
    )

    print(f"Saved {cfg.output.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

