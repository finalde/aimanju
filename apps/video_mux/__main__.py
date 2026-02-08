from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from libs.common.subprocess_runner import require_executable, run

from .app_config import AppConfig


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def _repo_root() -> Path:
    # .../aimanju/apps/video_mux/__main__.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def _resolve_from_repo_root(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (_repo_root() / pp).resolve()


def _escape_ffmpeg_subtitles_path(path: Path) -> str:
    """
    Escape for ffmpeg subtitles filter argument.
    We keep it minimal for typical Linux paths.
    """
    s = str(path)
    s = s.replace("\\", "\\\\")
    s = s.replace(":", "\\:")
    s = s.replace("'", "\\'")
    return s


def load_tts_mp3_paths(manifest_json: Path) -> List[Path]:
    data = json.loads(manifest_json.read_text(encoding="utf-8"))
    items = data.get("items", [])
    pairs = []
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = int(it.get("index", 0))
        mp3_path = str(it.get("mp3_path", "")).strip()
        if idx <= 0 or not mp3_path:
            continue
        pairs.append((idx, _resolve_from_repo_root(mp3_path)))
    pairs.sort(key=lambda x: x[0])
    return [p for _, p in pairs]


def concat_mp3s_to_voice(mp3_paths: List[Path], out_voice_mp3: Path) -> None:
    require_executable("ffmpeg")
    out_voice_mp3.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg concat demuxer requires a list file.
    list_file = out_voice_mp3.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join([f"file '{p.as_posix()}'" for p in mp3_paths]) + "\n",
        encoding="utf-8",
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(out_voice_mp3),
        ],
        check=True,
    )


def mux(
    *,
    video_mp4: Path,
    voice_mp3: Path,
    subtitles_srt: Path,
    subtitles_mode: str,
    subtitles_force_style: str,
    out_mp4: Path,
    vcodec: str,
    crf: int,
    preset: str,
    acodec: str,
    abitrate: str,
) -> None:
    require_executable("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if subtitles_mode == "none":
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_mp4),
                "-i",
                str(voice_mp3),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                acodec,
                "-b:a",
                abitrate,
                "-shortest",
                str(out_mp4),
            ],
            check=True,
        )
        return

    if subtitles_mode == "soft":
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_mp4),
                "-i",
                str(voice_mp3),
                "-i",
                str(subtitles_srt),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:s:0",
                "-c:v",
                "copy",
                "-c:a",
                acodec,
                "-b:a",
                abitrate,
                "-c:s",
                "mov_text",
                "-shortest",
                str(out_mp4),
            ],
            check=True,
        )
        return

    # burn_in (default): re-encode video to bake subtitles
    vf = f"subtitles={_escape_ffmpeg_subtitles_path(subtitles_srt)}"
    if subtitles_force_style.strip():
        vf = vf + f":force_style='{subtitles_force_style}'"
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_mp4),
            "-i",
            str(voice_mp3),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            vf,
            "-c:v",
            vcodec,
            "-crf",
            str(int(crf)),
            "-preset",
            preset,
            "-c:a",
            acodec,
            "-b:a",
            abitrate,
            "-shortest",
            str(out_mp4),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.video_mux",
        description="Mux video + TTS audio + subtitles into a final MP4.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    video_mp4 = _resolve_from_repo_root(cfg.inputs.video_mp4)
    tts_manifest = _resolve_from_repo_root(cfg.inputs.tts_manifest_json)
    subtitles_srt = _resolve_from_repo_root(cfg.inputs.subtitles_srt)

    out_dir = _resolve_from_repo_root(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_mp3 = _resolve_from_repo_root(cfg.output.voice_mp3)
    out_mp4 = _resolve_from_repo_root(cfg.output.out_mp4)

    mp3s = load_tts_mp3_paths(tts_manifest)
    if not mp3s:
        raise SystemExit(f"No mp3_path entries found in: {tts_manifest}")

    concat_mp3s_to_voice(mp3s, voice_mp3)

    mux(
        video_mp4=video_mp4,
        voice_mp3=voice_mp3,
        subtitles_srt=subtitles_srt,
        subtitles_mode=cfg.subtitles.mode,
        subtitles_force_style=cfg.subtitles.force_style,
        out_mp4=out_mp4,
        vcodec=cfg.encoding.vcodec,
        crf=cfg.encoding.crf,
        preset=cfg.encoding.preset,
        acodec=cfg.encoding.acodec,
        abitrate=cfg.encoding.abitrate,
    )

    print(f"Saved {out_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

