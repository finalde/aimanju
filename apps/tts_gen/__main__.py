from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from libs.common.subprocess_runner import require_executable, run

from .app_config import AppConfig


@dataclass(frozen=True)
class ParsedCue:
    index: int
    text: str


_PAREN_RE = re.compile(r"（[^）]*）")
_LEADING_PUNCT_RE = re.compile(r"^[，。！？、,.!?:;\\s]+")


def parse_script_lines(script: str) -> List[ParsedCue]:
    """
    Parse a multiline script from config:
    - each non-empty line is one cue
    - cue index starts at 1
    """
    lines = [ln.strip() for ln in script.splitlines()]
    cues: List[ParsedCue] = []
    idx = 1
    for ln in lines:
        if not ln:
            continue
        cues.append(ParsedCue(index=idx, text=ln))
        idx += 1
    return cues


def load_cues_from_script_manifest(path: Path) -> List[ParsedCue]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    cues: List[ParsedCue] = []
    for it in items:
        try:
            idx = int(it["index"])
            text = str(it["text"])
        except Exception:
            continue
        cues.append(ParsedCue(index=idx, text=text))
    cues.sort(key=lambda c: c.index)
    return cues


def _normalize_for_tts(text: str) -> str:
    """
    Make SRT dialogue more TTS-friendly:
    - Remove stage directions like （低声）.
    - Remove speaker prefix before the first Chinese colon ： (e.g. 小李：).
    - Normalize some punctuation that may be ignored.
    - Collapse whitespace/newlines.
    """
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned_lines: List[str] = []
    for ln in raw_lines:
        ln = _PAREN_RE.sub("", ln).strip()
        if "：" in ln:
            # Keep content after first speaker label.
            ln = ln.split("：", 1)[1].strip()
        # Normalize ellipsis/dashes to commas for more natural pauses.
        ln = ln.replace("……", "，").replace("——", "，").replace("…", "，")
        ln = _LEADING_PUNCT_RE.sub("", ln).strip()
        if ln:
            cleaned_lines.append(ln)
    return "。".join(cleaned_lines).strip()


def piper_tts(
    *,
    voice_model: Path,
    voice_config: Optional[Path],
    speaker_id: int,
    sample_rate: Optional[int],
    length_scale: float,
    noise_scale: float,
    noise_w_scale: float,
    sentence_silence: float,
    volume: float,
    no_normalize: bool,
    cuda: bool,
    text: str,
    out_wav: Path,
) -> None:
    # When running via `.venv/bin/python`, PATH is not automatically updated like
    # `source .venv/bin/activate` would do. Prefer the `piper` executable next to
    # the current Python interpreter, and fall back to PATH.
    piper_exe = Path(sys.executable).with_name("piper")
    piper_cmd = str(piper_exe) if piper_exe.exists() else require_executable("piper")
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        piper_cmd,
        "--model",
        str(voice_model),
        "--output_file",
        str(out_wav),
        "--length-scale",
        str(float(length_scale)),
        "--noise-scale",
        str(float(noise_scale)),
        "--noise-w-scale",
        str(float(noise_w_scale)),
        "--sentence-silence",
        str(float(sentence_silence)),
        "--volume",
        str(float(volume)),
    ]
    if sample_rate is not None:
        argv += ["--sample_rate", str(int(sample_rate))]
    if voice_config is not None:
        argv += ["--config", str(voice_config)]
    if speaker_id >= 0:
        argv += ["--speaker", str(int(speaker_id))]
    if no_normalize:
        argv += ["--no-normalize"]
    if cuda:
        argv += ["--cuda"]

    # Feed text via stdin to avoid shell quoting/encoding issues.
    run(argv, check=True, stdin_text=text + "\n")


def wav_to_mp3(in_wav: Path, out_mp3: Path) -> None:
    require_executable("ffmpeg")
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "4",
            str(out_mp3),
        ],
        check=True,
    )


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.tts_gen",
        description="Generate per-line WAVs from a script in config.yml (uses external Piper TTS).",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))
    if cfg.engine.type != "piper":
        raise SystemExit(f"Unsupported engine.type: {cfg.engine.type}")

    cues = load_cues_from_script_manifest(Path(cfg.input.script_manifest_json))

    out_dir = Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    voice_model = Path(cfg.engine.voice_model)
    voice_config = Path(cfg.engine.voice_config) if cfg.engine.voice_config.strip() else None
    if voice_config is None:
        # Best-effort autodetect: `<model>.json` and `<model>.onnx.json` are common.
        candidates = [
            Path(str(voice_model) + ".json"),
            Path(str(voice_model) + ".onnx.json"),
        ]
        for cand in candidates:
            if cand.exists():
                voice_config = cand
                break

    manifest_items: List[dict] = []
    for cue in cues:
        raw_text = cue.text
        norm_text = _normalize_for_tts(raw_text)
        if not norm_text:
            continue
        out_wav = out_dir / f"{cue.index:03d}.wav"
        out_mp3 = out_dir / f"{cue.index:03d}.mp3"
        out_txt = out_dir / f"{cue.index:03d}.txt"
        piper_tts(
            voice_model=voice_model,
            voice_config=voice_config,
            speaker_id=int(cfg.engine.speaker_id),
            sample_rate=(int(cfg.audio.sample_rate) if int(cfg.audio.sample_rate) > 0 else None),
            length_scale=float(cfg.engine.length_scale),
            noise_scale=float(cfg.engine.noise_scale),
            noise_w_scale=float(cfg.engine.noise_w_scale),
            sentence_silence=float(cfg.engine.sentence_silence),
            volume=float(cfg.engine.volume),
            no_normalize=bool(cfg.engine.no_normalize),
            cuda=bool(cfg.engine.cuda),
            text=norm_text,
            out_wav=out_wav,
        )
        if cfg.output.write_text_files:
            out_txt.write_text(norm_text + "\n", encoding="utf-8")
        if cfg.output.write_mp3_files:
            wav_to_mp3(out_wav, out_mp3)
        print(f"Saved {out_wav}")
        manifest_items.append(
            {
                "index": cue.index,
                "wav_path": str(out_wav),
                "mp3_path": str(out_mp3) if cfg.output.write_mp3_files else "",
                "text_raw": raw_text,
                "text_normalized": norm_text,
                "voice_model": str(voice_model),
                "voice_config": str(voice_config) if voice_config is not None else "",
            }
        )

    manifest_path = Path(cfg.output.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

