from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .app_config import AppConfig


@dataclass(frozen=True)
class Cue:
    index: int
    text: str


def _default_config_path() -> Path:
    return Path(__file__).with_name("config.yml")


def parse_lines(lines: str) -> List[Cue]:
    out: List[Cue] = []
    idx = 1
    for ln in (ln.strip() for ln in lines.splitlines()):
        if not ln:
            continue
        out.append(Cue(index=idx, text=ln))
        idx += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="apps.script_gen",
        description="Export a multiline script to outputs/scripts/* for downstream apps.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=str(_default_config_path()),
        help='Path to YAML config (default: module-local "config.yml").',
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_yaml(Path(args.config))

    out_dir = Path(cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cues = parse_lines(cfg.script.lines)

    items: List[dict] = []
    for cue in cues:
        txt_path = out_dir / f"{cue.index:03d}.txt"
        if cfg.output.write_text_files:
            txt_path.write_text(cue.text + "\n", encoding="utf-8")
        items.append(
            {
                "index": cue.index,
                "text": cue.text,
                "text_path": str(txt_path) if cfg.output.write_text_files else "",
            }
        )

    manifest_path = Path(cfg.output.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"items": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

