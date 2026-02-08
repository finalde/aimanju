from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .timecode import srt_timestamp


@dataclass(frozen=True)
class SrtCue:
    index: int
    start_seconds: float
    end_seconds: float
    text: str


def render_srt(cues: Iterable[SrtCue]) -> str:
    out: List[str] = []
    for cue in cues:
        out.append(str(cue.index))
        out.append(f"{srt_timestamp(cue.start_seconds)} --> {srt_timestamp(cue.end_seconds)}")
        out.append(cue.text.strip() if cue.text else "")
        out.append("")  # blank line
    return "\n".join(out).rstrip() + "\n"

