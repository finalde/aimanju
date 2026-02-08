from __future__ import annotations

import math


def srt_timestamp(seconds: float) -> str:
    """
    Format seconds -> SRT timestamp (HH:MM:SS,mmm).
    """
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clamp_positive(value: float, *, min_value: float = 0.001) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return max(float(value), float(min_value))

