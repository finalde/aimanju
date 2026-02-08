from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class CommandResult:
    argv: List[str]
    returncode: int
    stdout: str
    stderr: str


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        install_hint = ""
        if name == "ffmpeg":
            install_hint = " On Ubuntu/WSL: `sudo apt-get update && sudo apt-get install -y ffmpeg`."
        elif name == "piper":
            install_hint = " Install Piper TTS and ensure `piper` is in PATH."
        elif name.lower().startswith("rife"):
            install_hint = " Install a RIFE runner (e.g. `rife-ncnn-vulkan`) and ensure it's in PATH."
        raise FileNotFoundError(
            f"Required executable not found in PATH: {name}. "
            f"Please install it and try again.{install_hint}"
        )
    return path


def run(
    argv: Iterable[str],
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    stdin_text: Optional[str] = None,
) -> CommandResult:
    argv_list = list(argv)
    proc = subprocess.run(
        argv_list,
        cwd=str(cwd) if cwd else None,
        env=env,
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result = CommandResult(
        argv=argv_list,
        returncode=int(proc.returncode),
        stdout=proc.stdout,
        stderr=proc.stderr,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"argv: {result.argv}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
    return result

