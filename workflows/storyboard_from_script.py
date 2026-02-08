from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import typer

from .storyboard_schema import Shot, ShotList

app = typer.Typer(add_completion=False)


DEFAULT_GLOBAL_PROMPT = (
    "realistic cinematic, real photography look, 35mm lens look, shallow depth of field, "
    "afternoon natural window light, cool gray color grade, soft but contrasty lighting, "
    "subtle film grain, high detail, vertical 9:16 composition"
)

DEFAULT_GLOBAL_NEGATIVE = (
    "cartoon, anime, low resolution, blurry, ugly, deformed hands, extra fingers, "
    "extra limbs, multiple faces, unreadable text, watermark, logo"
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_shots_from_markdown(md: str) -> List[Dict[str, Any]]:
    """
    Parse the simple format used in `scripts/episode_01_thanks_network.md`:
      ### 镜头1（约2秒）— 钩子
      【画面】...
      台词...
    """
    pattern = re.compile(
        r"^###\s*镜头(?P<num>\d+)\s*（(?P<dur>[^）]*)）\s*(?:—\s*(?P<tag>.*))?$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(md))
    shots: List[Dict[str, Any]] = []

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        block = md[start:end].strip()

        num = int(m.group("num"))
        dur_raw = (m.group("dur") or "").strip()
        tag = (m.group("tag") or "").strip()

        # duration parsing: "约2秒" / "5秒" etc.
        dur_seconds: Optional[float] = None
        dur_m = re.search(r"(\d+(?:\.\d+)?)\s*秒", dur_raw)
        if dur_m:
            dur_seconds = float(dur_m.group(1))

        # Extract visual description between 【画面】 and end-of-line(s)
        visual = ""
        v_m = re.search(r"【画面】(.+)", block)
        if v_m:
            visual = v_m.group(1).strip()
        # Extract dialogue lines: everything except 【画面】 lines, keeping short
        dialogue_lines: List[str] = []
        for line in [l.strip() for l in block.splitlines() if l.strip()]:
            if line.startswith("【画面】"):
                continue
            if line.startswith("【字幕") or line.startswith("【字幕/屏幕字】"):
                continue
            dialogue_lines.append(line)
        dialogue = "\n".join(dialogue_lines).strip() or None

        shots.append(
            {
                "shot_id": f"{num:02d}",
                "duration_seconds": dur_seconds,
                "tag": tag,
                "visual": visual,
                "dialogue": dialogue,
            }
        )
    return shots


def _ollama_chat(
    prompt: str,
    model: str,
    base_url: str,
    timeout_seconds: int = 120,
) -> str:
    """
    Calls Ollama's OpenAI-compatible endpoint.
    Requires: `ollama serve` (default port 11434).
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a film storyboard assistant. Output STRICT JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }
    r = requests.post(url, json=payload, timeout=timeout_seconds)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def _build_default_prompts(episode: str, parsed: List[Dict[str, Any]]) -> ShotList:
    # Simple, deterministic prompts (no LLM required).
    # Keep screens textless; user will overlay subtitles later.
    # Keep prompts short: SDXL uses dual CLIP encoders with limited effective token budget.
    # Also, do NOT duplicate global_prompt inside per-shot prompts (renderer prepends global_prompt).
    character = "Asian male office worker, 25yo, short black hair, clean-shaven, white shirt, dark jacket, ID badge"
    scene = "modern open-plan office cubicle, desk with monitor and keyboard, glass partitions, blurred coworkers"

    shots: List[Shot] = []
    for s in parsed:
        num = s["shot_id"]
        visual = s.get("visual") or ""
        tag = s.get("tag") or ""
        summary = tag or visual or f"Shot {num}"

        # Rough framing heuristics per shot id for this script
        framing = {
            "01": "close-up, slight high angle",
            "02": "extreme close-up",
            "03": "medium shot with foreground occlusion",
            "04": "low angle medium close-up",
            "05": "close-up",
            "06": "close-up, dramatic push-in feel",
        }.get(num)

        # Per-shot prompt only (renderer will prepend global_prompt)
        # Avoid Chinese/quoted UI strings in prompts to reduce tokenizer bloat; keep intent concise.
        action_hint = {
            "01": "close-up: he freezes, shocked, staring at computer screen glow",
            "02": "extreme close-up: hand holding smartphone, chat UI blurred, trembling fingers",
            "03": "medium shot: he looks up pale, sweaty; coworkers as blurred foreground silhouettes",
            "04": "low angle: boss appears behind him, looming presence, tense atmosphere",
            "05": "close-up: boss hands him phone, error notification blurred, relief on his face",
            "06": "close-up: screen light hits his face; sudden panic, eyes widened",
        }.get(num, (visual or "office reaction shot"))

        prompt = (
            f"{character}, {scene}, {action_hint}, cinematic lighting, shallow depth of field, "
            "no readable text on screens, no subtitles"
        )

        shots.append(
            Shot(
                shot_id=num,
                duration_seconds=s.get("duration_seconds"),
                summary=summary,
                framing=framing,
                camera=None,
                action=visual or None,
                dialogue=s.get("dialogue"),
                prompt=prompt,
                negative_prompt=DEFAULT_GLOBAL_NEGATIVE,
                seed=1000 + int(num),
            )
        )

    return ShotList(
        episode=episode,
        style="realistic cinematic",
        global_prompt=DEFAULT_GLOBAL_PROMPT,
        global_negative_prompt=DEFAULT_GLOBAL_NEGATIVE,
        shots=shots,
    )


def _enhance_with_llm(
    shotlist: ShotList,
    script_md: str,
    ollama_model: str,
    ollama_base_url: str,
) -> ShotList:
    """
    Optional: use LLM to rewrite per-shot prompts and add camera/framing.
    We keep it safe: request STRICT JSON for shots only.
    """
    req = {
        "episode": shotlist.episode,
        "style": shotlist.style,
        "global_prompt": shotlist.global_prompt,
        "global_negative_prompt": shotlist.global_negative_prompt,
        "shots": [
            {
                "shot_id": s.shot_id,
                "duration_seconds": s.duration_seconds,
                "summary": s.summary,
                "action": s.action,
                "dialogue": s.dialogue,
            }
            for s in shotlist.shots
        ],
        "script": script_md,
    }

    prompt = (
        "Given the script and a draft shotlist, rewrite each shot into a high-quality "
        "text-to-image prompt for realistic cinematic storyboards.\n"
        "Constraints:\n"
        "- Keep the same shot count and shot_id.\n"
        "- Add framing and camera movement hints.\n"
        "- Ensure character consistency: same protagonist across shots.\n"
        "- Avoid generating readable UI/text; say 'no readable text on screens'.\n"
        "- Output STRICT JSON with schema:\n"
        '{ "global_prompt": string, "global_negative_prompt": string, "shots": ['
        '{ "shot_id": string, "framing": string, "camera": string, "prompt": string, '
        '"negative_prompt": string } ] }\n\n'
        f"INPUT:\n{json.dumps(req, ensure_ascii=False)}"
    )

    raw = _ollama_chat(prompt=prompt, model=ollama_model, base_url=ollama_base_url)
    data = json.loads(raw)

    # apply updates
    by_id = {s.shot_id: s for s in shotlist.shots}
    for upd in data.get("shots", []):
        sid = upd.get("shot_id")
        if sid not in by_id:
            continue
        s = by_id[sid]
        s.framing = upd.get("framing") or s.framing
        s.camera = upd.get("camera") or s.camera
        s.prompt = upd.get("prompt") or s.prompt
        s.negative_prompt = upd.get("negative_prompt") or s.negative_prompt

    shotlist.global_prompt = data.get("global_prompt") or shotlist.global_prompt
    shotlist.global_negative_prompt = (
        data.get("global_negative_prompt") or shotlist.global_negative_prompt
    )
    return shotlist


def _write_outputs(out_dir: Path, shotlist: ShotList) -> None:
    _ensure_dir(out_dir)
    json_path = out_dir / "shotlist.json"
    csv_path = out_dir / "shotlist.csv"
    prompts_path = out_dir / "prompts.md"

    json_path.write_text(shotlist.model_dump_json(indent=2), encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "shot_id": s.shot_id,
                "duration_seconds": s.duration_seconds,
                "summary": s.summary,
                "framing": s.framing,
                "camera": s.camera,
                "action": s.action,
                "dialogue": s.dialogue,
                "seed": s.seed,
            }
            for s in shotlist.shots
        ]
    )
    df.to_csv(csv_path, index=False)

    # prompts.md (human-friendly)
    lines: List[str] = []
    lines.append(f"## {shotlist.episode} prompts\n")
    lines.append("### Global prompt\n")
    lines.append(shotlist.global_prompt + "\n")
    lines.append("### Global negative prompt\n")
    lines.append(shotlist.global_negative_prompt + "\n")
    lines.append("---\n")
    for s in shotlist.shots:
        lines.append(f"## Shot {s.shot_id} — {s.summary}\n")
        if s.framing:
            lines.append(f"- framing: {s.framing}\n")
        if s.camera:
            lines.append(f"- camera: {s.camera}\n")
        if s.duration_seconds is not None:
            lines.append(f"- duration_seconds: {s.duration_seconds}\n")
        if s.dialogue:
            lines.append("\nDialogue:\n")
            lines.append("```text\n" + s.dialogue.strip() + "\n```\n")
        lines.append("\nPrompt:\n")
        lines.append("```text\n" + s.prompt.strip() + "\n```\n")
        lines.append("\nNegative prompt:\n")
        lines.append("```text\n" + (s.negative_prompt or "").strip() + "\n```\n")
        lines.append("---\n")

    prompts_path.write_text("".join(lines), encoding="utf-8")


@app.command()
def main(
    script: Path = typer.Option(..., exists=True, dir_okay=False),
    episode: str = typer.Option(..., help="Output episode folder name."),
    out_root: Path = typer.Option(
        Path("storyboards"), help="Root output folder (default: storyboards/)."
    ),
    llm: Optional[str] = typer.Option(
        None, help="Optional: 'ollama' to enhance prompts via local LLM."
    ),
    ollama_model: str = typer.Option(
        "qwen2.5:7b-instruct", help="Ollama model name."
    ),
    ollama_base_url: str = typer.Option(
        "http://localhost:11434", help="Ollama OpenAI-compatible base URL."
    ),
) -> None:
    md = _read_text(script)
    parsed = _parse_shots_from_markdown(md)
    if not parsed:
        raise typer.BadParameter("No shots found. Expect headings like '### 镜头1（约2秒）— ...'")

    shotlist = _build_default_prompts(episode=episode, parsed=parsed)
    if llm == "ollama":
        shotlist = _enhance_with_llm(
            shotlist=shotlist,
            script_md=md,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
        )

    out_dir = out_root / episode
    _write_outputs(out_dir=out_dir, shotlist=shotlist)
    typer.echo(f"Wrote: {out_dir}/shotlist.json, shotlist.csv, prompts.md")


if __name__ == "__main__":
    app()

