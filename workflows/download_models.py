from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import snapshot_download

app = typer.Typer(add_completion=False)


@app.command()
def sdxl(
    out_dir: Path = typer.Option(Path("models/sdxl-base-1.0")),
    repo_id: str = typer.Option("stabilityai/stable-diffusion-xl-base-1.0"),
    revision: Optional[str] = typer.Option(None, help="Optional HF revision."),
) -> None:
    """
    Download SDXL base weights to a local folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {repo_id} -> {out_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        revision=revision,
    )
    typer.echo("Done.")


@app.command()
def note() -> None:
    typer.echo(
        "\n".join(
            [
                "Notes:",
                "- SDXL is a solid open-weights choice for 16GB VRAM.",
                "- FLUX.1-dev often has stricter (typically non-commercial) licensing and heavier VRAM needs.",
                "- If a Hugging Face model is gated, you must `huggingface-cli login` first.",
            ]
        )
    )


if __name__ == "__main__":
    app()

