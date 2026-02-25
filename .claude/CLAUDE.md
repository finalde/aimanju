# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aimanju** is a local, open-source pipeline for converting scripts (剧本) into storyboards (分镜图):

```
Script (Markdown) → Shotlist (JSON/CSV) → Storyboard frames (images) → Video
```

The LLM stack uses Qwen2.5 via Ollama; image generation uses SDXL with optional ControlNet and IP-Adapter for consistency.

## Environment Setup

```bash
# Create virtualenv and install base dependencies
make venv
# or manually:
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/pip install -r requirements.txt

# Additional SDXL/diffusers dependencies (for image generation)
.venv/bin/pip install -r requirements-sdxl.txt
```

Always run Python via `.venv/bin/python` — there is no system-level install expected.

## Common Commands

### Core pipeline workflows
```bash
# Parse script → shotlist JSON/CSV/prompts.md (no LLM required)
.venv/bin/python -m workflows.storyboard_from_script \
  --script scripts/<episode>.md \
  --episode <episode_name>

# With Ollama LLM to auto-fill shot language and prompts
.venv/bin/python -m workflows.storyboard_from_script \
  --script scripts/<episode>.md \
  --episode <episode_name> \
  --llm ollama \
  --ollama-model qwen2.5:7b-instruct

# Render shotlist → batch images (SDXL diffusers)
.venv/bin/python -m workflows.render_sdxl_diffusers \
  --shot-json storyboards/<episode>/shotlist.json \
  --out-dir storyboards/<episode>/frames

# Download SDXL model weights
.venv/bin/python -m workflows.download_models sdxl
```

### Running individual apps
Each app reads its config from `apps/<app_name>/config.yml` by default:
```bash
.venv/bin/python -m apps.image_gen              # single image from prompt
.venv/bin/python -m apps.story_gen              # batch storyboard generation
.venv/bin/python -m apps.assets_gen -c apps/assets_gen/config.yml
.venv/bin/python -m apps.tts_gen
.venv/bin/python -m apps.video_mux
```

### ComfyUI (alternative UI)
```bash
make comfyui                         # Start on port 8188
make comfyui PORT=8189               # Custom port
make comfyui-run WORKFLOW=path/to/workflow_api.json
```

## Architecture

### Layered structure (DDD-inspired)

```
apps/           CLI entry points — each app owns its config.yml and __main__.py
workflows/      Standalone end-to-end pipeline scripts (not app-based)
libs/
  infrastructure/image_generation/   Low-level diffusers/ControlNet/IP-Adapter pipelines
  domain/storyboard/                 Core business models: Shot, ShotList (Pydantic)
  application/                       Orchestration layer (minimal currently)
  common/                            Shared utilities (YAML, timecode, SRT, subprocess)
external/ComfyUI/                    Bundled ComfyUI server with custom nodes
models/                              Local model checkpoints (.safetensors)
```

### Key design conventions

- **Configuration is explicit**: Every app has its own `config.yml`. No hidden defaults in code. All config classes use Pydantic for validation.
- **Infrastructure knows nothing about domain**: `libs/infrastructure/image_generation/` only handles `ImageGenerationRequest → ImageGenerationResult`. It has no knowledge of `ShotList`.
- **Apps are thin wrappers**: They load config, load domain data (shotlist), and call infrastructure. Minimal logic.
- **No test suite**: This is an integration-tested application; correctness is validated by running actual pipeline stages.

### Central domain model

`libs/domain/storyboard/shotlist.py` defines:

- `Shot`: `shot_id`, `prompt` (required), optional `framing`, `camera`, `action`, `dialogue`, `duration_seconds`, `negative_prompt`, `seed`
- `ShotList`: `episode`, `global_prompt`, `global_negative_prompt`, `shots: List[Shot]`

The `global_prompt` is prepended to each shot's prompt at render time.

### Image generation backends

All live in `libs/infrastructure/image_generation/`:

| Module | Use case |
|--------|----------|
| `sdxl_diffusers.py` | Basic txt2img / img2img |
| `sdxl_ip_adapter_diffusers.py` | Style/character consistency via reference image |
| `sdxl_controlnet_diffusers.py` | Pose/depth/canny control |
| `sdxl_ip_adapter_faceid.py` | Face identity consistency |

### Output locations

- `storyboards/<episode>/` — shotlist.json, shotlist.csv, prompts.md
- `outputs/assets/<episode>/` — character and background asset sheets
- `comfyui_workflows/` — ComfyUI workflow JSON definitions (also used as ComfyUI user dir)

## Notes

- Scripts and storyboard content are in Chinese (Simplified). Prompts sent to SDXL are in English.
- SDXL requires a CUDA GPU with ≥16GB VRAM for comfortable use; CPU fallback is slow.
- Ollama must be running locally at `http://localhost:11434` for LLM features.
- ComfyUI custom nodes are in `external/ComfyUI/custom_nodes/` (ControlNet, IP-Adapter, etc.).
- `flash-attn` build requires torch to be present at build time; see `pyproject.toml` for the uv workaround.
