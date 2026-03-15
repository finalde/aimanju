# ComfyUI Manual Model Downloads (Restoration / Face / ControlNet / Upscale)

Paths are relative to your **ComfyUI installation** (e.g. `external/ComfyUI` in this repo).  
So `models/checkpoints` = `{ComfyUI}/models/checkpoints`.

---

## Table: Name → Download URL → Save path

| Name | Download URL | Save path |
|------|---------------|-----------|
| Stable Diffusion (Realistic Vision V5.1) | https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors | `models/checkpoints/` |
| Florence-2 (microsoft/Florence-2-base) | Hugging Face: `microsoft/Florence-2-base` (auto-download) or clone repo | `~/.cache/huggingface/hub/` or extension’s local path |
| GFPGAN (GFPGANv1.4.pth) | https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth | `models/face_restore/` or `models/gfpgan/` or `models/facerestore_models/` |
| ReActor (inswapper_128.onnx) | https://huggingface.co/Aitrepreneur/insightface/resolve/main/inswapper_128.onnx | `models/insightface/models/` |
| ControlNet LineArt | https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth | `models/controlnet/` |
| ControlNet Depth | https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_depth.pth | `models/controlnet/` |
| ControlNet OpenPose | https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_openpose.pth | `models/controlnet/` |
| R-ESRGAN 4x+ (super-resolution) | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth | `models/upscale_models/` |
| ReActor face parsing (parsenet) | https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth | `models/facedetection/` |

---

## Notes

- **Realistic Vision V5.1**: No-VAE version; use a separate VAE node if needed (e.g. `vae-ft-mse-840000.safetensors` in `models/vae/`).
- **Florence-2**: Most nodes load via `transformers`/Hugging Face; first run will download to HF cache. For fully offline use, pre-download the repo and set `HF_HUB_OFFLINE=1` or the path your extension supports.
- **GFPGAN**: If your node expects a different folder (e.g. `gfpgan`, `facerestore_models`), create that folder under `models/` and put `GFPGANv1.4.pth` there.
- **ReActor / InsightFace**: `inswapper_128.onnx` in `models/insightface/models/`. Some workflows also need buffalo_l or antelopev2 models in `models/insightface/models/buffalo_l/` or `models/insightface/models/antelopev2/`; check the ReActor/InstantID node docs.
- **ControlNet**: All three go in `models/controlnet/`. Keep filenames as in the URLs (e.g. `control_sd15_depth.pth`).
- **R-ESRGAN**: Save as `RealESRGAN_x4plus.pth` or `RealESRGAN_x4.pth` in `models/upscale_models/` so the upscale node can list it.
- **ReActor face parsing**: When using ReActor with face restore (GFPGAN/CodeFormer) and “use_parse”, the node may try to download `parsing_parsenet.pth`. If download fails (e.g. no network), place it in `models/facedetection/parsing_parsenet.pth`. Create the folder if missing.

---

## Quick path reference (ComfyUI defaults)

| Folder under `models/` | Used for |
|------------------------|----------|
| `checkpoints/`         | SD checkpoints (.safetensors, .ckpt) |
| `controlnet/`         | ControlNet / T2I-Adapter weights |
| `upscale_models/`     | ESRGAN / Real-ESRGAN etc. |
| `face_restore/` or `gfpgan/` or `facerestore_models/` | GFPGAN, CodeFormer (node-dependent) |
| `insightface/models/`  | inswapper_128.onnx, buffalo_l, antelopev2 |
