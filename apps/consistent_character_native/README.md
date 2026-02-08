# consistent_character_native (FaceID -> ControlNet)

This module **replicates the ComfyUI consistent-character logic** natively:

1) **FaceID embedding** (InsightFace) from a reference image  
2) **IP-Adapter FaceID (SDXL)** to synthesize an identity-locked image  
3) **ControlNet OpenPose img2img** to force the target pose while keeping identity

It does **not** run ComfyUI — this is pure Python in our repo.

References:
- FaceID model card: https://huggingface.co/h94/IP-Adapter-FaceID
- InsightFace for embeddings

## Required models (download yourself)
Place in the paths referenced by `apps/consistent_character_native/config.yml`:

FaceID:
- `ip-adapter-faceid_sdxl.bin` (from `h94/IP-Adapter-FaceID`)

InsightFace:
- `buffalo_l` or `antelopev2` (folder)
  - place under `models/insightface/buffalo_l/...`
  - set `faceid.insightface_home: "models/insightface"`

ControlNet OpenPose (SDXL):
- a compatible SDXL OpenPose ControlNet, e.g. `controlnet-openpose-sdxl-1.0`

SDXL base:
- any SDXL base checkpoint (you already use `sdXL_v10VAEFix.safetensors`)

## Auto-generate OpenPose (optional)
If you don't have a pose image, this module can generate one automatically using
`controlnet-aux` (OpenPoseDetector). This will download a smaller OpenPose model
on first run.

Config:
- `pose.auto_generate: true`
- `pose.auto_from: "identity_image"` (or `"subject_image"`)
- `pose.control_image`: output path

## Run
```bash
.venv/bin/python -m apps.consistent_character_native -c apps/consistent_character_native/config.yml
```

Outputs:
- `output.identity_png`: identity-locked image from FaceID
- `output.posed_png`: pose-controlled image (identity preserved)
